"""Evaluate L2CS-Net checkpoint on Gaze360 test split with full/semi-front/front subsets.

L2CS's bundled Gaze360 dataset filters by angle at test time; this script bypasses that
filter and uses |gyaw| (from the label file) to define subsets.
"""
import os, sys, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from l2cs import L2CS, gazeto3d, angular


SUBSETS = {
    'full': float('inf'),
    'semi-front': math.pi / 2,
    'front': math.pi / 9,
}


class Gaze360TestDS(Dataset):
    def __init__(self, root, label_path, tf):
        self.root = root
        self.tf = tf
        with open(label_path) as f:
            f.readline()
            self.lines = [l.strip().split() for l in f if l.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx]
        face = parts[0]
        gaze2d = parts[5]
        yaw, pitch = (float(v) for v in gaze2d.split(','))
        img = Image.open(os.path.join(self.root, face)).convert('RGB')
        img = self.tf(img)
        # L2CS code names them "pitch","yaw" but treats label[0] as the first axis;
        # we hand it the same convention they trained on.
        cont_label = torch.tensor([yaw * 180 / np.pi, pitch * 180 / np.pi], dtype=torch.float32)
        return img, cont_label, float(yaw)


def build_model(arch='ResNet50', bins=90):
    if arch == 'ResNet18':
        return L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        return L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    return L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/path/to/gaze360')
    ap.add_argument('--test-label', default='/path/to/gaze360/test.label')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='./work_dir/L2CS_gaze360/eval_test_subsets.json')
    ap.add_argument('--arch', default='ResNet50')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = 0

    tf = transforms.Compose([
        transforms.Resize(448), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = Gaze360TestDS(args.data_root, args.test_label, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    model = build_model(args.arch, 90)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.cuda(device).eval()

    softmax = nn.Softmax(dim=1)
    idx_tensor = torch.FloatTensor(list(range(90))).cuda(device)

    errors = np.zeros(len(ds), dtype=np.float64)
    gyaws = np.zeros(len(ds), dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for images, cont_labels, gyaw in loader:
            images = images.cuda(device)
            lp = cont_labels[:, 0].float() * np.pi / 180  # = yaw_rad
            ly = cont_labels[:, 1].float() * np.pi / 180  # = pitch_rad
            pitch, yaw = model(images)
            pp = torch.sum(softmax(pitch) * idx_tensor, 1).cpu() * 4 - 180
            yp = torch.sum(softmax(yaw) * idx_tensor, 1).cpu() * 4 - 180
            pp = pp * np.pi / 180
            yp = yp * np.pi / 180
            for p, y, pl, yl, gy in zip(pp, yp, lp, ly, gyaw):
                errors[cursor] = angular(gazeto3d([p, y]), gazeto3d([pl, yl]))
                gyaws[cursor] = gy.item()
                cursor += 1
    assert cursor == len(ds)

    abs_yaw = np.abs(gyaws)
    out = {'checkpoint': args.ckpt, 'n_total': int(len(ds)), 'subsets': {}}
    for name, thresh in SUBSETS.items():
        mask = abs_yaw <= thresh
        e = errors[mask]
        out['subsets'][name] = {
            'n': int(mask.sum()),
            'angular_error_mean_deg': round(float(e.mean()), 2),
            'angular_error_std_deg': round(float(e.std()), 2),
        }
        print(f'  {name:12s}: {e.mean():.2f}° ± {e.std():.2f}°  (N={int(mask.sum())})')

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
