"""Evaluate GazeTR checkpoint on Gaze360 test split with full/semi-front/front subsets.

Saves work_dir/gaze360/eval_test_subsets.json with mean ± std per subset.
"""
import os, sys, json, argparse, math
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import model
import gtools

SUBSETS = {  # |gyaw| ≤ threshold (radians)
    'full': float('inf'),
    'semi-front': math.pi / 2,
    'front': math.pi / 9,
}


class Gaze360TestDS(Dataset):
    """Returns face image, label (yaw,pitch) tensor, gyaw_rad for filtering."""

    def __init__(self, root, label_path):
        self.root = root
        with open(label_path) as f:
            f.readline()  # header
            self.lines = [l.strip().split() for l in f if l.strip()]
        self.tf = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx]
        face = parts[0]
        gaze2d = parts[5]
        yaw, pitch = (float(v) for v in gaze2d.split(','))
        img = cv2.imread(os.path.join(self.root, face))
        img = self.tf(img)
        return img, torch.tensor([yaw, pitch], dtype=torch.float32), float(yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/path/to/gaze360')
    ap.add_argument('--test-label', default='/path/to/gaze360/test.label')
    ap.add_argument('--ckpt', default='./work_dir/gaze360/checkpoint/Iter_60_trans6.pt')
    ap.add_argument('--out', default='./work_dir/gaze360/eval_test_subsets.json')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=128)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = 0

    net = model.Model().cuda(device)
    sd = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=False)
    net.load_state_dict(sd); net.eval()

    ds = Gaze360TestDS(args.data_root, args.test_label)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    errors = np.zeros(len(ds), dtype=np.float64)
    gyaws = np.zeros(len(ds), dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for img, label, gyaw in loader:
            data = {'face': img.cuda(device), 'name': ['']*img.size(0)}
            pred = net(data).cpu().numpy()
            gt = label.numpy()
            for k in range(pred.shape[0]):
                errors[cursor] = gtools.angular(gtools.gazeto3d(pred[k]),
                                                gtools.gazeto3d(gt[k]))
                gyaws[cursor] = gyaw[k].item()
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
