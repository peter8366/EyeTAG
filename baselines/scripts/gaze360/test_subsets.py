"""Evaluate gaze360 (erkil1452 LSTM) checkpoint on test split with full/semi-front/front subsets."""
import os, sys, json, math, argparse
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
from model import GazeLSTM, build_model


SUBSETS = {
    'full': float('inf'),
    'semi-front': math.pi / 2,
    'front': math.pi / 9,
}


def parse_label_file(label_path):
    rows = []
    with open(label_path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            face_p, _l, _r, _o, gaze3d, gaze2d = parts[:6]
            face_parts = face_p.split('/')
            sid = face_parts[2]
            idx = int(face_parts[3].split('_')[1].split('.')[0])
            gx, gy, gz = (float(v) for v in gaze3d.split(','))
            gyaw, gpitch = (float(v) for v in gaze2d.split(','))
            rows.append({'sid': sid, 'idx': idx, 'face': face_p,
                         'gx': gx, 'gy': gy, 'gz': gz, 'gyaw': gyaw})
    return rows


class Gaze360SeqTestDS(Dataset):
    T = 7

    def __init__(self, data_root, label_path, tf):
        self.root = data_root
        self.tf = tf
        rows = parse_label_file(label_path)
        tracks = defaultdict(list)
        for r in rows:
            tracks[r['sid']].append(r)
        for sid in tracks:
            tracks[sid].sort(key=lambda x: x['idx'])
        self.tracks = tracks
        self.samples = []
        for sid, tr in tracks.items():
            for i in range(len(tr)):
                self.samples.append((sid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, target_pos = self.samples[idx]
        tr = self.tracks[sid]
        n = len(tr)
        frames = torch.empty(self.T, 3, 224, 224)
        for k, off in enumerate(range(-3, 4)):
            p = max(0, min(n - 1, target_pos + off))
            img = Image.open(os.path.join(self.root, tr[p]['face'])).convert('RGB')
            frames[k] = self.tf(img)
        frames = frames.view(self.T * 3, 224, 224)
        r = tr[target_pos]
        g = torch.tensor([r['gx'], r['gy'], r['gz']], dtype=torch.float32)
        g = g / (g.norm() + 1e-8)
        yaw_sph = math.atan2(g[0].item(), -g[2].item())
        pitch_sph = math.asin(g[1].item())
        return frames, torch.tensor([yaw_sph, pitch_sph], dtype=torch.float32), float(r['gyaw'])


def spherical2cartesian(x):
    out = torch.zeros(x.size(0), 3, device=x.device)
    out[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    out[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    out[:, 1] = torch.sin(x[:, 1])
    return out


def angular_errors(pred, gt):
    p = spherical2cartesian(pred)
    g = spherical2cartesian(gt)
    cos = (p * g).sum(dim=1).clamp(-0.99999, 0.99999)
    return torch.acos(cos) * 180.0 / math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/path/to/gaze360')
    ap.add_argument('--test-label', default='/path/to/gaze360/test.label')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='./work_dir/gaze360_lstm/eval_test_subsets.json')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=80)
    ap.add_argument('--backbone', default='r18', choices=['r18', 'vggface2-r50'])
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = 0

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    ds = Gaze360SeqTestDS(args.data_root, args.test_label, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)

    model = build_model(args.backbone).cuda(device)
    state = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state); model.eval()

    errors = np.zeros(len(ds), dtype=np.float64)
    gyaws = np.zeros(len(ds), dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for src, target, gyaw in loader:
            src = src.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            output, _ = model(src)
            err = angular_errors(output, target).cpu().numpy()
            n = err.shape[0]
            errors[cursor:cursor+n] = err
            gyaws[cursor:cursor+n] = gyaw.numpy()
            cursor += n
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
