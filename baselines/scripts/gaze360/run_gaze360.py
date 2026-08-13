"""End-to-end train + test for erkil1452/gaze360 on the EyeTag-processed Gaze360.

Original code expects raw `imgs/rec_001/head/000000/001620.jpg` paths with 7-frame
windows. The local dataset is the GazeHub-processed version (train/Face/{sid}/{sid}_{idx}.jpg
crops + train.label / test.label). This driver writes a small dataset adapter and runs
GazeLSTM + PinBall training, then evaluates on test split with mean ± std angular error.
"""
import os
import sys
import json
import math
import time
import random
import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
from model import GazeLSTM, PinBallLoss, build_model


# ────────────────────────────────────────────────────────────────────────────
# Dataset adapter — uses GazeHub-format label & Face crops with 7-frame windows.
# ────────────────────────────────────────────────────────────────────────────

def parse_label_file(label_path):
    rows = []
    with open(label_path, 'r') as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            face_p, _l, _r, origin_p, gaze3d, _gaze2d = parts[:6]
            face_parts = face_p.split('/')
            sid = face_parts[2]
            idx = int(face_parts[3].split('_')[1].split('.')[0])
            gx, gy, gz = (float(v) for v in gaze3d.split(','))
            rows.append({'sid': sid, 'idx': idx, 'face': face_p,
                         'gx': gx, 'gy': gy, 'gz': gz})
    return rows


class Gaze360SeqDataset(data.Dataset):
    """7-frame sequences, returns (T*3, 224, 224) tensor + spherical (yaw, pitch)."""

    T = 7

    def __init__(self, data_root, label_path, transform, train=True):
        self.root = data_root
        self.transform = transform
        self.train = train
        rows = parse_label_file(label_path)

        tracks = defaultdict(list)
        for r in rows:
            tracks[r['sid']].append(r)
        for sid in tracks:
            tracks[sid].sort(key=lambda x: x['idx'])
        # map (sid, idx) to position in sorted list for fast lookup
        self.tracks = tracks
        self.pos_in_track = {}
        for sid, tr in tracks.items():
            for i, r in enumerate(tr):
                self.pos_in_track[(sid, r['idx'])] = i

        self.samples = []  # (sid, target_pos)
        for sid, tr in tracks.items():
            for i in range(len(tr)):
                self.samples.append((sid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, target_pos = self.samples[idx]
        track = self.tracks[sid]
        n = len(track)
        # 7-frame window centered at target_pos: positions [-3..+3]
        positions = []
        for off in range(-3, 4):
            p = max(0, min(n - 1, target_pos + off))
            positions.append(p)
        frames = torch.empty(self.T, 3, 224, 224)
        for k, p in enumerate(positions):
            img = Image.open(os.path.join(self.root, track[p]['face'])).convert('RGB')
            frames[k] = self.transform(img)
        frames = frames.view(self.T * 3, 224, 224)

        # spherical(yaw, pitch) from 3D unit vector (gaze360 convention)
        r = track[target_pos]
        g = torch.tensor([r['gx'], r['gy'], r['gz']], dtype=torch.float32)
        g = g / (g.norm() + 1e-8)
        yaw = math.atan2(g[0].item(), -g[2].item())
        pitch = math.asin(g[1].item())
        return frames, torch.tensor([yaw, pitch], dtype=torch.float32)


# ────────────────────────────────────────────────────────────────────────────
# Training / eval
# ────────────────────────────────────────────────────────────────────────────

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


def setup_logger(workdir):
    os.makedirs(workdir, exist_ok=True)
    log = logging.getLogger('gaze360')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(workdir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def train(args, log, device):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
        transforms.ToTensor(), norm,
    ])
    train_ds = Gaze360SeqDataset(args.data_root, args.train_label, train_tf, train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    log.info(f'Train samples: {len(train_ds)}, batches/epoch: {len(train_loader)}')

    model = build_model(args.backbone).cuda(device)
    criterion = PinBallLoss().cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_err = math.inf
    last_ckpt = None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = sum_ang = nb = 0
        for i, (src, target) in enumerate(train_loader):
            src = src.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            output, var = model(src)
            loss = criterion(output, target, var)
            ang = angular_errors(output.detach(), target).mean().item()
            sum_loss += loss.item(); sum_ang += ang; nb += 1
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if (i + 1) % 100 == 0:
                log.info(f'  [{epoch+1}/{args.epochs}][{i+1}/{len(train_loader)}] '
                         f'loss={sum_loss/nb:.4f} ang={sum_ang/nb:.3f}°')
        dt = time.time() - t0
        log.info(f'Epoch {epoch+1} done in {dt:.1f}s — loss={sum_loss/nb:.4f} ang={sum_ang/nb:.3f}°')

        ckpt = os.path.join(args.work_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, ckpt)
        last_ckpt = ckpt

    return last_ckpt


def evaluate(args, log, device, ckpt_path):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    test_ds = Gaze360SeqDataset(args.data_root, args.test_label, test_tf, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    log.info(f'Test samples: {len(test_ds)}')

    model = build_model(args.backbone).cuda(device)
    state = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    # Strip DataParallel 'module.' prefix if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    errs = []
    with torch.no_grad():
        for src, target in test_loader:
            src = src.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            output, _ = model(src)
            errs.extend(angular_errors(output, target).cpu().tolist())

    errs = np.array(errs)
    res = {
        'checkpoint': ckpt_path,
        'n_samples': int(errs.size),
        'angular_error_mean_deg': float(errs.mean()),
        'angular_error_std_deg': float(errs.std()),
    }
    out = os.path.join(args.work_dir, 'eval_test.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    log.info(f'== Test: {res["angular_error_mean_deg"]:.2f}° ± {res["angular_error_std_deg"]:.2f}° '
             f'(N={res["n_samples"]}) ==')
    log.info(f'Saved -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/path/to/gaze360')
    ap.add_argument('--train-label', default='/path/to/gaze360/train.label')
    ap.add_argument('--test-label', default='/path/to/gaze360/test.label')
    ap.add_argument('--work-dir', default='work_dir/gaze360_lstm')
    ap.add_argument('--backbone', default='r18', choices=['r18', 'vggface2-r50'])
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--epochs', default=15, type=int)
    ap.add_argument('--batch-size', default=80, type=int)
    ap.add_argument('--workers', default=12, type=int)
    ap.add_argument('--lr', default=1e-4, type=float)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--test-ckpt', default=None)
    args = ap.parse_args()

    cudnn.benchmark = True
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = 0
    os.makedirs(args.work_dir, exist_ok=True)
    log = setup_logger(args.work_dir)
    log.info(f'args = {vars(args)}')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        items = sorted([f for f in os.listdir(args.work_dir) if f.endswith('.pth')],
                       key=lambda x: int(x.split('epoch')[1].split('.')[0]))
        ckpt = os.path.join(args.work_dir, items[-1])
    else:
        ckpt = train(args, log, device)

    evaluate(args, log, device, ckpt)


if __name__ == '__main__':
    main()
