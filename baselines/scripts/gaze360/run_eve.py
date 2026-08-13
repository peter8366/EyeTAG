"""Train + test gaze360 LSTM (erkil1452) on EVE (webcam_c).

7-frame sliding window over EVE webcam_c at target_hz subsample.
NOTE: at target_hz=10, 7-frame span is 0.6s (longer than gaze360 native 0.23s).
For closer-to-original behavior use --target-hz 30 (no subsample, full webcam_c rate).
"""
import os, sys, json, math, time, random, argparse, logging

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
sys.path.insert(0, os.path.dirname(os.path.dirname(base)))  # comparison/

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import GazeLSTM, PinBallLoss
from eve_common import EVESequenceDataset, DEFAULT_DATA_ROOT


# ── EVE → gaze360 LSTM format adapter ───────────────────────────────────────

class Gaze360EVEWrapper(Dataset):
    """Returns frames packed as (T*3, 224, 224) + target spherical (yaw, pitch)
    matching the gaze360 LSTM's expected input.
    """

    T = 7

    def __init__(self, base_ds):
        assert base_ds.num_frames == self.T, f'expected T={self.T}, got {base_ds.num_frames}'
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        # s['seq']: (T, 3, 224, 224) tensor (after per-frame transform)
        seq = s['seq']  # (T, 3, H, W)
        frames = seq.view(self.T * 3, seq.shape[-2], seq.shape[-1])  # (21, H, W)

        # gaze360 LSTM target: spherical (yaw, pitch) computed from 3D unit vec
        # EVE provides (pitch, yaw) radians directly — gaze360 LSTM's
        # spherical2cartesian formula:
        #   z = -cos(pitch)*cos(yaw); x = cos(pitch)*sin(yaw); y = sin(pitch)
        # so spherical[0]=yaw, spherical[1]=pitch (matching EVE convention)
        target = torch.tensor([s['yaw'].item(), s['pitch'].item()], dtype=torch.float32)
        return frames, target


def get_transforms(train=True):
    """gaze360 LSTM transforms: 224×224 + ImageNet normalize."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), norm,
    ])


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
    log = logging.getLogger('gaze360lstm_eve')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(workdir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def train_one(args, log, device):
    train_tf = get_transforms(train=True)
    base_train = EVESequenceDataset(args.data_root, split='train',
                                       num_frames=7, target_hz=args.target_hz,
                                       transform=train_tf)
    train_ds = Gaze360EVEWrapper(base_train)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
    log.info(f'Train samples: {len(train_ds)}, batches/epoch: {len(loader)}')

    model = GazeLSTM().cuda(device)
    criterion = PinBallLoss().cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    last_ckpt = None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = sum_ang = nb = 0
        for i, (src, target) in enumerate(loader):
            src = src.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            output, var = model(src)
            loss = criterion(output, target, var)
            ang = angular_errors(output.detach(), target).mean().item()
            sum_loss += loss.item(); sum_ang += ang; nb += 1
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if (i + 1) % 100 == 0:
                log.info(f'  [{epoch+1}/{args.epochs}][{i+1}/{len(loader)}] '
                         f'loss={sum_loss/nb:.4f} ang={sum_ang/nb:.3f}°')
        log.info(f'Epoch {epoch+1} done in {time.time()-t0:.1f}s — '
                 f'loss={sum_loss/nb:.4f} ang={sum_ang/nb:.3f}°')
        ckpt = os.path.join(args.work_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, ckpt)
        last_ckpt = ckpt
    return last_ckpt


def evaluate(args, log, device, ckpt_path):
    test_tf = get_transforms(train=False)
    base_test = EVESequenceDataset(args.data_root, split=args.test_split,
                                     num_frames=7, target_hz=args.target_hz,
                                     transform=test_tf)
    test_ds = Gaze360EVEWrapper(base_test)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    log.info(f'Test samples ({args.test_split}): {len(test_ds)}')

    model = GazeLSTM().cuda(device)
    state = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state); model.eval()

    errs = []
    with torch.no_grad():
        for src, target in loader:
            src = src.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            output, _ = model(src)
            errs.extend(angular_errors(output, target).cpu().tolist())

    errs = np.array(errs)
    res = {
        'checkpoint': ckpt_path,
        'test_split': args.test_split,
        'target_hz': args.target_hz,
        'camera': 'webcam_c',
        'n_samples': int(errs.size),
        'angular_error_mean_deg': float(errs.mean()),
        'angular_error_std_deg': float(errs.std()),
    }
    out = os.path.join(args.work_dir, f'eval_{args.test_split}_eve.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    log.info(f'== EVE {args.test_split}: {res["angular_error_mean_deg"]:.2f}° ± '
             f'{res["angular_error_std_deg"]:.2f}° (N={res["n_samples"]}) ==')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--work-dir', default='./work_dir/gaze360_lstm_eve')
    ap.add_argument('--target-hz', type=int, default=30,
                    help='30Hz native (default) matches EVE_final webcam_c temporal span. '
                         'gaze360 LSTM was originally designed for 30fps (7-frame span ~0.23s).')
    ap.add_argument('--test-split', default='test', choices=['val', 'test'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=80)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--test-ckpt', default=None)
    args = ap.parse_args()

    cudnn.benchmark = True
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = 0
    os.makedirs(args.work_dir, exist_ok=True)
    log = setup_logger(args.work_dir)
    log.info(f'=== gaze360 LSTM on EVE (webcam_c, T=7, hz={args.target_hz}) ===')
    log.info(f'args = {vars(args)}')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        items = sorted([f for f in os.listdir(args.work_dir) if f.endswith('.pth')],
                       key=lambda x: int(x.split('epoch')[1].split('.')[0]))
        ckpt = os.path.join(args.work_dir, items[-1])
    else:
        ckpt = train_one(args, log, device)
    evaluate(args, log, device, ckpt)


if __name__ == '__main__':
    main()
