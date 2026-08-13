"""Train + test GazeTR-Hybrid on EVE (webcam_c only).

Mirrors run_gaze360.py logic but swaps the dataset to EVEFrameDataset.
GazeTR expects: 224×224 face image (cv2-style BGR or HWC tensor), gaze label (pitch, yaw) radians.

Splits (from comparison/eve_common.py):
  train: train01..train34, val: train35..train39, test: val01..val05
"""
import os, sys, json, argparse, time, math
import logging

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.dirname(base_dir))  # comparison/

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

import model         # GazeTR Model
import gtools         # angular(), gazeto3d()
import ctools         # TimeCounter, GetLR

from eve_common import EVEFrameDataset, DEFAULT_DATA_ROOT


def setup_logger(workdir):
    os.makedirs(workdir, exist_ok=True)
    log = logging.getLogger('gazetr_eve')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(workdir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def get_transform():
    """GazeTR uses cv2.imread(BGR) → ToTensor(). Replicate via PIL RGB → BGR swap → ToTensor."""
    def _t(pil_img):
        # PIL RGB → numpy → BGR (to match cv2.imread behaviour)
        arr = np.array(pil_img)            # (H, W, 3) RGB uint8
        arr = arr[:, :, ::-1].copy()       # → BGR
        # GazeTR expects 224x224
        import cv2
        arr = cv2.resize(arr, (224, 224))
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
        return tensor
    return _t


def train_one(args, log, device):
    transform = get_transform()
    train_ds = EVEFrameDataset(args.data_root, split='train',
                                target_hz=args.target_hz, transform=transform,
                                return_format='pil')
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    log.info(f'Train batches/epoch: {len(loader)}')

    net = model.Model()
    net.train(); net.cuda()

    if args.pretrain and os.path.isfile(args.pretrain):
        sd = torch.load(args.pretrain, map_location='cuda:0', weights_only=False)
        net.load_state_dict(sd)
        log.info(f'Loaded pretrain {args.pretrain}')

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)
    if args.warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=args.warmup, after_scheduler=scheduler)

    savepath = os.path.join(args.work_dir, 'checkpoint')
    os.makedirs(savepath, exist_ok=True)

    n_per = len(loader); total = n_per * args.epochs
    timer = ctools.TimeCounter(total)

    optimizer.zero_grad(); optimizer.step(); scheduler.step()

    last_ckpt = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        sum_loss = 0.0; nb = 0
        for i, batch in enumerate(loader):
            data = {'face': batch['img'].cuda(non_blocking=True), 'name': [''] * batch['img'].size(0)}
            # GazeTR label = (yaw, pitch) per its tester convention? Inspect: tester uses
            #   gtools.gazeto3d(gaze) where gaze=[pitch, yaw]? Actually run_gaze360.py uses
            #   label = (yaw, pitch) consistent with how Decode_Gaze360 parses.
            # We follow the same: anno = [yaw, pitch] (rad) — matches Gaze360 .label 2D order.
            anno = torch.stack([batch['yaw'], batch['pitch']], dim=1).cuda(non_blocking=True)
            loss = net.loss(data, anno)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            rest = timer.step() / 3600
            sum_loss += loss.item(); nb += 1
            if i % 50 == 0:
                log.info(f'  [{epoch}/{args.epochs}][{i}/{n_per}] '
                         f'loss={loss.item():.4f} avg={sum_loss/nb:.4f} '
                         f'lr={ctools.GetLR(optimizer):.2e} rest={rest:.2f}h')
        scheduler.step()
        log.info(f'Epoch {epoch} done in {time.time()-t0:.1f}s — avg loss={sum_loss/nb:.4f}')

        if epoch % args.save_step == 0 or epoch == args.epochs:
            ckpt = os.path.join(savepath, f'Iter_{epoch}_eve.pt')
            torch.save(net.state_dict(), ckpt)
            last_ckpt = ckpt
            log.info(f'  saved {ckpt}')

    return last_ckpt


def evaluate(args, log, device, ckpt_path):
    transform = get_transform()
    test_ds = EVEFrameDataset(args.data_root, split=args.test_split,
                               target_hz=args.target_hz, transform=transform,
                               return_format='pil')
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    log.info(f'Test samples ({args.test_split}): {len(test_ds)}')

    net = model.Model().cuda()
    sd = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    net.load_state_dict(sd); net.eval()

    errors = []
    with torch.no_grad():
        for batch in loader:
            data = {'face': batch['img'].cuda(), 'name': [''] * batch['img'].size(0)}
            # GT: [yaw, pitch] rad (same order as training)
            gts = torch.stack([batch['yaw'], batch['pitch']], dim=1).numpy()
            preds = net(data).cpu().numpy()
            for p, g in zip(preds, gts):
                errors.append(gtools.angular(gtools.gazeto3d(p), gtools.gazeto3d(g)))

    errors = np.array(errors)
    out = {
        'checkpoint': ckpt_path,
        'test_split': args.test_split,
        'target_hz': args.target_hz,
        'camera': 'webcam_c',
        'n_samples': int(errors.size),
        'angular_error_mean_deg': float(errors.mean()),
        'angular_error_std_deg': float(errors.std()),
    }
    out_path = os.path.join(args.work_dir, f'eval_{args.test_split}_eve.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f'== EVE {args.test_split}: {out["angular_error_mean_deg"]:.2f}° ± '
             f'{out["angular_error_std_deg"]:.2f}° (N={out["n_samples"]}) ==')
    log.info(f'Saved -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--work-dir', default='./work_dir/gazetr_eve')
    ap.add_argument('--target-hz', type=int, default=30,
                    help='30Hz native (default) matches EVE_final webcam_c temporal span.')
    ap.add_argument('--test-split', default='test', choices=['val', 'test'])
    ap.add_argument('--pretrain', default='./pretrain/eth_xgaze_pretrain.pt',
                    help='ETH-XGaze pretrain (optional). Empty to skip.')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--decay', type=float, default=0.5)
    ap.add_argument('--decay-step', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--save-step', type=int, default=5)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--test-ckpt', default=None)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    cudnn.benchmark = True
    device = 0
    os.makedirs(args.work_dir, exist_ok=True)
    log = setup_logger(args.work_dir)
    log.info(f'=== GazeTR-Hybrid on EVE (webcam_c, target_hz={args.target_hz}) ===')
    log.info(f'args = {vars(args)}')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        ckdir = os.path.join(args.work_dir, 'checkpoint')
        items = sorted([f for f in os.listdir(ckdir) if f.endswith('.pt')],
                       key=lambda x: int(x.split('_')[1]))
        ckpt = os.path.join(ckdir, items[-1])
    else:
        ckpt = train_one(args, log, device)

    evaluate(args, log, device, ckpt)


if __name__ == '__main__':
    main()
