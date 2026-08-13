"""Train + test CrossGaze (InceptionResnet + VGGFace2 pretrain) on EVE (webcam_c).

Single-frame model. EVE labels (pitch, yaw) radians → 3D unit vector for training.
"""
import os, sys, json, math, time, random, argparse, logging

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.dirname(base_dir))   # comparison/

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from augmentation import RandAugmentPC
from utils import fixed_image_standardization, CosineSimilarityLoss, AngularDistance

from eve_common import EVEFrameDataset, pitchyaw_to_vector, DEFAULT_DATA_ROOT


class InceptionResnet(nn.Module):
    def __init__(self, pretrained='vggface2'):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained)
        self.backbone.last_bn = nn.Identity()
        self.backbone.last_linear = nn.Linear(in_features=1792, out_features=3, bias=True)

    def forward(self, x):
        return self.backbone(x)


def get_transforms(magnitude):
    """CrossGaze uses fixed_image_standardization (img-127.5)/128.0 on uint8."""
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        RandAugmentPC(n=2, m=magnitude),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    return train_tf, test_tf


class CrossGazeEVEWrapper(Dataset):
    """Wraps EVEFrameDataset to return CrossGaze's expected format."""

    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        # img is already transformed (Resize 224, RandAug, ToTensor, fixed_std)
        # Build 3D label from EVE (pitch, yaw): use the same pitchyaw_to_vector
        # convention as gazeto3d() so loss is consistent.
        vec = pitchyaw_to_vector(s['pitch'].item(), s['yaw'].item())
        return {
            'img': s['img'],
            'label3d': torch.from_numpy(vec),
            'pitch': s['pitch'],
            'yaw': s['yaw'],
        }


def setup_logger(workdir):
    os.makedirs(workdir, exist_ok=True)
    log = logging.getLogger('crossgaze_eve')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(workdir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def train_one(args, log, device):
    train_tf, _ = get_transforms(args.magnitude)
    base_train = EVEFrameDataset(args.data_root, split='train',
                                   target_hz=args.target_hz, transform=train_tf,
                                   return_format='pil')
    train_ds = CrossGazeEVEWrapper(base_train)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    log.info(f'Train samples: {len(train_ds)}, batches/epoch: {len(loader)}')

    model = InceptionResnet(pretrained='vggface2' if args.pretrained else None).cuda(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Params: {n_params/1e6:.2f}M, batch={args.batch_size}, lr={args.lr}')

    criterion = CosineSimilarityLoss().cuda(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    # Resume support: load weights and start from given epoch.
    if args.resume and os.path.isfile(args.resume):
        sd = torch.load(args.resume, map_location=f'cuda:{device}', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        # advance the scheduler to the right step
        for _ in range(args.resume_epoch):
            scheduler.step()
        log.info(f'Resumed from {args.resume}  (starting epoch {args.resume_epoch + 1})')

    last_ckpt = None
    for epoch in range(args.resume_epoch, args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = nb = 0
        for i, data in enumerate(loader):
            images = data['img'].cuda(device, non_blocking=True)
            labels = data['label3d'].cuda(device, non_blocking=True)
            pred = model(images)
            loss = criterion(pred, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            sum_loss += loss.item(); nb += 1
            if (i + 1) % 100 == 0:
                log.info(f'  [{epoch+1}/{args.epochs}][{i+1}/{len(loader)}] '
                         f'loss={sum_loss/nb:.4f} lr={optimizer.param_groups[0]["lr"]:.2e}')
        scheduler.step()
        log.info(f'Epoch {epoch+1} done in {time.time()-t0:.1f}s — loss={sum_loss/nb:.4f}')

        ckpt = os.path.join(args.work_dir, f'epoch{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt)
        last_ckpt = ckpt

    return last_ckpt


def evaluate(args, log, device, ckpt_path):
    _, test_tf = get_transforms(args.magnitude)
    base_test = EVEFrameDataset(args.data_root, split=args.test_split,
                                 target_hz=args.target_hz, transform=test_tf,
                                 return_format='pil')
    test_ds = CrossGazeEVEWrapper(base_test)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    log.info(f'Test samples ({args.test_split}): {len(test_ds)}')

    model = InceptionResnet(pretrained='vggface2' if args.pretrained else None).cuda(device)
    state = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.info(f'load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}')
    model.eval()

    errors = []
    with torch.no_grad():
        for data in loader:
            images = data['img'].cuda(device, non_blocking=True)
            labels = data['label3d'].cuda(device, non_blocking=True)
            pred = model(images)
            cos = torch.nn.functional.cosine_similarity(pred, labels, dim=1).clamp(-0.99999, 0.99999)
            err = torch.acos(cos) * 180.0 / math.pi
            errors.extend(err.cpu().tolist())

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
    ap.add_argument('--work-dir', default='./work_dir/crossgaze_eve')
    ap.add_argument('--target-hz', type=int, default=30,
                    help='30Hz native (default) matches EVE_final webcam_c temporal span.')
    ap.add_argument('--test-split', default='test', choices=['val', 'test'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--magnitude', type=int, default=3)
    ap.add_argument('--pretrained', type=lambda x: str(x).lower() != 'false', default=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--test-ckpt', default=None)
    ap.add_argument('--resume', default=None,
                    help='Path to .pth to resume training from')
    ap.add_argument('--resume-epoch', type=int, default=0,
                    help='Epoch index already completed (training starts at resume_epoch+1)')
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    cudnn.benchmark = True
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = 0
    os.makedirs(args.work_dir, exist_ok=True)
    log = setup_logger(args.work_dir)
    log.info(f'=== CrossGaze (IRes+VGGFace2) on EVE (webcam_c, hz={args.target_hz}) ===')
    log.info(f'args = {vars(args)}')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        items = sorted([f for f in os.listdir(args.work_dir) if f.endswith('.pth')],
                       key=lambda x: int(x.replace('epoch', '').replace('.pth', '')))
        ckpt = os.path.join(args.work_dir, items[-1])
    else:
        ckpt = train_one(args, log, device)

    evaluate(args, log, device, ckpt)


if __name__ == '__main__':
    main()
