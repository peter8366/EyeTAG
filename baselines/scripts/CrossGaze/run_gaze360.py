"""End-to-end train + test for CrossGaze (InceptionResnet + VGGFace2 pretrain) on Gaze360.

Uses the paper's main configuration: --model inception-resnet --pretrained True
(vggface2), simple trainer, angular (cosine-similarity) loss, RandAugment.

Adapted to GazeHub-format labels (6 columns) — original CrossGaze loader expected
9-column labels (identity / eye keypoints). The simple_trainer only uses face + 3D
gaze, so we provide a simplified loader.
"""
import os, sys, json, math, time, argparse, random, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
from augmentation import RandAugmentPC
from utils import fixed_image_standardization, CosineSimilarityLoss, AngularDistance


SUBSETS = {  # |gyaw| ≤ threshold (radians)
    'full': float('inf'),
    'semi-front': math.pi / 2,
    'front': math.pi / 9,
}


# ────────────────────────────────────────────────────────────────────────────
# Model — InceptionResnet face backbone (CrossGaze paper's main winner)
# ────────────────────────────────────────────────────────────────────────────

class InceptionResnet(nn.Module):
    def __init__(self, pretrained='vggface2'):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained)
        self.backbone.last_bn = nn.Identity()
        self.backbone.last_linear = nn.Linear(in_features=1792, out_features=3, bias=True)

    def forward(self, x):
        return self.backbone(x)


# ────────────────────────────────────────────────────────────────────────────
# Dataset — GazeHub format, 6 columns. Returns face image + 3D unit gaze vector.
# ────────────────────────────────────────────────────────────────────────────

class GazeHubFaceDS(Dataset):
    def __init__(self, root, label_path, transform):
        self.root = root
        self.transform = transform
        with open(label_path) as f:
            f.readline()  # header
            self.lines = [l.strip().split() for l in f if l.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx]
        face = parts[0]
        gaze3d = parts[4]
        gx, gy, gz = (float(v) for v in gaze3d.split(','))
        label3d = torch.tensor([gx, gy, gz], dtype=torch.float32)
        gaze2d = parts[5]
        gyaw, gpitch = (float(v) for v in gaze2d.split(','))
        img = Image.open(os.path.join(self.root, face))  # PIL RGB
        img = self.transform(img)
        return {'img': img, 'label3d': label3d, 'gyaw': float(gyaw)}


# ────────────────────────────────────────────────────────────────────────────
# Train / Eval
# ────────────────────────────────────────────────────────────────────────────

def setup_logger(workdir):
    os.makedirs(workdir, exist_ok=True)
    log = logging.getLogger('crossgaze')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(workdir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def get_transforms(magnitude):
    test_tf = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    train_tf = transforms.Compose([
        RandAugmentPC(n=2, m=magnitude),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    return train_tf, test_tf


def train(args, log, device):
    train_tf, test_tf = get_transforms(args.magnitude)
    train_ds = GazeHubFaceDS(args.data_root, args.train_label, train_tf)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    log.info(f'Train samples: {len(train_ds)}, batches/epoch: {len(loader)}')

    model = InceptionResnet(pretrained='vggface2' if args.pretrained else None).cuda(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Params: {n_params/1e6:.2f}M, batch={args.batch_size}, lr={args.lr}')

    criterion = CosineSimilarityLoss().cuda(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    last_ckpt = None
    for epoch in range(args.epochs):
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
            if (i + 1) % 50 == 0:
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
    test_ds = GazeHubFaceDS(args.data_root, args.test_label, test_tf)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    log.info(f'Test samples: {len(test_ds)}')

    model = InceptionResnet(pretrained='vggface2' if args.pretrained else None).cuda(device)
    state = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.info(f'load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}')
    model.eval()

    metric = AngularDistance().cuda(device)
    errors = np.zeros(len(test_ds), dtype=np.float64)
    gyaws = np.zeros(len(test_ds), dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for data in loader:
            images = data['img'].cuda(device, non_blocking=True)
            labels = data['label3d'].cuda(device, non_blocking=True)
            pred = model(images)
            # per-sample angular error in degrees
            cos = torch.nn.functional.cosine_similarity(pred, labels, dim=1).clamp(-0.99999, 0.99999)
            err = torch.acos(cos) * 180.0 / math.pi
            n = err.shape[0]
            errors[cursor:cursor+n] = err.cpu().numpy()
            gyaws[cursor:cursor+n] = data['gyaw'].numpy()
            cursor += n
    assert cursor == len(test_ds)

    abs_yaw = np.abs(gyaws)
    out = {'checkpoint': ckpt_path, 'n_total': int(len(test_ds)), 'subsets': {}}
    for name, thresh in SUBSETS.items():
        mask = abs_yaw <= thresh
        e = errors[mask]
        out['subsets'][name] = {
            'n': int(mask.sum()),
            'angular_error_mean_deg': round(float(e.mean()), 2),
            'angular_error_std_deg': round(float(e.std()), 2),
        }
        log.info(f'  {name:12s}: {e.mean():.2f}° ± {e.std():.2f}°  (N={int(mask.sum())})')

    out_path = os.path.join(args.work_dir, 'eval_test_subsets.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f'Saved -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/path/to/gaze360')
    ap.add_argument('--train-label', default='/path/to/gaze360/train.label')
    ap.add_argument('--test-label', default='/path/to/gaze360/test.label')
    ap.add_argument('--work-dir', default='./work_dir/crossgaze_iresnet')
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--magnitude', type=int, default=3)
    ap.add_argument('--pretrained', type=lambda x: str(x).lower() != 'false', default=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--test-ckpt', default=None)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    cudnn.benchmark = True
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = 0
    os.makedirs(args.work_dir, exist_ok=True)
    log = setup_logger(args.work_dir)
    log.info(f'args = {vars(args)}')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        items = sorted([f for f in os.listdir(args.work_dir) if f.endswith('.pth')],
                       key=lambda x: int(x.replace('epoch', '').replace('.pth', '')))
        ckpt = os.path.join(args.work_dir, items[-1])
    else:
        ckpt = train(args, log, device)

    evaluate(args, log, device, ckpt)


if __name__ == '__main__':
    main()
