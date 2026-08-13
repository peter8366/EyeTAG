"""Train + test L2CS-Net on EVE (webcam_c).

L2CS uses 448×448 ImageNet-normalized faces + (pitch_deg, yaw_deg) labels.
Classification (90 bins, binwidth=4) + MSE regression.
"""
import os, sys, json, math, time, argparse, logging

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.dirname(base_dir))   # comparison/

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from l2cs import L2CS, gazeto3d, angular
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from eve_common import EVEFrameDataset, DEFAULT_DATA_ROOT


# ────────────────────────────────────────────────────────────────────────────
# Model construction (matches comparison/L2CS-Net/run_gaze360.py)
# ────────────────────────────────────────────────────────────────────────────

def build_model(arch, bins=90):
    if arch == 'ResNet18':
        m = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        m = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    else:
        m = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    return m, pre_url


def get_ignored_params(model):
    b = [model.conv1, model.bn1, model.fc_finetune]
    for m in b:
        for mn, module in m.named_modules():
            if 'bn' in mn:
                module.eval()
            for _, p in module.named_parameters():
                yield p


def get_non_ignored_params(model):
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for m in b:
        for mn, module in m.named_modules():
            if 'bn' in mn:
                module.eval()
            for _, p in module.named_parameters():
                yield p


def get_fc_params(model):
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for m in b:
        for _, module in m.named_modules():
            for _, p in module.named_parameters():
                yield p


def load_filtered_state_dict(model, snapshot):
    md = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in md}
    md.update(snapshot)
    model.load_state_dict(md)


# ────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ────────────────────────────────────────────────────────────────────────────

class L2CSEVEWrapper(Dataset):
    """Wraps EVEFrameDataset → L2CS input format.
    Returns (img, binned_labels, cont_labels (pitch_deg, yaw_deg), name).
    """

    def __init__(self, base_ds, angle=180, binwidth=4):
        self.base = base_ds
        self.angle = angle
        self.binwidth = binwidth
        # bins range [-angle, angle) with binwidth
        self.bins = np.array(range(-angle, angle, binwidth))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        # EVE labels are in radians: pitch, yaw
        pitch_deg = float(s['pitch'].item()) * 180.0 / np.pi
        yaw_deg = float(s['yaw'].item()) * 180.0 / np.pi
        binned = np.digitize([pitch_deg, yaw_deg], self.bins) - 1
        labels = torch.from_numpy(binned).long()
        cont_labels = torch.FloatTensor([pitch_deg, yaw_deg])
        return s['img'], labels, cont_labels, f"{s['participant']}/{s['session']}"


def get_transform():
    return transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ────────────────────────────────────────────────────────────────────────────

def setup_logger(workdir):
    os.makedirs(workdir, exist_ok=True)
    log = logging.getLogger('l2cs_eve')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(workdir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def train_one(args, log, device):
    transform = get_transform()
    base_train = EVEFrameDataset(args.data_root, split='train',
                                   target_hz=args.target_hz, transform=transform,
                                   return_format='pil')
    train_ds = L2CSEVEWrapper(base_train, angle=180, binwidth=4)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
    log.info(f'Train samples: {len(train_ds)}, batches/epoch: {len(loader)}')

    model, pre_url = build_model(args.arch, 90)
    if args.resume and os.path.isfile(args.resume):
        sd = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(sd)
        log.info(f'Resumed from {args.resume}')
    else:
        load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    model.cuda(device)

    criterion = nn.CrossEntropyLoss().cuda(device)
    reg_criterion = nn.MSELoss().cuda(device)
    softmax = nn.Softmax(dim=1).cuda(device)
    idx_tensor = torch.FloatTensor(list(range(90))).cuda(device)

    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(model), 'lr': 0},
        {'params': get_non_ignored_params(model), 'lr': args.lr},
        {'params': get_fc_params(model), 'lr': args.lr},
    ], args.lr)

    log.info(f'Training {args.arch} | bs={args.batch_size} | lr={args.lr} | epochs={args.epochs}')

    last_ckpt = None
    start_epoch = args.resume_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        sum_y = sum_p = nb = 0.0
        for i, (images, labels, cont, _) in enumerate(loader):
            images = images.cuda(device, non_blocking=True)
            lp = labels[:, 0].cuda(device, non_blocking=True)
            ly = labels[:, 1].cuda(device, non_blocking=True)
            cp = cont[:, 0].cuda(device, non_blocking=True)
            cy = cont[:, 1].cuda(device, non_blocking=True)

            pitch, yaw = model(images)
            loss_p = criterion(pitch, lp)
            loss_y = criterion(yaw, ly)

            pp = torch.sum(softmax(pitch) * idx_tensor, 1) * 4 - 180
            yp = torch.sum(softmax(yaw) * idx_tensor, 1) * 4 - 180
            loss_p = loss_p + args.alpha * reg_criterion(pp, cp)
            loss_y = loss_y + args.alpha * reg_criterion(yp, cy)

            sum_p += loss_p.item(); sum_y += loss_y.item(); nb += 1

            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward([loss_p, loss_y],
                                    [torch.tensor(1.0).cuda(device)] * 2)
            optimizer.step()

            if (i + 1) % 100 == 0:
                log.info(f'  [{epoch+1}/{args.epochs}][{i+1}/{len(loader)}] '
                         f'pitch={sum_p/nb:.3f} yaw={sum_y/nb:.3f}')

        log.info(f'Epoch {epoch+1} done in {time.time()-t0:.1f}s — '
                 f'pitch={sum_p/nb:.3f} yaw={sum_y/nb:.3f}')
        ckpt = os.path.join(args.work_dir, f'_epoch_{epoch+1}.pkl')
        torch.save(model.state_dict(), ckpt)
        last_ckpt = ckpt
    return last_ckpt


def evaluate(args, log, device, ckpt_path):
    transform = get_transform()
    base_test = EVEFrameDataset(args.data_root, split=args.test_split,
                                 target_hz=args.target_hz, transform=transform,
                                 return_format='pil')
    test_ds = L2CSEVEWrapper(base_test, angle=180, binwidth=4)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    log.info(f'Test samples ({args.test_split}): {len(test_ds)}')

    model, _ = build_model(args.arch, 90)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.cuda(device).eval()

    softmax = nn.Softmax(dim=1)
    idx_tensor = torch.FloatTensor(list(range(90))).cuda(device)
    errors = []

    with torch.no_grad():
        for images, _, cont_labels, _ in loader:
            images = images.cuda(device)
            lp = cont_labels[:, 0].float() * np.pi / 180  # pitch in rad
            ly = cont_labels[:, 1].float() * np.pi / 180  # yaw in rad

            pitch, yaw = model(images)
            pp = torch.sum(softmax(pitch) * idx_tensor, 1).cpu() * 4 - 180
            yp = torch.sum(softmax(yaw) * idx_tensor, 1).cpu() * 4 - 180
            pp = pp * np.pi / 180
            yp = yp * np.pi / 180
            for p, y, pl, yl in zip(pp, yp, lp, ly):
                errors.append(angular(gazeto3d([p, y]), gazeto3d([pl, yl])))

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
    ap.add_argument('--work-dir', default='./work_dir/l2cs_eve')
    ap.add_argument('--target-hz', type=int, default=30,
                    help='30Hz native (default) matches EVE_final webcam_c temporal span.')
    ap.add_argument('--test-split', default='test', choices=['val', 'test'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--arch', default='ResNet50')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--test-ckpt', default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--resume-epoch', type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = 0
    os.makedirs(args.work_dir, exist_ok=True)
    log = setup_logger(args.work_dir)
    log.info(f'=== L2CS-Net ({args.arch}) on EVE (webcam_c, hz={args.target_hz}) ===')
    log.info(f'args = {vars(args)}')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        items = sorted([f for f in os.listdir(args.work_dir) if f.endswith('.pkl')],
                       key=lambda x: int(x.split('_')[2].split('.')[0]))
        ckpt = os.path.join(args.work_dir, items[-1])
    else:
        ckpt = train_one(args, log, device)

    evaluate(args, log, device, ckpt)


if __name__ == '__main__':
    main()
