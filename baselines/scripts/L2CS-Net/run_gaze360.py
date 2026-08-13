"""End-to-end train + test for L2CS-Net on Gaze360 (GazeHub-format labels).

Outputs:
  work_dir/L2CS_gaze360/_epoch_{N}.pkl   — checkpoints
  work_dir/L2CS_gaze360/train.log        — train log
  work_dir/L2CS_gaze360/eval_test.json   — final test result with mean & std
"""
import os
import sys
import json
import time
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from l2cs import L2CS, Gaze360 as L2CSGaze360Base, gazeto3d, angular
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SafeGaze360(L2CSGaze360Base):
    """Wrapper that retries on image-load errors so a single corrupt frame can't
    kill the DataLoader."""

    def __getitem__(self, idx):
        for attempt in range(5):
            try:
                return super().__getitem__(idx)
            except Exception:
                idx = (idx + 1) % len(self.lines)
        # last resort: re-raise
        return super().__getitem__(idx)


Gaze360 = SafeGaze360


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/path/to/gaze360')
    p.add_argument('--train-label', default='/path/to/gaze360/train.label')
    p.add_argument('--test-label', default='/path/to/gaze360/test.label')
    p.add_argument('--work-dir', default='work_dir/L2CS_gaze360')
    p.add_argument('--gpu', default=2, type=int)
    p.add_argument('--epochs', default=50, type=int)
    p.add_argument('--batch-size', default=32, type=int)
    p.add_argument('--workers', default=8, type=int)
    p.add_argument('--lr', default=1e-5, type=float)
    p.add_argument('--alpha', default=1.0, type=float)
    p.add_argument('--arch', default='ResNet50')
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--test-ckpt', default=None,
                   help='if set, skip training & test this checkpoint')
    p.add_argument('--resume', default=None,
                   help='resume training from this checkpoint (continues epoch numbering)')
    p.add_argument('--resume-epoch', type=int, default=0,
                   help='epoch index already completed (training starts at resume_epoch+1)')
    return p.parse_args()


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
        for module_name, module in m.named_modules():
            if 'bn' in module_name:
                module.eval()
            for _, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for m in b:
        for module_name, module in m.named_modules():
            if 'bn' in module_name:
                module.eval()
            for _, param in module.named_parameters():
                yield param


def get_fc_params(model):
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for m in b:
        for _, module in m.named_modules():
            for _, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def setup_logging(work_dir):
    os.makedirs(work_dir, exist_ok=True)
    log = logging.getLogger('l2cs')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(os.path.join(work_dir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def train(args, log, device):
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = Gaze360(args.train_label, args.data_root, transformations, 180, 4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    log.info(f'Train samples: {len(train_ds)}, batches/epoch: {len(train_loader)}')

    model, pre_url = build_model(args.arch, 90)
    if args.resume:
        log.info(f'Resuming from {args.resume} (epochs 1..{args.resume_epoch} skipped)')
        sd = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(sd)
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
        for i, (images, labels, cont, _) in enumerate(train_loader):
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
                log.info(f'  [{epoch+1}/{args.epochs}][{i+1}/{len(train_loader)}] '
                         f'pitch={sum_p/nb:.3f} yaw={sum_y/nb:.3f}')

        dt = time.time() - t0
        log.info(f'Epoch {epoch+1} done in {dt:.1f}s — pitch={sum_p/nb:.3f} yaw={sum_y/nb:.3f}')

        ckpt = os.path.join(args.work_dir, f'_epoch_{epoch+1}.pkl')
        torch.save(model.state_dict(), ckpt)
        last_ckpt = ckpt

    return last_ckpt


def evaluate(args, log, device, ckpt_path):
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = Gaze360(args.test_label, args.data_root, transformations, 180, 4, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    log.info(f'Test samples (semi-front filtered by L2CS): {len(test_ds)}')

    model, _ = build_model(args.arch, 90)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.cuda(device).eval()

    softmax = nn.Softmax(dim=1)
    idx_tensor = torch.FloatTensor(list(range(90))).cuda(device)
    errors = []

    with torch.no_grad():
        for images, _, cont_labels, _ in test_loader:
            images = images.cuda(device)
            lp = cont_labels[:, 0].float() * np.pi / 180
            ly = cont_labels[:, 1].float() * np.pi / 180

            pitch, yaw = model(images)
            pp = torch.sum(softmax(pitch) * idx_tensor, 1).cpu() * 4 - 180
            yp = torch.sum(softmax(yaw) * idx_tensor, 1).cpu() * 4 - 180
            pp = pp * np.pi / 180
            yp = yp * np.pi / 180
            for p, y, pl, yl in zip(pp, yp, lp, ly):
                errors.append(angular(gazeto3d([p, y]), gazeto3d([pl, yl])))

    errors = np.array(errors)
    result = {
        'checkpoint': ckpt_path,
        'n_samples': int(errors.size),
        'angular_error_mean_deg': float(errors.mean()),
        'angular_error_std_deg': float(errors.std()),
    }
    out = os.path.join(args.work_dir, 'eval_test.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f'== Test: {result["angular_error_mean_deg"]:.2f}° ± {result["angular_error_std_deg"]:.2f}° '
             f'(N={result["n_samples"]}) ==')
    log.info(f'Result saved -> {out}')
    return result


def main():
    args = parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = 0  # mapped via CUDA_VISIBLE_DEVICES

    os.makedirs(args.work_dir, exist_ok=True)
    log = setup_logging(args.work_dir)
    log.info(f'args = {vars(args)}')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        # find last checkpoint
        ckpts = sorted([f for f in os.listdir(args.work_dir) if f.endswith('.pkl')],
                       key=lambda x: int(x.split('_')[2].split('.')[0]))
        if not ckpts:
            raise SystemExit('no checkpoints found')
        ckpt = os.path.join(args.work_dir, ckpts[-1])
    else:
        ckpt = train(args, log, device)

    log.info(f'Evaluating {ckpt}')
    evaluate(args, log, device, ckpt)


if __name__ == '__main__':
    main()
