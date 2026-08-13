"""End-to-end train + test for GazeTR-Hybrid on Gaze360.

Modeled after trainer/total.py + tester/total.py but combined and patched to:
- Use a local working dir
- Run final test on the last checkpoint with per-sample errors → mean ± std
"""
import os, sys, json, argparse, time
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import yaml
import importlib
from easydict import EasyDict as edict
from warmup_scheduler import GradualWarmupScheduler

import model
import ctools, gtools


def setup_logger(logpath):
    import logging
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    log = logging.getLogger('gazetr')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def do_train(cfg, log):
    train = cfg.train
    dataloader_module = importlib.import_module('reader.' + train.reader)

    torch.cuda.set_device(train.device)
    cudnn.benchmark = True

    dataset = dataloader_module.loader(
        train.data, train.params.batch_size, shuffle=True, num_workers=8)
    log.info(f'Train batches/epoch: {len(dataset)}')

    net = model.Model()
    net.train(); net.cuda()

    if train.pretrain.enable:
        sd = torch.load(train.pretrain.path, map_location='cuda:0', weights_only=False)
        net.load_state_dict(sd)
        log.info(f'Loaded pretrain {train.pretrain.path}')

    optimizer = optim.Adam(net.parameters(), lr=train.params.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train.params.decay_step,
                                          gamma=train.params.decay)
    if train.params.warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=train.params.warmup, after_scheduler=scheduler)

    savepath = os.path.join(train.save.metapath, train.save.folder, 'checkpoint')
    os.makedirs(savepath, exist_ok=True)

    # one warmup step (matches original code)
    optimizer.zero_grad(); optimizer.step(); scheduler.step()

    n_per = len(dataset); total = n_per * train.params.epoch
    timer = ctools.TimeCounter(total)

    last_ckpt = None
    for epoch in range(1, train.params.epoch + 1):
        t0 = time.time()
        for i, (data, anno) in enumerate(dataset):
            for k in data:
                if k != 'name':
                    data[k] = data[k].cuda(non_blocking=True)
            anno = anno.cuda(non_blocking=True)
            loss = net.loss(data, anno)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            rest = timer.step() / 3600
            if i % 50 == 0:
                log.info(f'  [{epoch}/{train.params.epoch}][{i}/{n_per}] '
                         f'loss={loss.item():.4f} lr={ctools.GetLR(optimizer):.2e} rest={rest:.2f}h')
        scheduler.step()
        log.info(f'Epoch {epoch} done in {time.time()-t0:.1f}s')

        if epoch % train.save.step == 0 or epoch == train.params.epoch:
            ckpt = os.path.join(savepath, f'Iter_{epoch}_{train.save.model_name}.pt')
            torch.save(net.state_dict(), ckpt)
            last_ckpt = ckpt
            log.info(f'  saved {ckpt}')

    return last_ckpt


def do_test(cfg, log, ckpt_path):
    test = cfg.test
    train = cfg.train
    dataloader_module = importlib.import_module('reader.' + test.reader)
    torch.cuda.set_device(test.device)

    dataset = dataloader_module.loader(test.data, 64, num_workers=4, shuffle=False)
    log.info(f'Test samples loader has {len(dataset)} batches')

    net = model.Model().cuda()
    sd = torch.load(ckpt_path, map_location=f'cuda:{test.device}', weights_only=False)
    net.load_state_dict(sd); net.eval()

    errors = []
    with torch.no_grad():
        for j, (data, label) in enumerate(dataset):
            for k in data:
                if k != 'name':
                    data[k] = data[k].cuda()
            gts = label.numpy()
            preds = net(data).cpu().numpy()
            for p, g in zip(preds, gts):
                errors.append(gtools.angular(gtools.gazeto3d(p), gtools.gazeto3d(g)))

    errors = np.array(errors)
    out = {
        'checkpoint': ckpt_path,
        'n_samples': int(errors.size),
        'angular_error_mean_deg': float(errors.mean()),
        'angular_error_std_deg': float(errors.std()),
    }
    out_path = os.path.join(train.save.metapath, train.save.folder, 'eval_test.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f'== Test: {out["angular_error_mean_deg"]:.2f}° ± {out["angular_error_std_deg"]:.2f}° '
             f'(N={out["n_samples"]}) ==')
    log.info(f'Saved -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-cfg', default='config/train/config_gaze360.yaml')
    ap.add_argument('--test-cfg', default='config/test/config_gaze360.yaml')
    ap.add_argument('--gpu', type=int, default=None,
                    help='override device id (in yaml). 0 = first visible (use CUDA_VISIBLE_DEVICES to pick)')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--test-ckpt', default=None)
    args = ap.parse_args()

    train_cfg = edict(yaml.load(open(args.train_cfg), Loader=yaml.FullLoader))
    test_cfg = edict(yaml.load(open(args.test_cfg), Loader=yaml.FullLoader))
    cfg = edict({'train': train_cfg.train, 'test': test_cfg.test})

    if args.gpu is not None:
        cfg.train.device = args.gpu
        cfg.test.device = args.gpu

    workdir = os.path.join(cfg.train.save.metapath, cfg.train.save.folder)
    os.makedirs(workdir, exist_ok=True)
    log = setup_logger(os.path.join(workdir, 'train.log'))
    log.info('=== GazeTR-Hybrid on Gaze360 ===')

    if args.test_ckpt:
        ckpt = args.test_ckpt
    elif args.skip_train:
        ckdir = os.path.join(workdir, 'checkpoint')
        items = sorted([f for f in os.listdir(ckdir) if f.endswith('.pt')],
                       key=lambda x: int(x.split('_')[1]))
        ckpt = os.path.join(ckdir, items[-1])
    else:
        ckpt = do_train(cfg, log)

    do_test(cfg, log, ckpt)


if __name__ == '__main__':
    main()
