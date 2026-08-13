"""Train EyeTAG on Gaze360.

Reproduces the main result of the paper (Table 2, mean MAE 8.83 deg) with:

    python gaze360/train.py \\
        --data-root /path/to/gaze360 \\
        --work-dir  ./work_dir/eyetag_t48_delta \\
        --face-backbone vggface2 --eye-backbone resnet18 \\
        --pretrained-vggface checkpoints/resnet50_ft_weight.pkl \\
        --prev-mode mlp --prev-input delta \\
        --fusion-type cross_attn --temporal-type causal --gaze-space vector \\
        --num-frames 48 --seq-stride 4 --frame-stride 1 \\
        --epochs 15 --batch-size 32 --lr 3e-4 --warmup-epochs 3 \\
        --weight-decay 0.02 --grad-clip 1.0 \\
        --prev-gt-start 1.0 --prev-gt-end 0.0 --prev-gt-end-epoch 5 \\
        --cache-mode ar --autoreg-val --lr-restart-at-prev-zero \\
        --seed 42 --gpus 0

Training uses scheduled sampling: the gaze history fed to TAKE is drawn from the
ground truth with probability alpha, otherwise from a cache of the model's own
predictions that is refreshed once per epoch. alpha is annealed 1.0 -> 0.0 over
epochs 1-5, after which the history is fully autoregressive.
"""

# Allow running this file directly from anywhere: put the repo root on sys.path.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import logging
import math
import os
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import Gaze360Dataset
from eyetag.models import GazeEstimator
from utils import (angular_error_pitchyaw, collate_fn, save_checkpoint,
                   load_checkpoint, set_seed, vector_to_pitchyaw)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from torch.utils.tensorboard import SummaryWriter


# ── schedules ───────────────────────────────────────────────────────────

def get_alpha(epoch, args):
    if epoch >= args.alpha_end_epoch:
        return args.alpha_end
    progress = (epoch - 1) / max(1, args.alpha_end_epoch - 1)
    return args.alpha_start + progress * (args.alpha_end - args.alpha_start)


def get_prev_gt_ratio(epoch, args):
    if epoch >= args.prev_gt_end_epoch:
        return args.prev_gt_end
    progress = (epoch - 1) / max(1, args.prev_gt_end_epoch - 1)
    return args.prev_gt_start + progress * (args.prev_gt_end - args.prev_gt_start)


def make_lr_lambda(warmup_epochs, total_epochs, autoreg_epoch=None, autoreg_lr_factor=1.0,
                   lr_restart_at=None):
    """LR schedule:
       - warmup → cosine
       - if lr_restart_at: at this epoch, LR resets to 1.0 (= args.lr) and
         cosines down to 0 over the remaining epochs.
       - else if autoreg_epoch: legacy behavior (autoreg_lr_factor scaling).
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if lr_restart_at is not None and epoch >= lr_restart_at:
            progress = (epoch - lr_restart_at) / max(1, total_epochs - lr_restart_at)
            return 0.5 * (1 + np.cos(np.pi * progress))
        if autoreg_epoch is not None and epoch >= autoreg_epoch:
            progress = (epoch - autoreg_epoch) / max(1, total_epochs - autoreg_epoch)
            return autoreg_lr_factor * 0.5 * (1 + np.cos(np.pi * progress))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return lr_lambda


# ── prediction cache (scheduled sampling) ───────────────────────────────

@torch.no_grad()
def build_prediction_cache(model, args, device, log, batch_size=128):
    """Stride=1 inference over the training set to populate prev_gaze cache.

    Returns dict: (sid, frame_idx) → tuple of length D, where D=2 for
    prev_repr='pitchyaw' ((pitch_EVE, yaw_EVE)) or D=3 for prev_repr='vector'
    ((gx, gy, gz) unit vector).
    """
    t0 = time.time()
    model.eval()
    prev_repr = getattr(args, 'prev_repr', 'pitchyaw')

    cache_ds = Gaze360Dataset(
        data_root=args.data_root,
        split='train',
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        face_size=args.face_size,
        eye_size=args.eye_size,
        n_bins=args.n_bins,
        pitch_range=(args.pitch_min, args.pitch_max),
        yaw_range=(args.yaw_min, args.yaw_max),
        seq_stride=1,
        test_mode=True,
        augment=False,
        prediction_cache=None,
        prev_gt_ratio=1.0,
        left_pad=True,
        gaze_subset=args.gaze_subset,
        prev_input=args.prev_input,
        prev_repr=prev_repr,
    )

    cache_workers = max(args.workers * 2, 12)
    cache_loader = DataLoader(
        cache_ds, batch_size=batch_size, shuffle=False,
        num_workers=cache_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
        prefetch_factor=4, persistent_workers=False,
    )

    cache = {}
    for batch in cache_loader:
        face = batch['face'].to(device)
        left = batch['left_eye'].to(device)
        right = batch['right_eye'].to(device)
        prev = batch['prev_gaze'].to(device)

        with autocast(dtype=torch.bfloat16):
            if prev_repr == 'vector':
                pred_vec = model.predict_vector(face, left, right, prev,
                                                camera_id=None)  # (B, 3) unit vec
                pred_np = pred_vec.float().cpu().numpy()
            else:
                pred = model.predict(face, left, right, prev,
                                     camera_id=None)         # (B, 2) pitch/yaw EVE
                pred_np = pred.float().cpu().numpy()
        for i in range(pred_np.shape[0]):
            sid = batch['sid'][i]
            t_idx = batch['target_idx'][i]
            if isinstance(t_idx, torch.Tensor):
                t_idx = t_idx.item()
            cache[(sid, t_idx)] = tuple(float(v) for v in pred_np[i])

    log.info(f'  Cache built: {len(cache)} frames in {time.time()-t0:.1f}s '
             f'(prev_repr={prev_repr})')
    model.train()
    return cache


@torch.no_grad()
def build_prediction_cache_ar(model, args, device, log):
    """True sequential autoregressive cache builder.

    For each sid, predict frame-by-frame in time order. The prev_gaze fed to
    the model at frame i comes from the model's OWN previous predictions
    (zero for positions before any prediction exists), exactly matching
    deployment / autoreg eval distribution.

    Returns dict: (sid, frame_idx) → tuple of length D, where D=2 for
    prev_repr='pitchyaw' ((pitch_EVE, yaw_EVE)) or D=3 for prev_repr='vector'
    ((gx, gy, gz) unit vector).
    """
    import cv2
    from dataset import parse_label_file, build_sid_tracks, normalize_image
    t0 = time.time()
    model.eval()
    prev_repr = getattr(args, 'prev_repr', 'pitchyaw')
    # abs storage dim: 'tangent'/'angvel' roll the chain in pitchyaw space too.
    D = 3 if prev_repr == 'vector' else 2

    rows = parse_label_file(os.path.join(args.data_root, 'train.label'))
    tracks = build_sid_tracks(rows)

    T = args.num_frames
    fs = args.frame_stride
    face_size = args.face_size
    eye_size = args.eye_size

    cache = {}
    n_total = 0
    for sid in sorted(tracks.keys()):
        track = tracks[sid]
        n = len(track)
        if n == 0:
            continue

        # Preload all frames for this sid (no augmentation)
        face_np = np.zeros((n, 3, face_size, face_size), dtype=np.float32)
        left_np = np.zeros((n, 3, eye_size, eye_size), dtype=np.float32)
        right_np = np.zeros((n, 3, eye_size, eye_size), dtype=np.float32)
        for i, row in enumerate(track):
            f = cv2.cvtColor(cv2.imread(os.path.join(args.data_root, row['face']),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            l = cv2.cvtColor(cv2.imread(os.path.join(args.data_root, row['left']),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            r = cv2.cvtColor(cv2.imread(os.path.join(args.data_root, row['right']),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (face_size, face_size))
            l = cv2.resize(l, (eye_size, eye_size))
            r = cv2.resize(r, (eye_size, eye_size))
            face_np[i] = normalize_image(f)
            left_np[i] = normalize_image(l)
            right_np[i] = normalize_image(r)

        predictions = []  # list of D-tuples (pitch_EVE, yaw_EVE) or (gx, gy, gz)
        for i in range(n):
            raw = [i - k * fs for k in range(T - 1, -1, -1)]
            window = [max(0, p) for p in raw]

            face_t = torch.from_numpy(face_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            left_t = torch.from_numpy(left_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            right_t = torch.from_numpy(right_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)

            prev_gaze = np.zeros((T - 1, D), dtype=np.float32)
            for j, wi in enumerate(window[:-1]):
                if wi < len(predictions):
                    prev_gaze[j] = predictions[wi]
            # If model expects delta input, convert abs → delta (shape T-1 → T-2).
            if getattr(args, 'prev_input', 'abs') == 'delta':
                if T - 2 >= 1:
                    if prev_repr in ('tangent', 'angvel'):
                        from utils import delta_from_abs_pitchyaw
                        prev_gaze = delta_from_abs_pitchyaw(prev_gaze, prev_repr)
                    else:
                        prev_gaze = (prev_gaze[1:] - prev_gaze[:-1]).astype(np.float32)
                else:
                    prev_gaze = np.zeros((0, 2 if prev_repr == 'pitchyaw' else 3),
                                         dtype=np.float32)
            prev_t = torch.from_numpy(prev_gaze).unsqueeze(0).to(device)

            with autocast(dtype=torch.bfloat16):
                if prev_repr == 'vector':
                    pred_vec = model.predict_vector(face_t, left_t, right_t,
                                                    prev_t, camera_id=None)  # (1, 3)
                    pred_tuple = tuple(float(v) for v in pred_vec[0].cpu().tolist())
                else:
                    pred = model.predict(face_t, left_t, right_t,
                                         prev_t, camera_id=None)              # (1, 2)
                    pred_tuple = (float(pred[0, 0].cpu().item()),
                                  float(pred[0, 1].cpu().item()))
            predictions.append(pred_tuple)
            cache[(sid, track[i]['idx'])] = pred_tuple
        n_total += n

    log.info(f'  AR Cache built: {len(cache)} frames in {time.time()-t0:.1f}s '
             f'(prev_repr={prev_repr})')
    model.train()
    return cache


# ── evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
@torch.no_grad()
def evaluate(model, val_loader, device, zero_prev=False):
    model.eval()
    all_pp, all_py, all_gp, all_gy = [], [], [], []
    n_nonfinite = 0
    for batch in val_loader:
        face = batch['face'].to(device)
        left = batch['left_eye'].to(device)
        right = batch['right_eye'].to(device)
        prev = batch['prev_gaze'].to(device)
        if zero_prev:
            prev = torch.zeros_like(prev)
        pred = model.predict(face, left, right, prev, camera_id=None)  # (B, 2)
        for i in range(pred.size(0)):
            pp, py = pred[i, 0].cpu().item(), pred[i, 1].cpu().item()
            if not (math.isfinite(pp) and math.isfinite(py)):
                n_nonfinite += 1   # skip: a single bad sample must not nan the mean
                continue
            all_pp.append(pp)
            all_py.append(py)
            all_gp.append(batch['pitch'][i].item())
            all_gy.append(batch['yaw'][i].item())
    model.train()
    if n_nonfinite:
        print(f'[evaluate] WARNING: skipped {n_nonfinite} non-finite predictions')
    if not all_pp:
        return float('inf')
    return angular_error_pitchyaw(
        np.array(all_pp), np.array(all_py),
        np.array(all_gp), np.array(all_gy),
    )


@torch.no_grad()
def evaluate_autoreg_val(model, args, device, log):
    """Autoregressive validation: feed model predictions back as prev_gaze, per-sid."""
    from test import evaluate_autoregressive
    res = evaluate_autoregressive(
        model, args.data_root, split='val', num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        face_size=args.face_size, eye_size=args.eye_size, device=device,
        log=log,
        gaze_subset=args.gaze_subset,
        prev_input=getattr(args, 'prev_input', 'abs'),
        prev_repr=getattr(args, 'prev_repr', 'pitchyaw'),
    )
    return res['angular_error']


# ── main ────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs(args.work_dir, exist_ok=True)

    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('train')
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(os.path.join(args.work_dir, 'train.log'))
    fh.setFormatter(fmt)
    log.addHandler(fh)

    set_seed(args.seed)

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    use_dp = len(gpu_ids) > 1
    log.info(f'Device: {device}  GPUs: {gpu_ids}  DataParallel: {use_dp}  |  work_dir: {args.work_dir}')
    log.info(f'face={args.face_backbone}  eye={args.eye_backbone}  '
             f'prev={args.prev_mode}  temporal={args.temporal_type}  '
             f'fusion={args.fusion_type}  T={args.num_frames}  d={args.d_model}  '
             f'gaze_space={args.gaze_space}')
    log.info(f'Scheduled Sampling: prev_gt {args.prev_gt_start}→{args.prev_gt_end} '
             f'over epochs 1~{args.prev_gt_end_epoch}')

    if WANDB_AVAILABLE and args.wandb:
        wandb.init(project='eyetag-gaze360', name=os.path.basename(args.work_dir),
                   config=vars(args))

    tb_writer = SummaryWriter(log_dir=os.path.join(args.work_dir, 'tb_logs'))
    log.info(f'TensorBoard logs: {os.path.join(args.work_dir, "tb_logs")}')

    # ── Val dataset (fixed, no cache) ──
    val_ds = Gaze360Dataset(
        data_root=args.data_root,
        split='val',
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        face_size=args.face_size,
        eye_size=args.eye_size,
        n_bins=args.n_bins,
        pitch_range=(args.pitch_min, args.pitch_max),
        yaw_range=(args.yaw_min, args.yaw_max),
        test_mode=True,
        augment=False,
        left_pad=True,
        gaze_subset=args.gaze_subset,
        prev_input=args.prev_input,
        prev_repr=args.prev_repr,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=args.prefetch, persistent_workers=True,
    )
    log.info(f'Val: {len(val_ds)}')

    # ── Model ──
    model = GazeEstimator(
        num_frames=args.num_frames,
        face_backbone_type=args.face_backbone,
        eye_backbone_type=args.eye_backbone,
        prev_mode=args.prev_mode,
        prev_input=args.prev_input,
        prev_repr=args.prev_repr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        n_bins=args.n_bins,
        dropout=0.5,
        temporal_type=args.temporal_type,
        fusion_type=args.fusion_type,
        use_pog=args.use_pog,
        gaze_space=args.gaze_space,
        pretrained=True,
        pretrained_vggface=args.pretrained_vggface,
        freeze_backbone=args.freeze_backbone,
        use_camera_emb=False,
        use_head_pose=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f'Model params: {n_params:.2f}M (trainable: {n_trainable:.2f}M)')

    raw_model = model
    if use_dp:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        log.info(f'Wrapped model in DataParallel on GPUs {gpu_ids}')

    # ── Frozen teacher (optional self-distillation) ──
    teacher_model = None
    if args.distill_teacher:
        teacher_model = GazeEstimator(
            num_frames=args.num_frames,
            face_backbone_type=args.face_backbone,
            eye_backbone_type=args.eye_backbone,
            prev_mode=args.prev_mode,
            prev_input=args.prev_input,
            prev_repr=args.prev_repr,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            n_bins=args.n_bins, dropout=0.5,
            temporal_type=args.temporal_type, fusion_type=args.fusion_type,
            use_pog=args.use_pog, gaze_space=args.gaze_space,
            pretrained=False,
            freeze_backbone=False, use_camera_emb=False, use_head_pose=False,
        ).to(device)
        t_ckpt = torch.load(args.distill_teacher, map_location='cpu', weights_only=False)
        t_state = t_ckpt.get('model', t_ckpt.get('state_dict', t_ckpt))
        t_state = {k[7:] if k.startswith('module.') else k: v for k, v in t_state.items()}
        miss, unexp = teacher_model.load_state_dict(t_state, strict=False)
        log.info(f'distill_teacher loaded from {args.distill_teacher} '
                 f'(missing={len(miss)}, unexpected={len(unexp)})')
        teacher_model.eval()
        for p_ in teacher_model.parameters():
            p_.requires_grad = False

    # ── Optimizer with two LR groups (backbones 0.1×) ──
    backbone_param_ids = set()
    if hasattr(raw_model.face_backbone, 'backbone'):
        backbone_param_ids.update(id(p) for p in raw_model.face_backbone.backbone.parameters())
    else:
        backbone_param_ids.update(id(p) for p in raw_model.face_backbone.parameters())
    backbone_param_ids.update(id(p) for p in raw_model.eye_backbone.parameters())

    backbone_params, other_params = [], []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in backbone_param_ids:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    if args.fixed_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        autoreg_epoch = args.prev_gt_end_epoch if args.autoreg_lr_factor < 1.0 else None
        lr_restart_at = args.prev_gt_end_epoch if args.lr_restart_at_prev_zero else None
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, make_lr_lambda(args.warmup_epochs, args.epochs,
                                      autoreg_epoch=autoreg_epoch,
                                      autoreg_lr_factor=args.autoreg_lr_factor,
                                      lr_restart_at=lr_restart_at))
    scaler = GradScaler()

    # ── Resume / init_from ──
    start_epoch = 1
    best_val_err = float('inf')
    if args.resume:
        ckpt = load_checkpoint(args.resume, raw_model, optimizer, scheduler, device)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_err = ckpt.get('best_val_err', float('inf'))
        log.info(f'Resumed from epoch {start_epoch - 1}  |  best={best_val_err:.4f}°')
    elif args.init_from:
        ckpt_raw = torch.load(args.init_from, map_location='cpu', weights_only=False)
        src_state = ckpt_raw.get('model', ckpt_raw.get('state_dict', ckpt_raw))
        dst_state = raw_model.state_dict()
        loaded, skipped = [], []
        for k, v in src_state.items():
            key = k[7:] if k.startswith('module.') else k
            if key in dst_state and dst_state[key].shape == v.shape:
                dst_state[key] = v
                loaded.append(key)
            else:
                skipped.append(key)
        raw_model.load_state_dict(dst_state, strict=False)
        log.info(f'init_from: loaded {len(loaded)} tensors from {args.init_from} '
                 f'(skipped {len(skipped)})')

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        alpha = get_alpha(epoch, args)
        prev_gt_ratio = get_prev_gt_ratio(epoch, args)

        prediction_cache = None
        # When prev_mode='none', model ignores prev_gaze entirely. Cache is wasted.
        if prev_gt_ratio < 1.0 and args.prev_mode != 'none':
            log.info(f'[Epoch {epoch:03d}] Building prediction cache '
                     f'(mode={args.cache_mode}, prev_gt_ratio={prev_gt_ratio:.2f})...')
            if args.cache_mode == 'ar':
                prediction_cache = build_prediction_cache_ar(model, args, device, log)
            else:
                prediction_cache = build_prediction_cache(
                    model, args, device, log,
                    batch_size=args.cache_batch_size,
                )

        train_ds = Gaze360Dataset(
            data_root=args.data_root,
            split='train',
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            face_size=args.face_size,
            eye_size=args.eye_size,
            n_bins=args.n_bins,
            pitch_range=(args.pitch_min, args.pitch_max),
            yaw_range=(args.yaw_min, args.yaw_max),
            seq_stride=args.seq_stride,
            test_mode=False,
            augment=True,
            prediction_cache=prediction_cache,
            prev_gt_ratio=prev_gt_ratio,
            left_pad=True,
            gaze_subset=args.gaze_subset,
            prev_input=args.prev_input,
            prev_repr=args.prev_repr,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
            prefetch_factor=args.prefetch,
        )

        if epoch == start_epoch:
            log.info(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

        model.train()
        total_loss, total_cls, total_reg, total_ang, n = 0.0, 0.0, 0.0, 0.0, 0
        nan_count = 0
        global_step = (epoch - 1) * len(train_loader)

        for step, batch in enumerate(train_loader):
            face = batch['face'].to(device)
            left = batch['left_eye'].to(device)
            right = batch['right_eye'].to(device)
            prev = batch['prev_gaze'].to(device)
            prev_orig = prev
            if args.prev_noise_std > 0:
                prev = prev + torch.randn_like(prev) * args.prev_noise_std
            if args.prev_dropout_prob > 0:
                keep = (torch.rand(prev.shape[0], 1, 1, device=prev.device)
                        >= args.prev_dropout_prob).float()
                prev = prev * keep
            if args.zero_prev:
                prev = torch.zeros_like(prev)
            gt_vec = batch['gaze_vector'].to(device)
            validity = batch['validity'].to(device)

            optimizer.zero_grad()
            loss_kwargs = dict(
                gt_pitch=batch['pitch'].to(device),
                gt_yaw=batch['yaw'].to(device),
                gt_pitch_bin=batch['pitch_bin'].to(device),
                gt_yaw_bin=batch['yaw_bin'].to(device),
                gt_vec=gt_vec,
                gt_pog=None,
                validity=validity,
                gt_head_pose=None,
                head_pose_validity=None,
                alpha=alpha,
                lambda_pog=0.0,
                lambda_head=0.0,
            )

            if args.dual_pass:
                with autocast():
                    face_feat, eye_feat = raw_model.extract_visual(face, left, right)
                    pred_TF = raw_model.predict_from_features(face_feat, eye_feat, prev)
                    losses_TF, ang_TF = raw_model.compute_loss(pred_TF, **loss_kwargs)

                    if 'gaze_vector' in pred_TF:
                        pred_vec = pred_TF['gaze_vector'].detach().float()
                    else:
                        from utils import pitchyaw_to_vector
                        pred_vec = pitchyaw_to_vector(
                            pred_TF['pitch_reg'].detach().float(),
                            pred_TF['yaw_reg'].detach().float(),
                        )
                        pred_vec = torch.nn.functional.normalize(pred_vec, dim=-1, eps=1e-6)
                    # Build the prev token to inject back into the chain in the
                    # representation expected by the model.
                    if args.prev_repr == 'vector':
                        pred_prev = pred_vec.to(prev.dtype)                  # (B, 3)
                    else:
                        p_pitch, p_yaw = vector_to_pitchyaw(pred_vec)
                        pred_prev = torch.stack([p_pitch, p_yaw], dim=-1).to(prev.dtype)  # (B, 2)

                    prev_chain = prev.clone()
                    if args.dual_pass_replace == 'last':
                        prev_chain[:, -1, :] = pred_prev
                    elif args.dual_pass_replace == 'random':
                        T_prev = prev.shape[1]
                        idx = torch.randint(0, T_prev, (prev.shape[0],), device=prev.device)
                        prev_chain[torch.arange(prev.shape[0]), idx, :] = pred_prev
                    else:
                        prev_chain[:] = pred_prev.unsqueeze(1)

                    pred_C = raw_model.predict_from_features(face_feat, eye_feat, prev_chain)
                    losses_C, ang_C = raw_model.compute_loss(pred_C, **loss_kwargs)
                    loss_val = (args.dual_pass_alpha * losses_TF['loss_total']
                                + args.dual_pass_beta * losses_C['loss_total'])

                losses = {
                    'loss_total': loss_val,
                    'loss_cls': (losses_TF['loss_cls'] + losses_C['loss_cls']) * 0.5,
                    'loss_reg': (losses_TF['loss_reg'] + losses_C['loss_reg']) * 0.5,
                    'angular_error': (ang_TF + ang_C) * 0.5,
                    'angular_error_TF': ang_TF,
                    'angular_error_chain': ang_C,
                }
                ang = (ang_TF + ang_C) * 0.5
            else:
                with autocast():
                    pred_out = model(face, left, right, prev)
                    losses, ang = raw_model.compute_loss(pred_out, **loss_kwargs)
                loss_val = losses['loss_total']

                if teacher_model is not None and args.gaze_space == 'vector':
                    with torch.no_grad():
                        with autocast():
                            t_out = teacher_model(face, left, right, prev_orig)
                    s_vec = pred_out['gaze_vector'].float()
                    t_vec = t_out['gaze_vector'].float().detach()
                    cos = (s_vec * t_vec).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
                    loss_distill = (1.0 - cos).mean()
                    loss_val = loss_val + args.distill_lambda * loss_distill
                    losses['loss_distill'] = loss_distill

            if not torch.isfinite(loss_val):
                nan_count += 1
                if nan_count > len(train_loader) // 4:
                    log.error(f'Too many NaN batches ({nan_count}), stopping epoch')
                    break
                optimizer.zero_grad()
                continue

            scaler.scale(loss_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_val.item()
            total_cls += losses['loss_cls'].item()
            total_reg += losses['loss_reg'].item()
            total_ang += losses['angular_error'].item()
            n += 1
            global_step += 1

            if step % 50 == 0:
                lr = optimizer.param_groups[1]['lr']
                extra = ''
                if args.dual_pass:
                    extra = (f'  ang_TF={losses["angular_error_TF"].item():.2f}°'
                             f'  ang_C={losses["angular_error_chain"].item():.2f}°')
                if 'loss_distill' in losses:
                    extra += f'  distill={losses["loss_distill"].item():.4f}'
                log.info(f'[{epoch:03d}][{step:04d}/{len(train_loader)}] '
                         f'loss={loss_val.item():.4f}  '
                         f'ang={losses["angular_error"].item():.2f}°  '
                         f'cls={losses["loss_cls"].item():.4f}  '
                         f'reg={losses["loss_reg"].item():.4f}  '
                         f'lr={lr:.2e}  gt_ratio={prev_gt_ratio:.2f}' + extra)
                tb_writer.add_scalar('step/loss', loss_val.item(), global_step)
                tb_writer.add_scalar('step/angular_error', losses['angular_error'].item(), global_step)
                tb_writer.add_scalar('step/lr', lr, global_step)
                if WANDB_AVAILABLE and args.wandb:
                    log_dict = {
                        'step/loss': loss_val.item(),
                        'step/angular_error': losses['angular_error'].item(),
                        'step/lr': lr,
                        'global_step': global_step,
                    }
                    if args.dual_pass:
                        log_dict['step/ang_TF'] = losses['angular_error_TF'].item()
                        log_dict['step/ang_chain'] = losses['angular_error_chain'].item()
                    if 'loss_distill' in losses:
                        log_dict['step/loss_distill'] = losses['loss_distill'].item()
                    wandb.log(log_dict, step=global_step)

        scheduler.step()
        avg_loss = total_loss / max(n, 1)
        avg_ang = total_ang / max(n, 1)
        cur_lr = optimizer.param_groups[1]['lr']
        log.info(f'[Epoch {epoch:03d}] Train  loss={avg_loss:.4f}  '
                 f'ang={avg_ang:.2f}°  lr={cur_lr:.2e}  '
                 f'prev_gt_ratio={prev_gt_ratio:.2f}'
                 + (f'  (NaN: {nan_count})' if nan_count > 0 else ''))

        # TF val — bootstrap phase only (sanity check during scheduled sampling)
        val_err = None
        # TF val is used as bootstrap sanity (epochs 1..prev_gt_end_epoch) or as the
        # main metric in no-prev mode (where TF == AR).
        if epoch <= args.prev_gt_end_epoch or args.prev_mode == 'none':
            val_err = evaluate(model, val_loader, device, zero_prev=args.zero_prev)
            log.info(f'[Epoch {epoch:03d}] Val(TF) angular_error={val_err:.4f}°'
                     + ('  (zero_prev)' if args.zero_prev else '')
                     + '  [bootstrap sanity]')

        # AR val — every epoch from ep1 (main metric)
        autoreg_val_err = None
        # AR vs TF distinction is meaningless when model has no prev_gaze input.
        if args.autoreg_val and not args.zero_prev and args.prev_mode != 'none':
            log.info(f'[Epoch {epoch:03d}] Running autoregressive validation...')
            autoreg_val_err = evaluate_autoreg_val(model, args, device, log)
            log.info(f'[Epoch {epoch:03d}] Val(AR) angular_error={autoreg_val_err:.4f}°  ★ main metric')

        tb_writer.add_scalar('epoch/train_loss', avg_loss, epoch)
        tb_writer.add_scalar('epoch/train_angular_error', avg_ang, epoch)
        if val_err is not None:
            tb_writer.add_scalar('epoch/val_tf_angular_error', val_err, epoch)
        if autoreg_val_err is not None:
            tb_writer.add_scalar('epoch/val_ar_angular_error', autoreg_val_err, epoch)
        tb_writer.add_scalar('epoch/alpha', alpha, epoch)
        tb_writer.add_scalar('epoch/prev_gt_ratio', prev_gt_ratio, epoch)
        tb_writer.add_scalar('epoch/lr', cur_lr, epoch)
        tb_writer.flush()

        if WANDB_AVAILABLE and args.wandb:
            log_dict = {
                'epoch/train_loss': avg_loss,
                'epoch/train_angular_error': avg_ang,
                'epoch/alpha': alpha,
                'epoch/prev_gt_ratio': prev_gt_ratio,
                'epoch/lr': cur_lr,
                'epoch': epoch,
            }
            if val_err is not None:
                log_dict['epoch/val_tf_angular_error'] = val_err
            if autoreg_val_err is not None:
                log_dict['epoch/val_ar_angular_error'] = autoreg_val_err
            wandb.log(log_dict, step=global_step)

        save_checkpoint(
            os.path.join(args.work_dir, 'latest.pth'),
            epoch, raw_model, optimizer, scheduler, best_val_err, args)
        if args.save_epoch_ckpt and prev_gt_ratio <= 0.0:
            save_checkpoint(
                os.path.join(args.work_dir, f'epoch_{epoch:03d}.pth'),
                epoch, raw_model, optimizer, scheduler, best_val_err, args)

        # best.pth selection — AR val (main metric for prev-using models).
        # When prev_mode='none', AR and TF val are identical (model ignores prev),
        # so val_err (TF) is used as the selector.
        if autoreg_val_err is not None and autoreg_val_err < best_val_err:
            best_val_err = autoreg_val_err
            save_checkpoint(
                os.path.join(args.work_dir, 'best.pth'),
                epoch, raw_model, optimizer, scheduler, best_val_err, args)
            log.info(f'  *** New best AR: {best_val_err:.4f}°')
        elif args.prev_mode == 'none' and val_err is not None and val_err < best_val_err:
            best_val_err = val_err
            save_checkpoint(
                os.path.join(args.work_dir, 'best.pth'),
                epoch, raw_model, optimizer, scheduler, best_val_err, args)
            log.info(f'  *** New best val (no-prev): {best_val_err:.4f}°')

    tb_writer.close()
    log.info(f'Done. Best AR val angular_error: {best_val_err:.4f}°')


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='EyeTag-Gaze360 Training')

    # Data
    p.add_argument('--data-root', required=True,
                   help='Root of the preprocessed Gaze360 dataset')
    p.add_argument('--face-size', type=int, default=128)
    p.add_argument('--eye-size', type=int, default=128)

    # Model
    p.add_argument('--face-backbone', default='vggface2',
                   choices=['vggface2', 'resnet'])
    p.add_argument('--eye-backbone', default='resnet18',
                   choices=['resnet18', 'tsm', '3dcnn', 'efficientnet_b0'])
    p.add_argument('--prev-input', default='abs', choices=['abs', 'delta'],
                   help="prev_gaze input representation. abs: T-1 absolute past gazes "
                        "(default, baseline). delta: T-2 frame-to-frame differences "
                        "(velocity / motion cue). Both modes train the model to output "
                        "absolute gaze.")
    p.add_argument('--prev-repr', default='pitchyaw',
                   choices=['pitchyaw', 'vector', 'tangent', 'angvel'],
                   help="prev_gaze representation space. 'pitchyaw' (default): 2D "
                        "(pitch, yaw) angular space — current baseline behaviour. "
                        "'vector': 3D unit vector (gx, gy, gz) — matches the head's "
                        "output space, avoids yaw wrap-around at ±π, recommended for "
                        "Gaze360 with its panoramic yaw range.")
    p.add_argument('--prev-mode', default='mlp',
                   choices=['mlp', 'dilated', 'transformer', 'linear', 'tcn', 'none'])
    p.add_argument('--temporal-type', default='causal',
                   choices=['transformer', 'gru', 'hybrid', 'causal'])
    p.add_argument('--fusion-type', default='cross_attn',
                   choices=['add', 'separate', 'cross_attn', 'concat',
                            'bidir_cross_attn', 'self_attn_concat'])
    p.add_argument('--num-frames', type=int, default=7)
    p.add_argument('--frame-stride', type=int, default=1,
                   help='Within-window frame spacing. 1=30Hz, 2=15Hz, 3=10Hz '
                        '(gaze360 source is 30fps).')
    p.add_argument('--seq-stride', type=int, default=4,
                   help='Sliding-window stride between consecutive training samples.')
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--n-bins', type=int, default=90)
    p.add_argument('--pitch-min', type=float, default=-1.5707963267948966)
    p.add_argument('--pitch-max', type=float, default=1.5707963267948966)
    p.add_argument('--yaw-min', type=float, default=-3.141592653589793)
    p.add_argument('--yaw-max', type=float, default=3.141592653589793)
    p.add_argument('--gaze-space', default='vector',
                   choices=['pitchyaw', 'vector'],
                   help="'vector' is used for every result in the paper.")
    p.add_argument('--use-pog', action='store_true', default=False)
    p.add_argument('--pretrained-vggface',
                   default='checkpoints/resnet50_ft_weight.pkl',
                   help='Path to cydonia999 VGGFace2 ResNet-50 weights (.pkl).')
    p.add_argument('--freeze-backbone', action='store_true', default=False)

    # Alpha schedule (cls → reg, only used in pitchyaw mode)
    p.add_argument('--alpha-start', type=float, default=1.0)
    p.add_argument('--alpha-end', type=float, default=0.0)
    p.add_argument('--alpha-end-epoch', type=int, default=10)

    # Prev GT ratio schedule (scheduled sampling)
    p.add_argument('--prev-gt-start', type=float, default=1.0)
    p.add_argument('--prev-gt-end', type=float, default=0.0)
    p.add_argument('--prev-gt-end-epoch', type=int, default=10)

    # Training
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--cache-batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--backbone-lr-scale', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.02)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--grad-clip', type=float, default=1.0)

    # System
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--prefetch', type=int, default=4)
    p.add_argument('--gpus', default='0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--work-dir', default='work_dir')

    # Robust prev_gaze handling
    p.add_argument('--prev-noise-std', type=float, default=0.0)
    p.add_argument('--prev-dropout-prob', type=float, default=0.0)
    p.add_argument('--zero-prev', action='store_true', default=False)

    # Dual-pass chain
    p.add_argument('--dual-pass', action='store_true', default=False)
    p.add_argument('--dual-pass-alpha', type=float, default=1.0)
    p.add_argument('--dual-pass-beta', type=float, default=1.0)
    p.add_argument('--dual-pass-replace', default='last',
                   choices=['last', 'random', 'all'])

    # Self-distillation
    p.add_argument('--distill-teacher', default=None)
    p.add_argument('--distill-lambda', type=float, default=1.0)

    # Validation / scheduler / checkpoints
    p.add_argument('--autoreg-val', action='store_true')
    p.add_argument('--autoreg-lr-factor', type=float, default=1.0)
    p.add_argument('--fixed-lr', action='store_true')
    p.add_argument('--lr-restart-at-prev-zero', action='store_true',
                   help='Restart LR to args.lr at prev_gt_end_epoch and cosine down again.')
    p.add_argument('--save-epoch-ckpt', action='store_true')
    p.add_argument('--resume', default=None)
    p.add_argument('--init-from', default=None)
    p.add_argument('--wandb', action='store_true')

    # Prediction-cache mode
    p.add_argument('--cache-mode', default='ar', choices=['tf', 'ar'],
                   help="'ar' (default): true sequential autoregressive cache. "
                        "'tf' (legacy): single-shot TF predictions with GT prev.")

    # Gaze360 standard subset filter
    p.add_argument('--gaze-subset', default='full',
                   choices=['full', 'semi-front', 'front'],
                   help="'full' (default): all samples. "
                        "'semi-front': |yaw|≤90° (front-180). "
                        "'front': |yaw|≤20° (front-20).")

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
