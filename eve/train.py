"""Train EyeTAG on EVE.

Reproduces the EVE result of the paper (Table 2, 2.56 deg) with:

    python eve/train.py \
        --data-root /path/to/eve_dataset \
        --work-dir  ./work_dir/eyetag_eve_t48_delta \
        --cameras webcam_c --target-hz 30 \
        --face-backbone vggface2 --eye-backbone resnet18 \
        --pretrained-vggface checkpoints/resnet50_ft_weight.pkl \
        --prev-mode mlp --prev-input delta \
        --fusion-type cross_attn --temporal-type causal --gaze-space vector \
        --no-pog --num-frames 48 --seq-stride 4 \
        --epochs 15 --batch-size 32 --lr 3e-4 --warmup-epochs 3 \
        --weight-decay 0.02 --grad-clip 1.0 \
        --prev-gt-start 1.0 --prev-gt-end 0.0 --prev-gt-end-epoch 5 \
        --cache-mode ar --autoreg-val --lr-restart-at-prev-zero \
        --seed 42 --gpus 0

The paper uses the frontal webcam stream (`webcam_c`) only, and evaluates on the
official validation split because the EVE test annotations are not public.
"""

# Allow `python gaze360/train.py` from anywhere: put the repo root on sys.path.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import EVEDataset, get_participants
from eyetag.models import GazeEstimator
from utils import (angular_error_pitchyaw, collate_fn, save_checkpoint,
                   load_checkpoint, set_seed, vector_to_pitchyaw)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Alpha Schedule (cls → reg transition)
# ---------------------------------------------------------------------------

def get_alpha(epoch, args):
    if epoch >= args.alpha_end_epoch:
        return args.alpha_end
    progress = (epoch - 1) / max(1, args.alpha_end_epoch - 1)
    return args.alpha_start + progress * (args.alpha_end - args.alpha_start)


# ---------------------------------------------------------------------------
# Prev GT Ratio Schedule (scheduled sampling: GT → predicted prev_gaze)
# ---------------------------------------------------------------------------

def get_prev_gt_ratio(epoch, args):
    """Returns ratio of GT prev_gaze usage (1.0=all GT, 0.0=all predicted)."""
    if epoch >= args.prev_gt_end_epoch:
        return args.prev_gt_end
    progress = (epoch - 1) / max(1, args.prev_gt_end_epoch - 1)
    return args.prev_gt_start + progress * (args.prev_gt_end - args.prev_gt_start)


# ---------------------------------------------------------------------------
# LR Schedule (Warmup + Cosine Annealing)
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_epochs, total_epochs, autoreg_epoch=None, autoreg_lr_factor=1.0,
                   lr_restart_at=None):
    """LR schedule:
       - warmup → cosine
       - if lr_restart_at: at this epoch, LR resets to 1.0 (= args.lr) and
         cosines down to 0 over the remaining epochs.
       - else if autoreg_epoch: legacy autoreg_lr_factor scaling.
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


# ---------------------------------------------------------------------------
# Cache Generation: stride=1 inference over all training data (multi-camera)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_prediction_cache(model, args, participants, cameras, device, log, batch_size=128):
    """Run stride=1 inference on all training data to build prediction cache.

    Returns:
        dict mapping (participant, session, camera, frame_idx) → (pitch, yaw) as float tuple
    """
    t0 = time.time()
    model.eval()

    # stride=1 dataset: no augmentation, covers every frame
    cache_ds = EVEDataset(
        data_root=args.data_root,
        participants=participants,
        cameras=cameras,
        num_frames=args.num_frames,
        face_size=args.face_size,
        eye_size=args.eye_size,
        n_bins=args.n_bins,
        pitch_range=(args.pitch_min, args.pitch_max),
        yaw_range=(args.yaw_min, args.yaw_max),
        test_mode=True,
        augment=False,
        seq_stride=1,  # cover every frame
        prediction_cache=None,  # use GT while building the cache
        prev_gt_ratio=1.0,
        target_hz=args.target_hz,
        prev_input=args.prev_input,
    )

    cache_workers = max(args.workers * 2, 12)
    cache_loader = DataLoader(
        cache_ds, batch_size=batch_size, shuffle=False,
        num_workers=cache_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
        prefetch_factor=4, persistent_workers=False,
    )

    cache = {}
    use_cam = getattr(args, 'use_camera_emb', False)
    for batch in cache_loader:
        face = batch['face'].to(device)
        left = batch['left_eye'].to(device)
        right = batch['right_eye'].to(device)
        prev = batch['prev_gaze'].to(device)
        camera_id = batch['camera_id'].to(device) if use_cam else None

        with autocast(dtype=torch.bfloat16):
            pred = model.predict(face, left, right, prev, camera_id=camera_id)  # (B, 2)
        pred_np = pred.float().cpu().numpy()

        for i in range(pred_np.shape[0]):
            participant = batch['participant'][i]
            session = batch['session'][i]
            camera = batch['camera'][i]
            target_idx = batch['target_idx'][i]
            if isinstance(target_idx, torch.Tensor):
                target_idx = target_idx.item()
            cache[(participant, session, camera, target_idx)] = (
                float(pred_np[i, 0]), float(pred_np[i, 1])
            )

    elapsed = time.time() - t0
    log.info(f'  Cache built: {len(cache)} frames in {elapsed:.1f}s')
    model.train()
    return cache


@torch.no_grad()
def build_prediction_cache_ar(model, args, participants, cameras_list, device, log):
    """True sequential autoregressive cache builder for EVE.

    For each (participant, session, camera), sequentially predict frame-by-frame.
    The prev_gaze fed to the model is built from past self-predictions only
    (zero for positions before any prediction exists). Matches AR eval distribution.

    Returns: dict (participant, session, camera, subsampled_frame_idx) → (pitch, yaw)
    """
    import os as _os
    import os.path as _osp
    import cv2
    import h5py
    import decord as _decord
    _decord.bridge.set_bridge('native')

    from dataset import (get_subsample_rates, ALL_CAMERAS, normalize_image,
                         CAMERA_TO_ID)

    t0 = time.time()
    model.eval()

    if isinstance(cameras_list, str):
        if cameras_list == 'all':
            cameras_use = ALL_CAMERAS
        else:
            cameras_use = cameras_list.split(',')
    else:
        cameras_use = list(cameras_list)

    subsample_rates = get_subsample_rates(args.target_hz)
    use_cam = getattr(args, 'use_camera_emb', False)
    use_delta = (args.prev_input == 'delta')

    T = args.num_frames
    face_size = args.face_size
    eye_size = args.eye_size

    cache = {}
    n_total_frames = 0
    n_sessions = 0

    for participant in sorted(participants):
        pdir = _osp.join(args.data_root, participant)
        if not _osp.isdir(pdir):
            continue

        sessions = sorted([
            s for s in _os.listdir(pdir)
            if _osp.isdir(_osp.join(pdir, s))
            and s.startswith('step')
            and 'eye_tracker_calibration' not in s
        ])

        for session in sessions:
            sdir = _osp.join(pdir, session)
            for camera in cameras_use:
                rate = subsample_rates[camera]
                face_mp4 = _osp.join(sdir, f'{camera}_face.mp4')
                eyes_mp4 = _osp.join(sdir, f'{camera}_eyes.mp4')
                h5_path = _osp.join(sdir, f'{camera}.h5')
                if not (_osp.exists(face_mp4) and _osp.exists(eyes_mp4) and _osp.exists(h5_path)):
                    continue

                with h5py.File(h5_path, 'r') as h5:
                    n_raw = len(h5['face_g_tobii/data'])

                sub_indices = list(range(0, n_raw, rate))
                n_sub = len(sub_indices)
                if n_sub < T:
                    continue

                try:
                    fvr = _decord.VideoReader(face_mp4, num_threads=2)
                    evr = _decord.VideoReader(eyes_mp4, num_threads=2)
                    face_all = fvr.get_batch(sub_indices).asnumpy()
                    eyes_all = evr.get_batch(sub_indices).asnumpy()
                except Exception:
                    continue

                face_np = np.zeros((n_sub, 3, face_size, face_size), dtype=np.float32)
                left_np = np.zeros((n_sub, 3, eye_size, eye_size), dtype=np.float32)
                right_np = np.zeros((n_sub, 3, eye_size, eye_size), dtype=np.float32)
                for i in range(n_sub):
                    face = cv2.resize(face_all[i], (face_size, face_size))
                    eyes = eyes_all[i]
                    mid = eyes.shape[1] // 2
                    left = cv2.resize(eyes[:, :mid, :], (eye_size, eye_size))
                    right = cv2.resize(eyes[:, mid:, :], (eye_size, eye_size))
                    face_np[i] = normalize_image(face)
                    left_np[i] = normalize_image(left)
                    right_np[i] = normalize_image(right)
                del face_all, eyes_all

                predictions = []  # list of (pitch, yaw) per subsampled frame
                for i in range(n_sub):
                    start = max(0, i - T + 1)
                    window = list(range(start, i + 1))
                    while len(window) < T:
                        window = [window[0]] + window

                    face_t = torch.from_numpy(face_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
                    left_t = torch.from_numpy(left_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
                    right_t = torch.from_numpy(right_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)

                    prev_abs = np.zeros((T - 1, 2), dtype=np.float32)
                    for j, wi in enumerate(window[:-1]):
                        if wi < len(predictions):
                            prev_abs[j] = predictions[wi]
                    if use_delta:
                        if T - 2 >= 1:
                            prev_gaze = (prev_abs[1:] - prev_abs[:-1]).astype(np.float32)
                        else:
                            prev_gaze = np.zeros((0, 2), dtype=np.float32)
                    else:
                        prev_gaze = prev_abs
                    prev_t = torch.from_numpy(prev_gaze).unsqueeze(0).to(device)

                    cam_id_t = None
                    if use_cam:
                        cam_id_t = torch.tensor([CAMERA_TO_ID[camera]],
                                                dtype=torch.long, device=device)
                    with autocast(dtype=torch.bfloat16):
                        pred = model.predict(face_t, left_t, right_t, prev_t,
                                             camera_id=cam_id_t)
                    p_p = pred[0, 0].cpu().item()
                    p_y = pred[0, 1].cpu().item()
                    predictions.append((p_p, p_y))
                    # cache key uses ORIGINAL frame index (not subsampled),
                    # matching how dataset.py stores per-frame keys.
                    cache[(participant, session, camera, sub_indices[i])] = (p_p, p_y)

                n_total_frames += n_sub
                n_sessions += 1

    log.info(f'  AR Cache built: {len(cache)} frames from {n_sessions} sessions '
             f'in {time.time()-t0:.1f}s')
    model.train()
    return cache


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_autoreg_val(model, args, val_participants, cameras, device, log):
    """Autoregressive validation: session-by-session sequential inference for all cameras."""
    from test import evaluate_autoregressive
    results = evaluate_autoregressive(
        model, args.data_root, val_participants, cameras,
        args.num_frames, args.face_size, args.eye_size, device,
        batch_size=args.batch_size, log=log,
        target_hz=args.target_hz,
        prev_input=getattr(args, 'prev_input', 'abs'),
    )
    return results['angular_error']


@torch.no_grad()
def evaluate(model, val_loader, device, use_camera_emb=False, zero_prev=False):
    model.eval()
    all_preds_pitch, all_preds_yaw = [], []
    all_gt_pitch, all_gt_yaw = [], []

    for batch in val_loader:
        face = batch['face'].to(device)
        left = batch['left_eye'].to(device)
        right = batch['right_eye'].to(device)
        prev = batch['prev_gaze'].to(device)
        if zero_prev:
            prev = torch.zeros_like(prev)
        validity = batch['validity']
        camera_id = batch['camera_id'].to(device) if use_camera_emb else None

        pred = model.predict(face, left, right, prev, camera_id=camera_id)  # (B, 2)

        # Only evaluate valid frames
        for i in range(pred.size(0)):
            if validity[i]:
                all_preds_pitch.append(pred[i, 0].cpu().item())
                all_preds_yaw.append(pred[i, 1].cpu().item())
                all_gt_pitch.append(batch['pitch'][i].item())
                all_gt_yaw.append(batch['yaw'][i].item())

    model.train()

    if len(all_preds_pitch) == 0:
        return float('inf')

    return angular_error_pitchyaw(
        np.array(all_preds_pitch), np.array(all_preds_yaw),
        np.array(all_gt_pitch), np.array(all_gt_yaw),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.work_dir, exist_ok=True)

    # Logging
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

    # ── Parse cameras ──
    if args.cameras == 'all':
        cameras = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
    else:
        cameras = args.cameras.split(',')
    log.info(f'Cameras: {cameras}')

    # ── GPU setup ──
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    use_dp = len(gpu_ids) > 1
    log.info(f'Device: {device}  GPUs: {gpu_ids}  DataParallel: {use_dp}  |  work_dir: {args.work_dir}')
    log.info(f'face={args.face_backbone}  eye={args.eye_backbone}  '
             f'prev={args.prev_mode}  temporal={args.temporal_type}  '
             f'fusion={args.fusion_type}  T={args.num_frames}  d={args.d_model}')
    log.info(f'Scheduled Sampling: prev_gt {args.prev_gt_start}→{args.prev_gt_end} '
             f'over epochs 1~{args.prev_gt_end_epoch}')

    if WANDB_AVAILABLE and args.wandb:
        wandb.init(project='eyetag-eve-v2', name=os.path.basename(args.work_dir),
                   config=vars(args))

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.work_dir, 'tb_logs'))
    log.info(f'TensorBoard logs: {os.path.join(args.work_dir, "tb_logs")}')

    # ── Dataset ──
    train_participants = get_participants(args.data_root, 'train')
    val_participants = get_participants(args.data_root, 'val')

    # Validation dataset (fixed; no prediction cache needed)
    val_ds = EVEDataset(
        data_root=args.data_root,
        participants=val_participants,
        cameras=cameras,
        num_frames=args.num_frames,
        face_size=args.face_size,
        eye_size=args.eye_size,
        n_bins=args.n_bins,
        pitch_range=(args.pitch_min, args.pitch_max),
        yaw_range=(args.yaw_min, args.yaw_max),
        test_mode=True,
        augment=False,
        target_hz=args.target_hz,
        prev_input=args.prev_input,
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
        use_camera_emb=args.use_camera_emb,
        use_head_pose=args.use_head_pose,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f'Model params: {n_params:.2f}M (trainable: {n_trainable:.2f}M)')

    # ── DataParallel wrapping ──
    raw_model = model  # keep reference for save/load/compute_loss
    if use_dp:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        log.info(f'Wrapped model in DataParallel on GPUs {gpu_ids}')

    # ── Frozen teacher (for self-distillation) ──
    teacher_model = None
    if args.distill_teacher:
        teacher_model = GazeEstimator(
            num_frames=args.num_frames,
            face_backbone_type=args.face_backbone,
            eye_backbone_type=args.eye_backbone,
            prev_mode=args.prev_mode,
            prev_input=args.prev_input,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            n_bins=args.n_bins,
            dropout=0.5,
            temporal_type=args.temporal_type,
            fusion_type=args.fusion_type,
            use_pog=args.use_pog,
            gaze_space=args.gaze_space,
            pretrained=False,        # weights loaded from teacher checkpoint
            pretrained_vggface=None,
            freeze_backbone=False,
            use_camera_emb=args.use_camera_emb,
            use_head_pose=args.use_head_pose,
        ).to(device)
        t_ckpt = torch.load(args.distill_teacher, map_location='cpu', weights_only=False)
        t_state = t_ckpt.get('state_dict', t_ckpt)
        if 'state_dict' not in t_ckpt and 'model' in t_ckpt:
            t_state = t_ckpt['model']
        t_state = {k[7:] if k.startswith('module.') else k: v for k, v in t_state.items()}
        miss, unexp = teacher_model.load_state_dict(t_state, strict=False)
        log.info(f'distill_teacher: loaded from {args.distill_teacher} '
                 f'(missing={len(miss)} unexpected={len(unexp)})')
        teacher_model.eval()
        for p_ in teacher_model.parameters():
            p_.requires_grad = False

    # ── Optimizer ──
    # Reduce LR on pretrained backbones; full LR on heads, fusion, and any
    # newly-initialized modules attached on top of a frozen backbone
    # (e.g. FaceDINOv2's head_prompt + decoder).
    backbone_param_ids = set()
    if hasattr(raw_model.face_backbone, 'backbone'):
        # FaceDINOv2: only the actual pretrained model is in .backbone;
        # head_prompt / decoder / decoder_norm should get full LR.
        backbone_param_ids.update(id(p) for p in raw_model.face_backbone.backbone.parameters())
    else:
        backbone_param_ids.update(id(p) for p in raw_model.face_backbone.parameters())
    backbone_param_ids.update(id(p) for p in raw_model.eye_backbone.parameters())

    backbone_params = []
    other_params = []
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

    # ── Resume ──
    start_epoch = 1
    best_val_err = float('inf')

    if args.resume:
        ckpt = load_checkpoint(args.resume, raw_model, optimizer, scheduler, device)
        start_epoch = ckpt['epoch'] + 1
        best_val_err = ckpt.get('best_val_err', float('inf'))
        log.info(f'Resumed from epoch {start_epoch - 1}  |  best={best_val_err:.4f}°')
    elif args.init_from:
        ckpt_raw = torch.load(args.init_from, map_location='cpu', weights_only=False)
        src_state = ckpt_raw.get('state_dict', ckpt_raw)
        if 'state_dict' not in ckpt_raw and 'model' in ckpt_raw:
            src_state = ckpt_raw['model']
        dst_state = raw_model.state_dict()
        loaded, skipped_shape, skipped_missing = [], [], []
        for k, v in src_state.items():
            key = k[7:] if k.startswith('module.') else k
            if key not in dst_state:
                skipped_missing.append(key)
                continue
            if dst_state[key].shape != v.shape:
                skipped_shape.append(f'{key} {tuple(v.shape)}→{tuple(dst_state[key].shape)}')
                continue
            dst_state[key] = v
            loaded.append(key)
        raw_model.load_state_dict(dst_state, strict=False)
        log.info(f'init_from: loaded {len(loaded)} tensors from {args.init_from}')
        if skipped_shape:
            log.info(f'  skipped (shape mismatch, {len(skipped_shape)}): '
                     + '; '.join(skipped_shape[:5])
                     + (' ...' if len(skipped_shape) > 5 else ''))
        if skipped_missing:
            log.info(f'  skipped (not in model, {len(skipped_missing)})')

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        alpha = get_alpha(epoch, args)
        prev_gt_ratio = get_prev_gt_ratio(epoch, args)

        # -- Build the prediction cache (from epoch 2, only while ratio < 1.0) --
        # When prev_mode='none', cache is unused (model ignores prev_gaze).
        prediction_cache = None
        if prev_gt_ratio < 1.0 and args.prev_mode != 'none':
            log.info(f'[Epoch {epoch:03d}] Building prediction cache '
                     f'(mode={args.cache_mode}, prev_gt_ratio={prev_gt_ratio:.2f})...')
            if args.cache_mode == 'ar':
                prediction_cache = build_prediction_cache_ar(
                    model, args, train_participants, cameras, device, log)
            else:
                prediction_cache = build_prediction_cache(
                    model, args, train_participants, cameras, device, log,
                    batch_size=args.cache_batch_size,
                )

        # ── Create train dataset with current cache & ratio ──
        train_ds = EVEDataset(
            data_root=args.data_root,
            participants=train_participants,
            cameras=cameras,
            num_frames=args.num_frames,
            face_size=args.face_size,
            eye_size=args.eye_size,
            n_bins=args.n_bins,
            pitch_range=(args.pitch_min, args.pitch_max),
            yaw_range=(args.yaw_min, args.yaw_max),
            test_mode=False,
            augment=True,
            seq_stride=args.seq_stride,
            prediction_cache=prediction_cache,
            prev_gt_ratio=prev_gt_ratio,
            target_hz=args.target_hz,
            prev_input=args.prev_input,
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

        total_loss, total_cls, total_reg, total_pog, total_ang, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        nan_count = 0
        global_step = (epoch - 1) * len(train_loader)

        for step, batch in enumerate(train_loader):
            face = batch['face'].to(device)
            left = batch['left_eye'].to(device)
            right = batch['right_eye'].to(device)
            prev = batch['prev_gaze'].to(device)
            prev_orig = prev  # kept for the teacher in self-distillation
            if args.prev_noise_std > 0:
                prev = prev + torch.randn_like(prev) * args.prev_noise_std
            if args.prev_dropout_prob > 0:
                keep = (torch.rand(prev.shape[0], 1, 1, device=prev.device)
                        >= args.prev_dropout_prob).float()
                prev = prev * keep
            if args.zero_prev:
                prev = torch.zeros_like(prev)
            validity = batch['validity'].to(device)
            camera_id = batch['camera_id'].to(device) if args.use_camera_emb else None
            gt_head = batch['head_pose'].to(device) if args.use_head_pose else None
            head_valid = batch['head_pose_validity'].to(device) if args.use_head_pose else None

            optimizer.zero_grad()
            loss_kwargs = dict(
                gt_pitch=batch['pitch'].to(device),
                gt_yaw=batch['yaw'].to(device),
                gt_pitch_bin=batch['pitch_bin'].to(device),
                gt_yaw_bin=batch['yaw_bin'].to(device),
                gt_pog=batch['pog'].to(device) if args.use_pog else None,
                validity=validity,
                gt_head_pose=gt_head,
                head_pose_validity=head_valid,
                alpha=alpha,
                lambda_pog=args.lambda_pog,
                lambda_head=args.lambda_head,
            )

            if args.dual_pass:
                with autocast():
                    # Pass 1: TF — backbones forwarded once, cached for pass 2.
                    face_feat, eye_feat = raw_model.extract_visual(face, left, right)
                    pred_TF = raw_model.predict_from_features(
                        face_feat, eye_feat, prev, camera_id=camera_id)
                    losses_TF, ang_TF = raw_model.compute_loss(pred_TF, **loss_kwargs)

                    # Build prev_chain by replacing prev_gaze position(s) with detached
                    # pass-1 prediction (in pitch/yaw radians).
                    pred_vec = pred_TF['gaze_vector'].detach().float()
                    p_pitch, p_yaw = vector_to_pitchyaw(pred_vec)
                    pred_pitchyaw = torch.stack([p_pitch, p_yaw], dim=-1).to(prev.dtype)

                    prev_chain = prev.clone()
                    if args.dual_pass_replace == 'last':
                        prev_chain[:, -1, :] = pred_pitchyaw
                    elif args.dual_pass_replace == 'random':
                        T_prev = prev.shape[1]
                        idx = torch.randint(0, T_prev, (prev.shape[0],), device=prev.device)
                        prev_chain[torch.arange(prev.shape[0]), idx, :] = pred_pitchyaw
                    else:  # 'all'
                        prev_chain[:] = pred_pitchyaw.unsqueeze(1)

                    # Pass 2: Chain — reuse backbone features.
                    pred_C = raw_model.predict_from_features(
                        face_feat, eye_feat, prev_chain, camera_id=camera_id)
                    losses_C, ang_C = raw_model.compute_loss(pred_C, **loss_kwargs)

                    loss_val = (args.dual_pass_alpha * losses_TF['loss_total']
                                + args.dual_pass_beta * losses_C['loss_total'])

                losses = {
                    'loss_total': loss_val,
                    'loss_cls': (losses_TF['loss_cls'] + losses_C['loss_cls']) * 0.5,
                    'loss_reg': (losses_TF['loss_reg'] + losses_C['loss_reg']) * 0.5,
                    'loss_pog': (losses_TF['loss_pog'] + losses_C['loss_pog']) * 0.5,
                    'angular_error': (ang_TF + ang_C) * 0.5,
                    'angular_error_TF': ang_TF,
                    'angular_error_chain': ang_C,
                }
                ang = (ang_TF + ang_C) * 0.5
            else:
                with autocast():
                    pred_out = model(face, left, right, prev, camera_id=camera_id)
                    losses, ang = raw_model.compute_loss(pred_out, **loss_kwargs)
                loss_val = losses['loss_total']
                if teacher_model is not None and args.gaze_space == 'vector':
                    with torch.no_grad():
                        with autocast():
                            t_out = teacher_model(face, left, right, prev_orig,
                                                  camera_id=camera_id)
                    s_vec = pred_out['gaze_vector'].float()
                    t_vec = t_out['gaze_vector'].float().detach()
                    valid_mask = validity.bool()
                    if valid_mask.any():
                        s_v = s_vec[valid_mask]
                        t_v = t_vec[valid_mask]
                        cos = (s_v * t_v).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
                        loss_distill = (1.0 - cos).mean()
                    else:
                        loss_distill = torch.tensor(0.0, device=device)
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
            total_pog += losses['loss_pog'].item()
            total_ang += losses['angular_error'].item()
            n += 1
            global_step += 1

            # Step-level logging (every 50 steps)
            if step % 50 == 0:
                lr = optimizer.param_groups[1]['lr']
                dual_str = ''
                if args.dual_pass:
                    dual_str = (f'  ang_TF={losses["angular_error_TF"].item():.2f}°'
                                f'  ang_C={losses["angular_error_chain"].item():.2f}°')
                if 'loss_distill' in losses:
                    dual_str += f'  distill={losses["loss_distill"].item():.4f}'
                log.info(f'[{epoch:03d}][{step:04d}/{len(train_loader)}] '
                         f'loss={loss_val.item():.4f}  '
                         f'ang={losses["angular_error"].item():.2f}°  '
                         f'cls={losses["loss_cls"].item():.4f}  '
                         f'reg={losses["loss_reg"].item():.4f}  '
                         f'lr={lr:.2e}  '
                         f'gt_ratio={prev_gt_ratio:.2f}'
                         + dual_str)

                # TensorBoard step logging
                tb_writer.add_scalar('step/loss', loss_val.item(), global_step)
                tb_writer.add_scalar('step/angular_error', losses['angular_error'].item(), global_step)
                tb_writer.add_scalar('step/loss_cls', losses['loss_cls'].item(), global_step)
                tb_writer.add_scalar('step/loss_reg', losses['loss_reg'].item(), global_step)
                tb_writer.add_scalar('step/loss_pog', losses['loss_pog'].item(), global_step)
                tb_writer.add_scalar('step/lr', lr, global_step)
                if args.dual_pass:
                    tb_writer.add_scalar('step/ang_TF', losses['angular_error_TF'].item(), global_step)
                    tb_writer.add_scalar('step/ang_chain', losses['angular_error_chain'].item(), global_step)

                if WANDB_AVAILABLE and args.wandb:
                    wandb_log = {
                        'step/loss': loss_val.item(),
                        'step/angular_error': losses['angular_error'].item(),
                        'step/loss_cls': losses['loss_cls'].item(),
                        'step/loss_reg': losses['loss_reg'].item(),
                        'step/loss_pog': losses['loss_pog'].item(),
                        'step/lr': lr,
                        'global_step': global_step,
                    }
                    if args.dual_pass:
                        wandb_log['step/ang_TF'] = losses['angular_error_TF'].item()
                        wandb_log['step/ang_chain'] = losses['angular_error_chain'].item()
                    if 'loss_distill' in losses:
                        wandb_log['step/loss_distill'] = losses['loss_distill'].item()
                    wandb.log(wandb_log, step=global_step)

        scheduler.step()
        avg_loss = total_loss / max(n, 1)
        avg_cls = total_cls / max(n, 1)
        avg_reg = total_reg / max(n, 1)
        avg_pog = total_pog / max(n, 1)
        avg_ang = total_ang / max(n, 1)
        cur_lr = optimizer.param_groups[1]['lr']
        log.info(f'[Epoch {epoch:03d}] Train  loss={avg_loss:.4f}  '
                 f'cls={avg_cls:.4f}  reg={avg_reg:.4f}  pog={avg_pog:.4f}  '
                 f'ang={avg_ang:.2f}°  lr={cur_lr:.2e}  '
                 f'prev_gt_ratio={prev_gt_ratio:.2f}'
                 + (f'  (NaN: {nan_count})' if nan_count > 0 else ''))

        # Validation (teacher-forcing; with --zero-prev, prev=0 → TF == autoreg).
        val_err = evaluate(model, val_loader, device,
                           use_camera_emb=args.use_camera_emb,
                           zero_prev=args.zero_prev)
        log.info(f'[Epoch {epoch:03d}] Val    angular_error={val_err:.4f}°'
                 + ('  (zero_prev)' if args.zero_prev else ''))

        # Autoregressive validation (when gt_ratio=0 and enabled).
        # Skipped under --zero-prev because the student ignores prev → autoreg == TF.
        autoreg_val_err = None
        if args.autoreg_val and prev_gt_ratio <= 0.0 and not args.zero_prev:
            log.info(f'[Epoch {epoch:03d}] Running autoregressive validation...')
            autoreg_val_err = evaluate_autoreg_val(
                model, args, val_participants, cameras, device, log)
            log.info(f'[Epoch {epoch:03d}] Val(autoreg) angular_error={autoreg_val_err:.4f}°')

        # TensorBoard epoch logging
        tb_writer.add_scalar('epoch/train_loss', avg_loss, epoch)
        tb_writer.add_scalar('epoch/train_loss_cls', avg_cls, epoch)
        tb_writer.add_scalar('epoch/train_loss_reg', avg_reg, epoch)
        tb_writer.add_scalar('epoch/train_loss_pog', avg_pog, epoch)
        tb_writer.add_scalar('epoch/train_angular_error', avg_ang, epoch)
        tb_writer.add_scalar('epoch/val_angular_error', val_err, epoch)
        if autoreg_val_err is not None:
            tb_writer.add_scalar('epoch/val_autoreg_angular_error', autoreg_val_err, epoch)
        tb_writer.add_scalar('epoch/alpha', alpha, epoch)
        tb_writer.add_scalar('epoch/prev_gt_ratio', prev_gt_ratio, epoch)
        tb_writer.add_scalar('epoch/lr', cur_lr, epoch)
        tb_writer.flush()

        if WANDB_AVAILABLE and args.wandb:
            log_dict = {
                'epoch/train_loss': avg_loss,
                'epoch/train_loss_cls': avg_cls,
                'epoch/train_loss_reg': avg_reg,
                'epoch/train_loss_pog': avg_pog,
                'epoch/train_angular_error': avg_ang,
                'epoch/val_angular_error': val_err,
                'epoch/alpha': alpha,
                'epoch/prev_gt_ratio': prev_gt_ratio,
                'epoch/lr': cur_lr,
                'epoch': epoch,
            }
            if autoreg_val_err is not None:
                log_dict['epoch/val_autoreg_angular_error'] = autoreg_val_err
            if val_err < best_val_err:
                log_dict['epoch/best_val_angular_error'] = val_err
            wandb.log(log_dict, step=global_step)

        save_checkpoint(
            os.path.join(args.work_dir, 'latest.pth'),
            epoch, raw_model, optimizer, scheduler, best_val_err, args)

        # Save per-epoch checkpoint (after autoreg phase starts)
        if args.save_epoch_ckpt and prev_gt_ratio <= 0.0:
            save_checkpoint(
                os.path.join(args.work_dir, f'epoch_{epoch:03d}.pth'),
                epoch, raw_model, optimizer, scheduler, best_val_err, args)

        # Best checkpoint selection
        select_err = autoreg_val_err if autoreg_val_err is not None else val_err
        if select_err < best_val_err:
            best_val_err = select_err
            save_checkpoint(
                os.path.join(args.work_dir, 'best.pth'),
                epoch, raw_model, optimizer, scheduler, best_val_err, args)
            log.info(f'  *** New best: {best_val_err:.4f}°'
                     + (' (autoreg)' if autoreg_val_err is not None else ''))

    tb_writer.close()
    log.info(f'Done. Best val angular_error: {best_val_err:.4f}°')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='EyeTag-EVE Training (v2: Multi-Camera)')

    # Data
    p.add_argument('--data-root', required=True,
                   help='Root of the official EVE dataset')
    p.add_argument('--cameras', default='all',
                   help="Comma-separated cameras: 'all' or 'webcam_c,webcam_l' etc.")
    p.add_argument('--face-size', type=int, default=128)
    p.add_argument('--eye-size', type=int, default=128)

    # Model
    p.add_argument('--face-backbone', default='vggface2',
                   choices=['vggface2', 'resnet'])
    p.add_argument('--eye-backbone', default='resnet18',
                   choices=['resnet18', 'tsm', '3dcnn', 'efficientnet_b0'])
    p.add_argument('--prev-mode', default='mlp',
                   choices=['mlp', 'dilated', 'transformer', 'linear', 'tcn', 'none'])
    p.add_argument('--prev-input', default='abs', choices=['abs', 'delta'],
                   help="prev_gaze representation. abs: T-1 absolute past gazes (default). "
                        "delta: T-2 frame-to-frame differences (velocity / motion cue).")
    p.add_argument('--temporal-type', default='transformer',
                   choices=['transformer', 'gru', 'hybrid', 'causal'])
    p.add_argument('--fusion-type', default='add',
                   choices=['add', 'separate', 'cross_attn', 'concat', 'bidir_cross_attn'])
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--seq-stride', type=int, default=4,
                   help='Sliding window stride for training sequence construction')
    p.add_argument('--target-hz', type=int, default=10,
                   help='Frame sampling rate (10/15/30). Both training & inference must match.')
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--n-bins', type=int, default=36)
    p.add_argument('--pitch-min', type=float, default=-0.628)
    p.add_argument('--pitch-max', type=float, default=0.628)
    p.add_argument('--yaw-min', type=float, default=-0.628)
    p.add_argument('--yaw-max', type=float, default=0.628)
    p.add_argument('--use-pog', action='store_true', default=True)
    p.add_argument('--no-pog', dest='use_pog', action='store_false')
    p.add_argument('--use-camera-emb', action='store_true', default=False,
                   help='Add learnable camera_id embedding to visual tokens')
    p.add_argument('--use-head-pose', action='store_true', default=False,
                   help='Add auxiliary head-pose (pitch, yaw) regression head')
    p.add_argument('--lambda-head', type=float, default=0.0,
                   help='Weight on head_pose aux loss (only used if --use-head-pose)')
    p.add_argument('--gaze-space', default='pitchyaw',
                   choices=['pitchyaw', 'vector'],
                   help='Gaze output space: pitchyaw (cls+reg) or vector (3D angular loss)')
    p.add_argument('--pretrained-vggface',
                   default='checkpoints/resnet50_ft_weight.pkl',
                   help='VGGFace2 ResNet-50 weights (cydonia999 format).')
    p.add_argument('--freeze-backbone', action='store_true', default=False)

    # Alpha schedule (cls → reg)
    p.add_argument('--alpha-start', type=float, default=1.0)
    p.add_argument('--alpha-end', type=float, default=0.0)
    p.add_argument('--alpha-end-epoch', type=int, default=10)

    # Prev GT ratio schedule (GT → predicted)
    p.add_argument('--prev-gt-start', type=float, default=1.0,
                   help='Initial GT ratio for prev_gaze (1.0=all GT)')
    p.add_argument('--prev-gt-end', type=float, default=0.0,
                   help='Final GT ratio for prev_gaze (0.0=all predicted)')
    p.add_argument('--prev-gt-end-epoch', type=int, default=10,
                   help='Epoch at which prev_gt_ratio reaches prev-gt-end')

    # Training
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--cache-batch-size', type=int, default=128,
                   help='Batch size for cache generation inference (larger=faster)')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--backbone-lr-scale', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.02)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--lambda-pog', type=float, default=0.01)

    # System
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--prefetch', type=int, default=4)
    p.add_argument('--gpus', default='0',
                   help='Comma-separated GPU IDs (e.g., "0" or "1,2" for DataParallel)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--work-dir', default='work_dir')
    p.add_argument('--prev-noise-std', type=float, default=0.0,
                   help='Gaussian noise std (rad) added to prev_gaze during training (0=disabled)')
    p.add_argument('--prev-dropout-prob', type=float, default=0.0,
                   help='Probability of zeroing prev_gaze for each sample (0=disabled). '
                        'Forces robustness when prev_gaze is unreliable at inference.')
    p.add_argument('--dual-pass', action='store_true', default=False,
                   help='Enable dual-pass temporal chain training: forward backbones once, '
                        'temporal+head twice. Pass 2 substitutes a position of prev_gaze '
                        'with pass-1 (detached) prediction. Adds ~10-20%% compute, directly '
                        'targets exposure bias.')
    p.add_argument('--dual-pass-alpha', type=float, default=1.0,
                   help='Loss weight for TF pass (pass 1) in dual-pass training.')
    p.add_argument('--dual-pass-beta', type=float, default=1.0,
                   help='Loss weight for chain pass (pass 2) in dual-pass training.')
    p.add_argument('--dual-pass-replace', default='last',
                   choices=['last', 'random', 'all'],
                   help='Which prev_gaze position(s) to replace with TF prediction in pass 2.')
    p.add_argument('--distill-teacher', default=None,
                   help='Path to a frozen teacher checkpoint. Enables self-distillation: '
                        'teacher forwards with the original prev_GT, student matches teacher '
                        'gaze_vector via cosine loss. Pair with --zero-prev to train a '
                        'no-prev student that has no autoreg gap by construction.')
    p.add_argument('--distill-lambda', type=float, default=1.0,
                   help='Weight on the self-distillation loss term.')
    p.add_argument('--zero-prev', action='store_true', default=False,
                   help='Zero out prev_gaze during BOTH training and validation. '
                        'Forces the student to predict gaze without temporal prev signal, '
                        'making teacher-forcing == autoregressive at inference.')
    p.add_argument('--autoreg-val', action='store_true',
                   help='Run autoregressive validation after gt_ratio reaches 0')
    p.add_argument('--autoreg-lr-factor', type=float, default=1.0,
                   help='LR multiplier for autoregressive phase (e.g., 0.3).')
    p.add_argument('--fixed-lr', action='store_true',
                   help='Use fixed LR (no warmup, no cosine decay)')
    p.add_argument('--lr-restart-at-prev-zero', action='store_true', default=False,
                   help='Restart LR (warmup+cosine) at prev_gt_end_epoch. Helps AR-phase '
                        'learning when scheduled sampling ends.')
    p.add_argument('--cache-mode', default='ar', choices=['tf', 'ar'],
                   help="Prediction cache build mode. 'ar': true sequential autoregressive cache "
                        "(matches eval distribution). 'tf': teacher-forcing single-pass cache.")
    p.add_argument('--save-epoch-ckpt', action='store_true',
                   help='Save per-epoch checkpoints after autoreg phase starts')
    p.add_argument('--resume', default=None)
    p.add_argument('--init-from', default=None,
                   help='Load weights from a checkpoint, skipping shape-mismatched keys '
                        '(use for warm-start when changing T/num_frames). Unlike --resume, '
                        'optimizer/scheduler/epoch are NOT restored.')
    p.add_argument('--wandb', action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
