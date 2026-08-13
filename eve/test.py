"""Evaluate an EyeTAG checkpoint on EVE.

Runs per-camera autoregressive inference and reports the mean angular error,
overall and per camera.

    python eve/test.py --data-root /path/to/eve_dataset \
        --checkpoint work_dir/eyetag_eve_t48_delta/latest.pth \
        --split test --cameras webcam_c --target-hz 30 \
        --out work_dir/eyetag_eve_t48_delta/eval.json --gpu 0

Note: `--split test` selects the participants named `val01..val05`, i.e. the
official EVE *validation* split, which is what the paper reports (the official
test annotations are not publicly available).
"""

# Allow `python gaze360/train.py` from anywhere: put the repo root on sys.path.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import os
import os.path as osp
from collections import defaultdict

import cv2
import decord
import h5py
import numpy as np
import torch

from torch.utils.data import DataLoader

from dataset import EVEDataset, get_participants, ALL_CAMERAS, get_subsample_rates
from eyetag.models import GazeEstimator
from utils import angular_error_pitchyaw, collate_fn, load_checkpoint

decord.bridge.set_bridge('native')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_image(img):
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluate model with GT prev_gaze and return per-participant and per-camera results."""
    model.eval()

    results_by_participant = {}
    results_by_camera = defaultdict(lambda: {'pred_pitch': [], 'pred_yaw': [], 'gt_pitch': [], 'gt_yaw': []})
    all_preds_pitch, all_preds_yaw = [], []
    all_gt_pitch, all_gt_yaw = [], []

    use_cam = getattr(model, 'use_camera_emb', False)
    for batch in data_loader:
        face = batch['face'].to(device)
        left = batch['left_eye'].to(device)
        right = batch['right_eye'].to(device)
        prev = batch['prev_gaze'].to(device)
        cam_id = batch['camera_id'].to(device) if use_cam else None

        pred = model.predict(face, left, right, prev, camera_id=cam_id)  # (B, 2)

        for i in range(pred.size(0)):
            if not batch['validity'][i]:
                continue

            p_pitch = pred[i, 0].cpu().item()
            p_yaw = pred[i, 1].cpu().item()
            g_pitch = batch['pitch'][i].item()
            g_yaw = batch['yaw'][i].item()
            participant = batch['participant'][i]
            camera = batch['camera'][i]

            all_preds_pitch.append(p_pitch)
            all_preds_yaw.append(p_yaw)
            all_gt_pitch.append(g_pitch)
            all_gt_yaw.append(g_yaw)

            if participant not in results_by_participant:
                results_by_participant[participant] = {
                    'pred_pitch': [], 'pred_yaw': [],
                    'gt_pitch': [], 'gt_yaw': [],
                }
            results_by_participant[participant]['pred_pitch'].append(p_pitch)
            results_by_participant[participant]['pred_yaw'].append(p_yaw)
            results_by_participant[participant]['gt_pitch'].append(g_pitch)
            results_by_participant[participant]['gt_yaw'].append(g_yaw)

            results_by_camera[camera]['pred_pitch'].append(p_pitch)
            results_by_camera[camera]['pred_yaw'].append(p_yaw)
            results_by_camera[camera]['gt_pitch'].append(g_pitch)
            results_by_camera[camera]['gt_yaw'].append(g_yaw)

    # Overall angular error
    overall_err = angular_error_pitchyaw(
        np.array(all_preds_pitch), np.array(all_preds_yaw),
        np.array(all_gt_pitch), np.array(all_gt_yaw),
    )

    # Per-participant errors
    per_participant = {}
    for pid, data in results_by_participant.items():
        err = angular_error_pitchyaw(
            np.array(data['pred_pitch']), np.array(data['pred_yaw']),
            np.array(data['gt_pitch']), np.array(data['gt_yaw']),
        )
        per_participant[pid] = {'angular_error': err, 'n_samples': len(data['pred_pitch'])}

    # Per-camera errors
    per_camera = {}
    for cam, data in results_by_camera.items():
        err = angular_error_pitchyaw(
            np.array(data['pred_pitch']), np.array(data['pred_yaw']),
            np.array(data['gt_pitch']), np.array(data['gt_yaw']),
        )
        per_camera[cam] = {'angular_error': err, 'n_samples': len(data['pred_pitch'])}

    return {
        'angular_error': overall_err,
        'n_samples': len(all_preds_pitch),
        'per_participant': per_participant,
        'per_camera': per_camera,
    }


@torch.no_grad()
def evaluate_autoregressive(model, data_root, participants, cameras, num_frames,
                            face_size, eye_size, device, batch_size, log,
                            save_per_frame=None, zero_prev_gaze=False,
                            target_hz=10, prev_input='abs'):
    """Autoregressive evaluation: prev_gaze = model's own predictions.

    Multi-camera: sequential inference is run independently for each
    (session, camera) track.
    """
    model.eval()

    if isinstance(cameras, str):
        if cameras == 'all':
            cameras = ALL_CAMERAS
        else:
            cameras = cameras.split(',')

    subsample_rates = get_subsample_rates(target_hz)

    results_by_participant = {}
    results_by_camera = defaultdict(lambda: {'preds_pitch': [], 'preds_yaw': [], 'gt_pitch': [], 'gt_yaw': []})
    total_preds = 0

    for participant in sorted(participants):
        participant_dir = osp.join(data_root, participant)
        if not osp.isdir(participant_dir):
            continue

        part_preds_pitch, part_preds_yaw = [], []
        part_gt_pitch, part_gt_yaw = [], []

        sessions = sorted([
            s for s in os.listdir(participant_dir)
            if osp.isdir(osp.join(participant_dir, s))
            and s.startswith('step')
            and 'eye_tracker_calibration' not in s
        ])

        for session in sessions:
            session_dir = osp.join(participant_dir, session)

            for camera in cameras:
                subsample_rate = subsample_rates[camera]

                h5_path = osp.join(session_dir, f'{camera}.h5')
                face_path = osp.join(session_dir, f'{camera}_face.mp4')
                eyes_path = osp.join(session_dir, f'{camera}_eyes.mp4')

                if not all(osp.exists(p) for p in [h5_path, face_path, eyes_path]):
                    continue

                # Load GT labels
                with h5py.File(h5_path, 'r') as h5:
                    gaze_data = h5['face_g_tobii/data'][:]       # (N, 2)
                    gaze_valid = h5['face_g_tobii/validity'][:]   # (N,)

                total_raw = len(gaze_data)
                sub_indices = list(range(0, total_raw, subsample_rate))
                n_sub = len(sub_indices)
                if n_sub < num_frames:
                    continue

                # Decode all subsampled frames at once
                try:
                    face_vr = decord.VideoReader(face_path, num_threads=2)
                    eyes_vr = decord.VideoReader(eyes_path, num_threads=2)
                    face_all = face_vr.get_batch(sub_indices).asnumpy()
                    eyes_all = eyes_vr.get_batch(sub_indices).asnumpy()
                except Exception as e:
                    log.warning(f'Skip {participant}/{session}/{camera}: {e}')
                    continue

                # Preprocess
                face_np = np.zeros((n_sub, 3, face_size, face_size), dtype=np.float32)
                left_np = np.zeros((n_sub, 3, eye_size, eye_size), dtype=np.float32)
                right_np = np.zeros((n_sub, 3, eye_size, eye_size), dtype=np.float32)

                for i in range(n_sub):
                    face = cv2.resize(face_all[i], (face_size, face_size))
                    eyes = eyes_all[i]
                    mid = eyes.shape[1] // 2
                    left = cv2.resize(eyes[:, :mid, :], (eye_size, eye_size))
                    right = cv2.resize(eyes[:, mid:, :], (eye_size, eye_size))
                    face_np[i] = _normalize_image(face)
                    left_np[i] = _normalize_image(left)
                    right_np[i] = _normalize_image(right)

                del face_all, eyes_all

                # Autoregressive inference per (session, camera)
                T = num_frames
                predictions = []  # (pitch, yaw) per subsampled frame

                for i in range(n_sub):
                    # Build window [i-T+1, ..., i] with left padding
                    start = max(0, i - T + 1)
                    window = list(range(start, i + 1))
                    while len(window) < T:
                        window = [window[0]] + window

                    # Build tensors (B=1)
                    face_t = torch.from_numpy(face_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
                    left_t = torch.from_numpy(left_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
                    right_t = torch.from_numpy(right_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)

                    # prev_gaze from previous predictions (0 if not available)
                    prev_gaze = np.zeros((T - 1, 2), dtype=np.float32)
                    if not zero_prev_gaze:
                        for j, wi in enumerate(window[:-1]):
                            if wi < len(predictions):
                                prev_gaze[j] = predictions[wi]
                    if prev_input == 'delta':
                        if T - 2 >= 1:
                            prev_gaze = (prev_gaze[1:] - prev_gaze[:-1]).astype(np.float32)
                        else:
                            prev_gaze = np.zeros((0, 2), dtype=np.float32)

                    prev_t = torch.from_numpy(prev_gaze).unsqueeze(0).to(device)
                    cam_id_t = None
                    if getattr(model, 'use_camera_emb', False):
                        from dataset import CAMERA_TO_ID
                        cam_id_t = torch.tensor([CAMERA_TO_ID[camera]],
                                                dtype=torch.long, device=device)
                    pred = model.predict(face_t, left_t, right_t, prev_t,
                                         camera_id=cam_id_t)
                    pred_pitch = pred[0, 0].cpu().item()
                    pred_yaw = pred[0, 1].cpu().item()
                    predictions.append((pred_pitch, pred_yaw))

                    # Compute error if GT is valid
                    orig_idx = sub_indices[i]
                    if gaze_valid[orig_idx]:
                        part_preds_pitch.append(pred_pitch)
                        part_preds_yaw.append(pred_yaw)
                        part_gt_pitch.append(float(gaze_data[orig_idx, 0]))
                        part_gt_yaw.append(float(gaze_data[orig_idx, 1]))

                        # Per-camera tracking
                        results_by_camera[camera]['preds_pitch'].append(pred_pitch)
                        results_by_camera[camera]['preds_yaw'].append(pred_yaw)
                        results_by_camera[camera]['gt_pitch'].append(float(gaze_data[orig_idx, 0]))
                        results_by_camera[camera]['gt_yaw'].append(float(gaze_data[orig_idx, 1]))

                total_preds += len(predictions)

        if part_preds_pitch:
            err = angular_error_pitchyaw(
                np.array(part_preds_pitch), np.array(part_preds_yaw),
                np.array(part_gt_pitch), np.array(part_gt_yaw),
            )
            results_by_participant[participant] = {
                'angular_error': err,
                'n_samples': len(part_preds_pitch),
            }
            log.info(f'  {participant}: {err:.4f}° (N={len(part_preds_pitch)})')

    # Overall (weighted average)
    total_n = sum(info['n_samples'] for info in results_by_participant.values())
    weighted_sum = sum(info['angular_error'] * info['n_samples']
                       for info in results_by_participant.values())
    overall_err = weighted_sum / total_n if total_n > 0 else 0.0

    # Per-camera errors
    per_camera = {}
    for cam, data in results_by_camera.items():
        if data['preds_pitch']:
            err = angular_error_pitchyaw(
                np.array(data['preds_pitch']), np.array(data['preds_yaw']),
                np.array(data['gt_pitch']), np.array(data['gt_yaw']),
            )
            per_camera[cam] = {'angular_error': err, 'n_samples': len(data['preds_pitch'])}

    return {
        'angular_error': overall_err,
        'pog_error_px': None,
        'n_samples': total_n,
        'per_participant': results_by_participant,
        'per_camera': per_camera,
    }


def test(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('test')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # ── Restore config from checkpoint ──
    ckpt_raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = ckpt_raw.get('args', {})

    num_frames = ckpt_args.get('num_frames', args.num_frames)
    face_backbone = ckpt_args.get('face_backbone', args.face_backbone)
    eye_backbone = ckpt_args.get('eye_backbone', args.eye_backbone)
    prev_mode = ckpt_args.get('prev_mode', args.prev_mode)
    prev_input = ckpt_args.get('prev_input', getattr(args, 'prev_input', 'abs'))
    d_model = ckpt_args.get('d_model', args.d_model)
    nhead = ckpt_args.get('nhead', args.nhead)
    num_layers_model = ckpt_args.get('num_layers', args.num_layers)
    n_bins = ckpt_args.get('n_bins', args.n_bins)
    temporal_type = ckpt_args.get('temporal_type', args.temporal_type)
    fusion_type = ckpt_args.get('fusion_type', args.fusion_type)
    use_pog = ckpt_args.get('use_pog', True)
    gaze_space = ckpt_args.get('gaze_space', 'pitchyaw')
    face_size = ckpt_args.get('face_size', args.face_size)
    eye_size = ckpt_args.get('eye_size', args.eye_size)
    target_hz = ckpt_args.get('target_hz', args.target_hz)

    # Parse cameras from checkpoint or args
    cameras_str = ckpt_args.get('cameras', args.cameras)
    if cameras_str == 'all':
        cameras = ALL_CAMERAS
    else:
        cameras = cameras_str.split(',') if isinstance(cameras_str, str) else cameras_str

    log.info(f'Config: face={face_backbone} eye={eye_backbone} prev={prev_mode} '
             f'temporal={temporal_type} fusion={fusion_type} T={num_frames} d={d_model} '
             f'target_hz={target_hz}')
    log.info(f'Cameras: {cameras}')

    # ── Dataset ──
    split = args.split
    participants = get_participants(args.data_root, split)

    dataset = EVEDataset(
        data_root=args.data_root,
        participants=participants,
        cameras=cameras,
        num_frames=num_frames,
        face_size=face_size,
        eye_size=eye_size,
        n_bins=n_bins,
        test_mode=True,
        augment=False,
        target_hz=target_hz,
        prev_input=prev_input,
    )
    log.info(f'Test samples ({split}): {len(dataset)}')

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Model ──
    model = GazeEstimator(
        num_frames=num_frames,
        face_backbone_type=face_backbone,
        eye_backbone_type=eye_backbone,
        prev_mode=prev_mode,
        prev_input=prev_input,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers_model,
        n_bins=n_bins,
        temporal_type=temporal_type,
        fusion_type=fusion_type,
        use_pog=use_pog,
        gaze_space=gaze_space,
        pretrained=False,
        pretrained_vggface=None,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    log.info(f'Loaded: {args.checkpoint}')

    # ── Evaluate ──
    if args.autoregressive:
        if args.zero_prev_gaze:
            log.info('Mode: AUTOREGRESSIVE + ZERO prev_gaze (ablation)')
        else:
            log.info('Mode: AUTOREGRESSIVE (prev_gaze = model predictions)')
        results = evaluate_autoregressive(
            model, args.data_root, participants, cameras, num_frames,
            face_size, eye_size, device, args.batch_size, log,
            save_per_frame=args.save_per_frame,
            zero_prev_gaze=args.zero_prev_gaze,
            target_hz=target_hz,
            prev_input=prev_input,
        )
    else:
        log.info('Mode: GT prev_gaze')
        results = evaluate_model(model, data_loader, device)

    log.info(f'\n{"="*60}')
    log.info(f'Angular Error: {results["angular_error"]:.4f}°  (N={results["n_samples"]})')
    log.info(f'{"="*60}')

    # Per-camera breakdown
    if 'per_camera' in results and results['per_camera']:
        log.info('\nPer-camera:')
        log.info(f'  {"Camera":<15} | {"Angular Error":>15} | {"N":>8}')
        log.info(f'  {"-"*15}-+-{"-"*15}-+-{"-"*8}')
        for cam in ALL_CAMERAS:
            if cam in results['per_camera']:
                info = results['per_camera'][cam]
                log.info(f'  {cam:<15} | {info["angular_error"]:>14.4f}° | {info["n_samples"]:>8}')
        log.info(f'  {"-"*15}-+-{"-"*15}-+-{"-"*8}')
        log.info(f'  {"Overall":<15} | {results["angular_error"]:>14.4f}° | {results["n_samples"]:>8}')

    # Per-participant breakdown
    log.info('\nPer-participant:')
    for pid in sorted(results['per_participant'].keys()):
        info = results['per_participant'][pid]
        log.info(f'  {pid}: {info["angular_error"]:.4f}°  (N={info["n_samples"]})')

    # Save results
    if args.out:
        out_data = {
            'angular_error_deg': results['angular_error'],
            'n_samples': results['n_samples'],
            'checkpoint': args.checkpoint,
            'cameras': cameras,
            'per_participant': {
                k: {'angular_error': v['angular_error'], 'n_samples': v['n_samples']}
                for k, v in results['per_participant'].items()
            },
        }
        if 'per_camera' in results:
            out_data['per_camera'] = {
                k: {'angular_error': v['angular_error'], 'n_samples': v['n_samples']}
                for k, v in results['per_camera'].items()
            }
        with open(args.out, 'w') as f:
            json.dump(out_data, f, indent=2)
        log.info(f'Saved results to {args.out}')

    return results


def parse_args():
    p = argparse.ArgumentParser(description='EyeTag-EVE Evaluation (v2: Multi-Camera)')

    p.add_argument('--data-root', required=True,
                   help='Root of the official EVE dataset')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', default='test',
                   choices=['val', 'train', 'test',
                            'val_official', 'train_official', 'test_official'],
                   help="Default 'test' = val01..val05 (EVE official val, our held-out test). "
                        "'val' = train35..train39 (internal val for early stopping). "
                        "'train' = train01..train34. *_official = original EVE split names.")
    p.add_argument('--out', default=None)

    # Defaults (overridden by checkpoint)
    p.add_argument('--cameras', default='all')
    p.add_argument('--face-backbone', default='vggface2',
                   choices=['vggface2', 'resnet'])
    p.add_argument('--eye-backbone', default='resnet18')
    p.add_argument('--prev-mode', default='mlp')
    p.add_argument('--prev-input', default='abs', choices=['abs', 'delta'],
                   help='Fallback if checkpoint does not record prev_input.')
    p.add_argument('--temporal-type', default='transformer')
    p.add_argument('--fusion-type', default='add')
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--target-hz', type=int, default=10,
                   help='Frame sampling rate (overridden by checkpoint args if present)')
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--n-bins', type=int, default=36)
    p.add_argument('--face-size', type=int, default=128)
    p.add_argument('--eye-size', type=int, default=128)

    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--autoregressive', action='store_true', default=False,
                   help='Autoregressive inference: use model predictions as prev_gaze')
    p.add_argument('--zero-prev-gaze', action='store_true', default=False,
                   help='Always use zero prev_gaze (ablation)')
    p.add_argument('--save-per-frame', default=None,
                   help='Save per-frame errors to .npz (autoregressive only)')

    return p.parse_args()


if __name__ == '__main__':
    test(parse_args())
