"""Evaluate an EyeTAG checkpoint on Gaze360.

Two inference modes:
  --autoregressive  gaze history is built from the model's own predictions,
                    rolled out per subject track. This is the deployment
                    setting and the one reported in the paper.
  (default)         teacher forcing, i.e. ground-truth gaze history.

The paper reports the three standard yaw subsets; run once per subset:

    for S in full semi-front front; do
        python gaze360/test.py --data-root /path/to/gaze360 \
            --checkpoint work_dir/eyetag_t48_delta/best.pth \
            --split test --autoregressive --gaze-subset $S \
            --out work_dir/eyetag_t48_delta/eval_test_$S.json --gpu 0
    done
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
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (Gaze360Dataset, parse_label_file, build_sid_tracks,
                     normalize_image, GAZE360_SUBSET_YAW_THRESH_RAD)
from eyetag.models import GazeEstimator
from utils import (angular_error_pitchyaw, angular_error_vector,
                   collate_fn, load_checkpoint, pitchyaw_to_vector)


# ── per-sample std helper ───────────────────────────────────────────────

def per_sample_std(pred_pitch, pred_yaw, gt_pitch, gt_yaw):
    """Standard deviation of the per-sample angular errors (degrees)."""
    from utils import pitchyaw_to_vector
    pred_vec = pitchyaw_to_vector(pred_pitch, pred_yaw)
    gt_vec = pitchyaw_to_vector(gt_pitch, gt_yaw)
    pred_vec = pred_vec / (np.linalg.norm(pred_vec, axis=1, keepdims=True) + 1e-8)
    gt_vec = gt_vec / (np.linalg.norm(gt_vec, axis=1, keepdims=True) + 1e-8)
    dots = np.clip((pred_vec * gt_vec).sum(axis=1), -1.0, 1.0)
    errors = np.degrees(np.arccos(dots))
    return float(errors.std())


# ── per-yaw-bin error helper (gaze360 standard splits) ──────────────────

def split_by_yaw_bin(pred_p, pred_y, gt_p, gt_y, gt_yaw_g360):
    """Split errors by absolute gaze360 yaw bin.

    gaze360 standard:
      - frontal:           |yaw_g360| <= 20°
      - less-frontal:      20° < |yaw_g360| <= 90°
      - backward:          |yaw_g360| > 90°
    """
    bins = {'frontal': [], 'less_frontal': [], 'backward': []}
    abs_yaw_deg = np.abs(np.degrees(gt_yaw_g360))
    front_mask = abs_yaw_deg <= 20
    less_mask = (abs_yaw_deg > 20) & (abs_yaw_deg <= 90)
    back_mask = abs_yaw_deg > 90

    def err(mask):
        if not np.any(mask):
            return None, 0
        return angular_error_pitchyaw(
            pred_p[mask], pred_y[mask], gt_p[mask], gt_y[mask]), int(mask.sum())

    bins['frontal'] = err(front_mask)
    bins['less_frontal'] = err(less_mask)
    bins['backward'] = err(back_mask)
    return bins


# ── TF evaluation (GT prev_gaze) ────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    pp, py, gp, gy = [], [], [], []
    per_sid = defaultdict(lambda: {'pp': [], 'py': [], 'gp': [], 'gy': []})
    g360_yaw_list = []

    for batch in data_loader:
        face = batch['face'].to(device)
        left = batch['left_eye'].to(device)
        right = batch['right_eye'].to(device)
        prev = batch['prev_gaze'].to(device)
        pred = model.predict(face, left, right, prev, camera_id=None)  # (B, 2) pitch/yaw EVE
        for i in range(pred.size(0)):
            sid = batch['sid'][i]
            p_p, p_y = pred[i, 0].cpu().item(), pred[i, 1].cpu().item()
            g_p, g_y = batch['pitch'][i].item(), batch['yaw'][i].item()
            pp.append(p_p); py.append(p_y); gp.append(g_p); gy.append(g_y)
            per_sid[sid]['pp'].append(p_p)
            per_sid[sid]['py'].append(p_y)
            per_sid[sid]['gp'].append(g_p)
            per_sid[sid]['gy'].append(g_y)
            # gaze360 yaw = -yaw_EVE (we use this for bin splits)
            g360_yaw_list.append(-g_y)

    pp = np.array(pp); py = np.array(py); gp = np.array(gp); gy = np.array(gy)
    g360_yaw = np.array(g360_yaw_list)

    overall_err = angular_error_pitchyaw(pp, py, gp, gy)
    overall_std = float(per_sample_std(pp, py, gp, gy))
    per_sid_out = {}
    for sid, d in per_sid.items():
        e = angular_error_pitchyaw(
            np.array(d['pp']), np.array(d['py']),
            np.array(d['gp']), np.array(d['gy']))
        per_sid_out[sid] = {'angular_error': e, 'n_samples': len(d['pp'])}

    yaw_bins = split_by_yaw_bin(pp, py, gp, gy, g360_yaw)
    return {
        'angular_error': overall_err,
        'angular_error_std': overall_std,
        'n_samples': len(pp),
        'per_sid': per_sid_out,
        'per_yaw_bin': yaw_bins,
    }


# ── Autoregressive evaluation ───────────────────────────────────────────

@torch.no_grad()
def evaluate_autoregressive(model, data_root, split, num_frames, face_size, eye_size,
                            device, log, frame_stride=1, zero_prev_gaze=False,
                            gaze_subset='full', prev_input='abs',
                            prev_repr='pitchyaw'):
    """Per-sid sequential inference. prev_gaze comes from past predictions.

    frame_stride: within-window frame spacing matching the training config
                  (1=30Hz, 2=15Hz, 3=10Hz).

    gaze_subset: 'full' (default) / 'semi-front' / 'front'. Out-of-subset frames
                 are still PREDICTED (so the autoreg chain stays intact) but
                 their errors are NOT aggregated.

    prev_repr: 'pitchyaw' (2D angular) or 'vector' (3D unit vector). Must match
               the representation the model was trained with.
    """
    model.eval()

    if prev_repr not in ('pitchyaw', 'vector', 'tangent', 'angvel'):
        raise ValueError(f"prev_repr must be 'pitchyaw'/'vector'/'tangent'/'angvel', "
                         f"got {prev_repr}")
    # abs storage dim: 'tangent'/'angvel' roll the chain in pitchyaw space too.
    D = 3 if prev_repr == 'vector' else 2

    yaw_thresh_rad = GAZE360_SUBSET_YAW_THRESH_RAD[gaze_subset]

    rows = parse_label_file(osp.join(data_root, f'{split}.label'))
    tracks = build_sid_tracks(rows)

    per_sid_out = {}
    pp, py, gp, gy, g360_yaw_list = [], [], [], [], []

    fs = frame_stride
    for sid in sorted(tracks.keys()):
        track = tracks[sid]
        n = len(track)
        if n == 0:
            continue

        # Pre-load and preprocess all frames for this sid (no augmentation)
        face_np = np.zeros((n, 3, face_size, face_size), dtype=np.float32)
        left_np = np.zeros((n, 3, eye_size, eye_size), dtype=np.float32)
        right_np = np.zeros((n, 3, eye_size, eye_size), dtype=np.float32)
        for i, row in enumerate(track):
            f = cv2.cvtColor(cv2.imread(osp.join(data_root, row['face']),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            l = cv2.cvtColor(cv2.imread(osp.join(data_root, row['left']),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            r = cv2.cvtColor(cv2.imread(osp.join(data_root, row['right']),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (face_size, face_size))
            l = cv2.resize(l, (eye_size, eye_size))
            r = cv2.resize(r, (eye_size, eye_size))
            face_np[i] = normalize_image(f)
            left_np[i] = normalize_image(l)
            right_np[i] = normalize_image(r)

        T = num_frames
        # predictions store D-tuples (pitch_EVE, yaw_EVE) or (gx, gy, gz).
        # We also keep parallel pitch/yaw arrays for error aggregation.
        predictions = []
        pred_pitch_arr, pred_yaw_arr = [], []
        for i in range(n):
            # Build T positions: [i - (T-1)*fs, ..., i - fs, i], clamping negatives to 0.
            raw = [i - k * fs for k in range(T - 1, -1, -1)]
            window = [max(0, p) for p in raw]

            face_t = torch.from_numpy(face_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            left_t = torch.from_numpy(left_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            right_t = torch.from_numpy(right_np[window]).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)

            prev_gaze = np.zeros((T - 1, D), dtype=np.float32)
            if not zero_prev_gaze:
                for j, wi in enumerate(window[:-1]):
                    if wi < len(predictions):
                        prev_gaze[j] = predictions[wi]
            if prev_input == 'delta':
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

            if prev_repr == 'vector':
                pred_vec = model.predict_vector(face_t, left_t, right_t, prev_t,
                                                camera_id=None)  # (1, 3)
                pred_tuple = tuple(float(v) for v in pred_vec[0].cpu().tolist())
                # For error aggregation use pitch/yaw derived from the same vector.
                from utils import vector_to_pitchyaw
                p_pitch_t, p_yaw_t = vector_to_pitchyaw(pred_vec)
                p_p = p_pitch_t[0].cpu().item()
                p_y = p_yaw_t[0].cpu().item()
            else:
                pred = model.predict(face_t, left_t, right_t, prev_t,
                                     camera_id=None)              # (1, 2)
                p_p = pred[0, 0].cpu().item()
                p_y = pred[0, 1].cpu().item()
                pred_tuple = (p_p, p_y)
            predictions.append(pred_tuple)
            pred_pitch_arr.append(p_p)
            pred_yaw_arr.append(p_y)

            row = track[i]
            g_p = -row['gpitch']
            g_y = -row['gyaw']
            # Only aggregate errors for in-subset frames (autoreg chain still intact).
            if abs(row['gyaw']) <= yaw_thresh_rad:
                pp.append(p_p); py.append(p_y); gp.append(g_p); gy.append(g_y)
                g360_yaw_list.append(row['gyaw'])

        # Per-sid error — also filter to in-subset frames for fair comparison
        sid_keep_mask = np.array([abs(r['gyaw']) <= yaw_thresh_rad for r in track])
        if sid_keep_mask.any():
            sid_pp = np.array(pred_pitch_arr)[sid_keep_mask]
            sid_py = np.array(pred_yaw_arr)[sid_keep_mask]
            sid_gp = np.array([-r['gpitch'] for r in track])[sid_keep_mask]
            sid_gy = np.array([-r['gyaw'] for r in track])[sid_keep_mask]
            e = angular_error_pitchyaw(sid_pp, sid_py, sid_gp, sid_gy)
            per_sid_out[sid] = {'angular_error': e, 'n_samples': int(sid_keep_mask.sum())}
            log.info(f'  sid={sid}: {e:.4f}° (N={int(sid_keep_mask.sum())})')

    pp = np.array(pp); py = np.array(py); gp = np.array(gp); gy = np.array(gy)
    g360_yaw = np.array(g360_yaw_list)
    overall_err = angular_error_pitchyaw(pp, py, gp, gy)
    overall_std = float(per_sample_std(pp, py, gp, gy))
    yaw_bins = split_by_yaw_bin(pp, py, gp, gy, g360_yaw)
    return {
        'angular_error': overall_err,
        'angular_error_std': overall_std,
        'n_samples': len(pp),
        'per_sid': per_sid_out,
        'per_yaw_bin': yaw_bins,
    }


# ── main ────────────────────────────────────────────────────────────────

def test(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('test')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    ckpt_raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cargs = ckpt_raw.get('args', {})
    num_frames = cargs.get('num_frames', args.num_frames)
    face_backbone = cargs.get('face_backbone', args.face_backbone)
    eye_backbone = cargs.get('eye_backbone', args.eye_backbone)
    prev_mode = cargs.get('prev_mode', args.prev_mode)
    prev_input = cargs.get('prev_input', getattr(args, 'prev_input', 'abs'))
    prev_repr = cargs.get('prev_repr', getattr(args, 'prev_repr', 'pitchyaw'))
    d_model = cargs.get('d_model', args.d_model)
    nhead = cargs.get('nhead', args.nhead)
    num_layers_model = cargs.get('num_layers', args.num_layers)
    n_bins = cargs.get('n_bins', args.n_bins)
    temporal_type = cargs.get('temporal_type', args.temporal_type)
    fusion_type = cargs.get('fusion_type', args.fusion_type)
    use_pog = cargs.get('use_pog', False)
    gaze_space = cargs.get('gaze_space', 'vector')
    face_size = cargs.get('face_size', args.face_size)
    eye_size = cargs.get('eye_size', args.eye_size)

    log.info(f'Config: face={face_backbone} eye={eye_backbone} prev={prev_mode} '
             f'prev_input={prev_input} prev_repr={prev_repr} '
             f'temporal={temporal_type} fusion={fusion_type} T={num_frames} '
             f'd={d_model} gaze_space={gaze_space}')

    frame_stride = cargs.get('frame_stride', args.frame_stride)
    # subset: prefer CLI override, fall back to checkpoint's training subset, then 'full'
    gaze_subset = args.gaze_subset if args.gaze_subset else cargs.get('gaze_subset', 'full')
    log.info(f'gaze_subset: {gaze_subset}')

    dataset = Gaze360Dataset(
        data_root=args.data_root, split=args.split,
        num_frames=num_frames, frame_stride=frame_stride,
        face_size=face_size, eye_size=eye_size,
        n_bins=n_bins, test_mode=True, augment=False, left_pad=True,
        gaze_subset=gaze_subset,
        prev_input=prev_input,
        prev_repr=prev_repr,
    )
    log.info(f'Test samples ({args.split}): {len(dataset)}')

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )

    model = GazeEstimator(
        num_frames=num_frames,
        face_backbone_type=face_backbone, eye_backbone_type=eye_backbone,
        prev_mode=prev_mode, prev_input=prev_input, prev_repr=prev_repr,
        d_model=d_model, nhead=nhead, num_layers=num_layers_model,
        n_bins=n_bins, temporal_type=temporal_type, fusion_type=fusion_type,
        use_pog=use_pog, gaze_space=gaze_space, pretrained=False,
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    log.info(f'Loaded: {args.checkpoint}')

    if args.autoregressive:
        if args.zero_prev_gaze:
            log.info('Mode: AUTOREGRESSIVE + ZERO prev_gaze')
        else:
            log.info('Mode: AUTOREGRESSIVE')
        results = evaluate_autoregressive(
            model, args.data_root, args.split, num_frames, face_size, eye_size,
            device, log, frame_stride=frame_stride,
            zero_prev_gaze=args.zero_prev_gaze,
            gaze_subset=gaze_subset,
            prev_input=prev_input,
            prev_repr=prev_repr,
        )
    else:
        log.info('Mode: GT prev_gaze (TF)')
        results = evaluate_model(model, data_loader, device)

    log.info('=' * 60)
    log.info(f'Angular Error: {results["angular_error"]:.4f}°  (N={results["n_samples"]})')
    log.info('=' * 60)

    # Per-yaw-bin
    log.info('Per-yaw-bin (gaze360 convention):')
    log.info(f'  {"Bin":<15} | {"Angular Error":>15} | {"N":>8}')
    log.info(f'  {"-"*15}-+-{"-"*15}-+-{"-"*8}')
    for name in ['frontal', 'less_frontal', 'backward']:
        e, nn = results['per_yaw_bin'][name]
        e_s = f'{e:>14.4f}°' if e is not None else f'{"-":>15}'
        log.info(f'  {name:<15} | {e_s} | {nn:>8}')

    if args.out:
        out_data = {
            'angular_error_deg': results['angular_error'],
            'angular_error_std_deg': results.get('angular_error_std'),
            'n_samples': results['n_samples'],
            'checkpoint': args.checkpoint,
            'split': args.split,
            'autoregressive': args.autoregressive,
            'per_yaw_bin': {k: {'angular_error': v[0], 'n_samples': v[1]}
                            for k, v in results['per_yaw_bin'].items()},
            'per_sid': results['per_sid'],
        }
        with open(args.out, 'w') as f:
            json.dump(out_data, f, indent=2)
        log.info(f'Saved results to {args.out}')

    return results


def parse_args():
    p = argparse.ArgumentParser(description='EyeTag-Gaze360 Evaluation')
    p.add_argument('--data-root', required=True,
                   help='Root of the preprocessed Gaze360 dataset')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', default='val', choices=['val', 'train', 'test'])
    p.add_argument('--out', default=None)

    # Defaults (overridden by checkpoint args)
    p.add_argument('--face-backbone', default='vggface2',
                   choices=['vggface2', 'resnet'])
    p.add_argument('--eye-backbone', default='resnet18')
    p.add_argument('--prev-mode', default='mlp')
    p.add_argument('--prev-input', default='abs', choices=['abs', 'delta'],
                   help='Fallback if checkpoint does not record prev_input.')
    p.add_argument('--prev-repr', default='pitchyaw', choices=['pitchyaw', 'vector'],
                   help='Fallback if checkpoint does not record prev_repr.')
    p.add_argument('--temporal-type', default='causal')
    p.add_argument('--fusion-type', default='cross_attn')
    p.add_argument('--num-frames', type=int, default=7)
    p.add_argument('--frame-stride', type=int, default=1,
                   help='Within-window frame spacing (1=30Hz, 2=15Hz, 3=10Hz). '
                        'Overridden by checkpoint args if present.')
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--n-bins', type=int, default=90)
    p.add_argument('--face-size', type=int, default=128)
    p.add_argument('--eye-size', type=int, default=128)

    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--autoregressive', action='store_true', default=False)
    p.add_argument('--zero-prev-gaze', action='store_true', default=False)
    p.add_argument('--gaze-subset', default=None,
                   choices=[None, 'full', 'semi-front', 'front'],
                   help='Override subset. Default: read from checkpoint args (or full).')
    return p.parse_args()


if __name__ == '__main__':
    test(parse_args())
