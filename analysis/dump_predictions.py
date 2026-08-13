"""Dump per-frame autoregressive predictions on the Gaze360 test split to an NPZ.

The temporal-stability metrics in the paper (Table 3) are computed from
frame-ordered predictions, which the regular evaluation script does not keep.
This script runs the same autoregressive rollout as ``gaze360/test.py`` but
stores every prediction, ordered by (subject id, frame index).

Output NPZ:
    sid       (N,)   int32
    frame_idx (N,)   int32
    gt        (N, 3) float32  unit gaze vectors
    pred      (N, 3) float32  unit gaze vectors

Usage:
    python analysis/dump_predictions.py \
        --data-root /path/to/gaze360 \
        --ckpt work_dir/eyetag_t48_delta/best.pth \
        --out  analysis/data/eyetag.npz --gpu 0

Then:
    python analysis/temporal_stability.py --npz-dir analysis/data
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch


def parse_label(label_file):
    """Parse a Gaze360 `.label` file into rows sorted by (sid, frame index)."""
    rows = []
    with open(label_file) as f:
        next(f)  # header
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            face_p, left_p, right_p = p[0], p[1], p[2]
            sid = int(face_p.split('/')[2])
            idx = int(face_p.split('/')[3].split('_')[1].split('.')[0])
            gx, gy, gz = (float(v) for v in p[4].split(','))
            rows.append({'sid': sid, 'idx': idx, 'face': face_p,
                         'left': left_p, 'right': right_p,
                         'gx': gx, 'gy': gy, 'gz': gz})
    rows.sort(key=lambda r: (r['sid'], r['idx']))
    return rows


def normalize_vec(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def save_npz(out_path, rows_meta, preds):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sids = np.array([r['sid'] for r in rows_meta], dtype=np.int32)
    frame_idxs = np.array([r['idx'] for r in rows_meta], dtype=np.int32)
    gt_vec = normalize_vec(np.array([[r['gx'], r['gy'], r['gz']] for r in rows_meta],
                                    dtype=np.float32))
    pred_vec = normalize_vec(np.asarray(preds, dtype=np.float32))
    np.savez(out_path, sid=sids, frame_idx=frame_idxs, gt=gt_vec, pred=pred_vec)
    err = np.degrees(np.arccos(np.clip((gt_vec * pred_vec).sum(axis=1), -1.0, 1.0)))
    print(f'  saved {out_path}  N={len(preds)}  '
          f'mean={err.mean():.3f}deg  median={np.median(err):.3f}deg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='Gaze360 preprocessed root')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='output .npz path')
    ap.add_argument('--label-file', default=None,
                    help='defaults to <data-root>/test.label')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))

    from eyetag.models import GazeEstimator
    from eyetag.geometry import vector_to_pitchyaw, delta_from_abs_pitchyaw
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gaze360'))
    from dataset import normalize_image

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    label_file = Path(args.label_file) if args.label_file else data_root / 'test.label'

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    c = ckpt.get('args', {})
    c = vars(c) if hasattr(c, '__dict__') else c

    T = c.get('num_frames', 48)
    face_size = c.get('face_size', 128)
    eye_size = c.get('eye_size', 128)
    fs = c.get('frame_stride', 1)
    prev_mode = c.get('prev_mode') or 'mlp'
    prev_input = c.get('prev_input') or 'abs'
    prev_repr = c.get('prev_repr') or 'pitchyaw'
    abs_dim = 3 if prev_repr == 'vector' else 2
    print(f'  T={T} prev_mode={prev_mode} prev_input={prev_input} prev_repr={prev_repr}')

    model = GazeEstimator(
        num_frames=T,
        face_backbone_type=c.get('face_backbone', 'vggface2'),
        eye_backbone_type=c.get('eye_backbone', 'resnet18'),
        prev_mode=prev_mode, prev_input=prev_input, prev_repr=prev_repr,
        d_model=c.get('d_model', 256), nhead=c.get('nhead', 4),
        num_layers=c.get('num_layers', 2), n_bins=c.get('n_bins', 90),
        temporal_type=c.get('temporal_type', 'causal'),
        fusion_type=c.get('fusion_type', 'cross_attn'),
        gaze_space=c.get('gaze_space', 'vector'),
        use_pog=False, pretrained=False, pretrained_vggface=None,
    ).to(device).eval()

    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    info = model.load_state_dict(state, strict=False)
    assert not info.missing_keys, f'missing keys: {info.missing_keys[:5]}'

    rows = parse_label(label_file)
    by_sid = defaultdict(list)
    for r in rows:
        by_sid[r['sid']].append(r)
    for sid in by_sid:
        by_sid[sid].sort(key=lambda x: x['idx'])

    all_preds, all_rows = [], []
    with torch.no_grad():
        for sid in sorted(by_sid):
            track = by_sid[sid]
            n = len(track)
            face_np = np.zeros((n, 3, face_size, face_size), dtype=np.float32)
            left_np = np.zeros((n, 3, eye_size, eye_size), dtype=np.float32)
            right_np = np.zeros((n, 3, eye_size, eye_size), dtype=np.float32)
            for i, row in enumerate(track):
                f = cv2.cvtColor(cv2.imread(str(data_root / row['face'])), cv2.COLOR_BGR2RGB)
                l = cv2.cvtColor(cv2.imread(str(data_root / row['left'])), cv2.COLOR_BGR2RGB)
                r = cv2.cvtColor(cv2.imread(str(data_root / row['right'])), cv2.COLOR_BGR2RGB)
                face_np[i] = normalize_image(cv2.resize(f, (face_size, face_size)))
                left_np[i] = normalize_image(cv2.resize(l, (eye_size, eye_size)))
                right_np[i] = normalize_image(cv2.resize(r, (eye_size, eye_size)))

            preds_abs = []  # autoregressive chain in absolute gaze space
            for i in range(n):
                raw = [i - k * fs for k in range(T - 1, -1, -1)]
                w = [max(0, p) for p in raw]
                to_t = lambda arr: (torch.from_numpy(arr[w]).unsqueeze(0)
                                    .permute(0, 2, 1, 3, 4).to(device))
                face_t, left_t, right_t = to_t(face_np), to_t(left_np), to_t(right_np)

                prev_g = np.zeros((T - 1, abs_dim), dtype=np.float32)
                for j, wi in enumerate(w[:-1]):
                    if wi < len(preds_abs):
                        prev_g[j] = preds_abs[wi]
                if prev_input == 'delta':
                    if T - 2 >= 1:
                        if prev_repr in ('tangent', 'angvel'):
                            prev_g = delta_from_abs_pitchyaw(prev_g, prev_repr)
                        else:
                            prev_g = (prev_g[1:] - prev_g[:-1]).astype(np.float32)
                    else:
                        prev_g = np.zeros((0, abs_dim), dtype=np.float32)
                prev_t = torch.from_numpy(prev_g).unsqueeze(0).to(device)

                vec = model.predict_vector(face_t, left_t, right_t, prev_t, camera_id=None)
                gx, gy, gz = vec[0].cpu().tolist()
                if prev_repr == 'vector':
                    preds_abs.append((gx, gy, gz))
                else:
                    p_e, y_e = vector_to_pitchyaw(vec)
                    preds_abs.append((float(p_e[0].cpu()), float(y_e[0].cpu())))
                all_preds.append([gx, gy, gz])
                all_rows.append(track[i])

            if sid % 50 == 0:
                print(f'  sid={sid} done ({n} frames)', flush=True)

    save_npz(args.out, all_rows, all_preds)


if __name__ == '__main__':
    main()
