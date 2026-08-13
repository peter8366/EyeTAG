"""Temporal-stability metrics on Gaze360 (paper Table 3).

Given per-frame prediction dumps produced by ``analysis/dump_predictions.py``,
this reproduces the three metrics reported in the paper, plus the saccade-level
diagnostics used to verify that a low jitter is not simply a static predictor.

Frame-to-frame angular velocity (deg/frame), for ground truth and prediction:

    v_t = arccos(g_{t-1} . g_t) * 180 / pi

Metrics (only frame pairs that are consecutive within the same subject track
are used):

    J  Fixation Jitter  = mean(v_pred | v_gt <  1 deg/frame)     lower is better
    B  Saccade Bias     = mean(v_pred - v_gt | v_gt > 5)         closer to 0 is better
    M  Absolute Mean    = (J + |B|) / 2                          lower is better

    DirErr   angle between the GT and predicted displacement vector on saccade
             pairs -- penalises moving at the right speed in the wrong direction
    static%  fraction of saccade pairs where the prediction barely moves

Usage:
    python analysis/temporal_stability.py --npz-dir analysis/data
    python analysis/temporal_stability.py --npz ours.npz baseline.npz
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FIX_THR = 1.0     # deg/frame -- paper Table 3(a)
SAC_THR = 5.0     # deg/frame -- paper Table 3(b)
STATIC_THR = 0.5  # deg/frame -- "barely moving" prediction during a GT saccade


def angdeg(a, b):
    """Angle between corresponding rows of a and b (unit-normalised), degrees."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cos = np.clip((a * b).sum(1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def consecutive_pairs(d):
    """(gt_prev, gt_cur, pred_prev, pred_cur) for frame pairs adjacent in time
    within the same subject track."""
    sid, fidx = d['sid'], d['frame_idx']
    gt, pred = d['gt'].astype(np.float64), d['pred'].astype(np.float64)
    order = np.lexsort((fidx, sid))
    sid, fidx, gt, pred = sid[order], fidx[order], gt[order], pred[order]
    ok = (sid[1:] == sid[:-1]) & (fidx[1:] - fidx[:-1] == 1)
    return gt[:-1][ok], gt[1:][ok], pred[:-1][ok], pred[1:][ok]


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def evaluate(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    g0, g1, p0, p1 = consecutive_pairs(d)

    v_gt = angdeg(g0, g1)      # GT angular velocity
    v_pred = angdeg(p0, p1)    # predicted angular velocity
    fix = v_gt < FIX_THR
    sac = v_gt > SAC_THR

    J = v_pred[fix].mean()
    B = (v_pred[sac] - v_gt[sac]).mean()
    M = 0.5 * (J + abs(B))

    d_gt, d_pr = (g1 - g0)[sac], (p1 - p0)[sac]
    nz = np.linalg.norm(d_pr, axis=1) > 1e-9
    dir_err = angdeg(d_gt[nz], d_pr[nz])

    return {
        'model': Path(npz_path).stem,
        'n_pairs': int(len(v_gt)), 'n_fix': int(fix.sum()), 'n_sac': int(sac.sum()),
        'J_fixation_jitter': round(float(J), 2),
        'J_sem': round(float(sem(v_pred[fix])), 2),
        'B_saccade_bias': round(float(B), 2),
        'B_sem': round(float(sem(v_pred[sac] - v_gt[sac])), 2),
        'M_absolute_mean': round(float(M), 2),
        'dir_err_mean': round(float(dir_err.mean()), 2),
        'dir_err_sem': round(float(sem(dir_err)), 2),
        'pct_correct_hemisphere': round(100.0 * float((dir_err < 90).mean()), 1),
        'pct_near_static_pred': round(100.0 * float((v_pred[sac] < STATIC_THR).mean()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='*', default=None, help='explicit .npz paths')
    ap.add_argument('--npz-dir', default=None, help='directory of .npz dumps')
    ap.add_argument('--out', default=None, help='write results to this .json/.csv stem')
    args = ap.parse_args()

    paths = [Path(p) for p in (args.npz or [])]
    if args.npz_dir:
        paths += sorted(Path(args.npz_dir).glob('*.npz'))
    if not paths:
        ap.error('provide --npz and/or --npz-dir')

    rows = [evaluate(p) for p in paths]

    hdr = (f"{'Model':<26} {'N_fix':>6} {'N_sac':>6} | {'J':>6} {'B':>7} {'M':>6} | "
           f"{'DirErr':>8} {'<90deg%':>8} {'static%':>8}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['model']:<26} {r['n_fix']:>6} {r['n_sac']:>6} | "
              f"{r['J_fixation_jitter']:>6} {r['B_saccade_bias']:>7} "
              f"{r['M_absolute_mean']:>6} | {r['dir_err_mean']:>8} "
              f"{r['pct_correct_hemisphere']:>8} {r['pct_near_static_pred']:>8}")

    if args.out:
        stem = Path(args.out).with_suffix('')
        stem.parent.mkdir(parents=True, exist_ok=True)
        with open(f'{stem}.json', 'w') as f:
            json.dump(rows, f, indent=2)
        with open(f'{stem}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nsaved -> {stem}.json / {stem}.csv')


if __name__ == '__main__':
    main()
