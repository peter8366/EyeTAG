"""
Evaluate trained STAGE model on Gaze360 test set with per-sample angular errors,
splitting by yaw range (All / Semi-front 180 / Front 20) and reporting mean ± std.

Uses the same model + dataset code as core/tester.py, just collects per-sample
errors instead of running average. Loads the saved checkpoint (best by default,
or any --load_checkpoint_path).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import create_model
from datasources.Gaze360 import Gaze360Loader
from utils.checkpoints_manager import CheckpointsManager
from utils.core_utils import pitchyaw_to_vector
from utils.train_utils import my_collate


def angular_errors_per_sample(pred_pitchyaw, gt_pitchyaw, validity):
    """
    Compute angular error (degrees) for each sample in the batch, using only the
    LAST valid frame of each sequence (test convention: validity[-1] = 1).

    pred_pitchyaw: (B, T, 2) — model's pred in (pitch, yaw)
    gt_pitchyaw:   (B, T, 2) — ground truth
    validity:      (B, T)    — 1 for valid frames, 0 otherwise (test: only last=1)

    Returns: list of (angular_error_deg, gt_yaw_deg) per sample
    """
    out = []
    B = pred_pitchyaw.shape[0]
    for b in range(B):
        v = validity[b]
        valid_idx = (v > 0).nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            continue
        # use the last valid frame (test convention)
        j = valid_idx[-1].item()
        p = pred_pitchyaw[b, j]  # (2,)
        g = gt_pitchyaw[b, j]
        # convert to 3D unit vec
        p_vec = pitchyaw_to_vector(p.unsqueeze(0))  # (1, 3)
        g_vec = pitchyaw_to_vector(g.unsqueeze(0))
        sim = F.cosine_similarity(p_vec, g_vec, dim=1, eps=1e-8)
        sim = torch.clamp(sim, min=-1 + 1e-8, max=1 - 1e-8)
        err = torch.acos(sim) * (180.0 / np.pi)
        # gt is stored as [pitch, yaw] in radians (after [::-1] in dataset)
        gt_pitch_deg = g[0].item() * 180.0 / np.pi
        gt_yaw_deg = g[1].item() * 180.0 / np.pi
        out.append((err.item(), gt_pitch_deg, gt_yaw_deg))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, default='configs/stage_transformer.json')
    parser.add_argument('--save_path', type=str, default='./work_dir/stage_transformer')
    parser.add_argument('--load_checkpoint_path', type=str,
                        default='./work_dir/stage_transformer/checkpoints/best_checkpoint.pth.tar')
    parser.add_argument('--spatial_model', type=str, default='proposed')
    parser.add_argument('--out_json', type=str, default='./work_dir/stage_transformer/eval_with_std.json')
    args = parser.parse_args()

    # build config the same way main_gaze360.py does
    default_config = json.load(open('configs/default.json'))
    model_config = json.load(open(args.config_json))
    config = {**default_config, **model_config, **vars(args)}
    config['gaze360_path'] = 'data/Gaze360'
    config['tanh'] = False
    config['opt'] = 'sgd'
    config['skip_training'] = True
    config['load_step'] = 0
    config['learning_rate'] = config['base_learning_rate'] * config['batch_size']
    config = Namespace(**config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # build model & load weights
    model = create_model(config).to(device)
    ckpt_mgr = CheckpointsManager(network=model, output_dir=config.save_path)
    ckpt_mgr.load_checkpoint_frompath(config.load_checkpoint_path)
    model.eval()

    # test dataset
    test_ds = Gaze360Loader(source_path=config.gaze360_path, config=config, subset='test')
    print(f'Test dataset size: {len(test_ds)}')

    test_dl = DataLoader(test_ds,
                         batch_size=config.test_batch_size,
                         shuffle=False, drop_last=False,
                         num_workers=config.test_data_workers,
                         pin_memory=True, collate_fn=my_collate)

    all_errors = []  # list of (err_deg, gt_yaw_deg)

    with torch.no_grad():
        for input_data in tqdm(test_dl, desc='Eval'):
            if input_data is None:
                continue
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data[k] = v.to(device, non_blocking=True)

            # forward
            _, out_dict = model.compute_losses(input_data, only_3D=True)
            pred = out_dict['pred']  # (B, T, 2) pitch/yaw

            gt = input_data['face_g_tobii']
            validity = input_data['face_g_tobii_validity']
            errs = angular_errors_per_sample(pred, gt, validity)
            all_errors.extend(errs)

    print(f'\nTotal samples collected: {len(all_errors)}')
    errs = np.array([e for e, _, _ in all_errors], dtype=np.float64)
    pitches_abs_deg = np.array([abs(p) for _, p, _ in all_errors], dtype=np.float64)
    yaws_abs_deg = np.array([abs(y) for _, _, y in all_errors], dtype=np.float64)

    def report(name, mask):
        sel = errs[mask]
        if len(sel) == 0:
            return name, 0, float('nan'), float('nan')
        return name, len(sel), float(sel.mean()), float(sel.std(ddof=0))

    # STAGE upstream uses BOTH axes (|pitch| <= angle AND |yaw| <= angle).
    # We report that subset (matches Tester.gaze360_test) alongside the yaw-only
    # convention used by the Gaze360 paper.
    rows = []
    rows.append(report('All',                          np.ones_like(yaws_abs_deg, dtype=bool)))
    rows.append(report('Front 180  (STAGE: |p|,|y|<=90)', (pitches_abs_deg <= 90.0) & (yaws_abs_deg <= 90.0)))
    rows.append(report('Front 20   (STAGE: |p|,|y|<=20)', (pitches_abs_deg <= 20.0) & (yaws_abs_deg <= 20.0)))
    rows.append(report('Semi-front (yaw-only: |y|<=90)',   yaws_abs_deg <= 90.0))
    rows.append(report('Front     (yaw-only: |y|<=20)',    yaws_abs_deg <= 20.0))

    print('\n' + '=' * 76)
    print(f'{"Subset":<38} {"N":>7} {"Mean(°)":>10} {"Std(°)":>10}')
    print('-' * 76)
    for name, n, m, s in rows:
        print(f'{name:<38} {n:>7} {m:>10.2f} {s:>10.2f}')
    print('=' * 76)

    # save json
    payload = {
        'checkpoint': args.load_checkpoint_path,
        'n_total': int(len(errs)),
        'subsets': {name: {'n': int(n), 'mean': m, 'std': s} for name, n, m, s in rows},
    }
    with open(args.out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved -> {args.out_json}')


if __name__ == '__main__':
    main()
