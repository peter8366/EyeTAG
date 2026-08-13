"""STAGE inference on Gaze360 test → save per-frame predictions to NPZ
matching the schema of bmvc/jitter/data/*.npz (sid, frame_idx, gt, pred).
"""
import os, sys, json, argparse, warnings, numpy as np

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import Namespace
from tqdm import tqdm

from models import create_model
from utils.core_utils import my_collate
from utils.checkpoints_manager import CheckpointsManager
from main_gaze360 import Gaze360Loader


def pitchyaw_to_vector(py):
    """py: (..., 2) [pitch, yaw] → 3D unit vector in Gaze360 raw convention
       (matches the (gx, gy, gz) stored in our other NPZ files)."""
    pitch = py[..., 0]; yaw = py[..., 1]
    x =  torch.cos(pitch) * torch.sin(yaw)
    y =  torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=-1)


def parse_path(p):
    """Extract (sid, frame_idx) from a Gaze360 face_path."""
    parts = p.split('/')
    sid = int(parts[2])
    fname = parts[3]
    fidx = int(fname.split('_')[1].split('.')[0])
    return sid, fidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='work_dir/stage_transformer/checkpoints/best_checkpoint.pth.tar')
    ap.add_argument('--out', default='./npz_dumps/stage.npz')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    print(f'cwd: {os.getcwd()}')

    device = torch.device('cuda:0')

    # Build merged config (mimic main_gaze360.py)
    default_cfg = json.load(open('configs/default.json'))
    model_cfg   = json.load(open('configs/stage_transformer.json'))
    cfg = {**default_cfg, **model_cfg,
           'spatial_model': 'proposed',
           'tanh': False,                # main_gaze360.py forces this False
           'save_path': '/tmp/stage_g360_infer',
           'load_checkpoint_path': args.ckpt}
    cfg['learning_rate'] = cfg['base_learning_rate'] * cfg['batch_size']
    cfg = Namespace(**cfg)
    os.makedirs(cfg.save_path, exist_ok=True)

    # Model — load checkpoint manually with strict=False to tolerate
    # head-layer key drift between training and current code.
    model = create_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    weights = state.get('state_dict', state.get('model_state_dict', state))
    # CheckpointsManager strips 'module.' prefix — replicate
    cleaned = {}
    for k, v in weights.items():
        cleaned[k.replace('module.', '', 1) if k.startswith('module.') else k] = v
    info = model.load_state_dict(cleaned, strict=False)
    print(f'load_state_dict: missing={len(info.missing_keys)}  unexpected={len(info.unexpected_keys)}')
    if info.unexpected_keys[:3]:
        print(f'  first unexpected: {info.unexpected_keys[:3]}')
    if info.missing_keys[:3]:
        print(f'  first missing   : {info.missing_keys[:3]}')
    model.eval()

    # Dataset (test split, no subset angle filter)
    dataset = Gaze360Loader(source_path=cfg.gaze360_path, config=cfg, subset='test')
    print(f'dataset sequences: {len(dataset)}')

    # Build (sequence-index -> sid, frame_idx) using last frame of each sequence
    # last entry of path_source is the original Gaze360 path (e.g. rec_022/head/.../000131.jpg);
    # img2label_mapping turns it into the face-image path "test/Face/<sid>/<sid>_<idx>.jpg"
    # from which (sid, fidx) is read.
    seq_lookup = []
    bad = 0
    for path_source, _ in dataset.imgs:
        last_orig_path = path_source[-1]
        entry = dataset.img2label_mapping.get(last_orig_path)
        sid, fidx = -1, -1
        if entry is not None:
            face_path = entry['face_path']
            parts = face_path.split('/')
            if 'Face' in parts:
                try:
                    k = parts.index('Face')
                    sid = int(parts[k + 1])
                    fname = parts[k + 2]
                    fidx = int(fname.split('_')[1].split('.')[0])
                except Exception:
                    pass
        if sid < 0:
            bad += 1
        seq_lookup.append((sid, fidx))
    print(f'seq_lookup built: total={len(seq_lookup)}  unresolved={bad}')

    # Wrap dataset so we know which sequence index corresponds to each kept sample
    class WithIndex(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            if item is None: return None
            item['_idx'] = torch.tensor(idx, dtype=torch.long)
            return item

    loader = DataLoader(WithIndex(dataset), batch_size=args.batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=my_collate)

    sid_out, fidx_out, pred_out, gt_out = [], [], [], []
    n_seen = 0
    with torch.no_grad():
        for input_data in tqdm(loader):
            if input_data is None: continue
            idx_tensor = input_data.pop('_idx')
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data[k] = v.detach().to(device, non_blocking=True)

            try:
                out = model(input_data, {})
            except Exception:
                # some models expect compute_losses path
                _, out = model.compute_losses(input_data, only_3D=True)

            pred_py = out['pred']                          # (B, T, 2)
            gt_py   = input_data['face_g_tobii']           # (B, T, 2)
            validity = input_data['face_g_tobii_validity'] # (B, T)

            # Target = last valid frame in the sequence (the sequence's representative)
            # tester does this implicitly via validity[-1]=1 outside train.
            B, T, _ = pred_py.shape
            for b in range(B):
                seq_idx = int(idx_tensor[b].item())
                sid, fidx = seq_lookup[seq_idx]
                if sid < 0:
                    continue
                pred_vec = pitchyaw_to_vector(pred_py[b, -1]).cpu().numpy()
                gt_vec   = pitchyaw_to_vector(gt_py[b, -1]).cpu().numpy()
                sid_out.append(sid); fidx_out.append(fidx)
                pred_out.append(pred_vec); gt_out.append(gt_vec)
                n_seen += 1

    print(f'\ncollected: {n_seen} per-frame predictions')

    sid_arr  = np.array(sid_out,  dtype=np.int32)
    fidx_arr = np.array(fidx_out, dtype=np.int32)
    pred_arr = np.array(pred_out, dtype=np.float32)
    gt_arr   = np.array(gt_out,   dtype=np.float32)

    # Sanity: ang error
    pn = pred_arr / (np.linalg.norm(pred_arr, axis=1, keepdims=True) + 1e-8)
    gn = gt_arr   / (np.linalg.norm(gt_arr,   axis=1, keepdims=True) + 1e-8)
    dots = np.clip((pn * gn).sum(axis=1), -1, 1)
    err  = np.degrees(np.arccos(dots))
    print(f'overall angular error: {err.mean():.4f}° ± {err.std():.4f}°')

    np.savez(args.out,
             sid=sid_arr, frame_idx=fidx_arr,
             gt=gt_arr, pred=pred_arr)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
