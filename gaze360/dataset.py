"""Gaze360 dataset loader.

Expects the preprocessed Gaze360 layout (per-frame JPEGs plus a `.label` index):

    <data-root>/
        train.label, val.label, test.label
        train/Face/<sid>/<sid>_<idx>.jpg
        train/Left/...,  train/Right/...

Label line format:
    Face Left Right Origin 3DGaze 2DGaze
where 3DGaze is `gx,gy,gz` (unit vector) and 2DGaze is `gyaw,gpitch` in radians.

Sequences are sliding windows of T frames within a single subject id (sid); the
first frame is repeated to left-pad windows at the start of a track.

Internally, gaze is converted to the EVE angle convention
(pitch_EVE = -pitch_g360, yaw_EVE = -yaw_g360); the 3D vector is unchanged.
Horizontal-flip augmentation flips the image, swaps the left/right eye crops and
negates yaw_EVE (equivalently, the x component of the gaze vector).
"""

import os
import os.path as osp
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GAZE_INIT = np.array([0.0, 0.0], dtype=np.float32)  # (pitch=0, yaw=0)

# Gaze360 standard subsets — defined by max |yaw| in gaze360 convention (radians).
# Use 'full' (default) to match original Static-180 behavior.
GAZE360_SUBSET_YAW_THRESH_RAD = {
    'full': float('inf'),                  # all samples (incl. backward)
    'semi-front': float(np.pi / 2),        # |yaw| ≤ 90°   (front-180)
    'front': float(np.radians(20.0)),      # |yaw| ≤ 20°   (front-20)
}


def normalize_image(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """uint8 (H, W, 3) → float32 (3, H, W) ImageNet-normalized."""
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img.transpose(2, 0, 1)


def angle_to_bin(angle, min_val, max_val, n_bins):
    """Continuous angle → bin index [0, n_bins-1]."""
    bin_width = (max_val - min_val) / n_bins
    return int(np.clip((angle - min_val) / bin_width, 0, n_bins - 1))


def parse_label_file(label_path):
    """Read a gaze360 .label file → list of dicts (one per frame).

    Each line: face left right origin 3DGaze 2DGaze
      3DGaze: 'gx,gy,gz'   (unit vector, gaze360 convention)
      2DGaze: 'gyaw,gpitch'  (radians, gaze360 convention)

    Returns list of {sid, idx, face, left, right, origin, gx, gy, gz, gyaw, gpitch}.
    """
    out = []
    with open(label_path, 'r') as f:
        header = f.readline()  # skip header
        _ = header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            face_p, left_p, right_p, origin_p, gaze3d, gaze2d = parts[:6]
            # face = "<split>/Face/<sid>/<sid>_<idx>.jpg"
            face_parts = face_p.split('/')
            sid = face_parts[2]
            fname = face_parts[3]
            idx = int(fname.split('_')[1].split('.')[0])

            gx, gy, gz = (float(v) for v in gaze3d.split(','))
            gyaw, gpitch = (float(v) for v in gaze2d.split(','))

            out.append({
                'sid': sid,
                'idx': idx,
                'face': face_p,
                'left': left_p,
                'right': right_p,
                'origin': origin_p,
                'gx': gx, 'gy': gy, 'gz': gz,
                'gyaw': gyaw, 'gpitch': gpitch,
            })
    return out


def build_sid_tracks(rows):
    """Group rows by sid and sort by idx. Returns dict sid → list[row]."""
    tracks = defaultdict(list)
    for r in rows:
        tracks[r['sid']].append(r)
    for sid in tracks:
        tracks[sid].sort(key=lambda x: x['idx'])
    return tracks


class Gaze360Dataset(Dataset):
    """Gaze360 sequence dataset.

    Args:
        data_root:    gaze360 root (contains train/, val/, test/ image dirs and *.label files)
        split:        'train' / 'val' / 'test'
        num_frames:   T — sequence length
        frame_stride: spacing between consecutive frames in a window (1=native 30Hz,
                      2=15Hz, 3=10Hz). Gaze360 source is 30fps, so effective Hz = 30/frame_stride
                      for sids whose labeled frames are themselves consecutive (most are).
        face_size:    resize target for Face crop (square)
        eye_size:     resize target for each eye crop (square)
        n_bins:       classification bins per axis (only used if gaze_space='pitchyaw')
        pitch_range:  (min, max) radians for binning (EVE convention)
        yaw_range:    (min, max) radians for binning (EVE convention)
        seq_stride:   sliding-window stride within each sid (window step over target positions)
        test_mode:    disable augmentation
        augment:      enable augmentation (only effective when test_mode=False)
        prediction_cache: dict mapping (sid, idx) → (pitch_EVE, yaw_EVE) for scheduled sampling
        prev_gt_ratio: probability of using GT (1.0=all GT, 0.0=all cached pred)
        left_pad:     if True, pad short windows by repeating the first valid position (training default)
                      if False, skip windows that don't have T frames in bounds (clean evaluation default)
    """

    def __init__(self,
                 data_root,
                 split='train',
                 num_frames=7,
                 frame_stride=1,
                 face_size=128,
                 eye_size=128,
                 n_bins=90,
                 pitch_range=(-np.pi / 2, np.pi / 2),
                 yaw_range=(-np.pi, np.pi),
                 seq_stride=1,
                 test_mode=False,
                 augment=True,
                 prediction_cache=None,
                 prev_gt_ratio=1.0,
                 left_pad=True,
                 gaze_subset='full',
                 prev_input='abs',
                 prev_repr='pitchyaw'):
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.face_size = face_size
        self.eye_size = eye_size
        self.n_bins = n_bins
        self.pitch_range = pitch_range
        self.yaw_range = yaw_range
        self.seq_stride = seq_stride
        self.test_mode = test_mode
        self.augment = augment and not test_mode
        self.prediction_cache = prediction_cache
        self.prev_gt_ratio = prev_gt_ratio
        self.left_pad = left_pad
        self.gaze_subset = gaze_subset
        if prev_input not in ('abs', 'delta'):
            raise ValueError(f"prev_input must be 'abs' or 'delta', got {prev_input}")
        self.prev_input = prev_input
        if prev_repr not in ('pitchyaw', 'vector', 'tangent', 'angvel'):
            raise ValueError(f"prev_repr must be 'pitchyaw'/'vector'/'tangent'/'angvel', "
                             f"got {prev_repr}")
        if prev_repr in ('tangent', 'angvel') and prev_input != 'delta':
            raise ValueError(f"prev_repr='{prev_repr}' is a differential representation "
                             f"and requires prev_input='delta'.")
        self.prev_repr = prev_repr
        # abs storage dim (cache tuples / GT rows): pitchyaw-based reprs store 2D angles.
        self._abs_dim = 3 if prev_repr == 'vector' else 2
        # model-input dim after the (optional) delta step:
        # 2D (Δpitch, Δyaw) for 'pitchyaw'; 3D for 'vector'/'tangent'/'angvel'.
        self._prev_dim = 2 if prev_repr == 'pitchyaw' else 3

        if frame_stride < 1:
            raise ValueError(f'frame_stride must be >= 1, got {frame_stride}')
        if gaze_subset not in GAZE360_SUBSET_YAW_THRESH_RAD:
            raise ValueError(f"Unknown gaze_subset '{gaze_subset}'. "
                             f"Choose from {list(GAZE360_SUBSET_YAW_THRESH_RAD)}")
        self._yaw_thresh_rad = GAZE360_SUBSET_YAW_THRESH_RAD[gaze_subset]

        label_path = osp.join(data_root, f'{split}.label')
        if not osp.isfile(label_path):
            raise FileNotFoundError(f'Label file not found: {label_path}')

        rows = parse_label_file(label_path)
        self.tracks = build_sid_tracks(rows)

        # Build sequences. A window ending at target i contains positions:
        #   [i - (T-1)*fs, i - (T-2)*fs, ..., i - fs, i]
        # When left_pad=True, negative positions clamp to the first valid frame (0).
        T = num_frames
        fs = frame_stride

        self.sequences = []
        skipped_short = 0
        for sid, track in self.tracks.items():
            n = len(track)
            if left_pad:
                for i in range(0, n, seq_stride):
                    raw = [i - k * fs for k in range(T - 1, -1, -1)]
                    positions = [max(0, p) for p in raw]
                    self.sequences.append({'sid': sid, 'positions': positions, 'target_pos': i})
            else:
                min_target = (T - 1) * fs
                if n <= min_target:
                    skipped_short += 1
                    continue
                for i in range(min_target, n, seq_stride):
                    positions = [i - k * fs for k in range(T - 1, -1, -1)]
                    self.sequences.append({'sid': sid, 'positions': positions, 'target_pos': i})

        # Filter sequences by target frame's gaze360 yaw (in-subset only).
        # |gyaw|_g360 == |yaw|_EVE (sign flips but magnitudes match), so we can
        # threshold either; we use the raw gaze360 yaw for clarity.
        n_before_subset = len(self.sequences)
        if self._yaw_thresh_rad != float('inf'):
            self.sequences = [
                s for s in self.sequences
                if abs(self.tracks[s['sid']][s['target_pos']]['gyaw'])
                <= self._yaw_thresh_rad
            ]

        n_sids = len(self.tracks)
        subset_info = ''
        if gaze_subset != 'full':
            subset_info = (f', gaze_subset={gaze_subset} '
                           f'(|yaw|≤{np.degrees(self._yaw_thresh_rad):.0f}°: '
                           f'{n_before_subset}→{len(self.sequences)})')
        print(f'[Gaze360Dataset/{split}] {len(self.sequences)} sequences from '
              f'{n_sids} sids (T={num_frames}, frame_stride={frame_stride}, '
              f'seq_stride={seq_stride}, left_pad={left_pad}'
              + (f', skipped_short={skipped_short}' if skipped_short else '')
              + subset_info
              + ')')

    def __len__(self):
        return len(self.sequences)

    # ── image loading ─────────────────────────────────────────────────────

    def _load_image(self, rel_path):
        """Read JPG → uint8 RGB numpy."""
        p = osp.join(self.data_root, rel_path)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Failed to read {p}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── prev_gaze with scheduled sampling ─────────────────────────────────

    def _build_prev_gaze(self, sid, positions, gt_pitches_eve, gt_yaws_eve, flip):
        """Build (T-1, D) prev_gaze where D=2 for 'pitchyaw' or D=3 for 'vector'.

        - 'pitchyaw': each row is (pitch_EVE, yaw_EVE) in radians.
                      Flip negates yaw (index 1).
        - 'vector':   each row is (gx, gy, gz) unit vector (gaze360 convention,
                      identical to EVE for the 3D representation).
                      Flip negates gx (index 0, mirrors about y-z plane).
        """
        cache = self.prediction_cache
        prev = []
        track = self.tracks[sid]
        D = self._abs_dim
        flip_idx = 0 if self.prev_repr == 'vector' else 1
        for pos in positions[:-1]:  # all but the target
            use_gt = True
            if cache is not None and self.prev_gt_ratio < 1.0:
                row = track[pos]
                key = (sid, row['idx'])
                if key in cache and random.random() > self.prev_gt_ratio:
                    pg = np.array(cache[key], dtype=np.float32)  # length D tuple
                    if pg.shape[0] != D:
                        raise ValueError(
                            f"cache entry for ({sid},{row['idx']}) has dim {pg.shape[0]} "
                            f"but prev_repr={self.prev_repr} expects dim {D}.")
                    if flip:
                        pg[flip_idx] = -pg[flip_idx]
                    prev.append(pg)
                    use_gt = False
            if use_gt:
                row = track[pos]
                if self.prev_repr == 'vector':
                    pg = np.array([row['gx'], row['gy'], row['gz']], dtype=np.float32)
                else:  # 'pitchyaw' / 'tangent' / 'angvel' — abs stored as pitchyaw
                    pg = np.array([gt_pitches_eve[pos], gt_yaws_eve[pos]], dtype=np.float32)
                if flip:
                    pg[flip_idx] = -pg[flip_idx]
                prev.append(pg)
        abs_arr = np.stack(prev).astype(np.float32) if prev else np.zeros((0, D), dtype=np.float32)
        if self.prev_input == 'delta':
            # delta[j] = abs[j+1] - abs[j]  for j = 0..T-3 ; shape (T-2, D_out)
            # padded positions naturally produce 0 deltas (same gaze repeated).
            # 'tangent'/'angvel': geometrically exact differentials on the sphere.
            if abs_arr.shape[0] >= 2:
                if self.prev_repr in ('tangent', 'angvel'):
                    from utils import delta_from_abs_pitchyaw
                    return delta_from_abs_pitchyaw(abs_arr, self.prev_repr)
                return (abs_arr[1:] - abs_arr[:-1]).astype(np.float32)
            return np.zeros((max(0, self.num_frames - 2), self._prev_dim), dtype=np.float32)
        return abs_arr

    # ── __getitem__ ──────────────────────────────────────────────────────

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        sid = seq['sid']
        positions = seq['positions']
        target_pos = seq['target_pos']
        track = self.tracks[sid]
        T = self.num_frames

        # Augmentation params (per-sequence)
        flip = False
        do_color = False
        brightness = contrast = 1.0
        if self.augment:
            if random.random() < 0.5:
                flip = True
            if random.random() < 0.5:
                do_color = True
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)

        face_imgs, left_imgs, right_imgs = [], [], []
        for pos in positions:
            row = track[pos]
            face = self._load_image(row['face'])
            left = self._load_image(row['left'])
            right = self._load_image(row['right'])

            face = cv2.resize(face, (self.face_size, self.face_size))
            left = cv2.resize(left, (self.eye_size, self.eye_size))
            right = cv2.resize(right, (self.eye_size, self.eye_size))

            if flip:
                face = np.ascontiguousarray(np.fliplr(face))
                # swap left ↔ right and flip both
                left_aug = np.ascontiguousarray(np.fliplr(right))
                right_aug = np.ascontiguousarray(np.fliplr(left))
                left, right = left_aug, right_aug

            if do_color:
                shift = (brightness - 1.0) * 128.0
                face = np.clip(face.astype(np.float32) * contrast + shift, 0, 255).astype(np.uint8)
                left = np.clip(left.astype(np.float32) * contrast + shift, 0, 255).astype(np.uint8)
                right = np.clip(right.astype(np.float32) * contrast + shift, 0, 255).astype(np.uint8)

            face_imgs.append(normalize_image(face))
            left_imgs.append(normalize_image(left))
            right_imgs.append(normalize_image(right))

        face_tensor = torch.from_numpy(np.stack(face_imgs)).permute(1, 0, 2, 3)
        left_tensor = torch.from_numpy(np.stack(left_imgs)).permute(1, 0, 2, 3)
        right_tensor = torch.from_numpy(np.stack(right_imgs)).permute(1, 0, 2, 3)

        # ── labels ──
        gt_pitches_eve = np.array(
            [-track[p]['gpitch'] for p in range(len(track))], dtype=np.float32)
        gt_yaws_eve = np.array(
            [-track[p]['gyaw'] for p in range(len(track))], dtype=np.float32)

        target_row = track[target_pos]
        pitch_eve = float(gt_pitches_eve[target_pos])
        yaw_eve = float(gt_yaws_eve[target_pos])
        gt_vec = np.array([target_row['gx'], target_row['gy'], target_row['gz']],
                          dtype=np.float32)

        if flip:
            yaw_eve = -yaw_eve
            gt_vec = gt_vec.copy()
            gt_vec[0] = -gt_vec[0]  # x → -x mirrors yaw

        # normalize gt_vec to unit length (gaze360 values are already very close to unit)
        n = np.linalg.norm(gt_vec) + 1e-8
        gt_vec = gt_vec / n

        pitch_bin = angle_to_bin(pitch_eve, self.pitch_range[0], self.pitch_range[1], self.n_bins)
        yaw_bin = angle_to_bin(yaw_eve, self.yaw_range[0], self.yaw_range[1], self.n_bins)

        prev_gaze = self._build_prev_gaze(sid, positions, gt_pitches_eve, gt_yaws_eve, flip)

        return {
            'face': face_tensor,                                          # (3, T, H, W)
            'left_eye': left_tensor,
            'right_eye': right_tensor,
            'prev_gaze': torch.from_numpy(prev_gaze),                    # (T-1, 2) abs OR (T-2, 2) delta
            'pitch': torch.tensor(pitch_eve, dtype=torch.float32),
            'yaw': torch.tensor(yaw_eve, dtype=torch.float32),
            'pitch_bin': torch.tensor(pitch_bin, dtype=torch.long),
            'yaw_bin': torch.tensor(yaw_bin, dtype=torch.long),
            'gaze_vector': torch.from_numpy(gt_vec),                     # (3,)
            'validity': torch.tensor(True, dtype=torch.bool),
            # Convenience metadata
            'sid': sid,
            'target_idx': target_row['idx'],
            'origin': target_row['origin'],
        }


def get_splits(data_root):
    """Sanity check: return which splits exist."""
    return [s for s in ['train', 'val', 'test']
            if osp.isfile(osp.join(data_root, f'{s}.label'))]
