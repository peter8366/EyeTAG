"""EVE dataset loader.

Expects the official EVE release layout (MP4 video plus per-session HDF5
labels), read with decord and h5py:

    <data-root>/
        train01/, ..., val01/, ..., test01/
            step008_image_MIT-i2263005350/
                webcam_c.mp4, webcam_c_eyes.mp4, webcam_c.h5, ...

Sequences are sliding windows of T frames within one (participant, session,
camera) track. Frames are subsampled to `--target-hz` (30 Hz in the paper).
"""

import os
import os.path as osp
import random
from collections import defaultdict

import cv2
import decord
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Decord: numpy bridge (safe under multiprocessing)
decord.bridge.set_bridge('native')

# ImageNet normalization (for pretrained backbones)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default gaze for padding (looking forward)
GAZE_INIT = np.array([0.0, 0.0], dtype=np.float32)  # (pitch=0, yaw=0)

# Source frame rate per camera (raw video fps)
SOURCE_FPS = {'basler': 60, 'webcam_l': 30, 'webcam_c': 30, 'webcam_r': 30}

# Camera → subsample rate to achieve 10Hz (legacy constant; prefer get_subsample_rates())
CAMERA_SUBSAMPLE = {'basler': 6, 'webcam_c': 3, 'webcam_l': 3, 'webcam_r': 3}
ALL_CAMERAS = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
CAMERA_TO_ID = {'basler': 0, 'webcam_l': 1, 'webcam_c': 2, 'webcam_r': 3}


def get_subsample_rates(target_hz):
    """Per-camera integer subsample rate to reach target_hz from each camera's source fps.

    Raises ValueError if any camera's fps is not divisible by target_hz.
    """
    rates = {}
    for cam, fps in SOURCE_FPS.items():
        if fps % target_hz != 0:
            raise ValueError(
                f'Camera {cam} ({fps}Hz) cannot be cleanly subsampled to {target_hz}Hz '
                f'(fps {fps} not divisible by {target_hz})')
        rates[cam] = fps // target_hz
    return rates


def normalize_image(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Normalize image: (H, W, 3) uint8 → (3, H, W) float32 normalized."""
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img.transpose(2, 0, 1)  # (3, H, W)


def angle_to_bin(angle, min_val, max_val, n_bins):
    """Continuous angle → bin index [0, n_bins-1]."""
    bin_width = (max_val - min_val) / n_bins
    return int(np.clip((angle - min_val) / bin_width, 0, n_bins - 1))


class EVEDataset(Dataset):
    """
    EVE dataset loader with multi-camera and scheduled sampling support.

    Args:
        data_root:  EVE dataset root
        participants: list of participant IDs
        cameras:    camera name(s) — str or list/tuple
        num_frames: sequence length T
        face_size:  face crop resize target (default 128)
        eye_size:   eye crop resize target per eye (default 128)
        n_bins:     classification bins per axis
        pitch_range: (min, max) radians for binning
        yaw_range:   (min, max) radians for binning
        test_mode:  disable augmentation
        augment:    enable augmentation in training
        prediction_cache: dict mapping (participant, session, camera, frame_idx) → (pitch, yaw)
                          None means ground truth only
        prev_gt_ratio: probability of using GT (1.0 = all GT, 0.0 = all cached predictions)
    """

    def __init__(self,
                 data_root,
                 participants,
                 cameras='all',
                 num_frames=8,
                 face_size=128,
                 eye_size=128,
                 n_bins=36,
                 pitch_range=(-0.628, 0.628),
                 yaw_range=(-0.628, 0.628),
                 test_mode=False,
                 augment=True,
                 seq_stride=1,
                 prediction_cache=None,
                 prev_gt_ratio=1.0,
                 target_hz=10,
                 prev_input='abs'):
        self.data_root = data_root
        self.num_frames = num_frames
        self.face_size = face_size
        self.eye_size = eye_size
        self.n_bins = n_bins
        self.pitch_range = pitch_range
        self.yaw_range = yaw_range
        self.test_mode = test_mode
        self.augment = augment and not test_mode
        self.seq_stride = seq_stride
        self.prediction_cache = prediction_cache
        self.prev_gt_ratio = prev_gt_ratio
        self.target_hz = target_hz
        if prev_input not in ('abs', 'delta'):
            raise ValueError(f"prev_input must be 'abs' or 'delta', got {prev_input}")
        self.prev_input = prev_input
        self.subsample_rates = get_subsample_rates(target_hz)

        # Parse cameras
        if cameras == 'all':
            self.cameras = ALL_CAMERAS
        elif isinstance(cameras, str):
            self.cameras = cameras.split(',')
        else:
            self.cameras = list(cameras)

        # Build sequence index
        self.sequences = []
        self.participant_sessions = defaultdict(list)

        # Session-level label cache: (participant/session, camera) → {gaze, gaze_valid, pog, pog_valid}
        self.session_labels = {}

        self._build_index(participants)

    def _build_index(self, participants):
        """Scan dataset and build sequence index for all cameras."""
        sessions_seen = set()

        for participant in sorted(participants):
            participant_dir = osp.join(self.data_root, participant)
            if not osp.isdir(participant_dir):
                continue

            for session in sorted(os.listdir(participant_dir)):
                session_dir = osp.join(participant_dir, session)
                if not osp.isdir(session_dir):
                    continue

                for camera in self.cameras:
                    subsample_rate = self.subsample_rates[camera]

                    h5_path = osp.join(session_dir, f'{camera}.h5')
                    face_path = osp.join(session_dir, f'{camera}_face.mp4')
                    eyes_path = osp.join(session_dir, f'{camera}_eyes.mp4')

                    if not (osp.exists(h5_path) and osp.exists(face_path) and osp.exists(eyes_path)):
                        continue

                    # Read h5 to get frame count and validity
                    try:
                        with h5py.File(h5_path, 'r') as h5:
                            gaze_data = h5['face_g_tobii']['data'][:]          # (N, 2)
                            gaze_valid = h5['face_g_tobii']['validity'][:]     # (N,)
                            pog_data = h5['face_PoG_tobii']['data'][:]         # (N, 2)
                            pog_valid = h5['face_PoG_tobii']['validity'][:]    # (N,)
                            head_pose_data = h5['face_h']['data'][:]           # (N, 2) pitch/yaw
                            head_pose_valid = h5['face_h']['validity'][:]      # (N,)
                    except Exception:
                        continue

                    total_frames = len(gaze_data)

                    # Subsampled frame indices
                    sub_indices = list(range(0, total_frames, subsample_rate))
                    n_sub = len(sub_indices)

                    if n_sub < self.num_frames:
                        continue

                    # Store labels per (session, camera)
                    label_key = f'{participant}/{session}/{camera}'
                    self.session_labels[label_key] = {
                        'gaze': gaze_data,
                        'gaze_valid': gaze_valid,
                        'pog': pog_data,
                        'pog_valid': pog_valid,
                        'head_pose': head_pose_data,
                        'head_pose_valid': head_pose_valid,
                    }

                    # Create sequences: sliding window over subsampled indices
                    for seq_start in range(0, n_sub - self.num_frames + 1, self.seq_stride):
                        frame_indices = sub_indices[seq_start:seq_start + self.num_frames]
                        target_idx = frame_indices[-1]

                        # Check if target frame is valid
                        if not gaze_valid[target_idx]:
                            continue

                        self.sequences.append({
                            'participant': participant,
                            'session': session,
                            'camera': camera,
                            'session_dir': session_dir,
                            'label_key': label_key,
                            'frame_indices': frame_indices,
                            'target_idx': target_idx,
                        })

                    # Track sessions per participant
                    sess_key = (participant, session)
                    if sess_key not in sessions_seen:
                        sessions_seen.add(sess_key)
                        self.participant_sessions[participant].append(session)

        # Summary per camera
        cam_counts = defaultdict(int)
        for seq in self.sequences:
            cam_counts[seq['camera']] += 1

        cam_str = ', '.join(f'{c}={cam_counts[c]}' for c in self.cameras if c in cam_counts)
        print(f'[EVEDataset] {len(self.sequences)} sequences from '
              f'{len(self.participant_sessions)} participants, '
              f'cameras=[{",".join(self.cameras)}], T={self.num_frames} '
              f'({cam_str})')

    def __len__(self):
        return len(self.sequences)

    def _read_video_frames(self, video_path, frame_indices):
        """Read specific frames from mp4 video using Decord get_batch()."""
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, 3) RGB
            return [frames[i] for i in range(len(frame_indices))]
        except Exception:
            # Fallback: return black frames
            try:
                vr = decord.VideoReader(video_path, num_threads=1)
                h, w = vr[0].shape[:2]
            except Exception:
                h, w = 256, 256
            return [np.zeros((h, w, 3), dtype=np.uint8) for _ in frame_indices]

    def _split_eyes(self, eyes_frame):
        """Split concatenated eyes frame (256x128) into left and right (128x128 each)."""
        h, w, c = eyes_frame.shape
        mid = w // 2
        left = eyes_frame[:, :mid, :]
        right = eyes_frame[:, mid:, :]
        return left, right

    def _get_prev_gaze(self, participant, session, camera, frame_indices, gaze_data, gaze_valid, flip):
        """Construct prev_gaze with scheduled sampling support.

        Mix ground-truth and cached predictions according to prev_gt_ratio.
        - prev_gt_ratio=1.0: all ground truth
        - prev_gt_ratio=0.0: all cached predictions (zero fallback if absent)
        """
        prev_gaze_list = []
        cache = self.prediction_cache

        for fi in frame_indices[:-1]:  # All frames except the last (target)
            use_gt = True

            # Scheduled sampling: with a cache and ratio < 1.0, sample the prediction
            if cache is not None and self.prev_gt_ratio < 1.0:
                cache_key = (participant, session, camera, fi)
                if cache_key in cache:
                    if random.random() > self.prev_gt_ratio:
                        # use the cached prediction
                        cached = cache[cache_key]
                        pg = np.array([cached[0], cached[1]], dtype=np.float32)
                        if flip:
                            pg[1] = -pg[1]
                        prev_gaze_list.append(pg)
                        use_gt = False

            if use_gt:
                if gaze_valid[fi]:
                    pg = gaze_data[fi].copy()
                    if flip:
                        pg[1] = -pg[1]
                    prev_gaze_list.append(pg)
                else:
                    prev_gaze_list.append(GAZE_INIT.copy())

        abs_arr = np.stack(prev_gaze_list).astype(np.float32)  # (T-1, 2)
        if self.prev_input == 'delta':
            # delta[j] = abs[j+1] - abs[j]  for j = 0..T-3 ; shape (T-2, 2)
            # No dummy zero token — uses real transitions only.
            if abs_arr.shape[0] >= 2:
                return (abs_arr[1:] - abs_arr[:-1]).astype(np.float32)
            return np.zeros((max(0, self.num_frames - 2), 2), dtype=np.float32)
        return abs_arr

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        session_dir = seq['session_dir']
        label_key = seq['label_key']
        camera = seq['camera']
        frame_indices = seq['frame_indices']
        target_idx = seq['target_idx']

        labels = self.session_labels[label_key]
        gaze_data = labels['gaze']
        gaze_valid = labels['gaze_valid']
        pog_data = labels['pog']
        head_pose_data = labels['head_pose']
        head_pose_valid = labels['head_pose_valid']

        T = self.num_frames

        # ── Read video frames (camera-specific) ──
        face_path = osp.join(session_dir, f'{camera}_face.mp4')
        eyes_path = osp.join(session_dir, f'{camera}_eyes.mp4')

        face_frames = self._read_video_frames(face_path, frame_indices)
        eyes_frames = self._read_video_frames(eyes_path, frame_indices)

        # ── Process frames ──
        face_imgs = []
        left_imgs = []
        right_imgs = []

        # Augmentation params (consistent across all frames in sequence)
        flip = False
        color_jitter = False
        if self.augment:
            if random.random() < 0.5:
                flip = True
            if random.random() < 0.5:
                color_jitter = True
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)

        for i in range(T):
            face = face_frames[i]
            left, right = self._split_eyes(eyes_frames[i])

            # Resize
            face = cv2.resize(face, (self.face_size, self.face_size))
            left = cv2.resize(left, (self.eye_size, self.eye_size))
            right = cv2.resize(right, (self.eye_size, self.eye_size))

            # Flip augmentation
            if flip:
                face = np.fliplr(face).copy()
                left_aug = np.fliplr(right).copy()
                right_aug = np.fliplr(left).copy()
                left, right = left_aug, right_aug

            # Color jitter (simple brightness + contrast)
            if color_jitter:
                face = np.clip(face.astype(np.float32) * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)
                left = np.clip(left.astype(np.float32) * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)
                right = np.clip(right.astype(np.float32) * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)

            face_imgs.append(normalize_image(face))
            left_imgs.append(normalize_image(left))
            right_imgs.append(normalize_image(right))

        # Stack: (T, 3, H, W) → (3, T, H, W)
        face_tensor = torch.from_numpy(np.stack(face_imgs)).permute(1, 0, 2, 3)
        left_tensor = torch.from_numpy(np.stack(left_imgs)).permute(1, 0, 2, 3)
        right_tensor = torch.from_numpy(np.stack(right_imgs)).permute(1, 0, 2, 3)

        # ── Labels ──
        target_gaze = gaze_data[target_idx]  # (2,) pitch, yaw radians
        target_pog = pog_data[target_idx]    # (2,) screen pixels
        target_head = head_pose_data[target_idx].astype(np.float32)  # (2,) pitch, yaw
        target_head_valid = bool(head_pose_valid[target_idx])

        pitch = target_gaze[0]
        yaw = target_gaze[1]

        if flip:
            yaw = -yaw  # Flip yaw direction
            target_head = target_head.copy()
            target_head[1] = -target_head[1]  # Flip head yaw too

        # Bin labels
        pitch_bin = angle_to_bin(pitch, self.pitch_range[0], self.pitch_range[1], self.n_bins)
        yaw_bin = angle_to_bin(yaw, self.yaw_range[0], self.yaw_range[1], self.n_bins)

        # ── Previous gaze (with scheduled sampling) ──
        prev_gaze = self._get_prev_gaze(
            seq['participant'], seq['session'], camera,
            frame_indices, gaze_data, gaze_valid, flip,
        )

        # Validity for the target frame
        valid = gaze_valid[target_idx]

        return {
            'face': face_tensor,                                          # (3, T, H, W)
            'left_eye': left_tensor,                                      # (3, T, H, W)
            'right_eye': right_tensor,                                    # (3, T, H, W)
            'prev_gaze': torch.from_numpy(prev_gaze),                    # (T-1, 2)
            'pitch': torch.tensor(pitch, dtype=torch.float32),            # scalar
            'yaw': torch.tensor(yaw, dtype=torch.float32),               # scalar
            'pitch_bin': torch.tensor(pitch_bin, dtype=torch.long),       # scalar
            'yaw_bin': torch.tensor(yaw_bin, dtype=torch.long),           # scalar
            'pog': torch.tensor(target_pog, dtype=torch.float32),         # (2,)
            'validity': torch.tensor(valid, dtype=torch.bool),            # scalar
            'head_pose': torch.tensor(target_head, dtype=torch.float32),  # (2,)
            'head_pose_validity': torch.tensor(target_head_valid, dtype=torch.bool),
            'camera_id': torch.tensor(CAMERA_TO_ID[camera], dtype=torch.long),
            'participant': seq['participant'],
            'session': seq['session'],
            'camera': camera,
            'target_idx': target_idx,
            'frame_indices': torch.tensor(frame_indices, dtype=torch.long),  # (T,)
        }


def get_participants(data_root, split='train'):
    """Get participant list for a split.

    eyetag_EVE_final splits (3-way, subject-disjoint):
      - 'train': train01..train35       (35 participants) — model weight learning
      - 'val'  : train36..train39       (4 participants)  — early stopping / monitoring
      - 'test' : val01..val05           (5 participants)  — FINAL held-out eval

    NOTE: This splitting differs from EVE_v2 ('train' = all train01~39, 'val' = val01~05).
    Reason: EVE's official test labels (test01~10) are held by CodaBench, so val01~05 must
    serve as our test set. Using val01~05 for monitoring would leak test info into model
    selection. We therefore reserve train36~39 as the internal validation split.

    The original splits are still selectable via:
      'train_official': all train01~39
      'val_official':   val01~05
      'test_official':  test01~10
    """
    all_dirs = sorted(os.listdir(data_root))
    if split == 'train':
        # Internal-train: train01..train35
        return [f'train{i:02d}' for i in range(1, 36)
                if f'train{i:02d}' in all_dirs]
    elif split == 'val':
        # Internal-val: train36..train39
        return [f'train{i:02d}' for i in range(36, 40)
                if f'train{i:02d}' in all_dirs]
    elif split == 'test':
        # Held-out test: val01..val05 (EVE official val, our test)
        return [f'val{i:02d}' for i in range(1, 6)
                if f'val{i:02d}' in all_dirs]
    elif split == 'train_official':
        return [d for d in all_dirs if d.startswith('train')]
    elif split == 'val_official':
        return [d for d in all_dirs if d.startswith('val')]
    elif split == 'test_official':
        return [d for d in all_dirs if d.startswith('test')]
    else:
        raise ValueError(f'Unknown split: {split}')
