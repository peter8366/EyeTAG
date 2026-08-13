"""
models/head.py

Prediction heads:
  GazeHead       : Classification + Regression dual head (pitch/yaw)
  GazeVectorHead : 3D gaze vector regression head (x, y, z)
  PoGHead        : Point-of-Gaze regression head (screen coordinates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeHead(nn.Module):
    """Classification + Regression dual prediction head (L2CS-Net style)."""

    def __init__(self, in_dim=256, n_bins=36, dropout=0.5):
        super().__init__()
        self.pitch_cls = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(128, n_bins),
        )
        self.yaw_cls = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(128, n_bins),
        )
        self.pitch_reg = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(128, 1),
        )
        self.yaw_reg = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(128, 1),
        )

    def forward(self, x):
        """x: (B, in_dim) → dict"""
        return {
            'pitch_cls': self.pitch_cls(x),
            'yaw_cls': self.yaw_cls(x),
            'pitch_reg': self.pitch_reg(x).squeeze(-1),
            'yaw_reg': self.yaw_reg(x).squeeze(-1),
        }


class GazeVectorHead(nn.Module):
    """3D gaze vector regression head. Outputs L2-normalized (x, y, z)."""

    def __init__(self, in_dim=256, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(128, 3),
        )

    def forward(self, x):
        """x: (B, in_dim) → dict with 'gaze_vector': (B, 3) unit vector"""
        raw = self.head(x)  # (B, 3)
        gaze_vec = F.normalize(raw, dim=-1, eps=1e-6)
        return {'gaze_vector': gaze_vec}


class PoGHead(nn.Module):
    """Point-of-Gaze regression head (screen pixel coordinates)."""

    def __init__(self, in_dim=256, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),  # (x, y) screen pixels
        )

    def forward(self, x):
        """x: (B, in_dim) → (B, 2)"""
        return self.head(x)


class HeadPoseHead(nn.Module):
    """Head pose regression head (pitch, yaw in radians)."""

    def __init__(self, in_dim=256, out_dim=2, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        """x: (B, in_dim) → (B, out_dim)"""
        return self.head(x)
