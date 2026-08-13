"""
models/eye_net.py

Eye backbone — processes left/right eye images independently (shared weights).
Supports: 'resnet18' (ours), 'efficientnet_b0', 'tsm', '3dcnn'

Input per eye:  (B, 3, T, 128, 128)
Output per eye: (B, T, feat_dim)
Combined output (left+right cat): (B, T, feat_dim * 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)


class EyeResNet(nn.Module):
    """ResNet-18 per-frame eye feature extractor."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.feat_dim = 512

    def forward(self, x):
        """x: (B, 3, T, 128, 128) → (B, T, 512)"""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        return x.reshape(B, T, self.feat_dim)


class EyeTSM(nn.Module):
    """ResNet-18 + causal Temporal Shift Module (TSM).

    TSM (Lin et al., NeurIPS 2019) inserts channel shifts in time between
    residual blocks. Causal variant: shift only from past frames.

    Zero extra parameters vs EyeResNet — keeps ImageNet pretraining.
    Adds temporal modeling that EyeResNet (per-frame) lacks.
    """

    def __init__(self, pretrained=True, fold_div=8):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.fold_div = fold_div
        self.feat_dim = 512

    def forward(self, x):
        """x: (B, 3, T, 128, 128) → (B, T, 512)"""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = _temporal_shift_causal(x, T, self.fold_div)
        x = self.layer1(x)
        x = _temporal_shift_causal(x, T, self.fold_div)
        x = self.layer2(x)
        x = _temporal_shift_causal(x, T, self.fold_div)
        x = self.layer3(x)
        x = _temporal_shift_causal(x, T, self.fold_div)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        return x.reshape(B, T, self.feat_dim)


class Eye3DCNN(nn.Module):
    """Lightweight 3D CNN eye feature extractor.

    Unlike EyeResNet which processes each frame independently,
    this uses 3D convolutions to capture temporal patterns in eye movements.
    Temporal stride is kept at 1 to preserve T dimension for TemporalFusion.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # (B, 64, T, 32, 32)

        self.block1 = self._make_block(64, 128, temporal_stride=1, spatial_stride=2)
        # (B, 128, T, 16, 16)
        self.block2 = self._make_block(128, 256, temporal_stride=1, spatial_stride=2)
        # (B, 256, T, 8, 8)
        self.block3 = self._make_block(256, 512, temporal_stride=1, spatial_stride=2)
        # (B, 512, T, 4, 4)

        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep T, pool spatial
        self.feat_dim = 512

    @staticmethod
    def _make_block(in_ch, out_ch, temporal_stride, spatial_stride):
        stride = (temporal_stride, spatial_stride, spatial_stride)
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: (B, 3, T, 128, 128) → (B, T, 512)"""
        x = self.stem(x)       # (B, 64, T, 32, 32)
        x = self.block1(x)     # (B, 128, T, 16, 16)
        x = self.block2(x)     # (B, 256, T, 8, 8)
        x = self.block3(x)     # (B, 512, T, 4, 4)
        x = self.pool(x)       # (B, 512, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 512, T)
        return x.permute(0, 2, 1)      # (B, T, 512)


class EyeEfficientNetB0(nn.Module):
    """EfficientNet-B0 per-frame eye feature extractor (ImageNet pretrained).

    Same per-frame interface as EyeResNet: (B, 3, T, H, W) → (B, T, feat_dim).
    Uses MBConv blocks + Squeeze-and-Excitation (a different CNN family vs
    ResNet's plain BasicBlock / Bottleneck stack).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.features = base.features    # stem (Conv2dNormActivation) + 7 MBConv stages
        self.avgpool  = base.avgpool     # AdaptiveAvgPool2d((1, 1))
        self.feat_dim = 1280             # EfficientNet-B0 native output dim

    def forward(self, x):
        """x: (B, 3, T, H, W) → (B, T, 1280)"""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.features(x)                    # (BT, 1280, H', W')
        x = self.avgpool(x).flatten(1)          # (BT, 1280)
        return x.reshape(B, T, self.feat_dim)


class EyeBackbone(nn.Module):
    """Unified eye backbone wrapper. Processes L/R eyes with shared weights."""

    def __init__(self, backbone_type='resnet18', pretrained=True, freeze=True):
        super().__init__()
        if backbone_type == 'resnet18':
            self.net = EyeResNet(pretrained=pretrained)
        elif backbone_type == 'tsm':
            self.net = EyeTSM(pretrained=pretrained)
        elif backbone_type == '3dcnn':
            self.net = Eye3DCNN()
        elif backbone_type == 'efficientnet_b0':
            self.net = EyeEfficientNetB0(pretrained=pretrained)
        else:
            raise ValueError(f'Unknown eye backbone: {backbone_type}')

        self.feat_dim = self.net.feat_dim  # per-eye feature dim

    def forward(self, left_eye, right_eye):
        """
        left_eye:  (B, 3, T, H, W)
        right_eye: (B, 3, T, H, W) — already horizontally flipped if needed
        returns:   (B, T, feat_dim * 2)
        """
        left_feat = self.net(left_eye)    # (B, T, feat_dim)
        right_feat = self.net(right_eye)  # (B, T, feat_dim)
        return torch.cat([left_feat, right_feat], dim=-1)  # (B, T, feat_dim*2)
