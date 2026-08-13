"""
models/face_resnet.py

ResNet-18 per-frame face feature extractor (ImageNet pretrained).
Input : (B, 3, T, H, W)  — H,W typically 128x128 or 224x224
Output: (B, T, 512)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FaceResNet(nn.Module):
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
        """x: (B, 3, T, H, W) → (B, T, 512)"""
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
