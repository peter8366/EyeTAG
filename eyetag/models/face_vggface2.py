"""
models/face_vggface2.py

ResNet-50 face backbone pretrained on VGGFace2 (cydonia999/VGGFace2-pytorch
`resnet50_ft_weight.pkl`). Used by CrossGaze and similar SOTA — provides
face-domain features that ImageNet-pretrained ResNet lacks.

Input  : (B, 3, T, H, W)  — H,W typically 128x128. **The dataset feeds
         ImageNet-normalized RGB tensors**, so this module re-normalizes
         internally to VGGFace2 stats (BGR, mean-subtraction, 0-255 scale).
Output : (B, T, 2048)

Weight file (~165 MB):
  https://github.com/cydonia999/VGGFace2-pytorch (Google Drive)
  Local: checkpoints/resnet50_ft_weight.pkl
"""

import pickle
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50

logger = logging.getLogger(__name__)


# ImageNet normalization used by dataset.py (RGB, 0-1 scale)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# VGGFace2 normalization (BGR, 0-255 scale, mean-subtraction only)
_VGGFACE2_BGR_MEAN = torch.tensor([91.4953, 103.8827, 131.0912]).view(1, 3, 1, 1)


def _load_vggface2_pickle(path: str) -> dict:
    """Load cydonia999 VGGFace2 weights from pickle. Returns dict[str, Tensor]."""
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    out = {}
    for k, v in raw.items():
        out[k] = torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v
    return out


class FaceVGGFace2(nn.Module):
    """ResNet-50 face backbone, VGGFace2 pretrained (cydonia999 weights)."""

    def __init__(self, pretrained_path: str = None, freeze: bool = False):
        super().__init__()
        # torchvision resnet50 architecture (no ImageNet weights — we load VGGFace2)
        base = resnet50(weights=None)

        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        # fc layer dropped — gaze head is downstream

        self.feat_dim = 2048

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        # buffers for re-normalization (registered so they move with .to(device))
        self.register_buffer('_imagenet_mean', _IMAGENET_MEAN)
        self.register_buffer('_imagenet_std',  _IMAGENET_STD)
        self.register_buffer('_vggface2_mean', _VGGFACE2_BGR_MEAN)

    def _load_pretrained(self, path: str):
        path = str(path)
        if not Path(path).exists():
            raise FileNotFoundError(f'VGGFace2 weight file not found: {path}')
        weights = _load_vggface2_pickle(path)
        own = self.state_dict()
        loaded, skipped = [], []
        for k, v in weights.items():
            if k.startswith('fc.'):
                skipped.append(k)        # classifier head - not needed for gaze
                continue
            if k not in own:
                skipped.append(k)
                continue
            if own[k].shape != v.shape:
                skipped.append(f'{k} (shape mismatch {own[k].shape} vs {v.shape})')
                continue
            own[k] = v
            loaded.append(k)
        msg = self.load_state_dict(own, strict=False)
        logger.info(f'[FaceVGGFace2] loaded {len(loaded)} VGGFace2 tensors '
                    f'from {path} (skipped {len(skipped)})')

    def _imagenet_to_vggface2(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized RGB → VGGFace2 input (BGR, 0-255, mean-sub).

        x: (BT, 3, H, W) in ImageNet normalized space.
        """
        # un-normalize ImageNet
        x = x * self._imagenet_std + self._imagenet_mean      # → RGB 0-1
        # to 0-255
        x = x * 255.0                                          # → RGB 0-255
        # RGB → BGR (flip channel dim)
        x = x.flip(dims=[1])                                   # → BGR 0-255
        # subtract VGGFace2 mean
        x = x - self._vggface2_mean                            # → BGR mean-sub
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W) → (B, T, 2048)"""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # (BT, C, H, W)

        x = self._imagenet_to_vggface2(x)                       # re-norm

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)                                        # (BT, 2048)

        return x.reshape(B, T, self.feat_dim)
