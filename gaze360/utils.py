"""Gaze360-specific training utilities.

Gaze-space math (pitchyaw <-> vector, angular error, differential gaze) lives in
``eyetag.geometry`` and is re-exported here for convenience.
"""

import os
from typing import Any, Dict, List

import numpy as np
import torch

from eyetag.geometry import (
    pitchyaw_to_vector,
    vector_to_pitchyaw,
    angular_error_pitchyaw,
    angular_error_vector,
    angular_error_torch,
    delta_from_abs_pitchyaw,
)

__all__ = [
    'pitchyaw_to_vector', 'vector_to_pitchyaw', 'angular_error_pitchyaw',
    'angular_error_vector', 'angular_error_torch', 'delta_from_abs_pitchyaw',
    'collate_fn', 'set_seed', 'save_checkpoint', 'load_checkpoint',
]


def gaze360_to_eve_pitchyaw(g360_yaw, g360_pitch):
    """gaze360 (yaw, pitch) → EVE (pitch, yaw). Preserves the 3D direction."""
    return -g360_pitch, -g360_yaw


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader collate: stack tensors, keep metadata as lists."""
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(path, epoch, model, optimizer, scheduler,
                    best_val_err, args, **extra):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_err': best_val_err,
        'args': vars(args) if hasattr(args, '__dict__') else args,
    }
    state.update(extra)
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt
