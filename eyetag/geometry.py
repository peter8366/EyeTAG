"""Gaze-space geometry helpers shared by the Gaze360 and EVE pipelines.

Convention (used consistently across both datasets and by every model output):

    pitch, yaw  : radians
    unit vector : (x, y, z) with
                      x = -cos(pitch) * sin(yaw)
                      y = -sin(pitch)
                      z = -cos(pitch) * cos(yaw)

All functions accept either ``torch.Tensor`` or ``numpy.ndarray`` and return the
same type.
"""

import math

import numpy as np
import torch

__all__ = [
    'pitchyaw_to_vector',
    'vector_to_pitchyaw',
    'angular_error_pitchyaw',
    'angular_error_vector',
    'angular_error_torch',
    'delta_from_abs_pitchyaw',
]


def pitchyaw_to_vector(pitch, yaw):
    """(pitch, yaw) radians → (x, y, z) unit gaze vector. EVE convention.

    pitch=0, yaw=0 → (0, 0, -1).
    """
    if isinstance(pitch, torch.Tensor):
        cos_pitch = torch.cos(pitch)
        x = -cos_pitch * torch.sin(yaw)
        y = -torch.sin(pitch)
        z = -cos_pitch * torch.cos(yaw)
        return torch.stack([x, y, z], dim=-1)
    cos_pitch = np.cos(pitch)
    x = -cos_pitch * np.sin(yaw)
    y = -np.sin(pitch)
    z = -cos_pitch * np.cos(yaw)
    return np.stack([x, y, z], axis=-1)


def vector_to_pitchyaw(vec):
    """3D gaze vector (x, y, z) → (pitch, yaw) radians. EVE convention.

    vec: (..., 3)  returns (pitch, yaw) each (...,)
    """
    if isinstance(vec, torch.Tensor):
        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
        pitch = torch.asin(torch.clamp(-y, -1.0, 1.0))
        yaw = torch.atan2(-x, -z)
        return pitch, yaw
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    pitch = np.arcsin(np.clip(-y, -1.0, 1.0))
    yaw = np.arctan2(-x, -z)
    return pitch, yaw


def angular_error_pitchyaw(pred_pitch, pred_yaw, gt_pitch, gt_yaw):
    """Angular error in degrees from (pitch, yaw) pairs (EVE convention).

    Inputs: numpy arrays of shape (N,).  Returns scalar (degrees).
    """
    pred_vec = pitchyaw_to_vector(pred_pitch, pred_yaw)
    gt_vec = pitchyaw_to_vector(gt_pitch, gt_yaw)
    pred_vec = pred_vec / (np.linalg.norm(pred_vec, axis=1, keepdims=True) + 1e-8)
    gt_vec = gt_vec / (np.linalg.norm(gt_vec, axis=1, keepdims=True) + 1e-8)
    dots = np.clip((pred_vec * gt_vec).sum(axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).mean())


def angular_error_vector(pred_vec, gt_vec):
    """Angular error in degrees from (N, 3) unit-vector arrays."""
    pred_vec = pred_vec / (np.linalg.norm(pred_vec, axis=1, keepdims=True) + 1e-8)
    gt_vec = gt_vec / (np.linalg.norm(gt_vec, axis=1, keepdims=True) + 1e-8)
    dots = np.clip((pred_vec * gt_vec).sum(axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).mean())


def angular_error_torch(pred_pitch, pred_yaw, gt_pitch, gt_yaw):
    """Angular error in degrees (torch tensors). Returns scalar tensor."""
    pred_vec = pitchyaw_to_vector(pred_pitch, pred_yaw)
    gt_vec = pitchyaw_to_vector(gt_pitch, gt_yaw)
    pred_vec = torch.nn.functional.normalize(pred_vec, dim=-1, eps=1e-6)
    gt_vec = torch.nn.functional.normalize(gt_vec, dim=-1, eps=1e-6)
    dot = torch.clamp((pred_vec * gt_vec).sum(dim=-1), -1 + 1e-5, 1 - 1e-5)
    return torch.acos(dot).mean() * (180.0 / math.pi)


def delta_from_abs_pitchyaw(abs_arr, delta_repr):
    """Differential gaze sequence from an absolute pitchyaw sequence.

    Rebuttal Exp 2 (Reviewer SPwc): geometrically exact alternatives to the
    plain difference. All variants share |delta| == geodesic angle semantics
    for identical-frame padding (repeated gaze → zero row, same as baseline).

    abs_arr    : (N, 2) float32, EVE (pitch, yaw) radians
    delta_repr : 'pitchyaw' — (Δpitch, Δyaw) plain difference (baseline), (N-1, 2)
                 'tangent'  — sphere log-map velocity Log_{g_j}(g_{j+1}),
                              3D tangent vector at g_j, |·| = geodesic angle θ
                 'angvel'   — axis-angle rotation vector θ·(g_j × g_{j+1})/|g_j × g_{j+1}|,
                              the physical angular-velocity vector (rad/frame)
    returns    : (N-1, 2) or (N-1, 3) float32
    """
    abs_arr = np.asarray(abs_arr, dtype=np.float32)
    if abs_arr.shape[0] < 2:
        dim = 2 if delta_repr == 'pitchyaw' else 3
        return np.zeros((0, dim), dtype=np.float32)
    if delta_repr == 'pitchyaw':
        return (abs_arr[1:] - abs_arr[:-1]).astype(np.float32)

    v = pitchyaw_to_vector(abs_arr[:, 0], abs_arr[:, 1])   # (N, 3) unit
    v0, v1 = v[:-1], v[1:]
    cos = np.clip((v0 * v1).sum(axis=1), -1.0, 1.0)
    theta = np.arccos(cos)                                  # geodesic angle (rad)

    if delta_repr == 'tangent':
        u = v1 - cos[:, None] * v0                          # component of v1 ⊥ v0
    elif delta_repr == 'angvel':
        u = np.cross(v0, v1)                                # rotation axis (unnormalized)
    else:
        raise ValueError(f'unknown delta_repr: {delta_repr}')

    norm = np.linalg.norm(u, axis=1)
    out = np.zeros((v0.shape[0], 3), dtype=np.float32)
    ok = norm > 1e-8                                        # identical/antipodal-safe
    out[ok] = (theta[ok] / norm[ok])[:, None] * u[ok]
    return out
