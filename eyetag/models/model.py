"""EyeTAG — GazeEstimator.

Modular composition:
    DSVE (face backbone + eye backbone) + TAKE (prev_encoder)
    + CCAF (temporal_fusion) + GVP (head).

Backbone / fusion / prior options exposed via CLI flags:
    --face-backbone : vggface2 (ours) | resnet
    --eye-backbone  : resnet18 (ours) | efficientnet_b0 | tsm | 3dcnn
    --prev-input    : delta (kinematic prior, ours) | abs (normal prior)
    --prev-repr     : pitchyaw (ours) | vector | tangent | angvel
    --fusion-type   : cross_attn (ours) | bidir_cross_attn | self_attn_concat
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .face_resnet import FaceResNet
from .eye_net import EyeBackbone
from .prev_encoder import PrevLabelEncoder
from .temporal_fusion import TemporalFusion
from .head import GazeHead, GazeVectorHead, PoGHead, HeadPoseHead


class GazeEstimator(nn.Module):
    def __init__(self,
                 num_frames=7,
                 face_backbone_type='vggface2',
                 eye_backbone_type='resnet18',
                 prev_mode='mlp',
                 prev_input='abs',
                 prev_repr='pitchyaw',
                 d_model=256,
                 nhead=4,
                 num_layers=2,
                 n_bins=90,
                 dropout=0.5,
                 temporal_type='causal',
                 fusion_type='cross_attn',
                 use_pog=False,
                 gaze_space='vector',
                 pretrained=True,
                 pretrained_vggface=None,
                 freeze_backbone=False,
                 use_camera_emb=False,
                 use_head_pose=False,
                 num_cameras=1):
        super().__init__()
        self.num_frames = num_frames
        self.face_backbone_type = face_backbone_type
        self.prev_mode = prev_mode
        self.prev_input = prev_input
        if prev_input not in ('abs', 'delta'):
            raise ValueError(f"prev_input must be 'abs' or 'delta', got {prev_input}")
        self.prev_repr = prev_repr
        if prev_repr not in ('pitchyaw', 'vector', 'tangent', 'angvel'):
            raise ValueError(f"prev_repr must be 'pitchyaw'/'vector'/'tangent'/'angvel', "
                             f"got {prev_repr}")
        self.use_pog = use_pog
        self.gaze_space = gaze_space
        self.use_camera_emb = use_camera_emb
        self.use_head_pose = use_head_pose

        # ── Face backbone ──
        if face_backbone_type == 'vggface2':
            from .face_vggface2 import FaceVGGFace2
            self.face_backbone = FaceVGGFace2(
                pretrained_path=pretrained_vggface, freeze=False,
            )
            face_feat_dim = 2048
        elif face_backbone_type == 'resnet':
            self.face_backbone = FaceResNet(pretrained=pretrained)
            face_feat_dim = 512
        else:
            raise ValueError(f'Unknown face backbone: {face_backbone_type}')

        # ── Eye backbone ──
        eye_freeze = freeze_backbone
        self.eye_backbone = EyeBackbone(
            backbone_type=eye_backbone_type,
            pretrained=pretrained,
            freeze=eye_freeze,
        )
        eye_feat_dim = self.eye_backbone.feat_dim * 2  # left + right concatenated

        # ── Previous label encoder ──
        use_prev = (prev_mode != 'none')
        prev_out_dim = 64
        prev_num_tokens = 0
        if use_prev:
            # abs: T-1 past absolute gaze values
            # delta: T-2 frame-to-frame differences (one dummy entry would be unused)
            n_prev_steps = num_frames - 1 if prev_input == 'abs' else num_frames - 2
            if n_prev_steps < 1:
                raise ValueError(
                    f'num_frames={num_frames} too small for prev_input={prev_input} '
                    f'(needs >=2 for abs, >=3 for delta).')
            # 2D (pitch, yaw) for 'pitchyaw'; 3D for 'vector'/'tangent'/'angvel'.
            gaze_dim = 2 if prev_repr == 'pitchyaw' else 3
            self.prev_encoder = PrevLabelEncoder(
                num_prev=n_prev_steps,
                gaze_dim=gaze_dim,
                mode=prev_mode,
                hidden_dim=64,
                out_dim=prev_out_dim,
            )
            prev_num_tokens = self.prev_encoder.num_tokens
        else:
            self.prev_encoder = None

        # ── Temporal fusion ──
        self.temporal_fusion = TemporalFusion(
            face_dim=face_feat_dim,
            eye_dim=eye_feat_dim,
            prev_dim=prev_out_dim,
            prev_num_tokens=prev_num_tokens,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            num_frames=num_frames,
            use_prev=use_prev,
            temporal_type=temporal_type,
            fusion_type=fusion_type,
            use_camera_emb=use_camera_emb,
            num_cameras=num_cameras,
        )

        # ── Prediction heads ──
        if gaze_space == 'vector':
            self.gaze_head = GazeVectorHead(in_dim=d_model, dropout=dropout)
        else:
            self.gaze_head = GazeHead(in_dim=d_model, n_bins=n_bins, dropout=dropout)
        if use_pog:
            self.pog_head = PoGHead(in_dim=d_model, dropout=0.3)
        if use_head_pose:
            self.head_pose_head = HeadPoseHead(in_dim=d_model, out_dim=2, dropout=0.3)

        if freeze_backbone:
            for param in self.face_backbone.parameters():
                param.requires_grad = False

    # ── Helpers ──────────────────────────────────────────────────────────

    def _extract_face_feat(self, face_imgs):
        return self.face_backbone(face_imgs)

    @property
    def face_backbone_choices(self):
        return ['vggface2', 'resnet']

    def extract_visual(self, face_imgs, left_eye, right_eye):
        face_feat = self._extract_face_feat(face_imgs)
        eye_feat = self.eye_backbone(left_eye, right_eye)
        return face_feat, eye_feat

    def predict_from_features(self, face_feat, eye_feat, prev_gaze=None, camera_id=None):
        prev_feat = None
        if self.prev_encoder is not None and prev_gaze is not None:
            prev_feat = self.prev_encoder(prev_gaze)
        fused = self.temporal_fusion(face_feat, eye_feat, prev_feat, camera_id=camera_id)
        out = self.gaze_head(fused)
        if self.use_pog:
            out['pog'] = self.pog_head(fused)
        if self.use_head_pose:
            out['head_pose'] = self.head_pose_head(fused)
        return out

    def forward(self, face_imgs, left_eye, right_eye, prev_gaze=None, camera_id=None):
        face_feat, eye_feat = self.extract_visual(face_imgs, left_eye, right_eye)
        return self.predict_from_features(face_feat, eye_feat, prev_gaze, camera_id)

    def predict(self, face_imgs, left_eye, right_eye, prev_gaze=None, camera_id=None):
        """Inference helper.

        - vector mode: returns (pitch, yaw) derived from gaze_vector.
        - pitchyaw mode: returns (pitch_reg, yaw_reg) directly.
        Always (B, 2).
        """
        out = self.forward(face_imgs, left_eye, right_eye, prev_gaze, camera_id=camera_id)
        if self.gaze_space == 'vector':
            from ..geometry import vector_to_pitchyaw
            pitch, yaw = vector_to_pitchyaw(out['gaze_vector'])
            return torch.stack([pitch, yaw], dim=-1)
        return torch.stack([out['pitch_reg'], out['yaw_reg']], dim=-1)

    def predict_vector(self, face_imgs, left_eye, right_eye, prev_gaze=None, camera_id=None):
        """Return predicted 3D unit gaze vector (B, 3)."""
        out = self.forward(face_imgs, left_eye, right_eye, prev_gaze, camera_id=camera_id)
        if self.gaze_space == 'vector':
            return out['gaze_vector']
        from ..geometry import pitchyaw_to_vector
        vec = pitchyaw_to_vector(out['pitch_reg'], out['yaw_reg'])
        return F.normalize(vec, dim=-1, eps=1e-6)

    # ── Loss ─────────────────────────────────────────────────────────────

    def compute_loss(self, pred, gt_pitch, gt_yaw, gt_pitch_bin=None, gt_yaw_bin=None,
                     gt_vec=None, gt_pog=None, validity=None,
                     gt_head_pose=None, head_pose_validity=None,
                     alpha=1.0, lambda_pog=0.0, lambda_head=0.0):
        """Combined loss with optional validity masking.

        Args:
            pred: dict from forward()
            gt_pitch, gt_yaw: (B,) radians (EVE convention)
            gt_pitch_bin, gt_yaw_bin: (B,) long (used in pitchyaw mode only)
            gt_vec: (B, 3) unit vector — preferred GT for vector mode
            gt_pog, gt_head_pose: optional aux targets (unused on Gaze360 by default)
            validity: (B,) bool mask, or None (all valid)
            alpha: cls vs reg weight in pitchyaw mode (1=cls only, 0=reg only)

        Returns:
            losses_dict, angular_error_deg (scalar tensor)
        """
        from ..geometry import pitchyaw_to_vector

        if self.gaze_space == 'vector':
            return self._compute_loss_vector(
                pred, gt_pitch, gt_yaw, gt_vec, gt_pog, validity, lambda_pog,
                gt_head_pose=gt_head_pose, head_pose_validity=head_pose_validity,
                lambda_head=lambda_head)

        # ── pitchyaw mode ──
        if validity is not None:
            valid_mask = validity.bool()
            if valid_mask.sum() == 0:
                zero = torch.tensor(0.0, device=gt_pitch.device, requires_grad=True)
                return {'loss_total': zero, 'loss_cls': zero, 'loss_reg': zero,
                        'loss_ang': zero, 'loss_pog': zero, 'loss_head': zero,
                        'angular_error': zero}, zero
            pred_pitch_cls = pred['pitch_cls'][valid_mask]
            pred_yaw_cls = pred['yaw_cls'][valid_mask]
            pred_pitch_reg = pred['pitch_reg'][valid_mask]
            pred_yaw_reg = pred['yaw_reg'][valid_mask]
            gt_pitch = gt_pitch[valid_mask]
            gt_yaw = gt_yaw[valid_mask]
            gt_pitch_bin = gt_pitch_bin[valid_mask]
            gt_yaw_bin = gt_yaw_bin[valid_mask]
            pred_pog = pred['pog'][valid_mask] if 'pog' in pred else None
            if gt_pog is not None:
                gt_pog = gt_pog[valid_mask]
        else:
            pred_pitch_cls = pred['pitch_cls']
            pred_yaw_cls = pred['yaw_cls']
            pred_pitch_reg = pred['pitch_reg']
            pred_yaw_reg = pred['yaw_reg']
            pred_pog = pred.get('pog')

        loss_cls = (F.cross_entropy(pred_pitch_cls, gt_pitch_bin) +
                    F.cross_entropy(pred_yaw_cls, gt_yaw_bin))
        loss_reg = (F.mse_loss(pred_pitch_reg, gt_pitch) +
                    F.mse_loss(pred_yaw_reg, gt_yaw))

        pred_vec = pitchyaw_to_vector(pred_pitch_reg, pred_yaw_reg)
        gt_vec_local = pitchyaw_to_vector(gt_pitch, gt_yaw)
        pred_vec = F.normalize(pred_vec, dim=-1, eps=1e-6)
        gt_vec_local = F.normalize(gt_vec_local, dim=-1, eps=1e-6)
        dot = torch.clamp((pred_vec * gt_vec_local).sum(dim=-1), -1 + 1e-5, 1 - 1e-5)
        ang_error = torch.acos(dot).mean() * (180.0 / math.pi)
        loss_ang = torch.acos(dot).mean()

        loss_pog = torch.tensor(0.0, device=gt_pitch.device)
        if self.use_pog and pred_pog is not None and gt_pog is not None:
            loss_pog = F.mse_loss(pred_pog.float(), gt_pog.float())

        loss_head = torch.tensor(0.0, device=gt_pitch.device)
        if (self.use_head_pose and 'head_pose' in pred and gt_head_pose is not None
                and lambda_head > 0):
            pred_head = pred['head_pose']
            if validity is not None:
                pred_head = pred_head[valid_mask]
                gt_head_pose_v = (gt_head_pose[valid_mask]
                                  if gt_head_pose.shape[0] > pred_head.shape[0]
                                  else gt_head_pose)
            else:
                gt_head_pose_v = gt_head_pose
            if head_pose_validity is not None:
                hpv = head_pose_validity.bool()
                if validity is not None:
                    hpv = hpv[valid_mask]
                if hpv.any():
                    loss_head = F.mse_loss(pred_head[hpv].float(), gt_head_pose_v[hpv].float())
            else:
                loss_head = F.mse_loss(pred_head.float(), gt_head_pose_v.float())

        loss_total = (alpha * loss_cls + (1.0 - alpha) * loss_reg).float() + \
                     lambda_pog * loss_pog + lambda_head * loss_head
        return {
            'loss_total': loss_total,
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_ang': loss_ang,
            'loss_pog': loss_pog,
            'loss_head': loss_head,
            'angular_error': ang_error,
        }, ang_error

    def _compute_loss_vector(self, pred, gt_pitch, gt_yaw, gt_vec, gt_pog, validity,
                             lambda_pog, gt_head_pose=None,
                             head_pose_validity=None, lambda_head=0.0):
        """Compute loss in 3D gaze vector space."""
        from ..geometry import pitchyaw_to_vector

        if validity is not None:
            valid_mask = validity.bool()
            if valid_mask.sum() == 0:
                zero = torch.tensor(0.0, device=gt_pitch.device, requires_grad=True)
                return {'loss_total': zero, 'loss_cls': zero, 'loss_reg': zero,
                        'loss_ang': zero, 'loss_pog': zero, 'loss_head': zero,
                        'angular_error': zero}, zero
            pred_vec = pred['gaze_vector'][valid_mask]
            if gt_vec is not None:
                gt_vec_local = gt_vec[valid_mask]
            else:
                gt_pitch = gt_pitch[valid_mask]
                gt_yaw = gt_yaw[valid_mask]
                gt_vec_local = pitchyaw_to_vector(gt_pitch, gt_yaw)
            pred_pog = pred['pog'][valid_mask] if 'pog' in pred else None
            if gt_pog is not None:
                gt_pog = gt_pog[valid_mask]
        else:
            pred_vec = pred['gaze_vector']
            gt_vec_local = gt_vec if gt_vec is not None else pitchyaw_to_vector(gt_pitch, gt_yaw)
            pred_pog = pred.get('pog')

        gt_vec_local = F.normalize(gt_vec_local, dim=-1, eps=1e-6)
        dot = torch.clamp((pred_vec * gt_vec_local).sum(dim=-1), -1 + 1e-5, 1 - 1e-5)
        ang_error = torch.acos(dot).mean() * (180.0 / math.pi)
        loss_ang = torch.acos(dot).mean()

        loss_pog = torch.tensor(0.0, device=gt_pitch.device)
        if self.use_pog and pred_pog is not None and gt_pog is not None:
            loss_pog = F.mse_loss(pred_pog.float(), gt_pog.float())

        loss_head = torch.tensor(0.0, device=gt_pitch.device)
        if (self.use_head_pose and 'head_pose' in pred and gt_head_pose is not None
                and lambda_head > 0):
            pred_head = pred['head_pose']
            if validity is not None:
                pred_head = pred_head[valid_mask]
                gt_head_pose_v = (gt_head_pose[valid_mask]
                                  if gt_head_pose.shape[0] > pred_head.shape[0]
                                  else gt_head_pose)
            else:
                gt_head_pose_v = gt_head_pose
            if head_pose_validity is not None:
                hpv = head_pose_validity.bool()
                if validity is not None:
                    hpv = hpv[valid_mask]
                if hpv.any():
                    loss_head = F.mse_loss(pred_head[hpv].float(), gt_head_pose_v[hpv].float())
            else:
                loss_head = F.mse_loss(pred_head.float(), gt_head_pose_v.float())

        loss_total = loss_ang.float() + lambda_pog * loss_pog + lambda_head * loss_head
        zero = torch.tensor(0.0, device=gt_pitch.device)
        return {
            'loss_total': loss_total,
            'loss_cls': zero,
            'loss_reg': zero,
            'loss_ang': loss_ang,
            'loss_pog': loss_pog,
            'loss_head': loss_head,
            'angular_error': ang_error,
        }, ang_error
