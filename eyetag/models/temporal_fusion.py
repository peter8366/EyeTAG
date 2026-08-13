"""
models/temporal_fusion.py

Temporal fusion module — combines face/eye/prev features and models temporal dependencies.

Supports:
  'transformer' : Transformer Encoder (default)
  'gru'         : Bidirectional GRU
  'hybrid'      : Transformer Encoder → GRU
  'causal'      : Causal Transformer Decoder (no future frame access)

Fusion modes for face+eye:
  'add'              : element-wise addition (default)
  'separate'         : face and eye as separate tokens (2T tokens)
  'cross_attn'       : face→eye cross-attention before temporal module
  'concat'           : channel concat → lightweight attention → project
  'bidir_cross_attn' : face↔eye bidirectional cross-attention (DHECA-style),
                        followed by element-wise add of refined streams.
  'self_attn_concat' : each modality is first refined by its OWN self-attention
                        (intra-modal temporal refinement), then channel-concat
                        and projected back to d_model.

Optional camera conditioning:
  use_camera_emb=True adds a learnable per-camera embedding (nn.Embedding(4, d_model))
  that is broadcast across the visual tokens. Pass camera_id in forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusion(nn.Module):
    def __init__(self, face_dim, eye_dim, prev_dim=64, prev_num_tokens=1,
                 d_model=256, nhead=4, num_layers=2, dim_feedforward=1024,
                 dropout=0.1, num_frames=8, use_prev=True,
                 temporal_type='transformer', fusion_type='add',
                 use_camera_emb=False, num_cameras=4):
        super().__init__()
        self.use_prev = use_prev
        self.temporal_type = temporal_type
        self.fusion_type = fusion_type
        self.d_model = d_model
        self.num_frames = num_frames
        self.use_camera_emb = use_camera_emb

        # Feature projections
        self.face_proj = nn.Linear(face_dim, d_model)
        self.eye_proj = nn.Linear(eye_dim, d_model)

        if use_prev:
            self.prev_proj = nn.Linear(prev_dim, d_model)

        if use_camera_emb:
            self.camera_embedding = nn.Embedding(num_cameras, d_model)
            nn.init.normal_(self.camera_embedding.weight, std=0.02)

        # Fusion-specific modules
        if fusion_type == 'cross_attn':
            self.cross_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)

        elif fusion_type == 'bidir_cross_attn':
            self.face_attends_eye = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.eye_attends_face = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.face_norm = nn.LayerNorm(d_model)
            self.eye_norm = nn.LayerNorm(d_model)
            self.bidir_proj = nn.Linear(d_model * 2, d_model)

        elif fusion_type == 'concat':
            self.concat_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif fusion_type == 'self_attn_concat':
            # Independent self-attention on each modality, then concat + project.
            self.face_self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.eye_self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.face_self_norm = nn.LayerNorm(d_model)
            self.eye_self_norm  = nn.LayerNorm(d_model)
            self.self_attn_concat_proj = nn.Linear(d_model * 2, d_model)

        # Sequence length calculation
        if fusion_type == 'separate':
            visual_len = num_frames * 2  # face + eye as separate tokens
        else:
            visual_len = num_frames

        prev_tokens = prev_num_tokens if use_prev else 0
        seq_len = visual_len + prev_tokens

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Temporal modules
        if temporal_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation='gelu',
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        elif temporal_type == 'gru':
            self.temporal = nn.GRU(
                input_size=d_model, hidden_size=d_model,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False,
            )

        elif temporal_type == 'hybrid':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.gru = nn.GRU(
                input_size=d_model, hidden_size=d_model,
                num_layers=1, batch_first=True,
            )

        elif temporal_type == 'causal':
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation='gelu',
            )
            self.temporal = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            # Register causal mask buffer
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1),
            )

        else:
            raise ValueError(f'Unknown temporal_type: {temporal_type}')

    def _fuse_features(self, face_feat, eye_feat):
        """Fuse face and eye features according to fusion_type."""
        face = self.face_proj(face_feat)  # (B, T, d_model)
        eye = self.eye_proj(eye_feat)     # (B, T, d_model)

        if self.fusion_type == 'add':
            return face + eye  # (B, T, d_model)

        elif self.fusion_type == 'separate':
            # Interleave: [face_1, eye_1, face_2, eye_2, ...]
            B, T, D = face.shape
            tokens = torch.stack([face, eye], dim=2)  # (B, T, 2, D)
            return tokens.reshape(B, T * 2, D)  # (B, 2T, D)

        elif self.fusion_type == 'cross_attn':
            # face queries eye features via cross-attention
            attn_out, _ = self.cross_attn(face, eye, eye)
            return self.cross_norm(face + attn_out)  # (B, T, d_model)

        elif self.fusion_type == 'bidir_cross_attn':
            # face attends to eye, eye attends to face — DHECA-style
            face_attn, _ = self.face_attends_eye(face, eye, eye)
            eye_attn, _ = self.eye_attends_face(eye, face, face)
            face_ref = self.face_norm(face + face_attn)
            eye_ref = self.eye_norm(eye + eye_attn)
            fused = torch.cat([face_ref, eye_ref], dim=-1)  # (B, T, 2d)
            return self.bidir_proj(fused)  # (B, T, d_model)

        elif self.fusion_type == 'concat':
            combined = torch.cat([face, eye], dim=-1)  # (B, T, d_model*2)
            return self.concat_proj(combined)  # (B, T, d_model)

        elif self.fusion_type == 'self_attn_concat':
            # Refine each modality's temporal sequence with its own self-attention,
            # then channel-concat and project back to d_model.
            face_sa, _ = self.face_self_attn(face, face, face)
            eye_sa,  _ = self.eye_self_attn (eye,  eye,  eye)
            face_ref = self.face_self_norm(face + face_sa)   # (B, T, d_model)
            eye_ref  = self.eye_self_norm (eye  + eye_sa)    # (B, T, d_model)
            fused    = torch.cat([face_ref, eye_ref], dim=-1)  # (B, T, 2*d_model)
            return self.self_attn_concat_proj(fused)         # (B, T, d_model)

    def forward(self, face_feat, eye_feat, prev_feat=None, camera_id=None):
        """
        face_feat: (B, T, face_dim)
        eye_feat:  (B, T, eye_dim)
        prev_feat: (B, N_prev, prev_dim) or None
        camera_id: (B,) long, or None — required if use_camera_emb=True
        returns:   (B, d_model)
        """
        visual_tokens = self._fuse_features(face_feat, eye_feat)

        # Camera-conditional embedding broadcast across visual tokens
        if self.use_camera_emb:
            if camera_id is None:
                raise ValueError('camera_id required when use_camera_emb=True')
            cam_emb = self.camera_embedding(camera_id).unsqueeze(1)  # (B, 1, d_model)
            visual_tokens = visual_tokens + cam_emb

        if self.use_prev and prev_feat is not None:
            prev_tokens = self.prev_proj(prev_feat)  # (B, N_prev, d_model)
            tokens = torch.cat([prev_tokens, visual_tokens], dim=1)
        else:
            tokens = visual_tokens

        tokens = tokens + self.pos_embed[:, :tokens.size(1)]

        if self.temporal_type == 'transformer':
            out = self.temporal(tokens)
            return out[:, -1, :]  # last token

        elif self.temporal_type == 'gru':
            out, _ = self.temporal(tokens)
            return out[:, -1, :]  # last hidden

        elif self.temporal_type == 'hybrid':
            out = self.transformer(tokens)
            out, _ = self.gru(out)
            return out[:, -1, :]

        elif self.temporal_type == 'causal':
            # Self-attention with causal mask
            seq_len = tokens.size(1)
            mask = self.causal_mask[:seq_len, :seq_len]
            # Use decoder with memory=tokens (self-attention via cross-attention)
            out = self.temporal(tokens, tokens, tgt_mask=mask)
            return out[:, -1, :]
