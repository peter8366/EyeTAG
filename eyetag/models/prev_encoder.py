"""
models/prev_encoder.py

Previous gaze label encoder — encodes past gaze predictions into tokens.

Supports multiple encoding modes:
  'mlp'       : Flatten → MLP → single token  (EyeTag v3 default)
  'dilated'   : Dilated 1D Conv → single token (EyeTag v2 style)
  'transformer': 1D Transformer → CLS token    (new)
  'linear'    : Per-step Linear → T-1 tokens   (ARGaze style)

Input:  (B, T-1, 2)  — prev gaze (pitch, yaw) per frame
Output: (B, N, D)    — N=1 for mlp/dilated/transformer, N=T-1 for linear
"""

import torch
import torch.nn as nn


class PrevMLP(nn.Module):
    """Flatten → MLP → single token."""

    def __init__(self, num_prev, gaze_dim=2, hidden_dim=64, out_dim=64):
        super().__init__()
        in_dim = num_prev * gaze_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim
        self.num_tokens = 1

    def forward(self, x):
        """x: (B, T-1, 2) → (B, 1, out_dim)"""
        x = x.flatten(1)
        return self.mlp(x).unsqueeze(1)


class PrevDilatedConv(nn.Module):
    """Dilated 1D Conv → single token. (EyeTag v2 style)"""

    def __init__(self, num_prev, gaze_dim=2, hidden_dim=64, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(gaze_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim
        self.num_tokens = 1

    def forward(self, x):
        """x: (B, T-1, 2) → (B, 1, out_dim)"""
        x = x.permute(0, 2, 1)  # (B, 2, T-1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        return self.proj(x).unsqueeze(1)


class PrevTransformer(nn.Module):
    """1D Transformer Encoder → CLS token."""

    def __init__(self, num_prev, gaze_dim=2, d_model=64, nhead=4,
                 num_layers=1, out_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(gaze_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_prev + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.out_dim = out_dim
        self.num_tokens = 1

    def forward(self, x):
        """x: (B, T-1, 2) → (B, 1, out_dim)"""
        B = x.shape[0]
        x = self.input_proj(x)  # (B, T-1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, x], dim=1)  # (B, T, d_model)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        out = self.encoder(tokens)
        return self.out_proj(out[:, 0:1, :])  # CLS token → (B, 1, out_dim)


class PrevLinearTokens(nn.Module):
    """Per-step Linear projection → T-1 separate tokens. (ARGaze style)"""

    def __init__(self, num_prev, gaze_dim=2, out_dim=64):
        super().__init__()
        self.proj = nn.Linear(gaze_dim, out_dim)
        self.out_dim = out_dim
        self.num_tokens = num_prev  # T-1 tokens

    def forward(self, x):
        """x: (B, T-1, 2) → (B, T-1, out_dim)"""
        return self.proj(x)


class TCNBlock(nn.Module):
    """Single TCN block with residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding (left only)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.trim = padding  # trim right side for causal

        # Residual projection if channel mismatch
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """x: (B, C, L) → (B, out_ch, L)"""
        res = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        if self.trim > 0:
            out = out[:, :, :-self.trim]  # causal trim
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        if self.trim > 0:
            out = out[:, :, :-self.trim]
        out = self.dropout(out)
        return self.relu(out + res)


class PrevTCN(nn.Module):
    """TCN (Temporal Convolutional Network) for prev gaze encoding."""

    def __init__(self, num_prev, gaze_dim=2, hidden_dim=32, out_dim=64, dropout=0.1):
        super().__init__()
        self.block1 = TCNBlock(gaze_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = TCNBlock(hidden_dim, out_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_dim
        self.num_tokens = 1

    def forward(self, x):
        """x: (B, T-1, 2) → (B, 1, out_dim)"""
        x = x.permute(0, 2, 1)   # (B, 2, T-1)
        x = self.block1(x)        # (B, 32, T-1)
        x = self.block2(x)        # (B, 64, T-1)
        x = self.pool(x)          # (B, 64, 1)
        return x.permute(0, 2, 1)  # (B, 1, 64)


class PrevLabelEncoder(nn.Module):
    """Unified prev label encoder wrapper."""

    def __init__(self, num_prev, gaze_dim=2, mode='mlp', hidden_dim=64, out_dim=64):
        super().__init__()
        self.mode = mode

        if mode == 'mlp':
            self.encoder = PrevMLP(num_prev, gaze_dim, hidden_dim, out_dim)
        elif mode == 'dilated':
            self.encoder = PrevDilatedConv(num_prev, gaze_dim, hidden_dim, out_dim)
        elif mode == 'transformer':
            self.encoder = PrevTransformer(num_prev, gaze_dim, hidden_dim, 4, 1, out_dim)
        elif mode == 'linear':
            self.encoder = PrevLinearTokens(num_prev, gaze_dim, out_dim)
        elif mode == 'tcn':
            self.encoder = PrevTCN(num_prev, gaze_dim, hidden_dim, out_dim)
        else:
            raise ValueError(f'Unknown prev encoder mode: {mode}')

        self.out_dim = self.encoder.out_dim
        self.num_tokens = self.encoder.num_tokens

    def forward(self, x):
        """x: (B, T-1, 2) → (B, num_tokens, out_dim)"""
        return self.encoder(x)
