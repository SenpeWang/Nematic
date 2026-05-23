# -*- coding: utf-8 -*-
"""Basic building blocks used across all stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..q_tensor import QTensorTools
from ..utils import DropPath


class MLPBlock(nn.Module):
    """Channel-mixing MLP with residual connection and DropPath."""
    def __init__(self, dim, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False), nn.GroupNorm(16, hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False), nn.GroupNorm(16, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.mlp(x))


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class StageTransition(nn.Module):
    """Feature + Q 同步下采样。"""
    def __init__(self, feat_in, feat_out, Q_max_norm=1.0):
        super().__init__()
        self.feat_down = DownsampleLayer(feat_in, feat_out)
        self.Q_max_norm = Q_max_norm

    def forward(self, feat, Q):
        feat = self.feat_down(feat)
        Q = QTensorTools.downsample_q(Q[:, 0:1], Q[:, 1:2], feat.shape[2:], self.Q_max_norm)
        return feat, Q
