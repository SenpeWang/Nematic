# -*- coding: utf-8 -*-
"""Generic utility modules (DropPath, SpatialContextGate)."""

import torch
import torch.nn as nn


class SpatialContextGate(nn.Module):
    """Synergistic S-gate: feat and S co-determine the gate."""
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 4, 1, bias=False),
            nn.GroupNorm(8, feat_dim // 4), nn.GELU(),
            nn.Conv2d(feat_dim // 4, 1, 1),
        )
        self.s_proj = nn.Conv2d(1, 1, 1)
        self.residual = nn.Conv2d(feat_dim + 1, 1, 1)

    def forward(self, feat, S):
        r_feat = torch.sigmoid(self.feat_proj(feat))
        r_s = torch.sigmoid(self.s_proj(S))
        interaction = r_feat * r_s
        residual = self.residual(torch.cat([feat, S], dim=1))
        return torch.sigmoid(interaction + residual)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channel-first 2D feature maps.

    Permutes (B, C, H, W) → (B, H, W, C), applies LayerNorm,
    then permutes back. Matches VMamba v2's ln2d behavior.
    """
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()
