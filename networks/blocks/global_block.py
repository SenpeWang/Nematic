# -*- coding: utf-8 -*-
"""Global-stage blocks: SS2D semantic-nematic interaction."""

import torch
import torch.nn as nn
from ..q_tensor import QTensorTools
from ..utils import SpatialContextGate
from .base import MLPBlock
from .semantic_ssm import SemanticConditionedSSM
from .local_block import ConvQUpdater


class QConsistencyGate(nn.Module):
    """Director-field consistency gate.

    Measures spatial Laplacian of Q (curvature of director field).
    High consistency → strong feature enhancement.
    Low consistency (defects/boundaries) → suppressed enhancement.
    Replaces CrossModulation's redundant cross-attn + s_delta.
    """
    def __init__(self, dim):
        super().__init__()
        # Laplacian kernel for measuring director-field discontinuity
        self.q_lap = nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False)
        with torch.no_grad():
            lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            self.q_lap.weight.copy_(lap.view(1, 1, 3, 3).repeat(2, 1, 1, 1))

        # Consistency gate: |Laplacian| small → gate ≈ 1; |Laplacian| large → gate ≈ 0
        self.consistency_gate = nn.Sequential(
            nn.Conv2d(2, 4, 1, bias=False), nn.GELU(),
            nn.Conv2d(4, 1, 1), nn.Sigmoid(),
        )

        # Feature enhancement (only applied where director is consistent)
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(16, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x, Q):
        # Q Laplacian: |∇²Q| measures director-field curvature
        Q_lap = self.q_lap(Q)

        # Gate: high consistency (small |Lap|) → gate → 1
        g = self.consistency_gate(torch.abs(Q_lap))

        # Residual enhancement, modulated by director consistency
        return x + g * self.enhance(x)


class GlobalSGBlock(nn.Module):
    """Full semantic-nematic interaction block for Stage 2/3 (low resolution)."""

    def __init__(self, dim, d_state=1, ssm_ratio=2.0, drop_path=0.0, mlp_ratio=4,
                 Q_max_norm=1.0, axis_temperature=0.35, ssm_conv=3,
                 forward_type="v05_noz"):
        super().__init__()
        self.Q_max_norm = float(Q_max_norm)
        self.ssm = SemanticConditionedSSM(
            dim, d_state=d_state, ssm_ratio=ssm_ratio,
            ssm_conv=ssm_conv, forward_type=forward_type,
            drop_path=drop_path,
        )
        self.mlp_block = MLPBlock(dim, mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.q_updater = ConvQUpdater(dim)
        self.q_gate = QConsistencyGate(dim)
        # Pre-Norm layers aligned with VMamba VSSBlock style
        self.norm_mlp = nn.GroupNorm(16, dim)
        self.norm_gate = nn.GroupNorm(16, dim)

    def forward(self, x, Q):
        # ① SSM global scan + nematic recomposition
        x, ssm_auxiliary = self.ssm(x, Q)

        # ② MLP with Pre-Norm
        x = self.mlp_block(self.norm_mlp(x))

        # ③ Q update (structure-tensor grounded)
        Q, q_aux = self.q_updater(x, Q)

        # ④ Q-consistency gate: enhance features where director field is smooth
        x = x + self.q_gate(self.norm_gate(x), Q)

        s_out = QTensorTools.get_s(Q[:, 0:1], Q[:, 1:2])

        aux = {
            'S': s_out,
            **q_aux,
            **ssm_auxiliary,
        }
        return x, Q, aux
