# -*- coding: utf-8 -*-
"""Semantic SSM with nematic recomposition for global stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..q_tensor import QTensorTools
from ..utils import SpatialContextGate, DropPath, LayerNorm2d
from ..ss2d import SS2D


class NematicRecomposition(nn.Module):
    """几何显式向列重组：f_h/f_v 按 director 角度旋转到平行/垂直坐标系。

    projection_net 黑盒被替换为显式几何旋转 + 可学习各向异性强度。
    """
    def __init__(self):
        super().__init__()
        # 各向异性强度由 S 决定（有序度高 → 各向异性强）
        self.strength_fn = nn.Sequential(
            nn.Conv2d(1, 4, 1, bias=False), nn.GELU(),
            nn.Conv2d(4, 1, 1), nn.Sigmoid(),
        )
        self.contrast_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, f_h, f_v, Q_field):
        q1, q2 = Q_field[:, 0:1], Q_field[:, 1:2]
        theta = 0.5 * torch.atan2(q2, q1)
        cos2t = torch.cos(2 * theta)
        sin2t = torch.sin(2 * theta)
        S = torch.sqrt(q1.pow(2) + q2.pow(2) + 1e-8).clamp(0, 1)

        # 各向同性基底（保留 SS2D 方向扫描的平均信息）
        f_iso = 0.5 * (f_h + f_v)

        # 显式几何旋转：f_h/f_v 投影到 director 的平行/垂直方向
        # director 角度 θ，旋转矩阵 [cos2θ, sin2θ; -sin2θ, cos2θ] 作用于方向流
        f_parallel = f_h * cos2t + f_v * sin2t
        f_orthogonal = -f_h * sin2t + f_v * cos2t

        # 向列对比度：平行 vs 垂直的差异
        f_contrast = f_parallel - f_orthogonal

        # S 加权的向列增强
        strength = self.strength_fn(S)
        flow_nematic = f_iso + strength * 0.5 * f_contrast

        auxiliary = {
            'nematic_strength': strength,
            'contrast_scale': self.contrast_scale,
        }
        return flow_nematic, f_contrast, auxiliary


class SemanticConditionedSSM(nn.Module):
    def __init__(self, dim, d_state=1, ssm_ratio=2.0,
                 ssm_conv=3, forward_type="v05_noz", drop_path=0.0):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.ss2d = SS2D(
            d_model=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            d_conv=ssm_conv,
            conv_bias=False,
            forward_type=forward_type,
            channel_first=True,
        )
        self.s_gate = SpatialContextGate(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )
        # Zero-init proj so residual branch starts near zero
        nn.init.zeros_(self.proj[0].weight)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.nematic_recomposition = NematicRecomposition()

    def forward(self, x, Q):
        b, c, h, w = x.shape
        s = QTensorTools.get_s(Q[:, 0:1], Q[:, 1:2])
        x_norm = self.norm(x)

        # SS2D 4-direction scan; expose f_h / f_v for nematic recomposition
        y, f_h, f_v = self.ss2d(x_norm, return_directions=True)

        flow_nematic, f_contrast, recomposition_auxiliary = self.nematic_recomposition(f_h, f_v, Q)

        g = self.s_gate(x, s)
        out = flow_nematic + g * f_contrast * self.nematic_recomposition.contrast_scale

        ssm_auxiliary = {'ssm_gate': g, **recomposition_auxiliary}
        return x + self.drop_path(self.proj(out + y)), ssm_auxiliary
