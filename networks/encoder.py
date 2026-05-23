# -*- coding: utf-8 -*-
"""SGNet encoder with dual-backbone design.

Backbones:
  - LocalBackbone: Stage 0/1 (high-res), pure conv + local Q update.
  - GlobalBackbone: Stage 2/3 (low-res), SemanticConditionedSSM + MLP + CrossModulation.
"""

import torch
import torch.nn as nn
from .blocks.local_block import LocalSGBlock
from .blocks.global_block import GlobalSGBlock
from .blocks.base import StageTransition
from .q_tensor import QTensorTools


class LocalBackbone(nn.Module):
    """Backbone for high-resolution stages (0/1)."""
    def __init__(self, dims, depths, drop_path_rates, Q_max_norm=1.0):
        super().__init__()
        self.dims = list(dims)
        self.depths = list(depths)
        assert len(self.dims) == 3, "LocalBackbone expects 3 dims: [stage0, stage1, stage2_input]"
        assert len(self.depths) == 2, "LocalBackbone expects exactly 2 stages"
        
        self.stage0_blocks = nn.ModuleList()
        self.stage1_blocks = nn.ModuleList()
        
        idx = 0
        for _ in range(self.depths[0]):
            self.stage0_blocks.append(LocalSGBlock(
                self.dims[0], drop_path=drop_path_rates[idx], Q_max_norm=Q_max_norm
            ))
            idx += 1
        for _ in range(self.depths[1]):
            self.stage1_blocks.append(LocalSGBlock(
                self.dims[1], drop_path=drop_path_rates[idx], Q_max_norm=Q_max_norm
            ))
            idx += 1
            
        self.transitions = nn.ModuleList([
            StageTransition(self.dims[i], self.dims[i + 1], Q_max_norm=Q_max_norm)
            for i in range(len(self.dims) - 1)
        ])

    def forward(self, x, Q):
        stage_features, stage_Q, auxs = [], [], []
        
        # Stage 0
        stage_aux = None
        for block in self.stage0_blocks:
            x, Q, stage_aux = block(x, Q)
        stage_features.append(x)
        stage_Q.append(Q)
        auxs.append(stage_aux)
        x, Q = self.transitions[0](x, Q)
        
        # Stage 1
        stage_aux = None
        for block in self.stage1_blocks:
            x, Q, stage_aux = block(x, Q)
        stage_features.append(x)
        stage_Q.append(Q)
        auxs.append(stage_aux)
        
        # Stage 1 -> Stage 2 transition
        x, Q = self.transitions[1](x, Q)
        
        return x, Q, stage_features, stage_Q, auxs


class GlobalBackbone(nn.Module):
    """Backbone for low-resolution stages (2/3)."""
    def __init__(self, dims, depths, d_state=1, ssm_ratio=1.0,
                 drop_path_rates=None, Q_max_norm=1.0,
                 axis_temperature=0.35, ssm_conv=3,
                 forward_type="v05_noz"):
        super().__init__()
        self.dims = list(dims)
        self.depths = list(depths)
        assert len(self.dims) == 2, "GlobalBackbone expects 2 dims: [stage2, stage3]"
        assert len(self.depths) == 2, "GlobalBackbone expects exactly 2 stages"
        
        self.stage2_blocks = nn.ModuleList()
        self.stage3_blocks = nn.ModuleList()
        
        idx = 0
        for _ in range(self.depths[0]):
            self.stage2_blocks.append(GlobalSGBlock(
                self.dims[0],
                d_state=d_state, ssm_ratio=ssm_ratio, drop_path=drop_path_rates[idx],
                Q_max_norm=Q_max_norm, axis_temperature=axis_temperature,
                ssm_conv=ssm_conv, forward_type=forward_type,
            ))
            idx += 1
        for _ in range(self.depths[1]):
            self.stage3_blocks.append(GlobalSGBlock(
                self.dims[1],
                d_state=d_state, ssm_ratio=ssm_ratio, drop_path=drop_path_rates[idx],
                Q_max_norm=Q_max_norm, axis_temperature=axis_temperature,
                ssm_conv=ssm_conv, forward_type=forward_type,
            ))
            idx += 1
            
        self.transitions = nn.ModuleList([
            StageTransition(self.dims[i], self.dims[i + 1], Q_max_norm=Q_max_norm)
            for i in range(len(self.dims) - 1)
        ])

    def forward(self, x, Q):
        stage_features, stage_Q, auxs = [], [], []
        
        # Stage 2
        stage_aux = None
        for block in self.stage2_blocks:
            x, Q, stage_aux = block(x, Q)
        stage_features.append(x)
        stage_Q.append(Q)
        auxs.append(stage_aux)
        x, Q = self.transitions[0](x, Q)
        
        # Stage 3
        stage_aux = None
        for block in self.stage3_blocks:
            x, Q, stage_aux = block(x, Q)
        stage_features.append(x)
        stage_Q.append(Q)
        auxs.append(stage_aux)
        
        return x, Q, stage_features, stage_Q, auxs


class SGNetEncoder(nn.Module):
    def __init__(self, in_chans=8, embed_dim=96, depths=(2, 2, 8, 2),
                 d_state=1, ssm_ratio=1.0, drop_path_rate=0.2,
                 ssm_conv=3, forward_type="v05_noz",
                 Q_max_norm=1.0, Q_axis_temperature=0.35, **kwargs):
        super().__init__()
        self.num_stages = len(depths)
        self.depths = list(depths)
        mid_ch = embed_dim // 2
        self.stem_conv1 = nn.Sequential(
            nn.Conv2d(in_chans + 5, mid_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, mid_ch), nn.GELU(),
        )
        self.stem_conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, embed_dim), nn.GELU(),
        )
        self.dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        total_blocks = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        n_early = sum(self.depths[:2])
        dpr_early = dpr[:n_early]
        dpr_deep = dpr[n_early:]

        self.local_backbone = LocalBackbone(
            dims=self.dims[:3], depths=self.depths[:2],
            drop_path_rates=dpr_early, Q_max_norm=Q_max_norm,
        )
        self.global_backbone = GlobalBackbone(
            dims=self.dims[2:], depths=self.depths[2:],
            d_state=d_state, ssm_ratio=ssm_ratio, drop_path_rates=dpr_deep,
            Q_max_norm=Q_max_norm, axis_temperature=Q_axis_temperature,
            ssm_conv=ssm_conv, forward_type=forward_type,
        )

    def forward(self, x, nematic_prior):
        h = self.stem_conv1(torch.cat([x, nematic_prior], dim=1))
        h = self.stem_conv2(h)
        Q = QTensorTools.downsample_q(nematic_prior[:, 0:1], nematic_prior[:, 1:2], size=h.shape[2:])
        Q_stem = Q

        h, Q, stage_features_early, stage_Q_early, stage_aux_early = self.local_backbone(h, Q)
        h, Q, stage_features_deep, stage_Q_deep, stage_aux_deep = self.global_backbone(h, Q)

        stage_features = stage_features_early + stage_features_deep
        stage_Q = stage_Q_early + stage_Q_deep
        stage_aux = stage_aux_early + stage_aux_deep

        return stage_features, stage_Q, stage_aux, Q_stem
