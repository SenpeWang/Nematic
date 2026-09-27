# -*- coding: utf-8 -*-
"""NFAM-Net encoder with unified 4-stage StageBackbone and stem."""

import torch
import torch.nn as nn
from .ofm import OFMBlock
from .layer import StageTransition, _gn
from .ofe import OFEBlock
from .npr import NPR
from .qtensor import QTensorTools

class StageBackbone(nn.Module):
    """N-stage backbone with block-level NPR."""
    def __init__(self, dims, depths, block_cls, drop_path_rates,
                 q_updaters=None, chunk_size=64, stats_logger=None):
        super().__init__()
        self.dims = list(dims)
        self.depths = list(depths)
        self.q_updaters = q_updaters

        # ---- blocks per stage ----
        self.stages = nn.ModuleList()
        idx = 0
        for stage_i in range(len(self.depths)):
            blocks = nn.ModuleList()
            cls = block_cls[stage_i]
            for b in range(self.depths[stage_i]):
                kwargs = dict(drop_path=drop_path_rates[idx])
                if cls is OFMBlock:
                    kwargs.update(chunk_size=chunk_size,
                                  stats_logger=stats_logger,
                                  name=f'stage{stage_i}')
                blocks.append(cls(self.dims[stage_i], **kwargs))
                idx += 1
            self.stages.append(blocks)

        # ---- inter-stage transitions ----
        self.transitions = nn.ModuleList([
            StageTransition(self.dims[i], self.dims[i + 1])
            for i in range(len(self.depths) - 1)
        ])

    def forward(self, x, Q):
        stage_features, stage_Q = [], []

        for stage_i, blocks in enumerate(self.stages):
            for block in blocks:
                if self.q_updaters is not None and self.q_updaters[stage_i] is not None:
                    Q, _ = self.q_updaters[stage_i](x, Q)
                x = block(x, Q)

            stage_features.append(x)
            stage_Q.append(Q)

            if stage_i < len(self.transitions):
                x, Q = self.transitions[stage_i](x, Q)

        return x, Q, stage_features, stage_Q


class Encoder(nn.Module):
    """NFAM-Net Encoder with explicit stage dimensions."""
    def __init__(self, in_chans, config, stats_logger=None):
        super().__init__()

        dims = [96, 192, 384, 768]
        depths = list(config.encoder_depths)
        self.dims = dims
        self.depths = depths

        # ---- stem ----
        mid_ch = dims[0] // 2
        self.stem = nn.Sequential(
            nn.GroupNorm(1, in_chans + 3), nn.GELU(),
            nn.Conv2d(in_chans + 3, mid_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn(mid_ch), mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, dims[0], 3, stride=2, padding=1, bias=False),
        )

        # ---- drop path rates ----
        dpr_all = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(depths))]

        # ---- Q updaters (one per stage) ----
        self.q_updaters = nn.ModuleList([
            NPR(dims[i], stats_logger=stats_logger, name=f'stage{i}')
            for i in range(len(dims))
        ])

        # ---- backbone ----
        self.backbone = StageBackbone(
            dims=dims, depths=depths,
            block_cls=[OFEBlock, OFEBlock, OFMBlock, OFMBlock],
            drop_path_rates=dpr_all,
            q_updaters=list(self.q_updaters),
            chunk_size=64,
            stats_logger=stats_logger,
        )

    def forward(self, x, Q_prior):
        h = self.stem(torch.cat([x, Q_prior], dim=1))
        Q = QTensorTools.downsample_Q(Q_prior[:, 0:1], Q_prior[:, 1:2],
                                       size=h.shape[2:])
        Q_stem = Q
        h, Q, stage_features, stage_Q = self.backbone(h, Q)
        return stage_features, stage_Q, Q_stem
