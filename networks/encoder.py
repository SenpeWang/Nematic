# -*- coding: utf-8 -*-
"""
N²-Mamba 流形编码器
4 Stage 金字塔 + Block AttnRes (stage = block)

每个 stage = 一个 block:
  块内: 标准残差累加 → partial_block
  块间: softmax 注意力聚合 completed_blocks
  blocks 初始包含 token embedding (b_0)
  d_embed 共享嵌入空间处理跨 stage 通道/分辨率差异。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .block import NematicInteractionBlock, BlockAttnRes, DownsampleLayer


class NematicMambaEncoder(nn.Module):
    """
    流形编码器 (Block AttnRes)。

    PatchEmbed(stride-4) → 4 个 Stage, 每个 Stage 内多个 NIB
    每个 Stage = 一个 Block:
      块内: 标准残差累加 + 每层 NIB 前做 block_attn_res
      块间: softmax 注意力聚合 completed_blocks

    物理场通过 phys_pyramid 直接按 Stage 索引取用，无需 F.interpolate。
    """

    def __init__(self, in_chans=8, embed_dim=96, depths=(2, 2, 5, 2),
                 d_state=64, expand=2, drop_path_rate=0.2,
                 use_checkpoint=False, d_embed=64, mamba3_kwargs=None, **kwargs):
        super().__init__()
        self.num_stages = len(depths)
        self.depths = list(depths)
        self.use_checkpoint = use_checkpoint
        self.d_embed = d_embed
        self.mamba3_kwargs = dict(mamba3_kwargs or {})

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 4, 4),
            nn.GroupNorm(1, embed_dim),
        )

        self.dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        # 构建扁平的 Block 维度序列 (用于解码器兼容)
        self.all_block_dims = []
        for i, depth in enumerate(depths):
            self.all_block_dims.extend([self.dims[i]] * depth)
        total_blocks = len(self.all_block_dims)

        # DropPath 速率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.blocks = nn.ModuleList()
        idx = 0
        for i, depth in enumerate(depths):
            for j in range(depth):
                self.blocks.append(
                    NematicInteractionBlock(self.dims[i], d_state=d_state,
                                            expand=expand, drop_path=dpr[idx],
                                            mamba3_kwargs=self.mamba3_kwargs)
                )
                idx += 1

        # ── Block AttnRes: 每个 stage 一个 ──
        self.stage_attn_res = nn.ModuleList([
            BlockAttnRes(d_embed=d_embed) for _ in range(self.num_stages)
        ])

        # ── 跨 stage 投影: 各 stage 维度 ↔ d_embed ──
        self.embed_to_d = nn.Conv2d(embed_dim, d_embed, 1, bias=False)
        self.to_embed = nn.ModuleList([
            nn.Conv2d(self.dims[i], d_embed, 1, bias=False)
            for i in range(self.num_stages)
        ])
        self.from_embed = nn.ModuleList([
            nn.Conv2d(d_embed, self.dims[i], 1, bias=False)
            for i in range(self.num_stages)
        ])

        # Stage 边界处的下采样层
        self.downsamples = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.downsamples.append(DownsampleLayer(self.dims[i], self.dims[i + 1]))

    def forward(self, x, phys_pyramid=None):
        """
        Args:
            x: (B, in_chans, H, W)
            phys_pyramid: [(S0, Q1_0, Q2_0), ..., (S3, Q1_3, Q2_3)]
                          原生分辨率物理金字塔，按 Stage 索引取用
        Returns:
            feats: [f0, f1, f2, f3] 多尺度编码器特征
            completed_blocks: list of (B, d_embed, H_i, W_i) 块表示 (含 b_0)
        """
        x = self.patch_embed(x)
        feats = []

        # b_0 = token embedding, 始终作为第一个可检索源
        completed_blocks = [self.embed_to_d(x)]

        block_idx = 0
        for stage_i in range(self.num_stages):
            # 直接从金字塔取对应尺度的物理场 (无 F.interpolate)
            S_i, Q1_i, Q2_i = phys_pyramid[stage_i]

            # partial_block 初始化为当前 stage 输入的 embed 投影
            partial_block = self.to_embed[stage_i](x)

            for j in range(self.depths[stage_i]):
                # 块间注意力: 从所有历史块中选择性检索
                h_embed = self.stage_attn_res[stage_i](completed_blocks, partial_block, S=S_i, Q1=Q1_i, Q2=Q2_i)
                h = self.from_embed[stage_i](h_embed)   # 回到 stage 维度

                # NIB (NBE + GME + CFI)
                blk = self.blocks[block_idx]
                if self.use_checkpoint and h.requires_grad:
                    h = checkpoint(blk, h, S_i, Q1_i, Q2_i, use_reentrant=False)
                else:
                    h = blk(h, S=S_i, Q1=Q1_i, Q2=Q2_i)

                # 块内累加: NIB 输出投影到 d_embed, 累加到 partial_block
                layer_embed = self.to_embed[stage_i](h)
                partial_block = partial_block + layer_embed

                block_idx += 1

            # stage 结束 → 块表示存档
            completed_blocks.append(partial_block)

            # Stage 结束: 存多尺度特征 + 下采样
            feats.append(h)
            if stage_i < self.num_stages - 1:
                x = self.downsamples[stage_i](h)

        return feats, completed_blocks
