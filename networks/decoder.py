# -*- coding: utf-8 -*-
"""
N²-Mamba 测地线解码器
4 Stage 对称架构: 转置卷积上采样 → Concat 跳连 → 注意力残差块 (BlockAttnRes)

解码器继续使用编码器的 completed_blocks:
  每个 decoder stage = 一个 block
  块内: 注意力残差检索 → 基础卷积堆叠
  块间: softmax 注意力聚合 (编码器 blocks + 解码器已完成 blocks)

物理场通过 phys_pyramid 按 Stage 索引直接取用，无 F.interpolate。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import BlockAttnRes, UpsampleLayer


class BasicConvBlock(nn.Module):
    """
    常规的基础残差卷积块。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, in_channels)
        self.act2 = nn.GELU()

    def forward(self, x, **kwargs):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return x + h


class NematicDecoderStage(nn.Module):
    """
    单个解码器 Stage 标准范式：
    1. Transpose Conv 上采样 (UpsampleLayer)
    2. Concat 跳跃连接 (torch.cat + 1×1 融合)
    3. 注意力残差块 (BlockAttnRes + BasicConvBlock 堆叠)

    物理场直接接收已对齐的 (S_i, Q1_i, Q2_i)，无需内部 interpolate。
    """
    def __init__(self, in_ch, skip_ch, d_embed, n_blocks, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.d_embed = d_embed

        # Step 1: 转置卷积上采样
        if upsample:
            self.upsample_layer = UpsampleLayer(in_ch, skip_ch)
            # Step 2: Concat 融合 (skip_ch*2 → skip_ch)
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(skip_ch * 2, skip_ch, 1, bias=False),
                nn.GroupNorm(1, skip_ch),
                nn.GELU(),
            )
        else:
            self.upsample_layer = None
            self.skip_fuse = None

        # Step 3: 注意力残差 + 卷积堆叠
        self.stage_attn_res = BlockAttnRes(d_embed=d_embed)
        self.to_embed = nn.Conv2d(skip_ch, d_embed, 1, bias=False)
        self.from_embed = nn.Conv2d(d_embed, skip_ch, 1, bias=False)

        self.blocks = nn.ModuleList([
            BasicConvBlock(skip_ch) for _ in range(n_blocks)
        ])

    def forward(self, h, skip, completed_blocks, S_i, Q1_i, Q2_i):
        """
        数据流:
          h  ──[TransposeConv]──→ h_up
          h_up ──[cat(h_up, skip)]──→ fused
          fused ──[BlockAttnRes → BasicConvBlock × N]──→ output
        """
        # Step 1: 转置卷积上采样
        if self.upsample and skip is not None:
            h = self.upsample_layer(h, target_size=skip.shape[2:])
            # Step 2: Concat 跳跃连接
            h = self.skip_fuse(torch.cat([h, skip], dim=1))

        # Step 3: 注意力残差块 (BlockAttnRes 驱动的 Res-block)
        partial_block = self.to_embed(h)
        for blk in self.blocks:
            # 注意力检索: 从所有历史 completed_blocks 中提取信息
            h_embed = self.stage_attn_res(completed_blocks, partial_block, S=S_i, Q1=Q1_i, Q2=Q2_i)
            h_in = self.from_embed(h_embed)
            # 卷积残差精炼
            h = blk(h_in)
            # 累加到 partial_block
            layer_embed = self.to_embed(h)
            partial_block = partial_block + layer_embed

        return h, partial_block


class NematicMambaDecoder(nn.Module):
    """
    高度解耦的测地线解码器。
    各个层级由 NematicDecoderStage 构建。
    标准范式: TransposeConv ↑ → Concat Skip → Attention Res-block
    物理场通过 phys_pyramid 按 Stage 索引直接取用。
    """

    def __init__(self, encoder_dims, encoder_block_dims,
                 decoder_depths=(2, 2, 5, 2),
                 d_state=64, expand=2, d_embed=64, **kwargs):
        super().__init__()
        self.num_levels = len(encoder_dims) - 1   # 3 个上采样 level
        self.d_embed = d_embed
        self.stages = nn.ModuleList()

        # ── Bottleneck Stage (最深层) ──
        bottleneck_ch = encoder_dims[-1]
        n_bot = decoder_depths[0] if len(decoder_depths) > 0 else 2
        self.stages.append(NematicDecoderStage(
            in_ch=bottleneck_ch, skip_ch=bottleneck_ch, d_embed=d_embed,
            n_blocks=n_bot, upsample=False
        ))

        # ── 3 个上采样 Level ──
        for i in range(self.num_levels):
            deep_ch = encoder_dims[-(i + 1)]
            skip_ch = encoder_dims[-(i + 2)]
            n_blocks = decoder_depths[i + 1] if (i + 1) < len(decoder_depths) else 2
            self.stages.append(NematicDecoderStage(
                in_ch=deep_ch, skip_ch=skip_ch, d_embed=d_embed,
                n_blocks=n_blocks, upsample=True
            ))

        self.out_channels = encoder_dims[0]

    def forward(self, encoder_feats, completed_blocks, phys_pyramid=None):
        """
        Args:
            encoder_feats: [f0, f1, f2, f3] 从浅到深
            completed_blocks: 编码器的 completed_blocks (含 b_0)
            phys_pyramid: [(S0, Q1_0, Q2_0), ..., (S3, Q1_3, Q2_3)]
        """
        h = encoder_feats[-1]

        # Bottleneck 使用最深层物理场 phys_pyramid[-1]
        S_b, Q1_b, Q2_b = phys_pyramid[-1]
        h, partial_block = self.stages[0](
            h, skip=None, completed_blocks=completed_blocks,
            S_i=S_b, Q1_i=Q1_b, Q2_i=Q2_b
        )
        completed_blocks = completed_blocks + [partial_block]

        # 3 个上采样 Level: 使用对应尺度的物理场
        for i, stage in enumerate(self.stages[1:]):
            skip = encoder_feats[-(i + 2)]
            S_i, Q1_i, Q2_i = phys_pyramid[-(i + 2)]
            h, partial_block = stage(
                h, skip=skip, completed_blocks=completed_blocks,
                S_i=S_i, Q1_i=Q1_i, Q2_i=Q2_i
            )
            completed_blocks = completed_blocks + [partial_block]

        return h
