# -*- coding: utf-8 -*-
"""
N²-Mamba (Nematic-NeuroMamba) 顶层模型
PhysicsPriorStem (多尺度金字塔) + 流形编码器 + 测地线解码器 + 双头输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NematicMambaEncoder
from .decoder import NematicMambaDecoder


# ============================================================================
# 物理同构变换: raw (q1, q2) → (S, Q1, Q2)
# ============================================================================

def physics_isomorphism(raw):
    """
    将网络预测的 2 维原始向量变换为物理场 (S, Q1, Q2)。
    强制物理约束: S ≡ ||Q|| 且 S ∈ [0,1].
    """
    q1, q2 = raw[:, 0:1], raw[:, 1:2]
    r = torch.sqrt(q1.pow(2) + q2.pow(2) + 1e-8)
    S = torch.tanh(r)
    Q1 = S * q1 / (r + 1e-8)
    Q2 = S * q2 / (r + 1e-8)
    return S, Q1, Q2


# ============================================================================
# PhysicsPriorStem — 多尺度物理先验金字塔
# ============================================================================

class PhysicsPriorStem(nn.Module):
    """
    多尺度物理先验金字塔。

    通过与 Encoder 感受野严格对齐的 stride=2 下采样卷积,
    在 4 个原生分辨率尺度上预测物理场 (S, Q1, Q2)。
    彻底消灭 Encoder/Decoder 中的 F.interpolate 偏移。

    输出:
      feat:         (B, stem_channels, H/1, W/1)  先验干特征
      phys_pyramid: [(S0, Q1_0, Q2_0), ..., (S3, Q1_3, Q2_3)]
                    分辨率依次递减 (1/4, 1/8, 1/16, 1/32 of input)
                    注意: Encoder patch_embed stride=4, 故 Level-0 = H/4
    """

    def __init__(self, in_channels, stem_channels=64, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # 基础特征提取 (全分辨率)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(stem_channels, affine=True),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(stem_channels, affine=True),
            nn.GELU(),
        )

        # 全分辨率物理头 (用于蒸馏 Loss)
        self.head_full = nn.Conv2d(stem_channels, 2, 1, bias=True)

        # 多尺度下采样路径 + 每级物理头
        self.downsamples = nn.ModuleList()
        self.heads = nn.ModuleList()
        ch = stem_channels
        for i in range(num_levels):
            if i == 0:
                # Level-0: stride=4 匹配 Encoder patch_embed
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(ch, ch, 4, stride=4, padding=0, bias=False),
                    nn.GroupNorm(1, ch),
                    nn.GELU(),
                ))
            else:
                # Level-1,2,3: stride=2 匹配 Encoder 各级下采样
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(1, ch),
                    nn.GELU(),
                ))
            self.heads.append(nn.Conv2d(ch, 2, 1, bias=True))

    def forward(self, x):
        feat = self.conv(x)

        # 全分辨率物理场 (用于蒸馏 Loss 对齐)
        raw_full = self.head_full(feat)
        S_full, Q1_full, Q2_full = physics_isomorphism(raw_full)

        # 多尺度金字塔
        phys_pyramid = []
        h = feat
        for i in range(self.num_levels):
            h = self.downsamples[i](h)
            raw_i = self.heads[i](h)
            S_i, Q1_i, Q2_i = physics_isomorphism(raw_i)
            phys_pyramid.append((S_i, Q1_i, Q2_i))

        return feat, (S_full, Q1_full, Q2_full), phys_pyramid


# ============================================================================
# PosteriorPhysHead — 后验物理精炼头
# ============================================================================

class PosteriorPhysHead(nn.Module):
    """
    后验物理头: 从解码器特征精炼出最终 Q-Tensor 场。
    同样强制物理同构。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(1, in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 2, 1, bias=True),
        )

    def forward(self, feat):
        raw = self.proj(feat)
        return physics_isomorphism(raw)


# ============================================================================
# NematicNeuroMamba — 顶层模型
# ============================================================================

class NematicNeuroMamba(nn.Module):
    """
    N²-Mamba (Nematic-NeuroMamba).

    四大创新:
      1. NIB 双视角交互 (NBE + GME + CFI)
      2. GME 交叉扫描 + 物理后置门控 (Q1 方向路由 + S 显著性掩码)
      3. BlockAttnRes (Kimi Block Attention Residuals)
      4. 流形几何正则化 (Frank + Flow + Order)

    双预测头:
      - 分割头: (B, num_classes, H, W)
      - 后验物理头: 精炼 S, Q₁₁, Q₁₂
    """

    def __init__(self, input_channels=8, num_classes=2,
                 embed_dim=96,
                 encoder_depths=(2, 2, 6, 2),
                 decoder_depths=(2, 6, 2),
                 d_state=64, expand=2,
                 drop_path_rate=0.2, fpn_dim=256,
                 use_checkpoint=False, d_embed=64, mamba3_kwargs=None, **kwargs):
        super().__init__()

        self.input_norm = nn.GroupNorm(1, input_channels)
        self.mamba3_kwargs = dict(mamba3_kwargs or {})

        # 物理先验金字塔干
        stem_channels = max(embed_dim // 2, 32)
        num_enc_stages = len(encoder_depths)
        self.physics_stem = PhysicsPriorStem(
            input_channels, stem_channels=stem_channels,
            num_levels=num_enc_stages,
        )

        # 流形编码器
        self.encoder = NematicMambaEncoder(
            in_chans=input_channels + stem_channels,
            embed_dim=embed_dim,
            depths=encoder_depths,
            d_state=d_state,
            expand=expand,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            d_embed=d_embed,
            mamba3_kwargs=self.mamba3_kwargs,
        )

        # 测地线解码器 (全局 AttnRes 路由)
        self.decoder = NematicMambaDecoder(
            encoder_dims=self.encoder.dims,
            encoder_block_dims=self.encoder.all_block_dims,
            decoder_depths=decoder_depths,
            d_embed=d_embed,
        )

        # 上采样融合 (stride-4 → 原始分辨率)
        dec_out_ch = self.decoder.out_channels
        self.final_up = nn.Sequential(
            nn.Conv2d(dec_out_ch + input_channels, fpn_dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(1, fpn_dim // 2),
            nn.GELU(),
            nn.Conv2d(fpn_dim // 2, fpn_dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(1, fpn_dim // 4),
            nn.GELU(),
        )

        # 双预测头
        self.seg_head = nn.Conv2d(fpn_dim // 4, num_classes, 1)
        self.phys_head = PosteriorPhysHead(fpn_dim // 4)

    def forward(self, x):
        x_c = self.input_norm(x)
        B, C, H, W = x_c.shape

        # 物理先验金字塔预测
        stem_feat, (S_prior, Q1_prior, Q2_prior), phys_pyramid = self.physics_stem(x_c)

        # 编码器 (原始图像 + 先验干特征拼接, 物理金字塔对齐)
        enc_input = torch.cat([x_c, stem_feat], dim=1)
        feats, completed_blocks = self.encoder(enc_input, phys_pyramid=phys_pyramid)

        # 解码器 (物理金字塔 + completed_blocks)
        decoded = self.decoder(feats, completed_blocks, phys_pyramid=phys_pyramid)

        # 上采样 + 融合原始输入
        decoded_up = F.interpolate(decoded, (H, W), mode='bilinear', align_corners=False)
        fused = self.final_up(torch.cat([decoded_up, x_c], dim=1))

        # 分割头
        logits = self.seg_head(fused)

        # 后验物理头
        S_post, Q1_post, Q2_post = self.phys_head(fused)

        return {
            'logits': logits,
            'S': S_post,
            'Q11': Q1_post,
            'Q12': Q2_post,
            'S_prior': S_prior,
            'Q1_prior': Q1_prior,
            'Q2_prior': Q2_prior,
        }


# ============================================================================
# 工厂函数
# ============================================================================

def build_nematic_mamba(config):
    return NematicNeuroMamba(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        encoder_depths=config.encoder_depths,
        decoder_depths=config.decoder_depths,
        d_state=config.d_state,
        expand=config.expand,
        drop_path_rate=config.drop_path_rate,
        fpn_dim=config.fpn_dim,
        use_checkpoint=config.use_checkpoint,
        d_embed=config.d_embed,
        mamba3_kwargs=config.get_mamba3_kwargs(),
    )
