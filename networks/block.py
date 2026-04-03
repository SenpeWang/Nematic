# -*- coding: utf-8 -*-
"""
N²-Mamba 核心构建块
NBE (NematicBoundaryExtractor) / GME (GeodesicMambaEvolver)
CFI (CrossFeatureInteraction) / BlockAttnRes (Kimi Block Attention Residuals)
NIB (NematicInteractionBlock)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba3 import Mamba3


# ============================================================================
# DropPath
# ============================================================================

class DropPath(nn.Module):
    """随机深度 (Stochastic Depth)"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()


# ============================================================================
# NematicBoundaryExtractor (NBE) — 各向同性多尺度局部边界探测器 (局部流)
# ============================================================================

class NematicBoundaryExtractor(nn.Module):
    """
    Nematic Boundary Extractor (NBE).

    各向同性多尺度局部边界探测器:
      1. PreConv: 边界增强变换
      2. 通道压缩: Max+Avg → 2 通道薄特征图
      3. 多尺度特征提取 (并行空洞 Conv3×3, d=1/3/5)
      4. 抗网格化融合: Concat → Conv3×3 → Sigmoid
      5. 前景感知通道注意力: spatial_map 加权池化 → SE
    """

    def __init__(self, dim):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, dim),
        )

        # 多尺度并行空洞卷积
        self.spatial_d1 = nn.Conv2d(2, 1, 3, padding=1, dilation=1, bias=False)
        self.spatial_d3 = nn.Conv2d(2, 1, 3, padding=3, dilation=3, bias=False)
        self.spatial_d5 = nn.Conv2d(2, 1, 3, padding=5, dilation=5, bias=False)

        # 抗网格化融合
        self.fuse = nn.Conv2d(3, 1, 3, padding=1, bias=False)

        # 前景感知通道注意力
        r = max(dim // 4, 8)
        self.fc1 = nn.Linear(dim, r, bias=False)
        self.fc2 = nn.Linear(r, dim, bias=False)

    def forward(self, x):
        feat = self.pre_conv(x)

        # 空间压缩
        s_cat = torch.cat([
            feat.max(dim=1, keepdim=True)[0],
            feat.mean(dim=1, keepdim=True),
        ], dim=1)

        # 多尺度 + 抗网格化融合
        spatial_map = torch.sigmoid(self.fuse(torch.cat([
            self.spatial_d1(s_cat),
            self.spatial_d3(s_cat),
            self.spatial_d5(s_cat),
        ], dim=1)))

        # 前景感知通道注意力
        masked = feat * spatial_map
        desc = masked.sum(dim=(2, 3)) / (spatial_map.sum(dim=(2, 3)) + 1e-6)
        channel_map = torch.sigmoid(self.fc2(F.gelu(self.fc1(desc)))).unsqueeze(-1).unsqueeze(-1)

        return feat * spatial_map * channel_map


# ============================================================================
# GeodesicMambaEvolver (GME) — 交叉扫描 + 物理后置门控 (全局流)
# ============================================================================

class GeodesicMambaEvolver(nn.Module):
    """
    GME v2: 4 方向交叉扫描 + 物理后置门控.

    数据流:
      1. LayerNorm(x)
      2. 4 方向行列扫描 (H±, V±) → Mamba 长序列处理 → 还原 2D
      3. 物理后置门控 (Late-Gating):
         f_H = 0.5 * (H_fwd + H_bwd)   水平聚合
         f_V = 0.5 * (V_fwd + V_bwd)   垂直聚合
         α = sigmoid(Q1)               Q1>0 偏好水平, Q1<0 偏好垂直
         f_global = α·f_H + (1-α)·f_V  方向路由
         f_global *= (0.5 + 0.5·S)      S 显著性掩码
      4. out_proj (Conv1×1 + GN + GELU)

    Mamba 看到完整上下文 (包括所有方向纤维 + 背景),
    物理场仅在后置阶段做方向选择, 各司其职.
    """

    def __init__(self, dim, d_state=64, expand=2, mamba3_kwargs=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        mamba3_kwargs = dict(mamba3_kwargs or {})

        d_inner = int(dim * expand)
        headdim = 64
        for hd in [64, 32, 16, 8]:
            if d_inner % hd == 0:
                headdim = hd
                break

        self.mamba = Mamba3(
            d_model=dim, d_state=d_state, expand=expand,
            headdim=headdim, ngroups=1, is_mimo=False, chunk_size=64,
            **mamba3_kwargs,
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )

    def _scan_merge(self, x, dim_scan):
        """
        沿指定空间维度做双向 Mamba 扫描并取均值.

        Args:
            x: (B, C, H, W)
            dim_scan: 2 (沿 H, 列扫描) 或 3 (沿 W, 行扫描)
        Returns:
            (B, C, H, W) 双向融合结果
        """
        B, C, H, W = x.shape

        if dim_scan == 3:
            # 水平扫描: (B, C, H, W) → (B*H, W, C)
            seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        else:
            # 垂直扫描: (B, C, H, W) → (B*W, H, C)
            seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)

        fwd = self.mamba(seq)
        bwd = self.mamba(seq.flip(1)).flip(1)
        out = 0.5 * (fwd + bwd)

        if dim_scan == 3:
            return out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            return out.reshape(B, W, H, C).permute(0, 3, 2, 1)

    def forward(self, x, Q1, Q2, S):
        B, C, H, W = x.shape

        # LayerNorm
        x_norm = self.norm(x.reshape(B, C, H * W).transpose(1, 2))
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)

        # 4 方向交叉扫描 (水平 + 垂直, 各自双向)
        f_H = self._scan_merge(x_norm, dim_scan=3)  # 水平
        f_V = self._scan_merge(x_norm, dim_scan=2)  # 垂直

        # 物理后置门控
        alpha = torch.sigmoid(Q1)                    # (B, 1, H, W)
        f_global = alpha * f_H + (1 - alpha) * f_V   # Q1 方向路由
        f_global = f_global * (0.5 + 0.5 * S)        # S 显著性掩码

        return self.out_proj(f_global)


# ============================================================================
# CrossFeatureInteraction (CFI) — 交叉门控融合
# ============================================================================

class CrossFeatureInteraction(nn.Module):
    """
    Cross Feature Interaction (CFI) — 参考 MPFI/CFI 数据流。
    交叉调制 (Cross-modulation): 两路互补特征交叉加权后直接相加。
    O = σ(f_global) · f_local + σ(f_local) · f_global
    """

    def __init__(self, dim):
        super().__init__()
        self.gate_local = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.Sigmoid())
        self.gate_global = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.Sigmoid())

    def forward(self, f_local, f_global):
        x_local_prime = self.gate_global(f_global) * f_local
        x_global_prime = self.gate_local(f_local) * f_global
        return x_local_prime + x_global_prime


# ============================================================================
# SteerableMetricGating — 显式方向度量空间门控
# ============================================================================

class SteerableMetricGating(nn.Module):
    """
    将对齐子空间分为 0, 45, 90, 135 四个方向组。
    利用物理场 Q1, Q2 作为加权系数，对各自维度的特征进行独立显式门控聚焦。
    """
    def __init__(self, d_embed):
        super().__init__()
        assert d_embed % 4 == 0, "d_embed 维度必须能被 4 整除"
        self.d_embed = d_embed

    def forward(self, K, Q1, Q2):
        """
        K:  (N+1, B, H, W, d_embed)
        Q1: (1, B, H, W, 1) 或广播兼容型
        Q2: (1, B, H, W, 1)
        """
        g = self.d_embed // 4
        out = K.clone()
        out[..., :g]       *= (0.5 + 0.5 * Q1)     # 0°
        out[..., g:2*g]    *= (0.5 + 0.5 * Q2)     # 45°
        out[..., 2*g:3*g]  *= (0.5 - 0.5 * Q1)     # 90°
        out[..., 3*g:]     *= (0.5 - 0.5 * Q2)     # 135°
        return out


# ============================================================================
# BlockAttnRes — 块级注意力残差
# ============================================================================

class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals (块级深度注意力残差)。

    将网络层分成若干块 (每个 stage = 一个 block):
      块内: 标准残差累加 → partial_block
      块间: softmax 注意力聚合 completed_blocks

    共享对齐子空间 (Shared Alignment Space):
      1. 空间对齐: 双线性插值 + 3×3 DW-Conv 消除插值伪影
      2. 通道对齐: 由 encoder/decoder 的 to_embed (Conv1×1) 统一到 d_embed 维度
    每层通过可学习伪查询 w 从所有历史块中选择性检索信息。
    """

    def __init__(self, d_embed=64):
        super().__init__()
        self.d_embed = d_embed

        # 可学习伪查询向量 (input-independent), 零初始化确保训练初期等权平均
        self.proj = nn.Linear(d_embed, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

        # Key 归一化, 防止不同块间幅度差异偏置注意力权重
        self.norm = nn.RMSNorm(d_embed)

        # 空间对齐: 3×3 DW-Conv 消除插值伪影 (锯齿/像素偏移)
        self.align_conv = nn.Sequential(
            nn.Conv2d(d_embed, d_embed, 3, padding=1, groups=d_embed, bias=False),
            nn.GroupNorm(1, d_embed),
        )

        self.steerable_gating = SteerableMetricGating(d_embed)

    def _spatial_align(self, x, target_h, target_w):
        """空间对齐: 双线性插值 + DW-Conv 消除伪影。"""
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, (target_h, target_w), mode='bilinear', align_corners=False)
            x = self.align_conv(x)
        return x

    def forward(self, blocks, partial_block, S, Q1, Q2):
        """
        块间注意力: 对已完成块表示 + 当前块部分和做 softmax 聚合。

        Args:
            blocks: list of (B, d_embed, H_i, W_i) — 已完成的块表示 (含 b_0=embedding)
            partial_block: (B, d_embed, H, W) — 当前块的部分和
            S, Q1, Q2: 物理向列场 (B, 1, H, W)
        Returns:
            h: (B, d_embed, H, W) — 聚合后的输入
        """
        B, C, H, W = partial_block.shape

        # 空间对齐 + 构建源矩阵 V
        sources = [self._spatial_align(blk, H, W) for blk in blocks]
        sources.append(partial_block)

        V = torch.stack(sources, dim=0)            # [N+1, B, d_embed, H, W]

        # Key = RMSNorm(V), 在 channel 维度做归一化
        V_t = V.permute(0, 1, 3, 4, 2)            # [N+1, B, H, W, d_embed]
        K = self.norm(V_t)                          # [N+1, B, H, W, d_embed]

        # 方向基底门控
        Q1_r = Q1.squeeze(1).unsqueeze(0).unsqueeze(-1)
        Q2_r = Q2.squeeze(1).unsqueeze(0).unsqueeze(-1)
        K = self.steerable_gating(K, Q1_r, Q2_r)

        # 深度注意力路由: logits = w^T · K
        w = self.proj.weight.squeeze()              # [d_embed]
        logits = torch.einsum('c, n b h w c -> n b h w', w, K)

        # S 驱动的 Kimi 注意力温度缩放
        tau = S.squeeze(1).unsqueeze(0) + 1e-3  # [1, B, H, W]
        logits = logits / tau

        # softmax 聚合: h = Σ α · V
        alpha = F.softmax(logits.float(), dim=0).to(V.dtype)   # [N+1, B, H, W]
        h = torch.einsum('n b h w, n b c h w -> b c h w', alpha, V)

        return h


# ============================================================================
# NematicInteractionBlock (NIB) — 向列交互块
# ============================================================================

class NematicInteractionBlock(nn.Module):
    """
    Nematic Interaction Block (NIB).

    数据流:
      x → NBE (多尺度各向同性边界探测)                    → X_local
      x → GME (交叉扫描 + 物理后置门控)                   → X_global
      CFI (交叉调制): O = σ(X_g)·X_l + σ(X_l)·X_g
      残差 + MLP: x_out = (x + DropPath(O)) + DropPath(MLP(...))
    """

    def __init__(self, dim, d_state=64, expand=2, drop_path=0.0, mlp_ratio=4, mamba3_kwargs=None):
        super().__init__()
        self.nbe = NematicBoundaryExtractor(dim)
        self.gme = GeodesicMambaEvolver(dim, d_state=d_state, expand=expand, mamba3_kwargs=mamba3_kwargs)
        self.cfi = CrossFeatureInteraction(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden, 1, bias=False),
            nn.GroupNorm(1, mlp_hidden),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x, S, Q1, Q2):
        x_local = self.nbe(x)
        x_global = self.gme(x, Q1=Q1, Q2=Q2, S=S)
        o_mpfi = self.cfi(x_local, x_global)
        h = x + self.drop_path(o_mpfi)
        h = h + self.drop_path(self.mlp(h))
        return h


# ============================================================================
# 辅助层
# ============================================================================

class DownsampleLayer(nn.Module):
    """Stride-2 Conv3×3 下采样"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class UpsampleLayer(nn.Module):
    """转置卷积上采样 (Transpose Conv, stride=2)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()

    def forward(self, x, target_size=None):
        x = self.up(x)
        # 适配可能的奇数尺寸差异
        if target_size is not None:
            dh = x.shape[2] - target_size[0]
            dw = x.shape[3] - target_size[1]
            if dh != 0 or dw != 0:
                x = x[:, :, :target_size[0], :target_size[1]]
        return self.act(self.norm(x))
