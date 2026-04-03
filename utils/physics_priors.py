# -*- coding: utf-8 -*-
"""
物理先验工具模块
SobelGradient / 结构张量伪标签提取 (GPU Batch)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SobelGradient — Sobel 2D 梯度算子
# ============================================================================

class SobelGradient(nn.Module):
    """Sobel 2D 梯度 (多通道逐通道计算)"""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]) / 8.0
        ky = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]]) / 8.0
        self.register_buffer('kx', kx.reshape(1, 1, 3, 3))
        self.register_buffer('ky', ky.reshape(1, 1, 3, 3))

    def forward(self, field):
        """
        Args:  field (B, C, H, W)
        Returns: gx, gy 各 (B, C, H, W)
        """
        B, C, H, W = field.shape
        f = field.reshape(B * C, 1, H, W)
        f = F.pad(f, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(f, self.kx).reshape(B, C, H, W)
        gy = F.conv2d(f, self.ky).reshape(B, C, H, W)
        return gx, gy


# ============================================================================
# 结构张量伪标签提取 (辅助函数)
# ============================================================================

def _gaussian_kernel_2d(sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(math.ceil(3 * sigma)) * 2 + 1
    k = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gaussian1d = torch.exp(-0.5 * (k / sigma) ** 2)
    gaussian2d = gaussian1d.unsqueeze(0) * gaussian1d.unsqueeze(1)
    return gaussian2d / gaussian2d.sum()


# ============================================================================
# StructureTensorExtractor — GPU 批量结构张量物理场提取器
# ============================================================================

class StructureTensorExtractor(nn.Module):
    """
    基于结构张量提取物理场 (S, Q1, Q2) 伪标签。
    继承 nn.Module 使 Sobel buffer 自动跟随 .cuda()。
    支持 Batch 输入: (B, C, H, W) → (B, 1, H, W)。
    """

    def __init__(self, sigma_g=1.0, sigma_t=2.0):
        super().__init__()
        self.sigma_g = sigma_g
        self.sigma_t = sigma_t
        self.sobel = SobelGradient()

        # 预计算高斯核并注册为 buffer（自动跟随 .cuda()）
        kernel_g = _gaussian_kernel_2d(sigma_g).unsqueeze(0).unsqueeze(0)
        kernel_t = _gaussian_kernel_2d(sigma_t).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_g', kernel_g)
        self.register_buffer('kernel_t', kernel_t)

    @torch.no_grad()
    def forward(self, image):
        """
        Args:
            image: (B, C, H, W) torch.Tensor (GPU)
        Returns:
            S, Q1, Q2: 各 (B, 1, H, W) torch.Tensor
        """
        # 多通道 → 灰度
        if image.shape[1] > 1:
            img_gray = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            img_gray = image

        # 高斯预平滑
        pad_g = self.kernel_g.shape[-1] // 2
        img_smooth = F.conv2d(img_gray, self.kernel_g, padding=pad_g)

        # Sobel 梯度
        Ix, Iy = self.sobel(img_smooth)

        # 结构张量分量 → 批量高斯积分 (3 通道一次 conv2d)
        J_stack = torch.cat([Ix * Ix, Iy * Iy, Ix * Iy], dim=1)  # (B, 3, H, W)
        kernel_t_3ch = self.kernel_t.expand(3, -1, -1, -1)       # (3, 1, K, K)
        pad_t = self.kernel_t.shape[-1] // 2
        J_smooth = F.conv2d(J_stack, kernel_t_3ch, padding=pad_t, groups=3)
        J11_s, J22_s, J12_s = J_smooth[:, 0:1], J_smooth[:, 1:2], J_smooth[:, 2:3]

        trace = J11_s + J22_s
        diff = torch.sqrt(torch.clamp((J11_s - J22_s) ** 2 + 4 * J12_s ** 2, min=1e-8))

        S_st = diff / (trace + 1e-8)
        Q1_t = -(J11_s - J22_s) / (diff + 1e-8)
        Q2_t = -2 * J12_s / (diff + 1e-8)

        S = S_st
        Q1 = (S * Q1_t).clamp(-1, 1)
        Q2 = (S * Q2_t).clamp(-1, 1)

        return S.clamp(0, 1), Q1, Q2
