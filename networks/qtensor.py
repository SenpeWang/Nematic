# -*- coding: utf-8 -*-
"""Q-tensor related modules.

  - QImagePrior: image -> Q-tensor prior (fixed, no_grad)
  - QTensorTools: nematic Q-tensor ops (q1/q2 separate input, single entry)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared Sobel kernels (used by QImagePrior)
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0

class QTensorTools:
    """Nematic Q-tensor operations — single entry for all Q-tensor math.

    Every method receives q1 and q2 as separate arguments.
    No external script should perform Q-tensor math; only pass q1/q2 here.
    """

    # ── 基础量 ────────────────────────────────────────────────────────

    @staticmethod
    def get_S(q1, q2, eps=1e-8):
        """标量序参量 S = sqrt(q1^2 + q2^2). 当 Q 经过 get_Qtensor 归一化后值域 [0,1]."""
        return torch.sqrt(q1.pow(2) + q2.pow(2) + eps)

    # ── Q-tensor 构造与归一化 ────────────────────────────────────────

    @staticmethod
    def get_Qtensor(q1, q2, Q_max_norm=1.0, eps=1e-8):
        """归一化 Q-tensor (2ch), 模 capped at Q_max_norm."""
        Q = torch.cat([q1, q2], dim=1)
        n = QTensorTools.get_S(q1, q2, eps)
        scale = torch.clamp(float(Q_max_norm) / (n + eps), max=1.0)
        return Q * scale

    @staticmethod
    def get_Nematic(q1, q2, eps=1e-8):
        """返回 [Q, S] (3ch): q1, q2 加上标量序参量 S."""
        Q = torch.cat([q1, q2], dim=1)
        S = QTensorTools.get_S(q1, q2, eps)
        return torch.cat([Q, S], dim=1)


    # ── 上下采样 ──────────────────────────────────────────────────────

    @staticmethod
    def downsample_Q(q1, q2, size=None, Q_max_norm=1.0):
        """下采样 q1/q2 (双线性插值 + 重归一化)."""
        if size is None:
            size = (q1.shape[2] // 2, q1.shape[3] // 2)
        q1 = F.interpolate(q1, size=size, mode="bilinear", align_corners=False)
        q2 = F.interpolate(q2, size=size, mode="bilinear", align_corners=False)
        return QTensorTools.get_Qtensor(q1, q2, Q_max_norm)


    @staticmethod
    def get_unit_Q(q1, q2, threshold=0.05, eps=1e-6):
        """Q-tensor 单位方向 (cos2θ, sin2θ)，低 S 区域清零.

        Returns: (q1_hat, q2_hat), shape 同输入 q1/q2
        """
        S = QTensorTools.get_S(q1, q2)
        mask = (S > threshold).float()
        q1_hat = q1 / S.clamp_min(eps) * mask
        q2_hat = q2 / S.clamp_min(eps) * mask
        return q1_hat, q2_hat

    # ── 可视化 ────────────────────────────────────────────────────────

    @staticmethod
    def get_director_numpy(q1, q2, eps=1e-8):
        """numpy 版: 提取方向角 θ 和方向向量 (dx, dy)."""
        theta = 0.5 * np.arctan2(q2, q1 + eps)
        dx = np.cos(theta)
        dy = np.sin(theta)
        return theta, dx, dy

class QImagePrior(nn.Module):
    """Compute Q-tensor prior from raw image (fixed, no_grad)."""
    def __init__(self, sigma_g=0.0, sigma_t=1.0, tangent=True):
        super().__init__()
        self.tangent = tangent
        self.register_buffer("kx", SOBEL_X.reshape(1, 1, 3, 3))
        self.register_buffer("ky", SOBEL_Y.reshape(1, 1, 3, 3))
        self.register_buffer(
            "kernel_g",
            self._gaussian_kernel_2d(sigma_g).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "kernel_t",
            self._gaussian_kernel_2d(sigma_t).unsqueeze(0).unsqueeze(0),
        )

    @staticmethod
    def _gaussian_kernel_2d(sigma, kernel_size=None):
        if sigma < 1e-6:
            return torch.ones(1, 1, dtype=torch.float32)
        if kernel_size is None:
            kernel_size = int(math.ceil(3 * sigma)) * 2 + 1
        k = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-0.5 * (k / sigma) ** 2)
        g2 = g.unsqueeze(0) * g.unsqueeze(1)
        return g2 / g2.sum()

    def _sobel(self, field):
        b, c, h, w = field.shape
        x = field.reshape(b * c, 1, h, w)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x, self.kx).reshape(b, c, h, w)
        gy = F.conv2d(x, self.ky).reshape(b, c, h, w)
        return gx, gy

    @torch.no_grad()
    def forward(self, image):
        img = image.mean(dim=1, keepdim=True) if image.shape[1] > 1 else image
        pad_g = self.kernel_g.shape[-1] // 2
        img = F.conv2d(img, self.kernel_g, padding=pad_g)
        ix, iy = self._sobel(img)
        j = torch.cat([ix * ix, iy * iy, ix * iy], dim=1)
        k = self.kernel_t.expand(3, -1, -1, -1)
        pad_t = self.kernel_t.shape[-1] // 2
        js = F.conv2d(j, k, padding=pad_t, groups=3)
        j11, j22, j12 = js[:, 0:1], js[:, 1:2], js[:, 2:3]
        trace = j11 + j22
        diff_raw = (j11 - j22) ** 2 + 4 * j12 ** 2
        diff = torch.sqrt(diff_raw.clamp(min=1e-8))
        S = (diff / (trace + 1e-8)).clamp(0, 1)
        cos2theta = (j11 - j22) / diff
        sin2theta = (2 * j12) / diff
        q1 = S * cos2theta
        q2 = S * sin2theta
        # Zero out where diff_raw is too small (no reliable direction)
        valid = diff_raw > 1e-8
        q1 = q1 * valid.float()
        q2 = q2 * valid.float()
        S = S * valid.float()
        if self.tangent:
            q1, q2 = -q1, -q2
        return q1, q2, S
