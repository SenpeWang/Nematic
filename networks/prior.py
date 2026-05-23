# -*- coding: utf-8 -*-
"""Nematic image prior: structure-tensor based Q-tensor field estimation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0


def _gaussian_kernel_2d(sigma, kernel_size=None):
    if sigma < 1e-6:
        return torch.ones(1, 1, dtype=torch.float32)
    if kernel_size is None:
        kernel_size = int(math.ceil(3 * sigma)) * 2 + 1
    k = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-0.5 * (k / sigma) ** 2)
    g2 = g.unsqueeze(0) * g.unsqueeze(1)
    return g2 / g2.sum()


class Q_image_prior(nn.Module):
    """Per-channel structure tensor -> fused unified Q-tensor prior."""

    SCALE_PARAMS = [(0.0, 1.0), (1.0, 2.0), (1.5, 3.0), (1.5, 4.0)]

    def __init__(self, max_channels=16, scale_params=None, tangent=True):
        super().__init__()
        if scale_params is None:
            scale_params = self.SCALE_PARAMS
        self.max_channels = max_channels
        self.num_scales = len(scale_params)
        self.tangent = tangent
        self.register_buffer("kx", SOBEL_X.reshape(1, 1, 3, 3))
        self.register_buffer("ky", SOBEL_Y.reshape(1, 1, 3, 3))
        for k, (sg, st) in enumerate(scale_params):
            self.register_buffer(
                f"kernel_g_{k}",
                _gaussian_kernel_2d(sg).unsqueeze(0).unsqueeze(0),
            )
            self.register_buffer(
                f"kernel_t_{k}",
                _gaussian_kernel_2d(st).unsqueeze(0).unsqueeze(0),
            )
        self.scale_weights = nn.Parameter(torch.zeros(self.num_scales))

    @torch.no_grad()
    def _compute_channel_q(self, image, scale_idx=0):
        B, C, H, W = image.shape
        kernel_g = getattr(self, f"kernel_g_{scale_idx}")
        kernel_t = getattr(self, f"kernel_t_{scale_idx}")
        pad_g = kernel_g.shape[-1] // 2
        img_smooth = F.conv2d(image, kernel_g.expand(C, -1, -1, -1),
                              padding=pad_g, groups=C)
        img_pad = F.pad(img_smooth.reshape(B * C, 1, H, W),
                        (1, 1, 1, 1), mode='replicate')
        ix = F.conv2d(img_pad, self.kx).reshape(B, C, H, W)
        iy = F.conv2d(img_pad, self.ky).reshape(B, C, H, W)
        j11, j22, j12 = ix * ix, iy * iy, ix * iy
        j = torch.cat([j11, j22, j12], dim=1)
        pad_t = kernel_t.shape[-1] // 2
        k_t = kernel_t.expand(3 * C, -1, -1, -1)
        js = F.conv2d(j, k_t, padding=pad_t, groups=3 * C)
        j11s, j22s, j12s = js[:, 0:C], js[:, C:2*C], js[:, 2*C:3*C]
        trace = j11s + j22s
        q1 = torch.where(trace > 0, (j11s - j22s) / trace, torch.zeros_like(trace))
        q2 = torch.where(trace > 0, (2 * j12s) / trace, torch.zeros_like(trace))
        if self.tangent:
            q1, q2 = -q1, -q2
        return q1, q2

    def forward(self, image):
        B, C, H, W = image.shape
        if C != self.max_channels:
            raise ValueError(f"Input channels {C} != max_channels {self.max_channels}")

        scale_q1s, scale_q2s = [], []
        with torch.no_grad():
            for k in range(self.num_scales):
                q1_k, q2_k = self._compute_channel_q(image, scale_idx=k)
                scale_q1s.append(q1_k)
                scale_q2s.append(q2_k)
        q1_per_scale = [q.mean(dim=1, keepdim=True) for q in scale_q1s]
        q2_per_scale = [q.mean(dim=1, keepdim=True) for q in scale_q2s]

        w = F.softmax(self.scale_weights, dim=0)
        q1_out = sum(w[k] * q1_per_scale[k] for k in range(self.num_scales))
        q2_out = sum(w[k] * q2_per_scale[k] for k in range(self.num_scales))
        return q1_out, q2_out
