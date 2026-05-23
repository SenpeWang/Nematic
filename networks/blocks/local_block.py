# -*- coding: utf-8 -*-
"""Local-stage blocks: Riemannian diffusion + lightweight Q update."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..q_tensor import QTensorTools
from .base import MLPBlock


class RiemannianDiffusionBlock(nn.Module):
    """Q-tensor guided anisotropic neighbourhood aggregation."""

    def __init__(self, dim, K=3, Q_max_norm=1.0, alpha_hidden=8):
        super().__init__()
        self.dim = dim
        self.K = K
        self.Q_max_norm = float(Q_max_norm)

        offsets = [(dx, dy) for dy in range(-(K // 2), K // 2 + 1)
                           for dx in range(-(K // 2), K // 2 + 1)]
        K2 = K * K
        dx2 = torch.tensor([o[0] ** 2 for o in offsets], dtype=torch.float32)
        dxdy2 = torch.tensor([2.0 * o[0] * o[1] for o in offsets], dtype=torch.float32)
        dy2 = torch.tensor([o[1] ** 2 for o in offsets], dtype=torch.float32)
        self.register_buffer('_dx2', dx2.view(1, K2, 1, 1))
        self.register_buffer('_dxdy2', dxdy2.view(1, K2, 1, 1))
        self.register_buffer('_dy2', dy2.view(1, K2, 1, 1))

        self.alpha_net = nn.Sequential(
            nn.Conv2d(1, alpha_hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(alpha_hidden, 2, 1, bias=True),
        )
        with torch.no_grad():
            self.alpha_net[2].bias.copy_(torch.tensor([0.54, -3.0]))
            nn.init.zeros_(self.alpha_net[2].weight)
            nn.init.zeros_(self.alpha_net[0].weight)
            nn.init.zeros_(self.alpha_net[0].bias)

        self.tau_net = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.GroupNorm(8, dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1, bias=True),
        )
        with torch.no_grad():
            nn.init.constant_(self.tau_net[3].bias, 0.0)

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(16, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(16, dim),
        )
        self.refine_act = nn.GELU()
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def _build_metric(self, q1, q2, x):
        S = torch.sqrt(q1.pow(2) + q2.pow(2) + 1e-8).clamp(0, 1)
        safe_s = S.clamp(min=1e-6)
        inv_s = 1.0 / safe_s

        A = 0.5 + 0.5 * q1 * inv_s
        B = 0.5 * q2 * inv_s
        C = 0.5 - 0.5 * q1 * inv_s
        zero_mask = (S < 1e-6).float()
        A = A * (1 - zero_mask) + 0.5 * zero_mask
        B = B * (1 - zero_mask)
        C = C * (1 - zero_mask) + 0.5 * zero_mask

        alpha = F.softplus(self.alpha_net(S))
        alpha_along = alpha[:, 0:1]
        alpha_cross = alpha[:, 1:2]

        diff = alpha_along - alpha_cross
        M_11 = alpha_cross + diff * A
        M_12 = diff * B
        M_22 = alpha_cross + diff * C

        return M_11, M_12, M_22, S, alpha_along, alpha_cross, A, B

    def _diffusion(self, x, M_11, M_12, M_22, tau):
        B, C, H, W = x.shape
        K = self.K
        padding = K // 2

        x_unfold = F.unfold(x, kernel_size=K, padding=padding)
        x_unfold = x_unfold.view(B, C, K * K, H, W)

        d2 = (M_11 * self._dx2 + M_12 * self._dxdy2 + M_22 * self._dy2)

        tau_safe = tau.clamp(min=0.01)
        w = F.softmax(-d2 / tau_safe, dim=1)

        x_diffused = (x_unfold * w.unsqueeze(1)).sum(dim=2)

        return x_diffused, w

    def forward(self, x, Q):
        M_11, M_12, M_22, S, alpha_along, alpha_cross, A_diag, B_off = self._build_metric(
            Q[:, 0:1], Q[:, 1:2], x
        )

        tau = F.softplus(self.tau_net(x)) + 0.05

        x_diffused, w = self._diffusion(x, M_11, M_12, M_22, tau)

        x_refined = self.refine(x_diffused)
        x_out = x + self.refine_act(self.res_scale * x_refined)

        aux = {
            'S_mean': S.mean().item(),
            'alpha_along_mean': alpha_along.mean().item(),
            'alpha_cross_mean': alpha_cross.mean().item(),
            'tau_mean': tau.mean().item(),
            'M_trace_mean': (M_11 + M_22).mean().item(),
            'M_anisotropy': ((alpha_along - alpha_cross) / (alpha_along + alpha_cross + 1e-6)).mean().item(),
            'diffusion_weights': w.detach(),
            'director_A': A_diag.detach(),
            'director_B': B_off.detach(),
        }

        return x_out, aux


class StructureTensorBranch(nn.Module):
    """Explicit directional analysis via structure tensor on feature maps.

    Computes spatial gradients (Sobel) on each feature channel, builds
    the structure-tensor covariance matrix, and extracts nematic parameters
    via eigendecomposition. Operates on compressed θ-stream features.
    """
    def __init__(self, in_ch, out_dim=16, smooth_kernel=5, smooth_sigma=1.0):
        super().__init__()
        self.in_ch = in_ch

        # Sobel kernels for per-channel gradient estimation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # repeat for group conv: one filter per input channel
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))

        # Gaussian smoothing kernel
        k = smooth_kernel // 2
        x_coord = torch.arange(smooth_kernel, dtype=torch.float32) - k
        gauss_1d = torch.exp(-x_coord.pow(2) / (2 * smooth_sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        self.register_buffer('gauss_kernel', gauss_2d.view(1, 1, smooth_kernel, smooth_kernel))

        # Project (S_est, cos2θ_est, sin2θ_est) into learnable space
        self.project = nn.Sequential(
            nn.Conv2d(3, out_dim, 1, bias=False),
            nn.GroupNorm(16, out_dim), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        B, C, H, W = x.shape

        # Per-channel Sobel gradients (group conv)
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=C)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=C)

        # Structure-tensor components: average over channels
        J_xx = (gx ** 2).mean(dim=1, keepdim=True)
        J_xy = (gx * gy).mean(dim=1, keepdim=True)
        J_yy = (gy ** 2).mean(dim=1, keepdim=True)

        # Smooth the structure tensor
        pad = self.gauss_kernel.shape[-1] // 2
        J_xx = F.conv2d(J_xx, self.gauss_kernel, padding=pad)
        J_xy = F.conv2d(J_xy, self.gauss_kernel, padding=pad)
        J_yy = F.conv2d(J_yy, self.gauss_kernel, padding=pad)

        # Eigendecomposition
        trace = J_xx + J_yy + 1e-8
        diff = torch.sqrt((J_xx - J_yy) ** 2 + 4 * J_xy ** 2 + 1e-8)
        lambda1 = 0.5 * (trace + diff)
        lambda2 = 0.5 * (trace - diff)

        # Nematic parameters
        S_est = ((lambda1 - lambda2) / trace).clamp(0, 1)
        theta_est = 0.5 * torch.atan2(2 * J_xy, J_xx - J_yy)

        # Encode as features
        feat = torch.cat([S_est, torch.cos(2 * theta_est), torch.sin(2 * theta_est)], dim=1)
        struct_feat = self.project(feat)
        return struct_feat, S_est, theta_est


class ConvQUpdater(nn.Module):
    """Explicit (S, θ) parameterization with structure-tensor grounded θ update.

    1. Compress high-dim features to nematic-relevant channels (1×1).
    2. Explicitly split into S-stream and θ-stream.
    3. S-stream: 1×1 conv predictor (orderedness is a channel-wise statistic).
    4. θ-stream: StructureTensorBranch measures spatial directional coherence
       on the θ-features; θ update is driven by the discrepancy between
       the measured texture direction (theta_est) and current Q direction.
    Inter-block propagation stays q1,q2 (S·cos2θ, S·sin2θ).
    """
    def __init__(self, dim, compress_dim=None):
        super().__init__()
        if compress_dim is None:
            compress_dim = max(dim // 8, 64)
        self.compress_dim = compress_dim
        half = max(compress_dim // 2, 32)

        # 1. Compress: learn which channels encode nematic info
        self.compress = nn.Sequential(
            nn.Conv2d(dim, compress_dim, 1, bias=False),
            nn.GroupNorm(16, compress_dim), nn.GELU(),
        )

        # 2. Split: explicitly route to S-relevant vs θ-relevant subspaces
        self.split_s = nn.Conv2d(compress_dim, half, 1, bias=False)
        self.split_theta = nn.Conv2d(compress_dim, half, 1, bias=False)

        # 3. S branch (channel-wise, 1×1 conv is sufficient)
        self.s_gate = nn.Sequential(
            nn.Conv2d(1, half, 1, bias=False), nn.Sigmoid()
        )
        self.s_predictor = nn.Sequential(
            nn.Conv2d(half + 1, half, 1, bias=False),
            nn.GroupNorm(16, half), nn.GELU(),
            nn.Conv2d(half, 1, 1),
        )
        self.alpha_s = nn.Parameter(torch.tensor(0.1))

        # 4. θ branch: Structure Tensor replaces the geometrically-ungrounded
        #    theta_align/ortho. Direction must be measured, not guessed.
        self.struct_tensor = StructureTensorBranch(half, out_dim=16)

        # theta_predictor input: struct_feat (16ch) + sin/cos of direction error (2ch)
        self.theta_gate = nn.Sequential(
            nn.Conv2d(2, 16 + 2, 1, bias=False), nn.Sigmoid()
        )
        self.theta_predictor = nn.Sequential(
            nn.Conv2d(16 + 2, half, 1, bias=False),
            nn.GroupNorm(16, half), nn.GELU(),
            nn.Conv2d(half, 1, 1),
        )
        self.alpha_theta = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, Q):
        q1, q2 = Q[:, 0:1], Q[:, 1:2]
        eps = 1e-8
        S = torch.sqrt(q1.pow(2) + q2.pow(2) + eps).clamp(0, 1)
        theta = 0.5 * torch.atan2(q2, q1)

        # 1. Compress
        f = self.compress(x)

        # 2. Split into S-stream and θ-stream
        f_s = self.split_s(f)
        f_theta = self.split_theta(f)

        # 3. Structure tensor on θ-features: measures actual texture direction
        struct_feat, S_est, theta_est = self.struct_tensor(f_theta)

        # 4. S update (S_est provides physical reference for orderedness)
        s_gate = self.s_gate(S)
        s_input = torch.cat([f_s * s_gate, S_est], dim=1)
        dS = torch.tanh(self.s_predictor(s_input)) * torch.tanh(self.alpha_s)
        S_new = (S + dS).clamp(0, 1)

        # 5. θ update driven by direction discrepancy
        # theta_est = what the texture says; theta = what Q currently believes.
        # Encode the periodic angular error via sin/cos (no wrap-around issues).
        theta_err = theta_est - theta
        sin2_err = torch.sin(2 * theta_err)
        cos2_err = torch.cos(2 * theta_err)

        # All physical cues go to predictor; network learns when to update.
        theta_input = torch.cat([struct_feat, sin2_err, cos2_err], dim=1)
        theta_gate = self.theta_gate(Q)
        dtheta = torch.tanh(self.theta_predictor(theta_input * theta_gate)) * torch.tanh(self.alpha_theta)
        theta_new = theta + dtheta

        # 6. Compose back to q1, q2 = S * (cos 2θ, sin 2θ)
        two_theta = 2 * theta_new
        q1_new = S_new * torch.cos(two_theta)
        q2_new = S_new * torch.sin(two_theta)
        Q_new = torch.cat([q1_new, q2_new], dim=1)

        aux = {
            'dS': dS.detach(),
            'dtheta': dtheta.detach(),
            'S_new': S_new.detach(),
            'S_est': S_est.detach(),
            'theta_est': theta_est.detach(),
        }
        return Q_new, aux


class LocalSGBlock(nn.Module):
    """Lightweight block for Stage 0/1 (high resolution).

    Structure:
        RiemannianDiffusionBlock(x, Q) → MLPBlock → ConvQUpdater(x, Q)
    """
    def __init__(self, dim, drop_path=0.0, mlp_ratio=4, Q_max_norm=1.0):
        super().__init__()
        self.Q_max_norm = float(Q_max_norm)
        self.rdiff = RiemannianDiffusionBlock(dim, Q_max_norm=Q_max_norm)
        self.mlp_block = MLPBlock(dim, mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.q_updater = ConvQUpdater(dim)

    def forward(self, x, Q):
        x, aux = self.rdiff(x, Q)
        x = self.mlp_block(x)
        Q, q_aux = self.q_updater(x, Q)
        aux.update(q_aux)
        return x, Q, aux
