# -*- coding: utf-8 -*-
"""VMamba with Mamba3 SISO backend.

SS2D: Mamba3 统一 in_proj 参数化.
VSSBlock: SSM + MLP 自包含块.

扫描/合并: cross_scan_fn / cross_merge_fn
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .mamba3.triton.layernorm_gated import RMSNorm as RMSNormGated
from .mamba3.triton.mamba3_siso_combined import mamba3_siso_combined


def cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=0):
    """Cross scan: extract multi-directional sequences."""
    if in_channel_first:
        B, C, H, W = x.shape
    else:
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()

    L = H * W

    if scans == 0:  # cross2d: [行正, 列正, 行逆, 列逆]
        xs1 = x.reshape(B, C, L)
        xs2 = x.transpose(-1, -2).reshape(B, C, L)
        xs3 = xs1.flip(-1)
        xs4 = xs2.flip(-1)
        xs = torch.stack([xs1, xs2, xs3, xs4], dim=1)
    elif scans == 1:  # unidi
        xs1 = x.reshape(B, C, L)
        xs2 = xs1.flip(-1)
        xs = torch.stack([xs1, xs2], dim=1)
    else:  # bidi
        xs1 = x.reshape(B, C, L)
        xs = torch.stack([xs1, xs1.flip(-1)], dim=1)

    if out_channel_first:
        return xs
    else:
        return xs.permute(0, 1, 3, 2).contiguous()


def cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=0):
    """Cross merge: combine multi-directional sequences (simple sum, same as original VMamba)."""
    if not in_channel_first:
        ys = ys.permute(0, 1, 3, 2).contiguous()

    # 处理 5D 输入 (B, K, C, H, W) -> (B, K, C, L)
    if ys.dim() == 5:
        B, K, C, H, W = ys.shape
        ys = ys.reshape(B, K, C, H * W)

    B, K, C, L = ys.shape
    H = W = int(L ** 0.5)

    if scans == 0:  # cross2d: 行正+行逆, 列正+列逆, 然后相加
        y_row = ys[:, 0] + ys[:, 2].flip(-1)
        y_row = y_row.reshape(B, C, H, W)
        y_col = ys[:, 1] + ys[:, 3].flip(-1)
        y_col = y_col.reshape(B, C, W, H).transpose(-1, -2)
        y = y_row + y_col
    elif scans == 1:  # unidi
        y = (ys[:, 0] + ys[:, 1].flip(-1)).reshape(B, C, H, W)
    else:  # bidi
        y = (ys[:, 0] + ys[:, 1].flip(-1)).reshape(B, C, H, W)

    if not out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


# ═══════════════════════════════════════════════════════════════
#  辅助类
# ═══════════════════════════════════════════════════════════════

class Linear(nn.Linear):
    """Linear with channel_first support."""
    def __init__(self, *args, channel_first=False, groups=1, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.channel_first = channel_first
        self.groups = groups

    def forward(self, x: torch.Tensor):
        if self.channel_first:
            if len(x.shape) == 4:
                return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
            elif len(x.shape) == 3:
                return F.conv1d(x, self.weight[:, :, None], self.bias, groups=self.groups)
        else:
            return F.linear(x, self.weight, self.bias)


class LayerNorm(nn.LayerNorm):
    """LayerNorm with channel_first support."""
    def __init__(self, *args, channel_first=None, in_channel_first=False, out_channel_first=False, **kwargs):
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x


class Mlp(nn.Module):
    """MLP with channel_first support."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, channel_first=channel_first)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, channel_first=channel_first)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """DropPath (stochastic depth)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class SS2D(nn.Module):
    """SS2D with Mamba3 SISO SSM.

    流程: x → conv → cross_scan → SSM(Z门控) → cross_merge → out_proj
    """

    def __init__(self, dim, d_state=16, ssm_ratio=2.0, ngroups=1, chunk_size=64):
        super().__init__()
        d_inner = int(dim * ssm_ratio)
        self.d_inner = d_inner
        self.dim = dim
        self.d_state = d_state

        # V 侧参数
        nheads = max(2, d_inner // 64)
        self.nheads = nheads
        headdim_v = d_inner // nheads
        self.headdim_v = headdim_v

        # Q/K 侧参数
        self.ngroups = ngroups
        self.mimo_rank = 1
        self.headdim_qk = d_state

        # RoPE angles
        self.rope_fraction = 0.5
        self.split_tensor_size = int(d_state * self.rope_fraction)
        self.num_rope_angles = self.split_tensor_size // 2

        # Mamba3 全局参数
        self.chunk_size = chunk_size
        self.A_floor = 1e-4

        # 输入投影 + Z 门控
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)

        # 深度可分离卷积
        self.conv2d = nn.Sequential(
            nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner),
            nn.SiLU(),
        )

        # B/C projections
        self.B_proj = nn.Linear(d_inner, self.ngroups * self.mimo_rank * d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, self.ngroups * self.mimo_rank * d_state, bias=False)

        # B/C RMSNorm
        self.B_norm = RMSNormGated(d_state, eps=1e-5)
        self.C_norm = RMSNormGated(d_state, eps=1e-5)

        # B/C biases
        self.B_bias = nn.Parameter(1 + torch.zeros(nheads, self.mimo_rank, d_state))
        self.C_bias = nn.Parameter(1 + torch.zeros(nheads, self.mimo_rank, d_state))

        # dt projection
        self.dt_proj = nn.Linear(d_inner, nheads, bias=False)

        # dt_bias
        dt_min, dt_max = 0.001, 0.1
        _dt = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=1e-4)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        # 每方向独立的 A_log
        self.A_logs = nn.ParameterList([
            nn.Parameter(torch.log(torch.randn(nheads) + 1e-6))
            for _ in range(4)
        ])

        # Trap
        self.trap_proj = nn.Linear(d_inner, nheads, bias=False)

        # D 残差
        self.D = nn.Parameter(torch.ones(nheads))
        self.D._no_weight_decay = True

        # Angles
        self.angles_proj = nn.Linear(d_inner, self.num_rope_angles, bias=False)

        # 输出
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

        # 初始化
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        for a_log in self.A_logs:
            nn.init.uniform_(a_log, -8, -1)

    def forward(self, x, Q=None):
        """
        Args:
            x: (B, C, H, W) 输入特征 (channel-first)
            Q: (B, 2, H, W) Q-tensor (只读，用于计算S)
        Returns:
            (B, C, H, W)
        """
        B, C_in, H, W = x.shape
        L = H * W

        # 输入投影 + Z 门控
        xz = self.in_proj(x.permute(0, 2, 3, 1).reshape(B, L, C_in))
        x_spatial, z = xz.chunk(2, dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim_v)

        # 深度可分离卷积
        x_conv = x_spatial.permute(0, 2, 1).reshape(B, self.d_inner, H, W)
        x_conv = self.conv2d(x_conv)

        # 交叉扫描: 4 方向
        xs = cross_scan_fn(x_conv, in_channel_first=True, out_channel_first=True, scans=0)
        xs_seq = xs.permute(0, 1, 3, 2).contiguous().reshape(B * 4, L, self.d_inner)

        # z: no cross_scan, pass directly to kernel

        # 批量投影: 4 方向 B/C/dt/trap/angles
        B_all = self.B_proj(xs_seq)
        C_all = self.C_proj(xs_seq)
        B_all = rearrange(B_all, "bl l (r g n) -> bl l r g n", r=self.mimo_rank, g=self.ngroups)
        C_all = rearrange(C_all, "bl l (r g n) -> bl l r g n", r=self.mimo_rank, g=self.ngroups)
        B_all = self.B_norm(B_all)
        C_all = self.C_norm(C_all)
        B_all = B_all.reshape(B, 4, L, self.mimo_rank, self.ngroups, self.d_state)
        C_all = C_all.reshape(B, 4, L, self.mimo_rank, self.ngroups, self.d_state)

        # Trap 投影
        Trap_all = self.trap_proj(xs_seq).reshape(B, 4, L, self.nheads)

        # Angles 投影 + expand
        angles_all = self.angles_proj(xs_seq)
        angles_all = angles_all.reshape(B, 4, L, self.num_rope_angles)
        angles_all = angles_all.unsqueeze(-2).expand(-1, -1, -1, self.nheads, -1)

        # dt 投影
        dd_dt_all = self.dt_proj(xs_seq)
        dt_all = F.softplus(dd_dt_all + self.dt_bias).clamp(max=0.2)
        dt_all = dt_all.reshape(B, 4, L, self.nheads).permute(0, 1, 3, 2)

        # 逐方向调用 mamba3_siso_combined
        ys = []
        for k in range(4):
            x_k = xs_seq.reshape(B, 4, L, self.nheads, self.headdim_v)[:, k]
            B_k = B_all[:, k]
            C_k = C_all[:, k]
            z_k = z
            angles_k = angles_all[:, k]
            dt_k = dt_all[:, k]

            # A: -softplus + clamp
            A_k = -F.softplus(self.A_logs[k]).clamp(max=-self.A_floor)
            ADT_k = A_k.unsqueeze(0).unsqueeze(2) * dt_k

            # Trap
            Trap_k = Trap_all[:, k].permute(0, 2, 1)

            out_k = mamba3_siso_combined(
                Q=C_k.squeeze(2),
                K=B_k.squeeze(2),
                V=x_k,
                ADT=ADT_k,
                DT=dt_k,
                Trap=Trap_k,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles_k,
                D=self.D,
                Z=z_k,
                chunk_size=self.chunk_size,
            )
            if isinstance(out_k, tuple):
                out_k = out_k[0]

            # 处理 5D 输出 (B, L, nheads, headdim_v) -> (B, d_inner, H, W)
            if out_k.dim() == 4:
                out_k = out_k.reshape(B, L, self.d_inner)
            else:
                out_k = out_k.reshape(B, L, -1)
            out_k = out_k.permute(0, 2, 1).reshape(B, self.d_inner, H, W)
            ys.append(out_k)

        # Merge 4 directions (简单求和，和原始VMamba一致)
        ys = torch.stack(ys, dim=1)
        y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=0)

        # 输出投影
        y = y.permute(0, 2, 3, 1)
        y = self.out_norm(y.float())
        y = self.out_proj(y)

        return y.permute(0, 3, 1, 2)


class VSSBlock(nn.Module):
    """VSS Block: Mamba3 SSM + MLP self-contained.

    结构: x -> [LayerNorm -> SS2D -> +DropPath]
            -> [LayerNorm -> MLP -> +DropPath]
    """

    def __init__(self, dim, drop_path=0.0, chunk_size=64,
                 mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0):
        super().__init__()

        # SSM branch
        self.norm_ssm = LayerNorm(dim, channel_first=True)
        self.ssm = SS2D(dim, chunk_size=chunk_size)

        # MLP branch
        self.norm_mlp = LayerNorm(dim, channel_first=True)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
            channel_first=True,
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            (B, C, H, W) 输出特征
        """
        # SSM branch
        x = x + self.drop_path(self.ssm(self.norm_ssm(x)))

        # MLP branch
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))

        return x
