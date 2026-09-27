
import torch
import torch.nn as nn
from .qtensor import QTensorTools
from .VMamba.vmamba import VSSBlock, DropPath
from .layer import _gn


class ConvBlock(nn.Module):
    """深度可分离卷积 + 残差连接."""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(_gn(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(_gn(dim), dim),
            nn.GELU(),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.conv(x))


class _ChannelAttn(nn.Module):
    """SE 通道注意力 + S 调制."""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
        )
        self.s_mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 1, 1, bias=True),
        )

    def forward(self, x, S):
        se = self.se(x)
        s_mod = self.s_mod(S)
        return torch.sigmoid(se * (1.0 + s_mod))


class _SpatialAttn(nn.Module):
    """mean + max + S → 空间门控."""
    def __init__(self, dim, reduction=16):
        super().__init__()
        mid_dim = max(8, dim // reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(3, mid_dim, 1, bias=False),
            nn.GroupNorm(_gn(mid_dim), mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1, groups=mid_dim, bias=False),
            nn.GroupNorm(_gn(mid_dim), mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, S):
        mean_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([mean_feat, max_feat, S], dim=1)
        return self.conv(spatial_input)


class NematicStructureGate(nn.Module):
    """门控模块."""
    def __init__(self, dim, reduction=16, stats_logger=None, name=""):
        super().__init__()
        self.stats_logger = stats_logger
        self.name = name
        self.channel_attn = _ChannelAttn(dim, reduction)
        self.spatial_attn = _SpatialAttn(dim, reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(_gn(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(_gn(dim), dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.channel_attn.se[-1].weight)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, x, S):
        channel_attn = self.channel_attn(x, S)
        spatial_attn = self.spatial_attn(x, S)
        Gate = spatial_attn * channel_attn
        if self.stats_logger is not None:
            self.stats_logger.log_nsg(self.name, Gate.mean().item())
        x_masked = x * Gate
        x_refined = self.mlp(x_masked)
        return x + x_refined


class OFMBlock(nn.Module):
    """
    """
    def __init__(self, dim, drop_path=0.0, chunk_size=64,
                 mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                 stats_logger=None, name='', **kwargs):
        super().__init__()
        self.conv = ConvBlock(dim, drop_path=drop_path)
        self.vss = VSSBlock(
            dim=dim,
            drop_path=drop_path,
            chunk_size=chunk_size,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
        )
        self.gate = NematicStructureGate(dim, stats_logger=stats_logger, name=name)

    def forward(self, x, Q=None):
        S = QTensorTools.get_S(Q[:, 0:1], Q[:, 1:2])
        x = self.conv(x)
        x = self.vss(x)
        x = self.gate(x, S)
        return x
