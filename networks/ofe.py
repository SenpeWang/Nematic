import torch
import torch.nn as nn
from .qtensor import QTensorTools
from .layer import _gn, DropPath


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(_gn(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_gn(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.conv(x))


class NematicConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Sobel 梯度提取
        self.grad = nn.Conv2d(dim, dim * 2, 3, padding=1, groups=dim, bias=False)
        self._init_sobel(); self.grad.weight.requires_grad = False

        # 平行分支
        self.branch_par = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False),
            nn.GroupNorm(_gn(dim * 2), dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )

        # 垂直分支
        self.branch_perp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False),
            nn.GroupNorm(_gn(dim * 2), dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )

        # 自学习融合 (跨分支空间交互 + 非线性)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False),
            nn.GroupNorm(_gn(dim * 2), dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )

    @staticmethod
    def _sobel_kernels():
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
        return sx, sy

    def _init_sobel(self):
        sx, sy = self._sobel_kernels()
        with torch.no_grad():
            for i in range(self.dim):
                self.grad.weight.data[2 * i, 0] = sx
                self.grad.weight.data[2 * i + 1, 0] = sy

    def forward(self, f_sem: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f_sem.shape

        # ── 向列相基底解析 ──────────────────────────────────────
        q1_hat, q2_hat = QTensorTools.get_unit_Q(Q[:, 0:1], Q[:, 1:2])

        # 梯度提取
        R = self.grad(f_sem).view(B, C, 2, H, W)
        R_x = R[:, :, 0]
        R_y = R[:, :, 1]

        # 向列相投影
        R_par_x = 0.5 * ((1 + q1_hat) * R_x + q2_hat * R_y)
        R_par_y = 0.5 * (q2_hat * R_x + (1 - q1_hat) * R_y)
        R_perp_x = 0.5 * ((1 - q1_hat) * R_x - q2_hat * R_y)
        R_perp_y = 0.5 * (-q2_hat * R_x + (1 + q1_hat) * R_y)

        # 双分支编码
        F_par = self.branch_par(torch.cat([R_par_x, R_par_y], dim=1))
        F_perp = self.branch_perp(torch.cat([R_perp_x, R_perp_y], dim=1))

        # 自学习融合 + 残差
        F = self.fusion(torch.cat([F_par, F_perp], dim=1))
        return f_sem + F


class OFEBlock(nn.Module):

    def __init__(self, dim, drop_path=0.0, **kwargs):
        super().__init__()
        self.conv = ConvBlock(dim, drop_path=drop_path)
        self.nematic = NematicConv(dim)

    def forward(self, x, Q=None):
        x = self.conv(x)
        x = self.nematic(x, Q)
        return x
