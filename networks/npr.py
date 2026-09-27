# -*- coding: utf-8 -*-
"""NPR: Nematic Prior Refinement — Q-tensor 演化器。

根据当前语义特征和物理场，预测理想的 Q-tensor 状态，软性更新。

设计要点:
1. 跨界合流: 将 2 维物理场 Q 拼入语义流，让 1x1 卷积看清本底物理状态
2. 低秩压缩: dim+2 -> dim//4 -> 2，强制去噪、提炼核心物理驱动力
3. Tanh + get_Qtensor: 输出严格在 [-1, 1] 内，且模长 S <= 1
4. 软门控残差: Q_new = Q + gate * (Q_pred - Q)，凸组合形式
"""

import torch
import torch.nn as nn
from .qtensor import QTensorTools
from .layer import _gn


class NPR(nn.Module):
    """Nematic Prior Refinement.

    功能: 根据当前语义特征 x 和当前物理场 Q，更新 Q-tensor。
    
    更新公式:
        gate = sigmoid(scale) * conv(x)  in [0, 1]
        Q_pred = predictor(cat([x, Q]))  in [-1, 1], 模长 <= 1
        Q_new = Q + gate * (Q_pred - Q)
    
    物理意义:
        Q_pred: 网络推荐的理想 Q 状态
        gate:   门控权重，决定采纳建议的程度
        scale:  全局缩放因子
    """
    def __init__(self, dim, stats_logger=None, name='', **kwargs):
        super().__init__()
        self.stats_logger = stats_logger
        self.name = name

        # 理想 Q 状态预测器
        #   输入: 语义特征 (dim) + 当前物理场 (2) = dim + 2
        #   瓶颈: dim+2 -> dim//4 -> 2 (低秩压缩去噪)
        #   激活: Tanh -> 输出 q1, q2 in [-1, 1]
        self.q_pred = nn.Sequential(
            nn.GroupNorm(_gn(dim + 2), dim + 2), nn.GELU(),
            nn.Conv2d(dim + 2, dim // 4, 1, bias=False),
            nn.GroupNorm(_gn(dim // 4), dim // 4), nn.GELU(),
            nn.Conv2d(dim // 4, 2, 1),
            nn.Tanh(),
        )

        # 全局步长 (sigmoid 激活保证 in [0, 1])
        self.scale = nn.Parameter(torch.tensor(0.0))

        # 局部门控: 基于语义特征决定更新幅度
        self.conv = nn.Sequential(
            nn.Conv2d(dim, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, Q):
        """更新 Q-tensor.

        Args:
            x: 语义特征 (B, dim, H, W)
            Q: 当前物理场 (B, 2, H, W)
        Returns:
            Q_new: 更新后的物理场 (B, 2, H, W), 模长 <= 1
        """
        # 1. 跨界合流: 将物理场 Q 拼入语义特征
        inputs = torch.cat([x, Q], dim=1)  # (B, dim+2, H, W)

        # 2. 预测理想 Q 状态, Tanh 保证 q1, q2 in [-1, 1]
        Q_pred = self.q_pred(inputs)

        # 3. get_Qtensor 将模长 cap 到 1: 保证 S = sqrt(q1**2+q2**2) <= 1
        Q_pred = QTensorTools.get_Qtensor(Q_pred[:, 0:1], Q_pred[:, 1:2])

        # 4. 计算门控: scale 控制全局更新幅度, conv(x) 控制局部置信度
        gate = torch.sigmoid(self.scale) * self.conv(x)

        # 5. 软门控残差更新
        #    gate = 0: 保持当前 Q 不变
        #    gate = 1: 完全采用 Q_pred
        Q_new = Q + gate * (Q_pred - Q)

        # 6. 记录统计信息
        if self.stats_logger is not None:
            S_after = QTensorTools.get_S(Q_new[:, 0:1], Q_new[:, 1:2]).mean().item()
            self.stats_logger.log_q_update(
                self.name, gate.mean().item(),
                torch.sigmoid(self.scale).item(), S_after
            )

        return Q_new, {}