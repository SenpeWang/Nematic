# -*- coding: utf-8 -*-
"""Loss functions for NFAM-Net.

- Seg_loss: BCE + Dice segmentation loss (确定)
- SSupervisionLoss: S序参数监督损失 (可更新)
- NFAMNet_loss: 总损失封装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.qtensor import QTensorTools


class Seg_loss(nn.Module):
    """BCE + Dice segmentation loss."""
    def __init__(self, bce_weight=0.3, dice_weight=0.7, dice_eps=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_eps = dice_eps
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        pred = (logits[:, 1] - logits[:, 0]).unsqueeze(1)
        target = targets.float().unsqueeze(1) if targets.dim() == 3 else targets.float()
        bce = self.bce_fn(pred, target)
        probs = F.softmax(logits, dim=1)[:, 1]
        gt = targets.float().squeeze(1) if targets.dim() == 4 else targets.float()
        p_flat = probs.reshape(probs.size(0), -1)
        g_flat = gt.reshape(gt.size(0), -1)
        inter = (p_flat * g_flat).sum(1)
        union = p_flat.sum(1) + g_flat.sum(1)
        dice = 1.0 - ((2.0 * inter + self.dice_eps) / (union + self.dice_eps)).mean()
        return self.bce_weight * bce + self.dice_weight * dice



def frank_oseen_loss_2d(Q, mask=None):
    """2D Frank-Oseen 能量损失

    只考虑展曲和弯曲,只在前景区域计算
    """
    q1, q2 = Q[:, 0:1], Q[:, 1:2]

    # 计算 Q 的 2D 梯度
    dq1_dx = q1[:, :, :, 1:] - q1[:, :, :, :-1]
    dq2_dx = q2[:, :, :, 1:] - q2[:, :, :, :-1]
    dq1_dy = q1[:, :, 1:, :] - q1[:, :, :-1, :]
    dq2_dy = q2[:, :, 1:, :] - q2[:, :, :-1, :]

    # 裁剪到相同大小
    min_h = min(dq1_dx.shape[2], dq1_dy.shape[2])
    min_w = min(dq1_dx.shape[3], dq1_dy.shape[3])

    dq1_dx = dq1_dx[:, :, :min_h, :min_w]
    dq2_dx = dq2_dx[:, :, :min_h, :min_w]
    dq1_dy = dq1_dy[:, :, :min_h, :min_w]
    dq2_dy = dq2_dy[:, :, :min_h, :min_w]

    # 弹性能量: 每个像素的能量
    energy = dq1_dx**2 + dq2_dx**2 + dq1_dy**2 + dq2_dy**2

    # 只在前景区域计算
    if mask is not None:
        mask_2d = mask[:, :, :min_h, :min_w]
        energy = energy * mask_2d  # 只保留前景
        fg_count = mask_2d.sum().clamp(min=1)
        energy = energy.sum() / fg_count  # 只对前景像素平均
    else:
        energy = energy.mean()

    return energy


def _compute_mask_and_stats(S, targets):
    """计算 GT mask (下采样到 S 分辨率) 和 S 统计信息."""
    height, width = S.shape[2], S.shape[3]
    if targets.shape[1] != height or targets.shape[2] != width:
        mask = F.interpolate(targets.float().unsqueeze(1), size=(height, width), mode="nearest")
    else:
        mask = targets.float().unsqueeze(1) if targets.dim() == 3 else targets.float()

    foreground = (mask > 0.5).float()
    background = 1.0 - foreground
    fg_count = foreground.sum().clamp(min=1)
    bg_count = background.sum().clamp(min=1)

    return mask, foreground, background, fg_count, bg_count


def s_supervision_loss(Q_pred, targets, S, fg_weight=1.0, bg_weight=1.0):
    """S序参数监督损失: 前景S->1, 背景S->0.

    Args:
        Q_pred: (B, 2, H, W) Q-tensor
        targets: (B, H, W) 标签
        fg_weight: 前景损失权重
        bg_weight: 背景损失权重
        S: 预计算的序参量 (避免重复计算)
    Returns:
        loss: 标量损失
        components: 统计信息字典
    """

    _, foreground, background, fg_count, bg_count = _compute_mask_and_stats(S, targets)

    # 前景: S应接近1
    loss_foreground = ((1 - S) ** 2 * foreground).sum() / fg_count
    # 背景: S应接近0
    loss_background = (S ** 2 * background).sum() / bg_count

    loss = fg_weight * loss_foreground + bg_weight * loss_background

    components = {
        "S_foreground_loss": loss_foreground.item(),
        "S_background_loss": loss_background.item(),
        "S_mean": S.mean().item(),
        "S_foreground_mean": (S * foreground).sum().div(fg_count).item(),
        "S_background_mean": (S * background).sum().div(bg_count).item(),
    }

    return loss, components


class Nematic_loss(nn.Module):
    """向列相损失: S 监督 + Frank-Oseen 能量

    Args:
        stage_weights: 各stage权重, 内部相加为1
        fg_weight: 前景损失权重
        bg_weight: 背景损失权重
        fo_weight: Frank-Oseen 能量权重
    """
    def __init__(self, stage_weights=(0.4, 0.3, 0.2, 0.1), fg_weight=1.0, bg_weight=1.0, fo_weight=1.0):
        super().__init__()
        self.stage_weights = list(stage_weights)
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.fo_weight = fo_weight

    def forward(self, stage_Q, targets):
        device = stage_Q[0].device
        total_s_loss = torch.tensor(0.0, device=device)
        total_fo_loss = torch.tensor(0.0, device=device)
        components = {}

        for stage_index, Q_pred in enumerate(stage_Q):
            stage_weight = self.stage_weights[stage_index] if stage_index < len(self.stage_weights) else 0.0
            if stage_weight <= 0:
                continue

            # 每个 stage 只计算一次 S
            S = QTensorTools.get_S(Q_pred[:, 0:1], Q_pred[:, 1:2])
            mask, _, _, _, _ = _compute_mask_and_stats(S, targets)

            # S 监督损失 (传入预计算的 S, 避免重复)
            s_loss, s_components = s_supervision_loss(
                Q_pred, targets, S, self.fg_weight, self.bg_weight)
            total_s_loss = total_s_loss + stage_weight * s_loss

            # Frank-Oseen 能量损失 (2D 切片)
            fo_loss = frank_oseen_loss_2d(Q_pred, mask)
            total_fo_loss = total_fo_loss + stage_weight * self.fo_weight * fo_loss

            components[f"S/stage{stage_index}/fg_loss"] = s_components["S_foreground_loss"]
            components[f"S/stage{stage_index}/bg_loss"] = s_components["S_background_loss"]
            components[f"S/stage{stage_index}/fg_mean"] = s_components["S_foreground_mean"]
            components[f"S/stage{stage_index}/bg_mean"] = s_components["S_background_mean"]
            components[f"S/stage{stage_index}/mean"] = s_components["S_mean"]
            components[f"FO/stage{stage_index}"] = (stage_weight * self.fo_weight * fo_loss).item()

        # 分离 S 和 FO 损失
        components["S"] = total_s_loss.item()
        components["FO"] = total_fo_loss.item()
        components["Nematic"] = (total_s_loss + total_fo_loss).item()

        return total_s_loss + total_fo_loss, components


class NFAMNet_loss(nn.Module):
    """NFAM-Net总损失: Seg + Nematic."""

    # 损失函数参数 (直接定义，不从config读取)
    LAMBDA_SEG = 1.0           # 分割损失权重
    LAMBDA_NEMATIC = 0.3       # 向列相损失权重
    LOSS_WEIGHT = [0.3, 0.7]   # [bce_weight, dice_weight]
    Q_STAGE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # 各stage权重 (内部相加为1)

    def __init__(self):
        super().__init__()

        # 分割损失 (确定)
        self.seg_loss = Seg_loss(self.LOSS_WEIGHT[0], self.LOSS_WEIGHT[1])

        # 向列相损失 (S 监督 + Frank-Oseen 能量)
        self.s_loss = Nematic_loss(stage_weights=self.Q_STAGE_WEIGHTS, fo_weight=1.0)

        self.loss_components = {}

    def forward(self, outputs, targets):
        # 分割损失
        seg = self.seg_loss(outputs["logits"], targets)

        # 向列相损失 (S 统计已在内部一并计算, 无需重算)
        s_loss, s_comps = self.s_loss(outputs["stage_Q"], targets)

        # 总损失
        total = self.LAMBDA_SEG * seg + self.LAMBDA_NEMATIC * s_loss

        self.loss_components = {
            "Total": total.item(),
            "Seg": (self.LAMBDA_SEG * seg).item(),
            "S_w": (self.LAMBDA_NEMATIC * s_loss).item(),
            **s_comps,
        }
        return total


def get_loss_function(config=None):
    return NFAMNet_loss()
