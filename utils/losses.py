# -*- coding: utf-8 -*-
"""
损失函数模块
SegMainLoss (BCE + Dice) + PhysicsInformedLoss (Frank + Flow + Order) + TotalLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_priors import SobelGradient


# ============================================================================
# SegMainLoss (BCE + Dice)
# ============================================================================

class SegMainLoss(nn.Module):
    """
    分割复合损失 = bce_weight × BCE + dice_weight × Dice
    输入: logits (B, 2, H, W)  标签: targets (B, H, W)
    """

    def __init__(self, bce_weight=0.3, dice_weight=0.7, dice_eps=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_eps = dice_eps
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.loss_components = {}

    def compute_dice_loss(self, logits, targets):
        """Soft Dice Loss (前景通道)"""
        probs = F.softmax(logits, dim=1)[:, 1]
        target_flat = targets.float().squeeze(1) if targets.dim() == 4 else targets.float()
        pred_flat = probs.reshape(probs.size(0), -1)
        gt_flat = target_flat.reshape(target_flat.size(0), -1)
        inter = (pred_flat * gt_flat).sum(1)
        union = pred_flat.sum(1) + gt_flat.sum(1)
        return 1 - ((2 * inter + self.dice_eps) / (union + self.dice_eps)).mean()

    def compute_bce_loss(self, logits, targets):
        """BCE (前景-背景差分 logit)"""
        pred = (logits[:, 1] - logits[:, 0]).unsqueeze(1)
        target = targets.float().unsqueeze(1) if targets.dim() == 3 else targets.float()
        return self.bce_fn(pred, target)

    def forward(self, logits, targets):
        bce_loss = self.compute_bce_loss(logits, targets)
        dice_loss = self.compute_dice_loss(logits, targets)
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        self.loss_components = {
            'BCE': (self.bce_weight * bce_loss).item(),
            'Dice': (self.dice_weight * dice_loss).item(),
            'Seg': loss.item(),
        }
        return loss


# ============================================================================
# PhysicsInformedLoss — 流形几何正则化泛函
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    物理驱动正则化损失:

    L_phys = λ_frank·L_Frank + λ_flow·L_Flow + λ_order·L_Order

    ① FrankLoss — 2D 各向异性向列弹性能量 (Anisotropic Frank Energy)
       E = ∫ S² · [K₁₁·(∇·n)² + K₃₃·|n×(∇×n)|²] dΩ
       K₁₁ (Splay, 展曲) 和 K₃₃ (Bend, 弯曲) 为可学习参数。
       神经元纤维因细胞骨架刚性, 弯曲代价远高于展曲 → K₃₃ > K₁₁。
       ∇θ 从 Q-Tensor 梯度解析计算, 避免 atan2 不连续性。
       cosθ, sinθ 从 Q-Tensor 解析提取, 无需 atan2。
       S² 加权: 缺陷处 (S→0) 自然抑制, 有序区 (S→1) 严格平滑。

    ② ManifoldFlowLoss — 测地向列一致性流
       n = (cos θ, sin θ),  θ = 0.5·atan2(Q₁₂, Q₁₁)
       E = Σ[S · |n·∇P|²/(|∇P|+ε)] / Σ[|∇P|>ε处的像素数+ε]
       边缘归一化: 仅在梯度非零区域求均值

    ③ OrderLoss — 序参量双峰极化
       前景: (S-1)², 背景: S², 过渡区: S(1-S)
    """

    def __init__(self, lambda_frank=1.0, lambda_order=0.5,
                 lambda_flow=1.0, k11_init=1.0, k33_init=1.0,
                 s_threshold=0.3):
        super().__init__()
        self.lambda_frank = lambda_frank
        self.lambda_order = lambda_order
        self.lambda_flow = lambda_flow
        self.s_threshold = s_threshold
        self.sobel = SobelGradient()
        self.loss_components = {}

        # 可学习弹性常数 (softplus 保证正定)
        self.log_K11 = nn.Parameter(torch.tensor(float(k11_init)).log())
        self.log_K33 = nn.Parameter(torch.tensor(float(k33_init)).log())

    def compute_frank_loss(self, S, Q1, Q2):
        """
        2D 各向异性向列弹性能量:
          直接通过 Q-Tensor 代数计算 S²(∇·n)² 和 S²|n×(∇×n)|²
          无需重构存在符号歧义的 cosθ, sinθ，彻底规避 atan2 与 sqrt。
        """
        gx1, gy1 = self.sobel(Q1)
        gx2, gy2 = self.sobel(Q2)

        # 解析 ∇θ (避免 atan2 不连续性)
        Q_sq = Q1.pow(2) + Q2.pow(2) + 1e-8
        dtheta_dx = 0.5 * (Q1 * gx2 - Q2 * gx1) / Q_sq
        dtheta_dy = 0.5 * (Q1 * gy2 - Q2 * gy1) / Q_sq

        # 向列相纯张量展开:
        # S²(Splay)² = 0.5 S²(|∇θ|²) - S [0.5 Q1 (θ_x² - θ_y²) + Q2 θ_x θ_y]
        # S²(Bend)²  = 0.5 S²(|∇θ|²) + S [0.5 Q1 (θ_x² - θ_y²) + Q2 θ_x θ_y]
        A_term = 0.5 * S.pow(2) * (dtheta_dx.pow(2) + dtheta_dy.pow(2))
        B_term = S * (0.5 * Q1 * (dtheta_dx.pow(2) - dtheta_dy.pow(2)) + Q2 * dtheta_dx * dtheta_dy)

        # 可学习弹性常数 (softplus 保证正定)
        K11 = F.softplus(self.log_K11)
        K33 = F.softplus(self.log_K33)

        # S² 加权各向异性弹性能量
        energy = K11 * (A_term - B_term) + K33 * (A_term + B_term)
        return energy.mean()

    def compute_flow_loss(self, S, Q1, Q2, logits):
        """
        测地向列一致性流:
        完全使用向列相 Q-Tensor 代数展开，规避 atan2。
        S(n·∇P)² = 0.5 S |∇P|² + 0.5 Q1(Px² - Py²) + Q2 Px Py
        """
        mask_prob = torch.sigmoid(logits[:, 1:2] - logits[:, 0:1])
        mgx, mgy = self.sobel(mask_prob)
        grad_mag = torch.sqrt(mgx.pow(2) + mgy.pow(2) + 1e-8)

        term1 = 0.5 * S * (mgx.pow(2) + mgy.pow(2))
        term2 = 0.5 * Q1 * (mgx.pow(2) - mgy.pow(2))
        term3 = Q2 * mgx * mgy

        energy = (term1 + term2 + term3) / (grad_mag + 1e-4)

        # 边缘归一化: 仅在梯度明显区域统计
        edge_mask = (grad_mag > 1e-3).float()
        edge_count = edge_mask.sum() + 1e-6
        return (energy * edge_mask).sum() / edge_count

    def compute_order_loss(self, S, targets):
        """
        序参量双峰极化:
          前景: (S-1)² → 驱动高有序度
          背景:  S²    → 驱动低有序度
          过渡区: S(1-S) → 惩罚模糊边界
        """
        s = S.squeeze(1)
        fg = (targets == 1).float()
        bg = (targets == 0).float()

        loss_fg = ((s - 1).pow(2) * fg).sum() / (fg.sum() + 1e-8)
        loss_bg = (s.pow(2) * bg).sum() / (bg.sum() + 1e-8)

        mid_mask = ((s > self.s_threshold) & (s < 1 - self.s_threshold)).float()
        loss_bimodal = (mid_mask * s * (1 - s)).mean()
        return loss_fg + loss_bg + loss_bimodal

    def forward(self, logits, targets, S, Q1, Q2):
        """
        Args:
            logits: (B, 2, H, W) 分割 logits
            targets: (B, H, W)
            S:  (B, 1, H, W) 序参量
            Q1: (B, 1, H, W) Q-Tensor Q₁₁
            Q2: (B, 1, H, W) Q-Tensor Q₁₂
        """
        target_mask = targets.squeeze(1) if targets.dim() == 4 else targets

        frank_loss = self.compute_frank_loss(S, Q1, Q2)
        flow_loss = self.compute_flow_loss(S, Q1, Q2, logits)
        order_loss = self.compute_order_loss(S, target_mask)

        frank_weighted = self.lambda_frank * frank_loss
        flow_weighted = self.lambda_flow * flow_loss
        order_weighted = self.lambda_order * order_loss
        total = frank_weighted + flow_weighted + order_weighted

        self.loss_components = {
            'Frank': frank_weighted.item(),
            'Flow': flow_weighted.item(),
            'Order': order_weighted.item(),
            'Phys': total.item(),
        }
        return total


# ============================================================================
# PriorDistillLoss — 物理先验蒸馏损失
# ============================================================================

class PriorDistillLoss(nn.Module):
    """
    使 PhysicsPriorStem 输出的网络软标签与预处理（结构张量技术）计算的明确伪标签完全对齐。
    采用 L2（MSE）损失函数计算蒸馏。
    """
    def __init__(self, lambda_distill=10.0):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.loss_components = {}

    def forward(self, S_prior, Q1_prior, Q2_prior, S_target, Q1_target, Q2_target):
        loss_S = F.mse_loss(S_prior, S_target)
        loss_Q1 = F.mse_loss(Q1_prior, Q1_target)
        loss_Q2 = F.mse_loss(Q2_prior, Q2_target)
        
        total = self.lambda_distill * (loss_S + loss_Q1 + loss_Q2)
        
        self.loss_components = {
            'Distill': total.item()
        }
        return total


# ============================================================================
# TotalLoss — 联合损失
# ============================================================================

class TotalLoss(nn.Module):
    """L_total = λ_seg · L_Seg + L_Phys"""

    def __init__(self, config):
        super().__init__()
        self.seg_loss = SegMainLoss(
            bce_weight=config.loss_weight[0],
            dice_weight=config.loss_weight[1],
            dice_eps=config.dice_epsilon,
        )
        self.phys_loss = PhysicsInformedLoss(
            lambda_frank=config.lambda_frank,
            lambda_order=config.lambda_order,
            lambda_flow=config.lambda_flow,
            k11_init=getattr(config, 'k11_init', 1.0),
            k33_init=getattr(config, 'k33_init', 1.0),
            s_threshold=getattr(config, 's_threshold', 0.3),
        )
        self.distill_loss = PriorDistillLoss(
            lambda_distill=config.lambda_distill
        )
        self.lambda_seg = config.lambda_seg
        self.loss_components = {}

    def forward(self, outputs, targets, batch=None):
        """
        Args:
            outputs: dict {'logits', 'S', 'Q11', 'Q12', 'S_prior', 'Q1_prior', 'Q2_prior'}
            targets: (B, H, W)
            batch: dict 包含目标的 'S_target', 'Q1_target', 'Q2_target'
        """
        logits = outputs['logits']
        S = outputs['S']
        Q1 = outputs['Q11']
        Q2 = outputs['Q12']

        seg = self.seg_loss(logits, targets)
        phys = self.phys_loss(logits, targets, S, Q1, Q2)
        
        distill = self.distill_loss(
            outputs['S_prior'], outputs['Q1_prior'], outputs['Q2_prior'],
            batch['S_target'], batch['Q1_target'], batch['Q2_target']
        )

        total = self.lambda_seg * seg + phys + distill

        self.loss_components = {}
        for k, v in self.seg_loss.loss_components.items():
            self.loss_components[k] = self.lambda_seg * v
        self.loss_components.update(self.phys_loss.loss_components)
        self.loss_components.update(self.distill_loss.loss_components)
        self.loss_components['Total'] = total.item()
        return total


def get_loss_function(config):
    return TotalLoss(config)
