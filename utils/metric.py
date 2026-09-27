# -*- coding: utf-8 -*-
"""
评估指标模块 (二分类: 前景 vs 背景)

Dice / IoU / clDice / Precision / Recall
所有指标均为 Macro Average (样本间等权)。
"""

import numpy as np
import torch
from skimage.morphology import skeletonize


# ============================================================================
# Dice (Torch Tensor 版, 支持 Batch, 二分类)
# ============================================================================

def calculate_dice(pred, target):
    """
    批量二分类 Dice 指标 (Macro Average)。

    每个样本独立计算前景 Dice，最终取样本间算术平均。
    GT 全 0 且 Pred 全 0 的样本视为无效，不参与均值。

    Args:
        pred:   (B, C, H, W) logits 或 (B, H, W) 类别索引
        target: (B, H, W) 类别索引 {0, 1}
    Returns:
        (dice_sum, valid_count): 前景 Dice 累加和与有效样本数
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    batch_size = pred.shape[0]
    dice_sum, valid = 0.0, 0

    for i in range(batch_size):
        p = (pred[i] > 0).float()   # 前景 mask
        t = (target[i] > 0).float() # 前景 mask

        if p.sum() == 0 and t.sum() == 0:
            continue  # 跳过无效样本

        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice_sum += (2 * inter / union).item()
        valid += 1

    return dice_sum, valid


# ============================================================================
# clDice (中心线 Dice)
# ============================================================================

def cl_dice(pred_bin, gt_bin):
    """
    Centerline Dice: 基于骨架的拓扑连通性指标。
    衡量细长神经元结构的连续性。

    T_prec = |pred ∩ skel(gt)| / |skel(gt)|
    T_sens = |gt ∩ skel(pred)| / |skel(pred)|
    clDice = 2·T_prec·T_sens / (T_prec + T_sens)
    """
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    skel_pred = skeletonize(pred_bin.astype(bool))
    skel_gt = skeletonize(gt_bin.astype(bool))

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    t_prec = np.sum(pred_bin * skel_gt) / np.sum(skel_gt)
    t_sens = np.sum(gt_bin * skel_pred) / np.sum(skel_pred)
    return 2 * t_prec * t_sens / (t_prec + t_sens + 1e-8)

# ============================================================================
# 综合指标 (单样本, 二分类)
# ============================================================================

def calculate_sample_metrics(pred, gt, threshold=0.5):
    """
    计算单样本二分类完整指标集 (Macro Average)。

    GT 全 0 且 Pred 全 0 的样本标记为无效 (valid=False)，不参与均值。

    Args:
        pred:      (H, W) float 预测概率
        gt:        (H, W) {0,1} 标签
        threshold: 二值化阈值
    Returns:
        dict: dice, iou, precision, recall, cldice, valid
    """
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return {
            'dice': 0.0, 'iou': 0.0,
            'precision': 0.0, 'recall': 0.0,
            'cldice': 0.0,
            'valid': False,
        }

    inter = np.sum(pred_bin * gt_bin)
    pred_sum = np.sum(pred_bin)
    gt_sum = np.sum(gt_bin)
    union = pred_sum + gt_sum - inter

    TP = inter
    FP = pred_sum - inter
    FN = gt_sum - inter

    return {
        'dice': (2 * inter) / (pred_sum + gt_sum + 1e-8),
        'iou': inter / (union + 1e-8),
        'precision': TP / (TP + FP + 1e-8),
        'recall': TP / (TP + FN + 1e-8),
        'cldice': cl_dice(pred_bin, gt_bin),
        'valid': True,
    }
