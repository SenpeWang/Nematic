# -*- coding: utf-8 -*-
"""
评估指标模块
Dice / IoU / clDice (中心线Dice)
"""

import numpy as np
import torch
from skimage.morphology import skeletonize


# ============================================================================
# Dice (Torch Tensor 版, 支持 Batch)
# ============================================================================

def calculate_dice(pred, target, num_classes=2):
    """
    批量 Dice 指标。
    Args:
        pred: (B, C, H, W) logits 或 (B, H, W) 类别索引
        target: (B, H, W) 类别索引
    Returns:
        (dice_sum, valid_count): 前景类 Dice 累加和有效样本数
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    batch_size = pred.shape[0]
    dice_sum, valid = 0.0, 0

    for i in range(batch_size):
        p, t = pred[i], target[i]
        if (p == 0).all() and (t == 0).all():
            continue

        class_dices = []
        for c in range(1, num_classes):
            pc = (p == c).float()
            tc = (t == c).float()
            inter = (pc * tc).sum()
            union = pc.sum() + tc.sum()
            if union > 0:
                class_dices.append((2 * inter / union).item())
            else:
                class_dices.append(1.0 if tc.sum() == 0 else 0.0)

        if class_dices:
            dice_sum += np.mean(class_dices)
            valid += 1

    return dice_sum, valid


# ============================================================================
# IoU (Numpy 版)
# ============================================================================

def calculate_iou(pred_bin, gt_bin):
    """交并比"""
    inter = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - inter
    return inter / (union + 1e-8)


# ============================================================================
# clDice (中心线 Dice)
# ============================================================================

def cl_dice(pred_bin, gt_bin):
    """
    Centerline Dice: 基于骨架的拓扑连通性指标。
    衡量细长神经元结构的连续性。

    T_prec = |skel(pred) ∩ gt| / |skel(pred)|
    T_sens = |skel(gt) ∩ pred| / |skel(gt)|
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

    t_prec = np.sum(skel_pred * gt_bin) / np.sum(skel_pred)
    t_sens = np.sum(skel_gt * pred_bin) / np.sum(skel_gt)
    return 2 * t_prec * t_sens / (t_prec + t_sens + 1e-8)

# ============================================================================
# 综合指标 (单样本)
# ============================================================================

def calculate_sample_metrics(pred, gt, threshold=0.5):
    """
    计算单样本完整指标集。
    Args:
        pred: (H, W) float 预测概率
        gt:   (H, W) {0,1} 标签
    Returns:
        dict: dice, iou, precision, recall, cldice, valid
              valid=False 表示 GT=0 且 Pred=0，不应参与均值计算
    """
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    # GT=0 且 Pred=0: 标记无效，不参与均值
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
