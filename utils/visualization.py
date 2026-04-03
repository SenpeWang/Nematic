# -*- coding: utf-8 -*-
"""
可视化工具模块
预测对比图 (2×2) / 向列序场 (2×2) / 训练曲线
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================================
# plot_predictions — 2×2 分割对比
# ============================================================================

def plot_predictions(image, gt_mask, pred_mask, save_path,
                     title='', channel_idx=0):
    """
    2×2 分割对比:
      ┌─────────────┬─────────────┐
      │  Input (ch)  │     GT      │
      ├─────────────┼─────────────┤
      │  GT + Pred   │   Pred      │
      │  Overlay     │             │
      └─────────────┴─────────────┘
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    img_show = image[channel_idx] if image.ndim == 3 else image

    # Input
    axes[0, 0].imshow(img_show, cmap='gray')
    axes[0, 0].set_title('Input', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # GT
    axes[0, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # Overlay: GT(绿) + Pred(红), 重叠(黄)
    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)
    pred_bin = (pred_mask > 0).astype(np.float32)
    overlay[..., 0] = pred_bin          # R = Pred
    overlay[..., 1] = gt_bin            # G = GT
    overlay[..., 2] = 0
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (G=GT, R=Pred)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # Pred
    axes[1, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# plot_nematic_field — 2×2 向列序场
# ============================================================================

def plot_nematic_field(S, Q1, Q2, save_path, gt_mask=None,
                       stride=8, title=''):
    """
    2×2 向列序场:
      ┌──────────────┬──────────────┐
      │ S 热力图      │ θ 方向角图    │
      ├──────────────┼──────────────┤
      │ Director 矢量 │ S + GT 等高线 │
      └──────────────┴──────────────┘
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    H, W = S.shape

    theta = 0.5 * np.arctan2(Q2, Q1 + 1e-8)

    # S 热力图
    im0 = axes[0, 0].imshow(S, cmap='inferno', vmin=0, vmax=1)
    axes[0, 0].set_title('Order Parameter S', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # θ 方向角图 (周期色图)
    im1 = axes[0, 1].imshow(theta, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    axes[0, 1].set_title('Director Angle θ', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Director 矢量场 (Quiver)
    dx = np.cos(theta)
    dy = np.sin(theta)
    ys = np.arange(stride // 2, H, stride)
    xs = np.arange(stride // 2, W, stride)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    U = dx[Y, X] * S[Y, X]
    V = dy[Y, X] * S[Y, X]

    axes[1, 0].imshow(S, cmap='gray_r', vmin=0, vmax=1, alpha=0.3)
    axes[1, 0].quiver(X, Y, U, -V,
                       S[Y, X], cmap='coolwarm',
                       scale=25, width=0.003, clim=(0, 1))
    axes[1, 0].set_title('Director Field', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # S + GT 等高线
    axes[1, 1].imshow(S, cmap='inferno', vmin=0, vmax=1, alpha=0.7)
    if gt_mask is not None:
        axes[1, 1].contour(gt_mask, levels=[0.5], colors='lime', linewidths=1.5)
        axes[1, 1].set_title('S + GT Contour', fontsize=11, fontweight='bold')
    else:
        axes[1, 1].set_title('S Field', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# plot_loss_curves — 训练曲线
# ============================================================================

def plot_loss_curves(history, save_path, title='Training Curves'):
    """绘制 loss 和评估指标变化折线图"""
    loss_keys = [k for k in history if 'loss' in k.lower()]
    metric_keys = [k for k in history if k not in loss_keys and len(history[k]) > 0]

    n_panels = sum([bool(loss_keys), bool(metric_keys)])
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    if loss_keys:
        ax = axes[panel_idx]
        for k in loss_keys:
            ax.plot(range(1, len(history[k]) + 1), history[k], label=k, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    if metric_keys:
        ax = axes[panel_idx]
        for k in metric_keys:
            ax.plot(range(1, len(history[k]) + 1), history[k], label=k, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Metrics')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
