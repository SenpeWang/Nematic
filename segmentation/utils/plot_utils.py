# -*- coding: utf-8 -*-
"""Training curve plotting utilities."""

import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_progress_png(work_dir, epoch_losses, epoch_dices, epoch_times, epoch_lrs):
    """Save training curves: Loss, Dice, Time, LR."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if epoch_losses:
        axes[0, 0].plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=1)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    if epoch_dices:
        axes[0, 1].plot(range(1, len(epoch_dices) + 1), epoch_dices, 'r-', linewidth=1)
    axes[0, 1].set_title('Validation Dice')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].grid(True, alpha=0.3)

    if epoch_times:
        axes[1, 0].bar(range(1, len(epoch_times) + 1), epoch_times, color='g', alpha=0.7)
    axes[1, 0].set_title('Epoch Time (s)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].grid(True, alpha=0.3)

    if epoch_lrs:
        axes[1, 1].plot(range(1, len(epoch_lrs) + 1), epoch_lrs, 'm-', linewidth=1)
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(osp.join(work_dir, 'progress.png'), dpi=150)
    plt.close()
