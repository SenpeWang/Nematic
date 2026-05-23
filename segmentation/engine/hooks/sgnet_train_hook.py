# -*- coding: utf-8 -*-
"""SGNet3 custom training hook: early stop, best-dice save, progress curves."""

import os
import os.path as osp
import time

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from ...utils.plot_utils import save_progress_png


@HOOKS.register_module()
class SGNetTrainHook(Hook):
    """Custom training hook preserving SGNet3 logic."""

    priority = 'LOW'

    def __init__(self, start_epoch=70, patience=20):
        self.start_epoch = start_epoch
        self.patience = patience

        self.best_dice = 0.0
        self.best_loss = float('inf')
        self.no_improve_count = 0
        self.early_stopped = False

        self.epoch_losses = []
        self.epoch_dices = []
        self.epoch_times = []
        self.epoch_lrs = []
        self.current_epoch = 0
        self.epoch_start_time = None
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        self.val_loss_sum = 0.0
        self.val_loss_count = 0

        self.work_dir = None
        self.ckpt_dir = None
        self.log_path = None
        self.metrics_path = None
        self.iters_per_epoch = None

    def before_run(self, runner):
        self.work_dir = runner.work_dir
        self.ckpt_dir = osp.join(self.work_dir, 'checkpoints')
        self.log_path = osp.join(self.work_dir, 'log.txt')
        self.metrics_path = osp.join(self.work_dir, 'metrics.txt')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Compute iters per epoch from dataloader
        num_samples = len(runner.train_dataloader.dataset)
        batch_size = runner.train_dataloader.batch_size
        self.iters_per_epoch = num_samples // batch_size

        self.epoch_start_time = time.time()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if outputs and 'loss' in outputs:
            loss_val = outputs['loss'].item() if isinstance(outputs['loss'], torch.Tensor) else float(outputs['loss'])
        else:
            try:
                loss_val = runner.message_hub.get_scalar('train/loss').current()
            except KeyError:
                loss_val = 0.0
        self.epoch_loss_sum += loss_val
        self.epoch_loss_count += 1

    def after_val_epoch(self, runner, metrics=None):
        """Record metrics, check early stop, save best model."""
        # Read Dice directly from evaluator metrics
        dice_val = 0.0
        if metrics:
            for key, val in metrics.items():
                if 'mDice' in key:
                    dice_val = val / 100.0 if val > 1 else val
                    break

        avg_loss = self.epoch_loss_sum / max(self.epoch_loss_count, 1)
        try:
            lr_val = runner.optim_wrapper.get_lr()
            if isinstance(lr_val, dict):
                lr_val = list(lr_val.values())[0]
            while isinstance(lr_val, (list, tuple)):
                lr_val = lr_val[0]
            lr_val = float(lr_val)
        except Exception:
            lr_val = 0.0
        elapsed = time.time() - self.epoch_start_time

        self.epoch_losses.append(avg_loss)
        self.epoch_dices.append(dice_val)
        self.epoch_lrs.append(lr_val)
        self.epoch_times.append(elapsed)
        self.current_epoch = (runner.iter + 1) // self.iters_per_epoch

        # Attempt to get validation loss from message_hub (key may vary by mmengine version)
        val_loss = None
        for key in ('val/loss', 'loss', 'train/loss'):
            try:
                val_loss = runner.message_hub.get_scalar(key).current()
                break
            except KeyError:
                continue
        if val_loss is None:
            val_loss = avg_loss  # fallback to training loss for logging continuity

        improved_dice = dice_val > self.best_dice
        improved_loss = val_loss < self.best_loss

        if improved_dice:
            self.best_dice = dice_val
            self.no_improve_count = 0
            torch.save(runner.model.state_dict(), osp.join(self.ckpt_dir, 'best_dice_model.pth'))
        else:
            self.no_improve_count += 1

        if improved_loss:
            self.best_loss = val_loss

        torch.save(runner.model.state_dict(), osp.join(self.ckpt_dir, 'last_model.pth'))

        save_progress_png(self.work_dir, self.epoch_losses, self.epoch_dices,
                          self.epoch_times, self.epoch_lrs)

        with open(self.log_path, 'a') as f:
            f.write(f'[Epoch {self.current_epoch}] loss={avg_loss:.6f}, '
                    f'dice={dice_val:.6f}, lr={lr_val:.8f}, time={elapsed:.1f}s\n')

        with open(self.metrics_path, 'a') as f:
            f.write(f'[Epoch {self.current_epoch}] Val Dice={dice_val:.4f}, Val Loss={val_loss:.4f}\n')
            if improved_dice:
                f.write(f'  >> New Best Dice at Epoch {self.current_epoch}: Dice={dice_val:.4f}\n')
            if improved_loss:
                f.write(f'  * New Best Val Loss at Epoch {self.current_epoch}: Val Loss={val_loss:.4f}\n')

        print(f'[Epoch {self.current_epoch}] Loss={avg_loss:.4f}, '
              f'Dice={dice_val:.4f}, LR={lr_val:.8f}, Time={elapsed:.1f}s')

        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        self.val_loss_sum = 0.0
        self.val_loss_count = 0
        self.epoch_start_time = time.time()

        if (self.current_epoch >= self.start_epoch
                and self.no_improve_count >= self.patience):
            self.early_stopped = True
            print(f'Early stopping at epoch {self.current_epoch} (best_dice={self.best_dice:.6f})')
            runner._max_iters = runner.iter + 1

    def after_run(self, runner):
        summary_path = osp.join(self.work_dir, 'train_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('=' * 60 + '\n')
            f.write('Training Summary\n')
            f.write('=' * 60 + '\n')
            f.write(f'Work Dir: {self.work_dir}\n')
            f.write(f'Total Epochs: {self.current_epoch}\n')
            f.write(f'Best Dice: {self.best_dice:.6f}\n')
            f.write(f'Early Stopped: {self.early_stopped}\n')
            f.write('=' * 60 + '\n')
        save_progress_png(self.work_dir, self.epoch_losses, self.epoch_dices,
                          self.epoch_times, self.epoch_lrs)
