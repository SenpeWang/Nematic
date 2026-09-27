# -*- coding: utf-8 -*-
"""NFAM-Net training script."""

import os
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import warnings
import numpy as np
from collections import OrderedDict
import yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.config import config
from datasets.dataset import UniversalDataset
from networks.model import build_Model
from utils.losses import get_loss_function
from utils.stats_logger import NPRStats
from utils.metric import calculate_dice
from utils.visualization import plot_loss_curves

warnings.filterwarnings('ignore')


def cast_outputs_to_float(outputs):
    out_f32 = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            out_f32[k] = v.float()
        elif isinstance(v, list):
            out_f32[k] = [t.float() if isinstance(t, torch.Tensor) else t for t in v]
        else:
            out_f32[k] = v
    return out_f32


def save_training_state(path, model, optimizer, scheduler, scaler, epoch, best_dice, best_epoch, best_loss, best_loss_epoch, patience_counter, history):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'best_loss_epoch': best_loss_epoch,
        'patience_counter': patience_counter,
        'history': history,
    }, path)


def load_training_state(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return (
        ckpt.get('epoch', 0),
        ckpt.get('best_dice', 0.0),
        ckpt.get('best_epoch', 0),
        ckpt.get('best_loss', float('inf')),
        ckpt.get('best_loss_epoch', 0),
        ckpt.get('patience_counter', 0),
        ckpt.get('history', {'train_loss': [], 'val_loss': [], 'val_dice': []}),
    )


def train_epoch(loader, model, criterion, optimizer, scheduler, scaler, epoch, config):
    model.train()
    losses = []
    loss_meters = {}
    accum = max(int(config.grad_accum), 1)
    dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f'Train E{epoch}')
    for step, batch in enumerate(pbar):
        images = batch['image'].cuda(non_blocking=True).float()
        targets = batch['label'].cuda(non_blocking=True).long()
        with autocast(device_type="cuda", enabled=config.amp, dtype=dtype):
            outputs = model(images)
            outputs = cast_outputs_to_float(outputs)
            loss = criterion(outputs, targets)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / accum).backward()
        do_step = ((step + 1) % accum == 0) or ((step + 1) == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            has_nan = any(
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                for p in model.parameters() if p.grad is not None
            )
            if has_nan:
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 49:
            torch.cuda.empty_cache()
        for k, v in criterion.loss_components.items():
            if isinstance(v, (float, int)):
                loss_meters.setdefault(k, []).append(float(v))
        pbar.set_postfix(OrderedDict(loss=f'{np.mean(losses):.4f}'))
    scheduler.step()
    mean_loss = float(np.mean(losses)) if losses else float('nan')
    mean_comps = {k: float(np.mean(v)) for k, v in loss_meters.items()}
    return mean_loss, mean_comps


@torch.no_grad()
def valid_epoch(loader, model, criterion, epoch, config):
    model.eval()
    losses = []
    dice_sum, dice_valid = 0.0, 0
    dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16
    for batch in tqdm(loader, desc=f'Val E{epoch}'):
        images = batch['image'].cuda(non_blocking=True).float()
        targets = batch['label'].cuda(non_blocking=True).long()
        outputs = cast_outputs_to_float(model(images))
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        d_sum, d_cnt = calculate_dice(outputs["logits"], targets)
        dice_sum += d_sum
        dice_valid += d_cnt
    val_dice = dice_sum / dice_valid if dice_valid > 0 else 0.0
    return float(np.mean(losses)), float(val_dice)


def main():
    parser = argparse.ArgumentParser(description='NFAM-Net Training')
    parser.add_argument('--dataset', type=str, default='NEURO', choices=['NEURO', 'SY5Y', 'NEURITE', 'LS'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'ex'])
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from (model .pth or full training state .pt)')
    parser.add_argument('--diag', action='store_true', help='Append _diag to work_dir for diagnostic runs')
    args = parser.parse_args()

    config.update_for_dataset(args.dataset)
    suffix = '_ex' if args.mode == 'ex' else ''
    if args.diag:
        suffix += '_diag'
    config.work_dir = f'./Outputs/Train/{config.timestamp}_{config.project_name}_{config.dataset_name.lower()}{suffix}/'
    if args.epochs:
        config.epochs = args.epochs
        config.T_max = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr

    os.makedirs(config.work_dir, exist_ok=True)
    ckpt_dir = os.path.join(config.work_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(config.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print('#---------- Loading Data ----------#')
    if args.mode == 'ex':
        print('[EX Mode] Training on val set, evaluating on test set.')
        train_ds = UniversalDataset(config.dataset_name, 'val', transform=True)
        val_ds = UniversalDataset(config.dataset_name, 'test', transform=False)
    else:
        train_ds = UniversalDataset(config.dataset_name, 'train', transform=True)
        val_ds = UniversalDataset(config.dataset_name, 'val', transform=False)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    print('#---------- Building NFAM-Net ----------#')
    tb_logger = NPRStats(log_dir=os.path.join(config.work_dir, 'tensorboard'))
    model = build_Model(config, dataset_name=args.dataset, stats_logger=tb_logger).cuda().float()
    # Force float32
    torch.set_default_dtype(torch.float32)
    # Force all parameters to float32
    for p in model.parameters():
        p.data = p.data.float()
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f'Trainable Params: {trainable_p / 1e6:.2f}M | Total: {total_p / 1e6:.2f}M')

    criterion = get_loss_function().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
    scaler = GradScaler("cuda", enabled=(config.amp and config.amp_dtype == "float16"), init_scale=2.**16 if config.loss_scale == "dynamic" else 1.0)

    best_dice, best_epoch = 0.0, 0
    best_loss, best_loss_epoch = float('inf'), 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    start_epoch = 1

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        if resume_path.endswith('.pt'):
            start_epoch, best_dice, best_epoch, best_loss, best_loss_epoch, patience_counter, history = load_training_state(
                resume_path, model, optimizer, scheduler, scaler,
            )
            start_epoch += 1
            print(f'[RESUME] Full training state from {resume_path}, starting at E{start_epoch}')
        else:
            sd = torch.load(resume_path, map_location='cuda', weights_only=True)
            model.load_state_dict(sd)
            print(f'[RESUME] Model weights only from {resume_path}, starting at E1')

    print('#---------- Training ----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        tb_logger.set_step(epoch)
        train_loss, train_comps = train_epoch(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, config)
        for k, v in train_comps.items():
            tb_logger.scalar(f'loss/{k}', v)
        tb_logger.scalar('train/lr', optimizer.param_groups[0]['lr'])
        tb_logger.flush_epoch()
        val_loss, val_dice = valid_epoch(val_loader, model, criterion, epoch, config)
        torch.cuda.empty_cache()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        tag = ' [EX]' if args.mode == 'ex' else ''
        print(f'E{epoch}{tag} | dice={val_dice:.4f} | train loss={train_loss:.4f} | val loss={val_loss:.4f}')
        with open(os.path.join(config.work_dir, 'metrics.txt'), 'a') as f:
            f.write(f'[Epoch {epoch}]{tag} Val Dice={val_dice:.4f}, Val Loss={val_loss:.4f}\n')
        with open(os.path.join(config.work_dir, 'loss_details.txt'), 'a') as f:
            details = ', '.join([f'{k}={v:.4f}' for k, v in train_comps.items()])
            f.write(f'[Epoch {epoch}]{tag} Train Loss={train_loss:.4f}, {details}\n')
        if args.mode != 'ex':
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last_model.pth'))
            save_training_state(
                os.path.join(ckpt_dir, 'last_training_state.pt'),
                model, optimizer, scheduler, scaler, epoch,
                best_dice, best_epoch, best_loss, best_loss_epoch, patience_counter, history,
            )

        improved_dice = val_dice > best_dice
        improved_loss = val_loss < best_loss

        if improved_dice:
            best_dice, best_epoch = val_dice, epoch
            patience_counter = 0
            if args.mode != 'ex':
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_dice_model.pth'))
            print(f'>> New Best Dice (Dice={val_dice:.4f})')
            with open(os.path.join(config.work_dir, 'metrics.txt'), 'a') as f:
                f.write(f'  >> New Best Dice at Epoch {epoch}: Dice={val_dice:.4f}\n')
        elif epoch >= config.early_stopping_start:
            patience_counter += 1

        if improved_loss:
            best_loss, best_loss_epoch = val_loss, epoch
            if args.mode != 'ex':
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_loss_model.pth'))
            print(f' * New Best Loss (Val Loss={val_loss:.4f})')
            with open(os.path.join(config.work_dir, 'metrics.txt'), 'a') as f:
                f.write(f'  * New Best Val Loss at Epoch {epoch}: Val Loss={val_loss:.4f}\n')
        if args.mode != 'ex' and epoch % config.val_interval == 0:
            plot_loss_curves(history, os.path.join(config.work_dir, 'training_curves.png'))
        if config.early_stopping and epoch >= config.early_stopping_start and patience_counter >= config.early_patience:
            print(f'Early stopping at E{epoch}')
            break
    tb_logger.close()
    print(f'Done! Best Dice={best_dice:.4f} at E{best_epoch} | Best Val Loss={best_loss:.4f} at E{best_loss_epoch}')
    if args.mode != 'ex':
        plot_loss_curves(history, os.path.join(config.work_dir, 'training_curves.png'))
    with open(os.path.join(config.work_dir, 'summary.json'), 'w') as f:
        json.dump({
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'best_val_loss': best_loss,
            'best_loss_epoch': best_loss_epoch,
            'dataset': config.dataset_name,
            'checkpoints': {
                'best_dice': 'checkpoints/best_dice_model.pth',
                'best_loss': 'checkpoints/best_loss_model.pth',
                'last': 'checkpoints/last_model.pth',
            },
        }, f, indent=2)


if __name__ == '__main__':
    main()
