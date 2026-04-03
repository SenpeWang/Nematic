# -*- coding: utf-8 -*-
"""
N²-Mamba 训练脚本
AMP bfloat16 + 梯度裁剪 + NaN 防护 + 物理 loss 实时打印 + 流形可视化
"""

import os
import argparse

# GPU 选择 (必须在 import torch 之前)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu

import json
import warnings
import numpy as np
from datetime import datetime
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
from networks.model import build_nematic_mamba
from utils.losses import get_loss_function
from utils.metric import calculate_dice
from utils.visualization import plot_predictions, plot_nematic_field, plot_loss_curves
from utils.physics_priors import StructureTensorExtractor

warnings.filterwarnings("ignore")


def apply_runtime_overrides(args):
    config.update_for_dataset(args.dataset)
    config.apply_experiment_settings(
        experiment_name=args.experiment,
        ssm_variant=args.ssm_variant,
        ssm_dt_scale=args.ssm_dt_scale,
        ssm_a_scale=args.ssm_a_scale,
        ssm_trap_scale=args.ssm_trap_scale,
        ssm_angle_scale=args.ssm_angle_scale,
    )
    config.refresh_timestamp()
    config.work_dir = config.build_output_dir('train')

    if args.epochs is not None:
        config.epochs = args.epochs
        config.T_max = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr


def train_epoch(loader, model, criterion, optimizer, scheduler, scaler, epoch, config, physics_extractor):
    model.train()
    losses = []
    loss_meters = {}
    accum = config.grad_accum
    dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16

    pbar = tqdm(loader, desc=f'Train E{epoch}')
    for step, batch in enumerate(pbar):
        images = batch['image'].cuda(non_blocking=True).float()
        targets = batch['label'].cuda(non_blocking=True).long()

        # GPU 批量计算物理伪标签 (替代 CPU DataLoader 逐样本计算)
        S_target, Q1_target, Q2_target = physics_extractor(images)
        batch['S_target'] = S_target
        batch['Q1_target'] = Q1_target
        batch['Q2_target'] = Q2_target

        if step % accum == 0:
            optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=config.amp, dtype=dtype):
            outputs = model(images)
            out_f32 = {k: v.float() if isinstance(v, torch.Tensor) else v
                       for k, v in outputs.items()}
            loss = criterion(out_f32, targets, batch)

        # NaN / Inf 防护
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        (loss / accum).backward()

        if (step + 1) % accum == 0:
            # 梯度 NaN 检查
            has_nan = any(
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                for p in model.parameters() if p.grad is not None
            )
            if has_nan:
                optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.append(loss.item())

        # 收集物理 loss 分量
        for k, v in criterion.loss_components.items():
            loss_meters.setdefault(k, []).append(v)

        pbar.set_postfix(loss=f'{np.mean(losses):.4f}')

    scheduler.step()
    avg_loss = float(np.mean(losses)) if losses else float('nan')
    avg_meters = {k: float(np.mean(v)) for k, v in loss_meters.items()}
    return avg_loss, avg_meters


@torch.no_grad()
def valid_epoch(loader, model, criterion, epoch, config, physics_extractor):
    model.eval()
    losses = []
    loss_meters = {}
    dice_sum, dice_valid = 0.0, 0
    dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16

    for batch in tqdm(loader, desc=f'Val E{epoch}'):
        images = batch['image'].cuda(non_blocking=True).float()
        targets = batch['label'].cuda(non_blocking=True).long()

        S_target, Q1_target, Q2_target = physics_extractor(images)
        batch['S_target'] = S_target
        batch['Q1_target'] = Q1_target
        batch['Q2_target'] = Q2_target

        with autocast(device_type='cuda', enabled=config.amp, dtype=dtype):
            outputs = model(images)
        out_f32 = {k: v.float() if isinstance(v, torch.Tensor) else v
                   for k, v in outputs.items()}
        loss = criterion(out_f32, targets, batch)
        losses.append(loss.item())
        for k, v in criterion.loss_components.items():
            loss_meters.setdefault(k, []).append(v)
            
        d_sum, d_cnt = calculate_dice(out_f32['logits'], targets, config.num_classes)
        dice_sum += d_sum
        dice_valid += d_cnt

    val_dice = dice_sum / dice_valid if dice_valid > 0 else 0.0
    avg_loss = float(np.mean(losses)) if losses else float('nan')
    avg_meters = {k: float(np.mean(v)) for k, v in loss_meters.items()}
    return avg_loss, val_dice, avg_meters


@torch.no_grad()
def visualize_samples(model, dataset, epoch, config, n=3):
    model.eval()
    vis_dir = os.path.join(config.work_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16

    for idx in range(min(n, len(dataset))):
        sample = dataset[idx]
        img = sample['image'].unsqueeze(0).cuda().float()
        gt = sample['label'].numpy()

        with autocast(device_type='cuda', enabled=config.amp, dtype=dtype):
            out = model(img)
        pred = torch.argmax(out['logits'].float(), dim=1)[0].cpu().numpy()

        plot_predictions(
            sample['image'].numpy(), gt, pred,
            os.path.join(vis_dir, f'e{epoch}_s{idx}_pred.png'),
            title=f'Epoch {epoch}',
        )

        S = out['S'][0, 0].float().cpu().numpy()
        Q1 = out['Q11'][0, 0].float().cpu().numpy()
        Q2 = out['Q12'][0, 0].float().cpu().numpy()
        plot_nematic_field(
            S, Q1, Q2,
            os.path.join(vis_dir, f'e{epoch}_s{idx}_nematic.png'),
            gt_mask=gt, stride=8, title=f'Nematic E{epoch}',
        )


def main():
    parser = argparse.ArgumentParser(description='N²-Mamba Training')
    parser.add_argument('--dataset', type=str, default='NEURO',
                        choices=['NEURO', 'HNSCC', 'BR', 'LN', 'PR', 'TONSIL-1'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--ssm_variant', type=str, default=None)
    parser.add_argument('--ssm_dt_scale', type=float, default=None)
    parser.add_argument('--ssm_a_scale', type=float, default=None)
    parser.add_argument('--ssm_trap_scale', type=float, default=None)
    parser.add_argument('--ssm_angle_scale', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    apply_runtime_overrides(args)

    os.makedirs(config.work_dir, exist_ok=True)
    ckpt_dir = os.path.join(config.work_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(config.work_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print('#---------- Run Config ----------#')
    print(f'Dataset: {config.dataset_name}')
    print(f'Experiment: {config.experiment_name}')
    print(f'SSM Variant: {config.ssm_variant}')
    print(f'SSM Scales: dt={config.ssm_dt_scale}, A={config.ssm_a_scale}, trap={config.ssm_trap_scale}, angle={config.ssm_angle_scale}')
    print(f'Work Dir: {config.work_dir}')

    print('#---------- Loading Data ----------#')
    train_ds = UniversalDataset(config.dataset_name, 'train', transform=True, config=config)
    val_ds = UniversalDataset(config.dataset_name, 'val', transform=False, config=config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    print('#---------- Building N²-Mamba ----------#')
    model = build_nematic_mamba(config).cuda()
    total_p = sum(p.numel() for p in model.parameters())
    print(f'Params: {total_p / 1e6:.2f}M')

    # GPU 物理先验提取器 (替代 CPU DataLoader 中的逐样本计算)
    physics_extractor = StructureTensorExtractor().cuda()

    criterion = get_loss_function(config).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            betas=config.betas, eps=config.eps,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min)

    use_scaler = config.amp and config.amp_dtype == 'float16'
    scaler = GradScaler('cuda', enabled=use_scaler)

    best_dice, best_epoch = 0.0, 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    loss_log_path = os.path.join(config.work_dir, 'loss_log.txt')

    print('#---------- Training ----------#')
    for epoch in range(1, config.epochs + 1):
        train_loss, train_meters = train_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            scaler, epoch, config, physics_extractor)
        val_loss, val_dice, val_meters = valid_epoch(val_loader, model, criterion, epoch, config, physics_extractor)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f'E{epoch}  | dice={val_dice:.4f} | train loss={train_loss:.4f} | val loss={val_loss:.4f}')

        # 动态创建表头并写入数据
        if epoch == 1:
            headers = ["Epoch", "Val_Dice", "Tr_Total", "Val_Total"]
            for k in train_meters.keys():
                headers.append(f"Tr_{k}")
            for k in val_meters.keys():
                headers.append(f"Val_{k}")
            with open(loss_log_path, 'w') as f:
                f.write("\t".join(headers) + "\n")
                
        row_data = [f"{epoch}", f"{val_dice:.6f}", f"{train_loss:.6f}", f"{val_loss:.6f}"]
        for k in train_meters.keys():
            row_data.append(f"{train_meters[k]:.6f}")
        for k in val_meters.keys():
            row_data.append(f"{val_meters[k]:.6f}")
            
        with open(loss_log_path, 'a') as f:
            f.write("\t".join(row_data) + "\n")

        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, 'last_model.pth'))

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, 'best_model.pth'))
            print(f' ★ New Best (Dice={val_dice:.4f})')
            visualize_samples(model, val_ds, epoch, config)
        else:
            if epoch >= config.early_stopping_start:
                patience_counter += 1
        if epoch % config.val_interval == 0:
            plot_loss_curves(history,
                             os.path.join(config.work_dir, 'training_curves.png'))

        if (config.early_stopping and epoch >= config.early_stopping_start
                and patience_counter >= config.early_patience):
            print(f'Early stopping at E{epoch}')
            break

    print(f'Done! Best Dice={best_dice:.4f} at E{best_epoch}')
    plot_loss_curves(history, os.path.join(config.work_dir, 'training_curves.png'))
    with open(os.path.join(config.work_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'dataset': config.dataset_name,
            'experiment_name': config.experiment_name,
            'ssm_variant': config.ssm_variant,
            'ssm_dt_scale': config.ssm_dt_scale,
            'ssm_a_scale': config.ssm_a_scale,
            'ssm_trap_scale': config.ssm_trap_scale,
            'ssm_angle_scale': config.ssm_angle_scale,
            'work_dir': config.work_dir,
        }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
