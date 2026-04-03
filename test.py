# -*- coding: utf-8 -*-
"""
N?-Mamba ????
?????? ? ?? ? Dice/IoU/clDice ? ?????? ? JSON ??
"""

import os
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu

import json
import re
import sys
import warnings
import numpy as np
import yaml
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from configs.config import config
from datasets.dataset import UniversalDataset
from networks.model import build_nematic_mamba
from utils.metric import calculate_sample_metrics
from utils.visualization import plot_predictions, plot_nematic_field

warnings.filterwarnings("ignore")


def resolve_checkpoint_path(checkpoint_arg: str):
    checkpoint_arg = os.path.abspath(checkpoint_arg)
    if os.path.isdir(checkpoint_arg):
        run_dir = checkpoint_arg.rstrip('/\\')
        ckpt_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        return ckpt_path, run_dir

    ckpt_path = checkpoint_arg
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    return ckpt_path, run_dir


def load_train_config(run_dir: str):
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path, 'r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f) or {}
    for key, value in loaded.items():
        setattr(config, key, value)


def main():
    parser = argparse.ArgumentParser(description='N?-Mamba Testing')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['NEURO', 'HNSCC', 'BR', 'LN', 'PR', 'TONSIL-1'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--ssm_variant', type=str, default=None)
    parser.add_argument('--ssm_dt_scale', type=float, default=None)
    parser.add_argument('--ssm_a_scale', type=float, default=None)
    parser.add_argument('--ssm_trap_scale', type=float, default=None)
    parser.add_argument('--ssm_angle_scale', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_vis', action='store_true', default=True)
    args = parser.parse_args()

    try:
        ckpt_path, run_dir = resolve_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        print(f'[ERROR] ???: {exc}')
        sys.exit(1)

    load_train_config(run_dir)

    if args.dataset is not None:
        config.update_for_dataset(args.dataset)
    elif not getattr(config, 'dataset_name', None):
        config.update_for_dataset('NEURO')

    config.apply_experiment_settings(
        experiment_name=args.experiment,
        ssm_variant=args.ssm_variant,
        ssm_dt_scale=args.ssm_dt_scale,
        ssm_a_scale=args.ssm_a_scale,
        ssm_trap_scale=args.ssm_trap_scale,
        ssm_angle_scale=args.ssm_angle_scale,
    )

    train_ts_match = re.search(r'(\d{8}_\d{6})', ckpt_path)
    train_ts = train_ts_match.group(1) if train_ts_match else config.timestamp
    config.test_dir = config.build_output_dir('test', timestamp=train_ts)
    os.makedirs(config.test_dir, exist_ok=True)

    print('#---------- Run Config ----------#')
    print(f'Dataset: {config.dataset_name}')
    print(f'Experiment: {config.experiment_name}')
    print(f'SSM Variant: {config.ssm_variant}')
    print(f'Test Dir: {config.test_dir}')

    print('#---------- Loading ----------#')
    test_ds = UniversalDataset(config.dataset_name, 'test', transform=False, config=config)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_nematic_mamba(config).cuda()

    sd = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f'Loaded: {ckpt_path}')

    pred_dir = os.path.join(config.test_dir, 'pred')
    phys_dir = os.path.join(config.test_dir, 'phys')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(phys_dir, exist_ok=True)

    all_metrics = []
    dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16

    print('#---------- Testing ----------#')
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            images = batch['image'].cuda().float()
            targets = batch['label'].numpy()[0]
            case_name = batch['case_name'][0]

            with autocast(device_type='cuda', enabled=config.amp, dtype=dtype):
                outputs = model(images)

            pred_prob = F.softmax(outputs['logits'].float(), dim=1)[0, 1].cpu().numpy()
            pred_mask = (pred_prob >= config.threshold).astype(np.uint8)

            metrics = calculate_sample_metrics(pred_prob, targets, threshold=config.threshold)
            metrics['case'] = case_name
            all_metrics.append(metrics)

            if args.save_vis:
                plot_predictions(
                    batch['image'][0].numpy(), targets, pred_mask,
                    os.path.join(pred_dir, f'{case_name}.png'),
                    title=case_name,
                )
                S = outputs['S'][0, 0].float().cpu().numpy()
                Q1 = outputs['Q11'][0, 0].float().cpu().numpy()
                Q2 = outputs['Q12'][0, 0].float().cpu().numpy()
                plot_nematic_field(
                    S, Q1, Q2,
                    os.path.join(phys_dir, f'{case_name}.png'),
                    gt_mask=targets, stride=8,
                    title=f'Nematic: {case_name}',
                )

    valid_metrics = [m for m in all_metrics if m.get('valid', True)]
    skipped = len(all_metrics) - len(valid_metrics)
    print('\n' + '=' * 60)
    metric_names = ['dice', 'iou', 'cldice', 'precision', 'recall']
    summary = {}
    for name in metric_names:
        vals = [m[name] for m in valid_metrics if np.isfinite(m[name])]
        mean_v = np.mean(vals) if vals else float('nan')
        std_v = np.std(vals) if vals else float('nan')
        summary[name] = {'mean': float(mean_v), 'std': float(std_v)}
        print(f'  {name:>10s}: {mean_v:.4f} ? {std_v:.4f}')

    print(f'  Valid: {len(valid_metrics)} / Total: {len(all_metrics)} (skipped {skipped} empty)')
    print('=' * 60)

    report = {
        'dataset': config.dataset_name,
        'checkpoint': ckpt_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_name': config.experiment_name,
        'ssm_variant': config.ssm_variant,
        'ssm_dt_scale': config.ssm_dt_scale,
        'ssm_a_scale': config.ssm_a_scale,
        'ssm_trap_scale': config.ssm_trap_scale,
        'ssm_angle_scale': config.ssm_angle_scale,
        'num_samples': len(all_metrics),
        'summary': summary,
        'per_sample': all_metrics,
    }
    with open(os.path.join(config.test_dir, 'test_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(os.path.join(config.test_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f'N?-Mamba Test | {config.dataset_name} | {len(all_metrics)} samples\n\n')
        for name in metric_names:
            s = summary[name]
            f.write(f'{name:>10s}: {s["mean"]:.4f} ? {s["std"]:.4f}\n')

    print(f'Saved: {config.test_dir}')


if __name__ == '__main__':
    main()
