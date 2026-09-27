# -*- coding: utf-8 -*-
"""NFAM-Net testing script with explicit checkpoint selection.

默认模式: 只保存 pred (二值掩码 png)
--vis 模式: 额外保存 comparison (四格图) + nematic (向列序场)
"""

import os
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu

import json
import re
import warnings
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from configs.config import config
from datasets.dataset import UniversalDataset
from networks.model import build_Model
from networks.qtensor import QTensorTools
from utils.metric import calculate_sample_metrics
from utils.visualization import plot_predictions, plot_nematic_field

warnings.filterwarnings('ignore')


CKPT_FILES = {
    'best_dice': 'best_dice_model.pth',
    'best_loss': 'best_loss_model.pth',
    'last': 'last_model.pth',
}
CKPT_TYPES_BY_FILE = {v: k for k, v in CKPT_FILES.items()}


def infer_dataset_from_path(path, default='NEURO'):
    low = str(path).lower()
    if '_sy5y' in low or '/sy5y' in low:
        return 'SY5Y'
    if '_neurite' in low or '/neurite' in low:
        return 'NEURITE'
    if '_neuro' in low or '/neuro' in low:
        return 'NEURO'
    if '_ls' in low or '/ls' in low:
        return 'LS'
    return default


def infer_ckpt_type_from_path(path, default='best_dice'):
    return CKPT_TYPES_BY_FILE.get(os.path.basename(path), default)


def resolve_checkpoint(train_dir=None, ckpt_type='best_dice', ckpt=None):
    if ckpt:
        ckpt_path = os.path.abspath(ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        train_dir = os.path.dirname(os.path.dirname(ckpt_path))
        ckpt_type = infer_ckpt_type_from_path(ckpt_path, ckpt_type)
        return ckpt_path, train_dir, ckpt_type
    if not train_dir:
        raise ValueError('Either --ckpt/--checkpoint or --train_dir is required')
    if ckpt_type not in CKPT_FILES:
        raise ValueError(f'Unknown ckpt_type={ckpt_type}; choose one of {sorted(CKPT_FILES)}')
    ckpt_path = os.path.join(train_dir, 'checkpoints', CKPT_FILES[ckpt_type])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
    return ckpt_path, train_dir, ckpt_type


def main():
    parser = argparse.ArgumentParser(description='NFAM-Net Testing')
    parser.add_argument('--dataset', type=str, default=None, choices=['NEURO', 'SY5Y', 'NEURITE', 'LS'])
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--ckpt', '--checkpoint', dest='ckpt', type=str, default=None)
    parser.add_argument('--ckpt_type', type=str, default='best_dice', choices=['best_dice', 'best_loss', 'last'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--vis', action='store_true', help='Save comparison (4-grid) + nematic field in addition to pred')
    args = parser.parse_args()

    ckpt_path, train_dir, ckpt_type = resolve_checkpoint(args.train_dir, args.ckpt_type, args.ckpt)
    dataset_name = args.dataset or infer_dataset_from_path(train_dir)

    train_ts_match = re.search(r'(\d{8}_\d{6})', ckpt_path)
    train_ts = train_ts_match.group(1) if train_ts_match else config.timestamp
    config.test_dir = f'./Outputs/Test/{train_ts}_{config.project_name}_{dataset_name.lower()}_{ckpt_type}/'
    os.makedirs(config.test_dir, exist_ok=True)

    print('#---------- Loading ----------#')
    test_ds = UniversalDataset(dataset_name, 'test', transform=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_Model(config, dataset_name=dataset_name).cuda()
    sd = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f'Loaded: {ckpt_path}')

    # pred 目录始终创建
    pred_dir = os.path.join(config.test_dir, 'pred')
    os.makedirs(pred_dir, exist_ok=True)

    # --vis 模式: 额外创建 comparison + nematic 目录
    comparison_dir = None
    stage_dirs = None
    if args.vis:
        comparison_dir = os.path.join(config.test_dir, 'comparison')
        nematic_dir = os.path.join(config.test_dir, 'nematic')
        os.makedirs(comparison_dir, exist_ok=True)
        os.makedirs(nematic_dir, exist_ok=True)
        stage_dirs = []
        for i in range(4):
            stage_dir = os.path.join(nematic_dir, f'stage{i}')
            os.makedirs(stage_dir, exist_ok=True)
            stage_dirs.append(stage_dir)

    all_metrics = []
    dtype = torch.bfloat16 if config.amp_dtype == 'bfloat16' else torch.float16

    mode_str = 'ALL (pred + comparison + nematic)' if args.vis else 'pred only'
    print(f'#---------- Testing ({mode_str}) ----------#')

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

            # 始终保存 pred
            pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            pred_mask_img.save(os.path.join(pred_dir, f'{case_name}.png'))

            # --vis: 额外保存 comparison + nematic
            if args.vis:
                # 分割对比图 (四格)
                plot_predictions(
                    batch['image'][0].numpy(), targets, pred_mask,
                    os.path.join(comparison_dir, f'{case_name}.png'),
                    title=case_name,
                )

                # 向列序场图 (每个 stage)
                stage_Q = outputs['stage_Q']
                for stage_idx, Q in enumerate(stage_Q):
                    Q_np = Q[0].cpu().numpy()
                    Q1 = Q_np[0]
                    Q2 = Q_np[1]
                    S = QTensorTools.get_S(
                        torch.tensor(Q1).unsqueeze(0).unsqueeze(0),
                        torch.tensor(Q2).unsqueeze(0).unsqueeze(0)
                    ).numpy()[0, 0]

                    H_s, W_s = S.shape
                    if targets.shape[0] != H_s or targets.shape[1] != W_s:
                        gt_downsampled = F.interpolate(
                            torch.tensor(targets).float().unsqueeze(0).unsqueeze(0),
                            size=(H_s, W_s), mode='nearest'
                        ).numpy()[0, 0]
                    else:
                        gt_downsampled = targets

                    plot_nematic_field(
                        S, Q1, Q2,
                        os.path.join(stage_dirs[stage_idx], f'{case_name}.png'),
                        gt_mask=gt_downsampled,
                        title=f'{case_name} Stage{stage_idx}',
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
        print(f'  {name:>10s}: {mean_v:.4f} +/- {std_v:.4f}')
    print(f'  Valid: {len(valid_metrics)} / Total: {len(all_metrics)} (skipped {skipped} empty)')
    print('=' * 60)

    report = {
        'dataset': dataset_name,
        'checkpoint': ckpt_path,
        'ckpt_type': ckpt_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': len(all_metrics),
        'summary': summary,
        'per_sample': all_metrics,
    }
    with open(os.path.join(config.test_dir, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(config.test_dir, 'test_report.txt'), 'w') as f:
        f.write(f'NFAM-Net Test | {dataset_name} | {ckpt_type} | {len(all_metrics)} samples\n\n')
        for name in metric_names:
            s = summary[name]
            f.write(f'{name:>10s}: {s["mean"]:.4f} +/- {s["std"]:.4f}\n')

    print(f'Saved: {config.test_dir}')
    print(f'  - pred/: 二值掩码 png')
    if args.vis:
        print(f'  - comparison/: 分割对比图 (四格)')
        print(f'  - nematic/: 向列序场图 (stage0~3)')


if __name__ == '__main__':
    main()
