#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize test results with prediction overlays.

Usage:
    python tools/analysis_tools/visualize_test_results.py \
        config.py checkpoint.pth --show-dir ./vis
"""

import argparse
import os
import os.path as osp
import sys

_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
_SEG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
for _p in (_PROJECT_ROOT, _SEG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from tqdm import tqdm

import segmentation.model  # noqa: F401
from segmentation.utils.visualization import plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize test results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show-dir', required=True, help='directory to save visualizations')
    parser.add_argument('--max-samples', type=int, default=None, help='max number of samples to visualize')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.load_from = args.checkpoint

    runner = Runner.from_cfg(cfg)
    test_dataloader = runner.test_dataloader
    model = runner.model
    model.eval()

    os.makedirs(args.show_dir, exist_ok=True)
    total = min(len(test_dataloader), args.max_samples) if args.max_samples else len(test_dataloader)

    print(f'Generating prediction visuals for {total} samples...')
    for idx, data in enumerate(tqdm(test_dataloader, desc='Vis')):
        if args.max_samples and idx >= args.max_samples:
            break

        with torch.no_grad():
            outputs = model.test_step(data)

        for batch_idx, data_sample in enumerate(data['data_samples']):
            img_path = data_sample.img_path
            gt = data_sample.gt_sem_seg.data.cpu().numpy().squeeze()
            pred = outputs[batch_idx].pred_sem_seg.data.cpu().numpy().squeeze()

            if pred.ndim == 3 and pred.shape[0] == 2:
                pred = pred.argmax(axis=0)
            pred = (pred > 0).astype(np.uint8)
            gt = (gt > 0).astype(np.uint8)

            img = Image.open(img_path)
            img_np = np.array(img)
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.ndim == 3 and img_np.shape[2] > 3:
                img_np = img_np[:, :, :3]

            basename = osp.splitext(osp.basename(img_path))[0]
            save_path = osp.join(args.show_dir, f'{basename}.png')
            plot_predictions(img_np, gt, pred, save_path, title=basename)

    print(f'All visuals saved to: {args.show_dir}')


if __name__ == '__main__':
    main()
