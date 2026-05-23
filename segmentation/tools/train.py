# Copyright (c) OpenMMLab. All rights reserved.
"""SGNet3 MMSeg training script (standard MMSeg / VMamba style).

Usage:
    python segmentation/tools/train.py segmentation/configs/sgnet/upernet_sgnet_neurite.py
    python segmentation/tools/train.py config.py --work-dir ./Outputs/my_exp
"""

import argparse
import os
import os.path as osp
import sys

# Ensure project root and segmentation/ are on Python path.
_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
_SEG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
for _p in (_PROJECT_ROOT, _SEG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Trigger registration of custom modules via segmentation package.
# (custom_imports in config also loads these; kept here for robustness.)
import segmentation  # noqa: F401

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP is already enabled.', logger='current')
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    cfg.resume = args.resume

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
