# -*- coding: utf-8 -*-
import argparse
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from configs.config import config
from networks.block import GeodesicMambaEvolver


def main():
    parser = argparse.ArgumentParser(description='Mamba3 smoke test')
    parser.add_argument('--dim', type=int, default=96)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--height', type=int, default=16)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--ssm_variant', type=str, default='baseline')
    parser.add_argument('--ssm_dt_scale', type=float, default=1.0)
    parser.add_argument('--ssm_a_scale', type=float, default=1.0)
    parser.add_argument('--ssm_trap_scale', type=float, default=1.0)
    parser.add_argument('--ssm_angle_scale', type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the Mamba3 smoke test.')

    config.apply_experiment_settings(
        experiment_name='smoke',
        ssm_variant=args.ssm_variant,
        ssm_dt_scale=args.ssm_dt_scale,
        ssm_a_scale=args.ssm_a_scale,
        ssm_trap_scale=args.ssm_trap_scale,
        ssm_angle_scale=args.ssm_angle_scale,
    )
    kwargs = config.get_mamba3_kwargs()

    model = GeodesicMambaEvolver(
        dim=args.dim,
        d_state=args.d_state,
        expand=args.expand,
        mamba3_kwargs=kwargs,
    ).cuda().eval()

    x = torch.randn(args.batch_size, args.dim, args.height, args.width, device='cuda')
    Q1 = torch.randn(args.batch_size, 1, args.height, args.width, device='cuda')
    Q2 = torch.randn(args.batch_size, 1, args.height, args.width, device='cuda')
    S = torch.sigmoid(torch.randn(args.batch_size, 1, args.height, args.width, device='cuda'))

    with torch.no_grad():
        y = model(x, Q1=Q1, Q2=Q2, S=S)

    print({
        'input_shape': tuple(x.shape),
        'output_shape': tuple(y.shape),
        'variant': args.ssm_variant,
        'dt_scale': args.ssm_dt_scale,
        'a_scale': args.ssm_a_scale,
        'trap_scale': args.ssm_trap_scale,
        'angle_scale': args.ssm_angle_scale,
    })


if __name__ == '__main__':
    main()
