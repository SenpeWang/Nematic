# Copyright (c) OpenMMLab. All rights reserved.
"""SGNet3 MMSeg test script (standard MMSeg / VMamba style).

Usage:
    python segmentation/tools/test.py config.py checkpoint.pth
    python segmentation/tools/test.py config.py checkpoint.pth --show-dir ./vis
    python segmentation/tools/test.py config.py checkpoint.pth --no-custom-eval
"""

import argparse
import json
import os
import os.path as osp
import sys

# Ensure project root and segmentation/ are on Python path.
_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
_SEG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
for _p in (_PROJECT_ROOT, _SEG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Trigger registration
import segmentation.model  # noqa: F401

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='output directory')
    parser.add_argument('--out', type=str, help='output prediction directory')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show-dir', help='directory for painted images')
    parser.add_argument('--wait-time', type=float, default=2, help='show interval (s)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--no-custom-eval',
        action='store_true',
        help='Skip custom evaluation (four-panel visuals, CLDice, per-sample metrics). '
             'By default custom eval is always run during test.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    """Trigger visualization hook based on --show / --show-dir."""
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks. '
            'refer to usage "visualization=dict(type=\'SegVisualizationHook\')"')
    return cfg


# ============================================================================
# Custom evaluation helpers (VMamba-style)
# ============================================================================

def _cl_dice(v_p, v_l):
    """Compute CLDice (Centerline Dice)."""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return None
    v_p = (v_p > 0).astype(np.uint8)
    v_l = (v_l > 0).astype(np.uint8)
    if v_p.sum() == 0 and v_l.sum() == 0:
        return None
    if v_p.sum() == 0 or v_l.sum() == 0:
        return 0.0
    s_p = skeletonize(v_p)
    s_l = skeletonize(v_l)
    t_prec = np.sum(s_p & (v_l > 0)) / np.sum(s_p) if np.sum(s_p) > 0 else 0.0
    t_sens = np.sum(s_l & (v_p > 0)) / np.sum(s_l) if np.sum(s_l) > 0 else 0.0
    if t_prec + t_sens == 0:
        return 0.0
    return 2 * t_prec * t_sens / (t_prec + t_sens)


def _compute_sample_metrics(pred, gt):
    """Compute per-sample metrics: Dice, IoU, Precision, Recall, CLDice."""
    pred_binary = (pred > 0).astype(np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)

    if pred_binary.sum() == 0 and gt_binary.sum() == 0:
        return None

    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    cldice = _cl_dice(pred_binary, gt_binary)

    return {
        'Dice': float(dice),
        'IoU': float(iou),
        'Precision': float(precision),
        'Recall': float(recall),
        'CLDice': float(cldice) if cldice is not None else None,
    }


def _create_confusion_matrix_overlay(rgb_image, prediction, ground_truth, alpha=0.6):
    """Create confusion matrix overlay: TP=Yellow, FN=Red, FP=Green."""
    pred = (prediction > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)
    mask_tp = (gt == 1) & (pred == 1)
    mask_fn = (gt == 1) & (pred == 0)
    mask_fp = (gt == 0) & (pred == 1)
    overlay_layer = rgb_image.copy()
    overlay_layer[mask_tp] = [180, 180, 0]   # Yellow
    overlay_layer[mask_fn] = [255, 0, 0]     # Red
    overlay_layer[mask_fp] = [0, 255, 0]     # Green
    mask_any = mask_tp | mask_fn | mask_fp
    blended = rgb_image.copy()
    blended[mask_any] = (overlay_layer[mask_any] * alpha +
                         rgb_image[mask_any] * (1 - alpha)).astype(np.uint8)
    return blended


def _save_four_panels(output_dir, img_name, image, pred, gt):
    """Save 2×2 panel: original, overlay, GT, Pred."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image
    rgb_gray = np.stack([gray, gray, gray], axis=-1)

    overlay = _create_confusion_matrix_overlay(rgb_gray, pred, gt)
    gt_mask = ((gt > 0).astype(np.uint8) * 255)
    pred_mask = ((pred > 0).astype(np.uint8) * 255)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('Overlay (Y=TP, R=FN, G=FP)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('GT')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Pred')
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = osp.join(output_dir, f'{img_name}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_custom_eval(runner, work_dir, ckpt_path=''):
    """Custom evaluation: per-sample metrics + four-panel visuals."""
    import matplotlib
    matplotlib.use('Agg')

    model = runner.model
    model.eval()

    four_panels_dir = osp.join(work_dir, 'four_panels')
    os.makedirs(four_panels_dir, exist_ok=True)

    all_metrics = []
    sample_results = []

    with torch.no_grad():
        for idx, data_batch in enumerate(runner.test_dataloader):
            data = model.data_preprocessor(data_batch, False)
            results = model._run_forward(data, mode='predict')
            data_samples = data_batch.get('data_samples', None)

            for i, result in enumerate(results):
                pred = result.pred_sem_seg.data.squeeze().cpu().numpy()

                gt = None
                if data_samples is not None and i < len(data_samples):
                    ds = data_samples[i]
                    if hasattr(ds, 'gt_sem_seg'):
                        gt = ds.gt_sem_seg.data.squeeze().cpu().numpy()
                if gt is None and hasattr(result, 'gt_sem_seg'):
                    gt = result.gt_sem_seg.data.squeeze().cpu().numpy()
                if gt is None:
                    continue

                img_path = result.metainfo.get('img_path', f'sample_{idx}_{i}')
                img_name = osp.splitext(osp.basename(img_path))[0]

                if 'inputs' in data_batch:
                    raw_img = data_batch['inputs'][i].cpu().numpy().transpose(1, 2, 0)
                else:
                    raw_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

                metrics = _compute_sample_metrics(pred, gt)
                if metrics is not None:
                    all_metrics.append(metrics)
                    sample_results.append({'name': img_name, **metrics})

                _save_four_panels(four_panels_dir, img_name, raw_img, pred, gt)

    avg_metrics = {}
    for key in ['Dice', 'IoU', 'Precision', 'Recall', 'CLDice']:
        values = [m[key] for m in all_metrics if m[key] is not None]
        avg_metrics[key] = float(np.mean(values)) if values else 0.0

    metrics_path = osp.join(work_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'average': avg_metrics,
            'per_sample': sample_results,
            'num_valid_samples': len(all_metrics),
        }, f, indent=2, ensure_ascii=False)

    summary_path = osp.join(work_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('Test Summary\n')
        f.write('=' * 60 + '\n')
        f.write(f'Checkpoint: {ckpt_path}\n')
        f.write(f'Valid Samples: {len(all_metrics)}\n')
        f.write('-' * 60 + '\n')
        for key, val in avg_metrics.items():
            f.write(f'{key}: {val:.6f}\n')
        f.write('=' * 60 + '\n')

    print(f'\nCustom Test Results:')
    for key, val in avg_metrics.items():
        print(f'  {key}: {val:.6f}')
    print(f'Results saved to: {work_dir}')
    return avg_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    runner = Runner.from_cfg(cfg)

    runner.test()
    if not args.no_custom_eval:
        run_custom_eval(runner, cfg.work_dir, args.checkpoint)


if __name__ == '__main__':
    main()
