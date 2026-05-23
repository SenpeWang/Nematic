# -*- coding: utf-8 -*-
"""Four-panel visualization for segmentation test results.

Layout (2x2):
  - Top-left:   Original image
  - Top-right:  Ground Truth (green foreground)
  - Bottom-left: Prediction (red foreground)
  - Bottom-right: Difference map (TP=white, FP=red, FN=green)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch


def _read_original_image(img_path):
    """Read original image from file path."""
    img = Image.open(img_path)
    img = np.array(img)
    # Handle grayscale / single channel
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        # Multi-channel (e.g. NEURO 8ch): take first 3 channels
        img = img[:, :, :3]
    return img


def _compute_difference(gt, pred):
    """Compute TP/FP/FN/TN masks."""
    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    tn = (gt == 0) & (pred == 0)
    return tp, fp, fn, tn


def save_four_panel(image_path, gt_mask, pred_mask, save_path,
                    title_prefix=''):
    """Save a single four-panel figure.

    Args:
        image_path (str): Path to original image file.
        gt_mask (np.ndarray): Ground truth mask (H, W), values 0/1.
        pred_mask (np.ndarray): Prediction mask (H, W), values 0/1.
        save_path (str): Output PNG path.
        title_prefix (str): Optional prefix for title.
    """
    img = _read_original_image(image_path)
    tp, fp, fn, tn = _compute_difference(gt_mask, pred_mask)

    # Build difference RGB image
    diff = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    diff[tp] = [255, 255, 255]   # True Positive: white
    diff[fp] = [255, 0, 0]       # False Positive: red
    diff[fn] = [0, 255, 0]       # False Negative: green
    # TN remains black

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'{title_prefix}Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_mask, cmap='Greens', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(diff)
    axes[1, 1].set_title('Diff (TP=W, FP=R, FN=G)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_four_panels_from_runner(runner, save_dir, max_samples=None):
    """Generate four-panel figures after runner.test().

    Iterates over the test dataloader, runs inference, and saves panels.
    """
    model = runner.model
    model.eval()
    dataloader = runner.test_dataloader

    os.makedirs(save_dir, exist_ok=True)

    total = min(len(dataloader), max_samples) if max_samples else len(dataloader)
    print(f'Generating four-panel visuals for {total} samples...')

    for idx, data in enumerate(dataloader):
        if max_samples and idx >= max_samples:
            break

        with torch.no_grad():
            results = model.test_step(data)

        # data['data_samples'] is a list (batch_size=1 for test)
        for batch_idx, data_sample in enumerate(data['data_samples']):
            img_path = data_sample.img_path
            gt = data_sample.gt_sem_seg.data.cpu().numpy().squeeze()
            pred = results[batch_idx].pred_sem_seg.data.cpu().numpy().squeeze()

            # Binarize prediction (if logits) or take argmax
            if pred.ndim == 3 and pred.shape[0] == 2:
                pred = pred.argmax(axis=0)
            pred = (pred > 0).astype(np.uint8)
            gt = (gt > 0).astype(np.uint8)

            basename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(save_dir, f'{basename}_fourpanel.png')
            save_four_panel(img_path, gt, pred, save_path,
                            title_prefix=f'[{idx}] ')

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f'  {idx + 1}/{total} done')

    print(f'All panels saved to: {save_dir}')
