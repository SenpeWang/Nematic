# Copyright (c) OpenMMLab. All rights reserved.
"""Convert NEURO dataset to mmsegmentation format.

Expected raw structure:
    neuro/
    ├── train/
    │   ├── images/*.tif   (8ch uint16)
    │   └── labels/*.png
    ├── val/
    │   ├── images/*.tif
    │   └── labels/*.png
    └── test/
        ├── images/*.tif
        └── labels/*.png

Note: Images are multi-channel TIFF. The MMSeg dataloader uses
`to_float32=True` in the pipeline. Labels are binarized to 0/1.
"""

import argparse
import json
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import mkdir_or_exist, scandir
from PIL import Image


def check_and_normalize_labels(label_dir, output_dir):
    """Ensure labels are binary (0/1)."""
    mkdir_or_exist(output_dir)
    for label_name in scandir(label_dir, suffix='.png'):
        label_path = osp.join(label_dir, label_name)
        img = mmcv.imread(label_path, flag='grayscale')
        if img.max() > 1:
            img = (img > 0).astype(np.uint8)
        out_path = osp.join(output_dir, label_name)
        mmcv.imwrite(img, out_path)


def compute_stats(image_dir):
    """Compute per-channel mean/std for NEURO images."""
    stats = {'mean': [], 'std': [], 'min': [], 'max': []}
    from PIL import Image
    accum = []
    for img_name in scandir(image_dir, suffix='.tif'):
        img_path = osp.join(image_dir, img_name)
        img = np.array(Image.open(img_path))
        if img.ndim == 2:
            img = img[..., None]
        accum.append(img.reshape(-1, img.shape[-1]))
    if not accum:
        return stats
    data = np.concatenate(accum, axis=0).astype(np.float32)
    stats['mean'] = data.mean(axis=0).tolist()
    stats['std'] = data.std(axis=0).tolist()
    stats['min'] = data.min(axis=0).tolist()
    stats['max'] = data.max(axis=0).tolist()
    return stats


def generate_list(image_dir, list_file):
    with open(list_file, 'w') as f:
        for img_name in sorted(scandir(image_dir, suffix='.tif')):
            name = osp.splitext(img_name)[0]
            f.write(f'{name}\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert NEURO dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='NEURO dataset path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--no-norm',
        action='store_true',
        help='skip label normalization')
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='only compute and save stats.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir if args.out_dir else dataset_path
    mkdir_or_exist(out_dir)

    if args.stats_only:
        img_dir = osp.join(dataset_path, 'train', 'images')
        print('Computing statistics...')
        stats = compute_stats(img_dir)
        with open(osp.join(out_dir, 'NEURO_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print('Stats saved.')
        return

    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = osp.join(dataset_path, split, 'images')
        label_dir = osp.join(dataset_path, split, 'labels')
        if not osp.exists(img_dir):
            print(f'Skip {split}: images dir not found')
            continue
        if not osp.exists(label_dir):
            print(f'Skip {split}: labels dir not found')
            continue

        out_label_dir = osp.join(out_dir, split, 'labels')
        if not args.no_norm:
            print(f'Normalizing {split} labels...')
            check_and_normalize_labels(label_dir, out_label_dir)
        else:
            mkdir_or_exist(out_label_dir)

        list_file = osp.join(out_dir, f'{split}.txt')
        print(f'Generating {list_file}...')
        generate_list(img_dir, list_file)

    # Also compute stats for train set
    train_img_dir = osp.join(dataset_path, 'train', 'images')
    if osp.exists(train_img_dir):
        print('Computing statistics...')
        stats = compute_stats(train_img_dir)
        with open(osp.join(out_dir, 'NEURO_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    print('Done!')


if __name__ == '__main__':
    main()
