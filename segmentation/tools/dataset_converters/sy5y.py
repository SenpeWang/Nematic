# Copyright (c) OpenMMLab. All rights reserved.
"""Convert SY5Y dataset to mmsegmentation format.

Expected raw structure:
    sy5y/
    ├── train/
    │   ├── images/*.tif (or .png)
    │   └── labels/*.png
    ├── val/
    │   ├── images/*.tif
    │   └── labels/*.png
    └── test/
        ├── images/*.tif
        └── labels/*.png

Note: Images are single-channel. Labels are binarized to 0/1.
"""

import argparse
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import mkdir_or_exist, scandir


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


def generate_list(image_dir, list_file, suffixes=('.tif', '.png')):
    with open(list_file, 'w') as f:
        names = []
        for suffix in suffixes:
            names.extend(scandir(image_dir, suffix=suffix))
        for img_name in sorted(set(names)):
            name = osp.splitext(img_name)[0]
            f.write(f'{name}\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SY5Y dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='SY5Y dataset path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--no-norm',
        action='store_true',
        help='skip label normalization')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir if args.out_dir else dataset_path
    mkdir_or_exist(out_dir)

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

    print('Done!')


if __name__ == '__main__':
    main()
