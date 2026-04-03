# -*- coding: utf-8 -*-
"""
通用数据集加载器
TIFF 多通道 + 0-1 归一化 + albumentations 增强 (弹性形变 + 网格畸变)
"""

import os
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

from .dataset_config import get_dataset_paths, get_dataset_config


def load_and_process_label(label_path: str) -> np.ndarray:
    """统一标签加载: PNG/TIFF → (H, W) int64 {0, 1}"""
    if label_path.endswith(('.tif', '.tiff')):
        label = tifffile.imread(label_path)
    else:
        label = np.array(Image.open(label_path))
    if label.ndim == 3:
        label = np.max(label, axis=2)
    return (label > 0).astype(np.int64)


def build_train_transforms(config):
    """
    训练增强管线 (albumentations)。
    图像和标签通过 albumentations 自动严格对齐。
    """
    transforms = []

    if config.aug_random_crop:
        h, w = config.aug_crop_size
        # 模仿 nnU-Net / U-Mamba 采样策略:
        # 保证预设比例 (如 33.3%) 的 Patch 必然包含前景标签（抑制全负样本导致无效学习）
        oversample_p = getattr(config, 'oversample_foreground_percent', 0.333)
        
        transforms.append(
            A.OneOf([
                A.CropNonEmptyMaskIfExists(height=h, width=w, p=oversample_p),
                A.RandomCrop(height=h, width=w, p=1.0 - oversample_p)
            ], p=1.0)
        )

    transforms.append(A.HorizontalFlip(p=config.aug_hflip_p))
    transforms.append(A.VerticalFlip(p=config.aug_vflip_p))
    transforms.append(A.RandomRotate90(p=0.5))

    return A.Compose(transforms)


class UniversalDataset(Dataset):
    """
    通用 MIF 数据集。
    TIFF 多通道 + 0-1 归一化 + albumentations 增强 + NaN 拦截。

    物理先验伪标签 (S, Q1, Q2) 已迁移至 GPU 批量计算，不再在此处提取。
    """

    def __init__(self, dataset_name, split='train', transform=True,
                 img_size=512, config=None):
        self.dataset_name = dataset_name.upper()
        self.split = split
        self.img_size = img_size
        self.do_augment = transform and (split == 'train')

        self.images_dir, self.labels_dir = get_dataset_paths(dataset_name, split)
        ds_info = get_dataset_config(dataset_name)
        self.channels = ds_info['channels']
        self.shape_format = ds_info['shape_format']

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith(('.tif', '.tiff'))
        ])

        if self.do_augment and config is not None:
            self.aug_pipeline = build_train_transforms(config)
        else:
            self.aug_pipeline = None

        print(
            f"[UniversalDataset] {self.dataset_name}/{split}: "
            f"{self.images_dir} ({len(self.image_files)} samples, "
            f"ch={self.channels}, aug={self.do_augment})"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # 读取图像
        img_path = os.path.join(self.images_dir, img_name)
        image = tifffile.imread(img_path)
        if self.shape_format == 'HWC':
            image = np.transpose(image, (2, 0, 1))     # → (C, H, W)

        # 读取标签
        label_name = img_name.replace('.tif', '.png').replace('.tiff', '.png')
        label_path = os.path.join(self.labels_dir, label_name)
        label = load_and_process_label(label_path)      # (H, W) {0, 1}

        # 通道独立 Z-score 归一化 (Channel-wise Z-Score Normalization)
        image = image.astype(np.float32)
        for c in range(image.shape[0]):
            c_mean = image[c].mean()
            c_std = image[c].std()
            if c_std > 1e-8:
                image[c] = (image[c] - c_mean) / c_std
            else:
                image[c] = image[c] - c_mean

        # 数据增强 (albumentations 要求 HWC 格式)
        if self.aug_pipeline is not None:
            image_hwc = np.transpose(image, (1, 2, 0))  # (C, H, W) → (H, W, C)
            augmented = self.aug_pipeline(image=image_hwc, mask=label.astype(np.uint8))
            image = np.transpose(augmented['image'], (2, 0, 1))  # → (C, H, W)
            label = augmented['mask'].astype(np.int64)

        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # NaN / Inf 拦截
        if torch.isnan(image).any() or torch.isinf(image).any():
            image = torch.zeros_like(image)

        return {
            'image': image,
            'label': label,
            'case_name': img_name.rsplit('.', 1)[0],
            'folder_name': self.dataset_name,
        }
