# -*- coding: utf-8 -*-
"""Dataset loader for NFAM-Net."""

import os
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import rotate as ndi_rotate, zoom as ndi_zoom
from .dataset_config import get_dataset_paths, get_dataset_config

# ============================================================================
# 模块常量
# ============================================================================

AUG_HFLIP_P = 0.5
AUG_VFLIP_P = 0.5
AUG_ROTATE_LIMIT = 35
AUG_ROTATE_P = 0.5
AUG_SCALE_RANGE = (0.7, 1.4)
AUG_SCALE_P = 0.5
AUG_CROP_P = 0.5
DEFAULT_IMG_SIZE = 512
IMAGE_EXTENSIONS = ('.tif', '.tiff', '.png')


# ============================================================================
# 图像 I/O
# ============================================================================

def _load_image(path, shape_format):
    if path.lower().endswith(('.tif', '.tiff')):
        image = tifffile.imread(path)
    else:
        image = np.array(Image.open(path))

    if shape_format == 'HWC':
        image = np.transpose(image, (2, 0, 1))
    elif shape_format == 'HW':
        if image.ndim == 3:
            image = np.max(image, axis=2)
        image = image[np.newaxis, :]

    image = image.astype(np.float32)
    max_value = image.max() if image.size > 0 else 0.0
    image = image / 65535.0 if max_value > 255 else image / 255.0
    return image.astype(np.float32)


def _load_label(label_path):
    label = tifffile.imread(label_path) if label_path.endswith(('.tif', '.tiff')) else np.array(Image.open(label_path))
    if label.ndim == 3:
        label = np.max(label, axis=2)
    return (label > 0).astype(np.int64)


# ============================================================================
# 数据增强
# ============================================================================

def _resize_spatial(image, label, target_size):
    height, width = image.shape[-2], image.shape[-1]
    if (height, width) == (target_size, target_size):
        return image, label
    zoom_h = target_size / height
    zoom_w = target_size / width
    image = ndi_zoom(image, (1, zoom_h, zoom_w), order=1)
    label = ndi_zoom(label.astype(np.float32), (zoom_h, zoom_w), order=0).astype(np.int64)
    return image, label


def apply_train_aug(image, label):
    # 1. 随机缩放
    if np.random.rand() < AUG_SCALE_P:
        scale = np.random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
        h, w = image.shape[1], image.shape[2]

        # 缩放图像 (双线性插值)
        image = ndi_zoom(image, (1, scale, scale), order=1)
        # 缩放标签 (最近邻插值，保持离散值)
        label = ndi_zoom(label, (scale, scale), order=0).astype(np.int64)

        # 确保大小一致 (裁剪或填充)
        c, h_new, w_new = image.shape
        if h_new != h or w_new != w:
            # 裁剪到目标大小
            h_crop = min(h_new, h)
            w_crop = min(w_new, w)
            image = image[:, :h_crop, :w_crop]
            label = label[:h_crop, :w_crop]

            # 如果需要，填充到目标大小
            c, h_cur, w_cur = image.shape
            if h_cur < h or w_cur < w:
                pad_h = h - h_cur
                pad_w = w - w_cur
                image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
                label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant')

    # 2. 随机裁剪
    if np.random.rand() < AUG_CROP_P:
        h, w = image.shape[1], image.shape[2]
        crop_size = min(h, w, 512)

        if h > crop_size and w > crop_size:
            top = np.random.randint(0, h - crop_size)
            left = np.random.randint(0, w - crop_size)

            image = image[:, top:top+crop_size, left:left+crop_size]
            label = label[top:top+crop_size, left:left+crop_size]

    # 3. 随机翻转
    if np.random.rand() < AUG_HFLIP_P:
        image = image[:, :, ::-1]
        label = label[:, ::-1]

    if np.random.rand() < AUG_VFLIP_P:
        image = image[:, ::-1, :]
        label = label[::-1, :]

    # 4. 随机旋转
    if np.random.rand() < AUG_ROTATE_P and AUG_ROTATE_LIMIT > 0:
        angle = float(np.random.uniform(-AUG_ROTATE_LIMIT, AUG_ROTATE_LIMIT))
        image = ndi_rotate(image, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)
        label = ndi_rotate(label, angle, reshape=False, order=0, mode='constant', cval=0).astype(np.int64)

    return image, label


# ============================================================================
# Dataset 类
# ============================================================================

class UniversalDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=True, img_size=None):
        self.dataset_name = dataset_name.upper()
        self.split = split
        self.img_size = img_size if img_size is not None else DEFAULT_IMG_SIZE
        self.do_augment = transform and (split == 'train')

        self.images_dir, self.labels_dir = get_dataset_paths(dataset_name, split)
        ds_info = get_dataset_config(dataset_name)
        self.channels = ds_info['channels']
        self.shape_format = ds_info['shape_format']
        self.label_suffix = ds_info.get('label_suffix', '')

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        print(
            f"[DatasetInfo] {self.dataset_name}/{split}: {self.images_dir} "
            f"({len(self.image_files)} samples, ch={self.channels}, "
            f"aug={self.do_augment})"
        )

    def __len__(self):
        return len(self.image_files)

    def _label_path(self, img_name):
        base = os.path.splitext(img_name)[0]
        if self.label_suffix:
            label_name = base + self.label_suffix + '.png'
        else:
            label_name = base + '.png'
        path = os.path.join(self.labels_dir, label_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing label for {img_name}: expected {path}')
        return path

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        case_name = img_name.rsplit('.', 1)[0]
        image = _load_image(os.path.join(self.images_dir, img_name), self.shape_format)
        label = _load_label(self._label_path(img_name))

        image, label = _resize_spatial(image, label, self.img_size)

        if self.do_augment:
            image, label = apply_train_aug(image, label)

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()

        if torch.isnan(image).any() or torch.isinf(image).any():
            image = torch.zeros_like(image)

        return {
            'image': image,
            'label': label,
            'case_name': case_name,
        }
