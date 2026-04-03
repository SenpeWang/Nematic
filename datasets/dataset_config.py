# -*- coding: utf-8 -*-
"""
数据集注册表
字典管理多数据集路径 / 通道数 / shape_format
"""

import os

# 数据根目录
_DATA_ROOT = '/home/wangshengping/04_Nero/data'

# ============================================================================
# 数据集注册表
# ============================================================================

DATASET_REGISTRY = {
    'NEURO': {
        'root': '/home/wangshengping/DataSet/Neuro',
        'channels': 8,
        'shape_format': 'CHW',
        'num_classes': 2,
    },
    'HNSCC': {
        'root': os.path.join(_DATA_ROOT, 'HNSCC'),
        'channels': 8,
        'shape_format': 'HWC',
        'num_classes': 2,
    },
    'BR': {
        'root': os.path.join(_DATA_ROOT, 'BR_2D'),
        'channels': 8,
        'shape_format': 'HWC',
        'num_classes': 2,
    },
    'LN': {
        'root': os.path.join(_DATA_ROOT, 'LN_2D'),
        'channels': 8,
        'shape_format': 'HWC',
        'num_classes': 2,
    },
    'PR': {
        'root': os.path.join(_DATA_ROOT, 'PR_2D'),
        'channels': 8,
        'shape_format': 'HWC',
        'num_classes': 2,
    },
    'TONSIL-1': {
        'root': os.path.join(_DATA_ROOT, 'Tonsil-1_2D'),
        'channels': 8,
        'shape_format': 'HWC',
        'num_classes': 2,
    },
}


def get_dataset_config(name: str) -> dict:
    name = name.upper()
    if name not in DATASET_REGISTRY:
        raise ValueError(f'未知数据集: {name}，可选: {list(DATASET_REGISTRY.keys())}')
    return DATASET_REGISTRY[name]


def get_dataset_paths(name: str, split: str):
    """返回 (images_dir, labels_dir)"""
    info = get_dataset_config(name)
    root = info['root']
    images_dir = os.path.join(root, split, 'images')
    labels_dir = os.path.join(root, split, 'labels')
    return images_dir, labels_dir

