# -*- coding: utf-8 -*-
"""
数据集注册表
字典管理多数据集路径 / 通道数 / shape_format
"""
import os

# ============================================================================
# 数据集注册表
# ============================================================================

DATASET_REGISTRY = {
    'NEURO': {
        'root': '/home/wangshengping/DataSet/Neuro',
        'channels': 8,
        'shape_format': 'CHW',
    },
    'SY5Y': {
        'root': '/home/wangshengping/DataSet/SY5Y',
        'channels': 1,
        'shape_format': 'HW',
        'label_suffix': '',
    },
    'NEURITE': {
        'root': '/home/wangshengping/DataSet/Neurite',
        'channels': 1,
        'shape_format': 'HW',
    },
    'LS': {
        'root': '/home/wangshengping/DataSet/LS',
        'channels': 1,
        'shape_format': 'HW',
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
