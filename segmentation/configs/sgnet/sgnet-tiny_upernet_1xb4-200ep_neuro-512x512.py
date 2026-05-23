# SGNet3 + UPerHead 训练配置 — NEURO 数据集
# 8ch TIFF, 二值分割


_base_ = [
    '../_base_/models/upernet_sgnet.py',
    '../_base_/datasets/neuro.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_45k.py',
]

crop_size = (512, 512)

# NEURO: 8ch TIFF, 不归一化到 ImageNet, 直接缩放到 ~0-1
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0] * 8,
    std=[1] * 8,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(input_channels=8),
)
