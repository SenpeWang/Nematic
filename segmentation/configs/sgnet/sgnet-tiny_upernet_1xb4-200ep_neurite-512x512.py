# SGNet3 + UPerHead 训练配置 — Neurite 数据集
# 二值分割: BCE + Dice Loss (+ Nematic auxiliary)


_base_ = [
    '../_base_/models/upernet_sgnet.py',
    '../_base_/datasets/neurite_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_30k.py',
]

crop_size = (512, 512)

# Neurite: 3ch PNG, 使用 ImageNet 归一化
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(input_channels=3),
)
