# SGNet3 + UPerHead 训练配置 — SY5Y 数据集
# 1ch TIFF/PNG, 二值分割


_base_ = [
    '../_base_/models/upernet_sgnet.py',
    '../_base_/datasets/sy5y.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_120k.py',
]

crop_size = (512, 512)

# SY5Y: 1ch 灰度图
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0],
    std=[1],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(input_channels=1),
)
