# SGNet3 + UPerHead 训练配置 — LS (Livecellsstaining) 数据集
# 二值分割: BCE + Dice Loss (+ Nematic auxiliary)
# 注意: 前景占比仅 ~5.6%, 存在类别不平衡

_base_ = [
    '../_base_/models/upernet_sgnet.py',
    '../_base_/datasets/ls_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_56k_ls.py',
]

crop_size = (512, 512)

# LS: 3ch PNG (伪3通道, R=G=B), percentile normalization
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[1, 1, 1],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(input_channels=3),
)
