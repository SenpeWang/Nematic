# LS (Livecellsstaining) 数据集配置
# 二值分割: 活细胞染色前景 vs 背景
# 图像: 512x512x3 uint8 PNG (伪3通道, R=G=B), 标签: 512x512 uint8 PNG (0/255)
# 前景占比约 5.6%, 存在类别不平衡

dataset_type = 'BaseSegDataset'
data_root = '/home/wangshengping/DataSet/LS'

crop_size = (512, 512)

# 训练 pipeline: 加载 → 归一化 → 标签映射 → 增强 → 打包
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PercentileNormalize', lower=1, upper=99, per_channel=True),
    dict(type='LoadAnnotations', label_mapping={255: 1}),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-30, 30)),
    dict(type='PackSegInputs')
]

# 测试/验证 pipeline: 同样需要 PercentileNormalize 保证分布一致
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PercentileNormalize', lower=1, upper=99, per_channel=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', label_mapping={255: 1}),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'LS'), palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'LS'), palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'LS'), palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'aAcc'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'aAcc'])
