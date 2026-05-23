# SGNet3 + UPerHead 通用模型配置
# 不含 dataset-specific 参数（data_preprocessor / backbone.input_channels 在子配置中覆盖）
norm_cfg = dict(type='SyncBN', requires_grad=True)

# data_preprocessor is defined in the top-level dataset-specific config
# via model=dict(data_preprocessor=...)

model = dict(
    type='SGNetSegmentor',
    # Nematic/Q-pred hyperparameters
    Q_max_norm=1.0,
    lambda_seg=1.0,
    lambda_nematic=0.3,
    nematic_ordered_weight=1.0,
    nematic_boundary_weight=1.0,
    Q_stage_weights=(1.0, 0.5, 0.2, 0.0),
    backbone=dict(
        type='MM_SGNetBackbone',
        out_indices=(0, 1, 2, 3),
        input_channels=3,
        num_classes=2,
        embed_dim=96,
        encoder_depths=(2, 2, 8, 2),
        d_state=1,
        ssm_ratio=1.0,
        drop_path_rate=0.2,
        ssm_conv=3,
        Q_max_norm=1.0,
        Q_axis_temperature=0.35,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                naive_dice=True,
                eps=1e-5,
                loss_weight=1.0,
                loss_name='loss_dice',
            ),
        ],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                naive_dice=True,
                eps=1e-5,
                loss_weight=0.4,
                loss_name='loss_dice',
            ),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
