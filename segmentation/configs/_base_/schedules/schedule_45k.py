# NEURO schedule: 918 samples, batch_size=4, 200 epochs
# iters/epoch = 229, max_iters = 45800

optim_wrapper = dict(_delete_=True, 
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm_mlp': dict(decay_mult=0.),
            'norm_gate': dict(decay_mult=0.),
            'input_norm': dict(decay_mult=0.),
        }),
    loss_scale='dynamic',
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=45800,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=45800,
    val_interval=229,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(_delete_=True, 
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=229,
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

custom_hooks = [dict(type='SGNetTrainHook', start_epoch=70, patience=20, priority='LOW')]

randomness = dict(seed=42)
work_dir = './Outputs/train_results'
