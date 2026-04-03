# -*- coding: utf-8 -*-
"""
Nematic 全局配置
Data / Model / Train / Output 四组管理
"""

import os
import re
from datetime import datetime


class config:

    # ================================================================
    #  Data
    # ================================================================
    dataset_name    = 'NEURO'
    input_channels  = 8
    num_classes     = 2
    img_size        = 512

    # 数据增强 (albumentations)
    aug_random_crop        = True
    aug_crop_size          = (512, 512)
    aug_hflip_p            = 0.5
    aug_vflip_p            = 0.5

    # ================================================================
    #  Model
    # ================================================================
    embed_dim           = 96
    encoder_depths      = [2, 2, 5, 2]
    decoder_depths      = [2, 2, 5, 2]
    d_state             = 64
    expand              = 2
    drop_path_rate      = 0.2
    fpn_dim             = 256
    use_checkpoint      = False
    d_embed             = 64              # 共享嵌入空间维度

    # Nematic Manifold 物理门控
    s_threshold         = 0.3

    # ???????? (???? baseline ??)
    experiment_name     = 'baseline'
    ssm_variant         = 'baseline'
    ssm_dt_scale        = 1.0
    ssm_a_scale         = 1.0
    ssm_trap_scale      = 1.0
    ssm_angle_scale     = 1.0

    # ================================================================
    #  Train
    # ================================================================
    epochs          = 200
    batch_size      = 4
    grad_accum      = 16
    num_workers     = 4

    optimizer       = 'AdamW'
    lr              = 3e-4
    betas           = (0.9, 0.999)
    eps             = 1e-8
    weight_decay    = 0.01

    scheduler       = 'CosineAnnealingLR'
    T_max           = 200
    eta_min         = 1e-6

    amp             = True
    amp_dtype       = 'bfloat16'

    # 损失权重
    lambda_seg      = 1.0
    lambda_frank    = 1.0               # Frank 弹性能量 (各向异性)
    k11_init        = 1.0               # Splay (展曲) 弹性常数初始值
    k33_init        = 1.0               # Bend (弯曲) 弹性常数初始值
    lambda_order    = 0.5               # 序参量双峰约束
    lambda_flow     = 1.0               # 测地向列一致性流
    lambda_distill  = 0.1               # 物理先验蒸馏
    loss_weight     = [0.3, 0.7]        # [BCE, Dice]
    dice_epsilon    = 1e-5

    early_stopping       = False
    early_stopping_start = 120
    early_patience       = 20
    seed                 = 42

    # ================================================================
    #  Output
    # ================================================================
    project_name    = 'Nematic'
    runtime_python  = '/home/wangshengping/myconda/envs/sp_mamba/bin/python'
    tmux_session_prefix = 'nematic'
    timestamp       = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir        = f'./Outputs/Train/{timestamp}_{project_name}/'
    test_dir        = f'./Outputs/Test/{timestamp}_{project_name}/'

    val_interval    = 1
    threshold       = 0.5

    # ================================================================
    #  便捷方法
    # ================================================================

    @classmethod
    def update_for_dataset(cls, dataset_name: str):
        from datasets.dataset_config import get_dataset_config
        cls.dataset_name = dataset_name.upper()
        dataset_info = get_dataset_config(cls.dataset_name)
        cls.input_channels = dataset_info['channels']
        cls.num_classes = dataset_info.get('num_classes', 2)

    @classmethod
    def sanitize_tag(cls, value: str) -> str:
        value = str(value).strip().lower()
        value = re.sub(r'[^a-zA-Z0-9._-]+', '-', value)
        return value.strip('-') or 'default'

    @classmethod
    def refresh_timestamp(cls):
        cls.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return cls.timestamp

    @classmethod
    def get_experiment_tag(cls) -> str:
        return f"{cls.sanitize_tag(cls.experiment_name)}_{cls.sanitize_tag(cls.ssm_variant)}"

    @classmethod
    def build_output_dir(cls, split: str, timestamp: str = None) -> str:
        split_name = 'Train' if split.lower() == 'train' else 'Test'
        run_ts = timestamp or cls.timestamp
        dataset_tag = cls.sanitize_tag(cls.dataset_name)
        exp_tag = cls.get_experiment_tag()
        return f'./Outputs/{split_name}/{run_ts}_{cls.project_name}_{dataset_tag}_{exp_tag}/'

    @classmethod
    def apply_experiment_settings(
        cls,
        experiment_name=None,
        ssm_variant=None,
        ssm_dt_scale=None,
        ssm_a_scale=None,
        ssm_trap_scale=None,
        ssm_angle_scale=None,
    ):
        if experiment_name is not None:
            cls.experiment_name = experiment_name
        if ssm_variant is not None:
            cls.ssm_variant = ssm_variant
        if ssm_dt_scale is not None:
            cls.ssm_dt_scale = float(ssm_dt_scale)
        if ssm_a_scale is not None:
            cls.ssm_a_scale = float(ssm_a_scale)
        if ssm_trap_scale is not None:
            cls.ssm_trap_scale = float(ssm_trap_scale)
        if ssm_angle_scale is not None:
            cls.ssm_angle_scale = float(ssm_angle_scale)

    @classmethod
    def get_mamba3_kwargs(cls):
        return {
            'ssm_variant': cls.ssm_variant,
            'ssm_dt_scale': cls.ssm_dt_scale,
            'ssm_a_scale': cls.ssm_a_scale,
            'ssm_trap_scale': cls.ssm_trap_scale,
            'ssm_angle_scale': cls.ssm_angle_scale,
        }

    @classmethod
    def to_dict(cls):
        result = {}
        for k, v in cls.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (classmethod, staticmethod)):
                continue
            if callable(v):
                continue
            result[k] = v
        return result
