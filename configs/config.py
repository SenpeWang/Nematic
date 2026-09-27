# -*- coding: utf-8 -*-
"""Global configuration for Nematic-Q multi-task segmentation."""

from datetime import datetime


class config():
    encoder_depths = [2, 3, 6, 2]
    drop_path_rate = 0.2

    epochs = 200
    batch_size = 8
    grad_accum = 1
    num_workers = 4

    lr = 3e-4
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-4

    T_max = 200
    eta_min = 1e-6

    amp = False
    amp_dtype = 'bfloat16'
    loss_scale = 'dynamic'


    early_stopping = False
    early_stopping_start = 120
    early_patience = 20
    seed = 1234

    project_name = 'NFAM-Net'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = f'./Outputs/Train/{timestamp}_{project_name}/'
    test_dir = f'./Outputs/Test/{timestamp}_{project_name}/'

    val_interval = 1
    threshold = 0.5

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
    @classmethod
    def update_for_dataset(cls, dataset_name):
        """Update config based on dataset."""
        cls.dataset_name = dataset_name.upper()
        # Update work_dir with dataset name
        cls.work_dir = f'./Outputs/Train/{cls.timestamp}_{cls.project_name}_{cls.dataset_name.lower()}/'
        cls.test_dir = f'./Outputs/Test/{cls.timestamp}_{cls.project_name}_{cls.dataset_name.lower()}/'
