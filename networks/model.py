# -*- coding: utf-8 -*-
"""SGNet top-level model (encoder + prior only; decoder delegated to MMSeg UPerHead)."""

import torch
import torch.nn as nn
from .encoder import SGNetEncoder
from .prior import Q_image_prior
from .q_tensor import QTensorTools


class SGNet(nn.Module):
    def __init__(self, input_channels=8, num_classes=2, embed_dim=96,
                 encoder_depths=(2, 2, 5, 2),
                 d_state=64, ssm_ratio=2.0, drop_path_rate=0.2,
                 ssm_conv=3,
                 Q_max_norm=1.0, Q_axis_temperature=0.35,
                 **kwargs):
        super().__init__()
        self.input_norm = nn.GroupNorm(1, input_channels)
        self.nematic_prior = Q_image_prior(max_channels=input_channels, tangent=True)
        self.encoder = SGNetEncoder(
            in_chans=input_channels, embed_dim=embed_dim, depths=encoder_depths,
            d_state=d_state, ssm_ratio=ssm_ratio, drop_path_rate=drop_path_rate,
            ssm_conv=ssm_conv, Q_max_norm=Q_max_norm,
            Q_axis_temperature=Q_axis_temperature,
        )

    def _stage_prior(self, x):
        q1, q2 = self.nematic_prior(x)
        Q = QTensorTools.get_qtensor(q1, q2, 1.0)
        S = QTensorTools.get_s(q1, q2)
        director = QTensorTools.get_axial(Q[:, 0:1], Q[:, 1:2])
        return torch.cat([Q, S, director], dim=1)  # (B, 5, H, W)

    def forward(self, x):
        x_c = self.input_norm(x)
        nematic_prior = self._stage_prior(x)
        stage_features, stage_Q, stage_aux, Q_stem = self.encoder(x_c, nematic_prior)
        return {
            'stage_features': stage_features,
            'stage_Q': stage_Q,
            'stage_aux': stage_aux,
            'Q_stem': Q_stem,
        }


def build_Model(config):
    import inspect
    from datasets.dataset_config import get_dataset_config
    sig = inspect.signature(SGNet.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ('self', 'kwargs'):
            continue
        if name in ('input_channels', 'num_classes'):
            continue
        if hasattr(config, name):
            kwargs[name] = getattr(config, name)
    if hasattr(config, 'dataset_name'):
        ds = get_dataset_config(config.dataset_name)
        kwargs['input_channels'] = ds['channels']
        kwargs['num_classes'] = ds.get('num_classes', 2)
    return SGNet(**kwargs)
