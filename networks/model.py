# -*- coding: utf-8 -*-
"""NFAM-Net top-level model."""

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .qtensor import QImagePrior, QTensorTools
from datasets.dataset_config import get_dataset_config


class NFAMNet(nn.Module):
    """NFAMNet: Nematic Field-Aware Modeling network.

    Args:
        config: experiment config (encoder_depths, drop_path_rate)
        in_chans: input image channels (from dataset registry)
    """
    def __init__(self, config, in_chans, stats_logger=None):
        super().__init__()
        self.input_norm = nn.GroupNorm(1, in_chans)
        self.Q_image_prior = QImagePrior(tangent=True)
        self.encoder = Encoder(in_chans, config, stats_logger)
        self.decoder = Decoder(self.encoder.dims)

    @torch.no_grad()
    def _normalize_for_Q_prior(self, x):
        x_float = x.float()
        x_min = x_float.amin(dim=(-2, -1), keepdim=True)
        x_max = x_float.amax(dim=(-2, -1), keepdim=True)
        span = (x_max - x_min).clamp_min(1e-6)
        return (x_float - x_min) / span

    def _stage_prior(self, x):
        x_q = self._normalize_for_Q_prior(x)
        q1, q2, _ = self.Q_image_prior(x_q)
        Q_S = QTensorTools.get_Nematic(q1, q2)
        return Q_S

    def forward(self, x):
        x_c = self.input_norm(x)
        Q_image_prior = self._stage_prior(x)
        stage_features, stage_Q, Q_stem = self.encoder(x_c, Q_image_prior)
        logits = self.decoder(stage_features, x_c)
        return {
            "logits": logits,
            "stage_Q": stage_Q,
        }


def build_Model(config, dataset_name='NEURO', stats_logger=None):
    """Build NFAMNet from config.

    input_channels comes from dataset registry, not from config.
    """
    ds = get_dataset_config(dataset_name)
    return NFAMNet(config, ds['channels'], stats_logger)
