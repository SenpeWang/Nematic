# -*- coding: utf-8 -*-
"""MMSeg integration for SGNet3.

MM_SGNetBackbone:
  Wraps SGNet encoder + input_norm + nematic_prior.
  Outputs list[Tensor] (features at out_indices) and stores last_stage_Q.

SGNetSegmentor:
  Extends EncoderDecoder with Nematic/Q-pred auxiliary loss.
  Keeps decode_head unmodified (standard UPerHead).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.registry import MODELS as MODELS_MMSEG

def _import_abspy(name="networks", path="../"):
    import sys
    import importlib
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    assert os.path.isdir(path), f"Path {path} does not exist"
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


_networks = _import_abspy("networks", "../")
SGNet = _networks.SGNet
QTensorTools = _networks.QTensorTools

from .models.losses.nematic_loss import NematicLossAdapter


@MODELS_MMSEG.register_module()
class MM_SGNetBackbone(BaseModule):
    """MMSeg-compatible backbone for SGNet3.

    Wraps the SGNet encoder (stem + local + global stages).
    The nematic prior is computed on-the-fly from the input image.
    stage_Q from the last forward pass is stored in ``last_stage_Q``
    so that the segmentor can retrieve it for Nematic loss computation.
    """

    def __init__(self,
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
                 init_cfg=None):
        super().__init__(init_cfg)
        # Build a full SGNet only to reuse its encoder / prior / norm.
        self.sgnet = SGNet(
            input_channels=input_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            encoder_depths=encoder_depths,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            drop_path_rate=drop_path_rate,
            ssm_conv=ssm_conv,
            Q_max_norm=Q_max_norm,
            Q_axis_temperature=Q_axis_temperature,
        )
        self.encoder = self.sgnet.encoder
        self.nematic_prior = self.sgnet.nematic_prior
        self.input_norm = self.sgnet.input_norm
        self.out_indices = out_indices
        self.last_stage_Q = None

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): (B, C, H, W) pre-processed image.

        Returns:
            list[Tensor]: Feature maps at ``out_indices``.
        """
        x_c = self.input_norm(x)
        nematic_prior = self.sgnet._stage_prior(x)
        stage_features, stage_Q, _, _ = self.encoder(x_c, nematic_prior)
        self.last_stage_Q = stage_Q
        return [stage_features[i] for i in self.out_indices]


@MODELS_MMSEG.register_module()
class SGNetSegmentor(EncoderDecoder):
    """Custom segmentor that adds Nematic/Q-pred loss on top of UPerHead.

    The decode_head is left completely unmodified (standard UPerHead).
    The extra loss is injected by overriding ``loss()``.
    """

    def __init__(self,
                 Q_max_norm=1.0,
                 lambda_seg=1.0,
                 lambda_nematic=0.3,
                 nematic_ordered_weight=1.0,
                 nematic_boundary_weight=1.0,
                 Q_stage_weights=(1.0, 0.5, 0.2, 0.0),
                 **kwargs):
        super().__init__(**kwargs)
        self.Q_max_norm = Q_max_norm
        self.lambda_seg = lambda_seg
        self.lambda_nematic = lambda_nematic

        # Lightweight Q-pred head (mirrors UPerQDecoder.q_refiner)
        channels = self.decode_head.channels
        self.q_refiner = nn.Sequential(
            nn.Conv2d(channels + 2, max(channels // 2, 16), 3, padding=1, bias=False),
            nn.BatchNorm2d(max(channels // 2, 16)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 2, 16), 2, 1, bias=False),
        )
        self.q_eta = nn.Parameter(torch.tensor(0.2))

        # Stand-alone Nematic loss (no config dependency)
        self.nematic_loss = NematicLossAdapter(
            Q_max_norm=Q_max_norm,
            ordered_weight=nematic_ordered_weight,
            boundary_weight=nematic_boundary_weight,
            Q_stage_weights=list(Q_stage_weights),
        )

    def loss(self, inputs, data_samples):
        """Compute segmentation + Nematic losses."""
        # 1. Extract features (also populates backbone.last_stage_Q)
        features = self.extract_feat(inputs)

        # 2. Segmentation head: get bottleneck feature + logits
        #    We call _forward_feature + cls_seg manually so that we can reuse
        #    the bottleneck feature for Q-pred without modifying the head.
        feat = self.decode_head._forward_feature(features)
        seg_logits = self.decode_head.cls_seg(feat)

        # 3. Standard segmentation losses (CE + Dice, etc.)
        losses = self.decode_head.loss_by_feat(seg_logits, data_samples)

        # 4. Nematic / Q-pred loss
        if (hasattr(self.backbone, 'last_stage_Q')
                and self.backbone.last_stage_Q is not None):

            stage_Q = self.backbone.last_stage_Q

            # Upsample deepest stage_Q to bottleneck resolution
            Q4_up = resize(
                stage_Q[-1],
                size=feat.shape[2:],
                mode='bilinear',
                align_corners=self.decode_head.align_corners,
            )

            # Refine Q-prediction
            dq = torch.tanh(
                self.q_refiner(torch.cat([feat, Q4_up], dim=1))
            ) * self.Q_max_norm
            Q_pred = QTensorTools.get_qtensor(
                (Q4_up + self.q_eta.tanh() * dq)[:, 0:1],
                (Q4_up + self.q_eta.tanh() * dq)[:, 1:2],
                self.Q_max_norm,
            )

            # Gather GT labels from data_samples
            gt = torch.stack([s.gt_sem_seg.data for s in data_samples], dim=0)
            # Squeeze possible channel dim (MMSeg stores H,W or 1,H,W)
            if gt.dim() == 4 and gt.shape[1] == 1:
                gt = gt.squeeze(1)

            # Compute Nematic loss
            outputs = {
                'Q_pred': Q_pred,
                'stage_Q': stage_Q,
            }
            nematic_loss, nematic_comps = self.nematic_loss(outputs, gt)
            losses['loss_nematic'] = self.lambda_nematic * nematic_loss

            # Optional: log components for debugging
            losses['n_ordered'] = torch.tensor(nematic_comps['N_ordered'])
            losses['n_boundary'] = torch.tensor(nematic_comps['N_boundary'])

        # Auxiliary head losses (standard MMSeg behaviour)
        if self.with_auxiliary_head:
            losses_aux = self._auxiliary_head_forward_train(
                features, data_samples)
            losses.update(losses_aux)

        return losses
