# -*- coding: utf-8 -*-
"""Standalone Nematic loss adapted from utils/losses.Nematic.

Removes config dependency and the unused `batch` argument.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import QTensorTools


class NematicLossAdapter(nn.Module):
    _BOUNDARY_WIDTH = 2
    _DEFECT_S_THRESHOLD = 0.1
    _Q_FINAL_WEIGHT = 1.0
    _AXIS_TEMPERATURE = 0.35

    def __init__(self, Q_max_norm=1.0, ordered_weight=1.0,
                 boundary_weight=1.0, Q_stage_weights=(1.0, 0.5, 0.2, 0.0)):
        super().__init__()
        self.Q_max_norm = float(Q_max_norm)
        self.ordered_weight = float(ordered_weight)
        self.boundary_weight = float(boundary_weight)
        self.Q_stage_weights = list(Q_stage_weights)
        self.boundary_width = self._BOUNDARY_WIDTH
        self.defect_s_thr = self._DEFECT_S_THRESHOLD
        self.axis_temp = self._AXIS_TEMPERATURE
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
        self.register_buffer('sobel_kx', kx.reshape(1, 1, 3, 3))
        self.register_buffer('sobel_ky', ky.reshape(1, 1, 3, 3))

    def _resize(self, x, size, mode='bilinear'):
        if x.shape[2:] == size:
            return x
        if mode == 'nearest':
            return F.interpolate(x, size=size, mode=mode)
        return F.interpolate(x, size=size, mode=mode, align_corners=False)

    def _weighted_mean(self, value, weight):
        return (value * weight).sum() / weight.sum().clamp_min(1e-6)

    def _Q_items(self, outputs):
        Q_fields = outputs['stage_Q']
        items = []
        if 'Q_pred' in outputs:
            items.append(('final', outputs['Q_pred'], self._Q_FINAL_WEIGHT))
        weights = self.Q_stage_weights
        for i, Q in enumerate(Q_fields):
            w = float(weights[i]) if i < len(weights) else 0.0
            items.append((f'stage{i}', Q, w))
        return items

    def _fg_mask(self, targets, size):
        return self._resize((targets > 0).float().unsqueeze(1), size, mode='nearest')

    def _boundary_band(self, fg_mask):
        w = self.boundary_width
        k = 2 * w + 1
        fg_dilated = F.max_pool2d(fg_mask, kernel_size=k, stride=1, padding=w)
        fg_eroded = -F.max_pool2d(-fg_mask, kernel_size=k, stride=1, padding=w)
        return (fg_dilated - fg_eroded).clamp(0, 1)

    def _w_dist(self, fg_mask):
        band = self._boundary_band(fg_mask)
        diffused = band
        for _ in range(4):
            diffused = F.max_pool2d(diffused, kernel_size=3, stride=1, padding=1)
            diffused = diffused * fg_mask
        amax = diffused.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        diffused = diffused / amax
        return (diffused * fg_mask).clamp(0.05, 1.0)

    def _sobel_mask(self, mask):
        b, _, h, w = mask.shape
        f = mask.reshape(b, 1, h, w)
        f = F.pad(f, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(f, self.sobel_kx).reshape(b, 1, h, w)
        gy = F.conv2d(f, self.sobel_ky).reshape(b, 1, h, w)
        return gx, gy

    def L_ordered(self, Q_items, targets):
        total = Q_items[0][1].new_tensor(0.0)
        norm = 0.0
        for _, Q, w in Q_items:
            if w <= 0:
                continue
            fg = self._fg_mask(targets, Q.shape[2:])
            S_pred = QTensorTools.get_s(Q[:, 0:1], Q[:, 1:2])
            s_local = F.avg_pool2d(S_pred * fg, kernel_size=2, stride=1, padding=0)
            fg_local = F.avg_pool2d(fg, kernel_size=2, stride=1, padding=0)
            s_local_mean = s_local / fg_local.clamp_min(1e-6)
            loss_map = (0.7 - s_local_mean).clamp(min=0)
            mask = fg_local >= 0.5
            if mask.sum() > 0:
                total = total + w * loss_map[mask].mean()
                norm += w
        return total / norm if norm > 0 else Q_items[0][1].new_tensor(0.0)

    def L_boundary(self, Q_items, targets):
        total = Q_items[0][1].new_tensor(0.0)
        norm = 0.0
        for _, Q, w in Q_items:
            if w <= 0:
                continue
            fg = self._fg_mask(targets, Q.shape[2:])
            band = self._boundary_band(fg)
            weight = band * fg
            if weight.sum() < 1e-6:
                norm += w
                continue
            gx, gy = self._sobel_mask(fg)
            theta_t = torch.atan2(-gy, gx + 1e-8)
            theta_pred = QTensorTools.get_theta(Q[:, 0:1], Q[:, 1:2])
            diff = theta_pred - theta_t
            diff = torch.remainder(diff + math.pi / 2, math.pi) - math.pi / 2
            loss = diff.pow(2)
            total = total + w * self._weighted_mean(loss, weight)
            norm += w
        return total / norm if norm > 0 else Q_items[0][1].new_tensor(0.0)

    def forward(self, outputs, targets):
        Q_items = self._Q_items(outputs)
        if not Q_items:
            return outputs[list(outputs.keys())[0]].new_tensor(0.0), {}
        l_ord = self.L_ordered(Q_items, targets)
        l_bnd = self.L_boundary(Q_items, targets)
        total = (
            self.ordered_weight * l_ord
            + self.boundary_weight * l_bnd
        )
        comps = {
            'Nematic': total.item(),
            'N_ordered': l_ord.item(),
            'N_boundary': l_bnd.item(),
        }
        return total, comps
