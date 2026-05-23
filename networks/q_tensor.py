# -*- coding: utf-8 -*-
"""Q-tensor mathematical operations."""

import torch
import torch.nn.functional as F


class QTensorTools:
    """Nematic Q-tensor operations — single entry for all Q-tensor math.

    Every method receives q1 and q2 as separate arguments.
    No external script should perform Q-tensor math; only pass q1/q2 here.
    """

    @staticmethod
    def get_norm(q1, q2, eps=1e-8):
        return torch.sqrt(q1.pow(2) + q2.pow(2) + eps)

    @staticmethod
    def get_theta(q1, q2):
        return 0.5 * torch.atan2(q2, q1)

    @staticmethod
    def get_qtensor(q1, q2, Q_max_norm=1.0):
        Q = torch.cat([q1, q2], dim=1)
        n = QTensorTools.get_norm(q1, q2)
        scale = torch.clamp(float(Q_max_norm) / n, max=1.0)
        return Q * scale

    @staticmethod
    def get_s(q1, q2):
        return QTensorTools.get_norm(q1, q2).clamp(0, 1)

    @staticmethod
    def get_axial(q1, q2):
        n = QTensorTools.get_norm(q1, q2)
        cos2t = torch.where(n > 0, q1 / n, torch.zeros_like(q1))
        sin2t = torch.where(n > 0, q2 / n, torch.zeros_like(q2))
        return torch.cat([cos2t, sin2t], dim=1)

    @staticmethod
    def downsample_q(q1, q2, size=None, Q_max_norm=1.0):
        if size is None:
            size = (q1.shape[2] // 2, q1.shape[3] // 2)
        q1 = F.interpolate(q1, size=size, mode="area")
        q2 = F.interpolate(q2, size=size, mode="area")
        return QTensorTools.get_qtensor(q1, q2, Q_max_norm)
