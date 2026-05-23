# Copyright (c) OpenMMLab. All rights reserved.
"""Percentile-based normalization for microscopy images.

Avoids outlier distortion by using p1/p99 instead of min/max.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PercentileNormalize(BaseTransform):
    """Normalize image using percentile-based min-max scaling.

    Args:
        lower (float): Lower percentile (e.g. 1.0). Defaults to 1.
        upper (float): Upper percentile (e.g. 99.0). Defaults to 99.
        per_channel (bool): Whether to compute percentiles per channel.
            Defaults to True.
    """

    def __init__(self, lower=1.0, upper=99.0, per_channel=True):
        self.lower = float(lower)
        self.upper = float(upper)
        self.per_channel = per_channel

    def transform(self, results):
        """Normalize image in results['img']."""
        img = results['img']  # numpy array, typically (C, H, W) or (H, W, C)
        img = img.astype(np.float32)

        if self.per_channel and img.ndim == 3:
            # Assume channel-first (C, H, W) — standard after LoadImageFromFile
            # with tifffile backend. If channel-last, the logic still works
            # because we iterate over the first dimension.
            for c in range(img.shape[0]):
                ch = img[c]
                p_low = np.percentile(ch, self.lower)
                p_high = np.percentile(ch, self.upper)
                scale = p_high - p_low
                if scale < 1e-8:
                    scale = 1.0
                img[c] = np.clip((ch - p_low) / scale, 0.0, 1.0)
        else:
            p_low = np.percentile(img, self.lower)
            p_high = np.percentile(img, self.upper)
            scale = p_high - p_low
            if scale < 1e-8:
                scale = 1.0
            img = np.clip((img - p_low) / scale, 0.0, 1.0)

        results['img'] = img
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(lower={self.lower}, "
                f"upper={self.upper}, per_channel={self.per_channel})")
