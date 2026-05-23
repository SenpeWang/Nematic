# -*- coding: utf-8 -*-
"""Custom transform: convert label values (e.g. 255 -> 1)."""

from mmengine.dataset import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConvertLabel(BaseTransform):
    """Convert source label value to destination value.

    Args:
        src_val (int): Source value to replace.
        dst_val (int): Destination value.
    """

    def __init__(self, src_val, dst_val):
        self.src_val = int(src_val)
        self.dst_val = int(dst_val)

    def transform(self, results):
        """Transform function to convert label values."""
        if 'gt_seg_map' in results and results['gt_seg_map'] is not None:
            gt = results['gt_seg_map']
            gt[gt == self.src_val] = self.dst_val
            results['gt_seg_map'] = gt
        return results
