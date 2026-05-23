# Copyright (c) OpenMMLab. All rights reserved.
"""Extended LoadAnnotations with optional label mapping.

Replaces the separate ConvertLabel step by embedding label value
conversion directly into annotation loading.
"""

from mmseg.datasets.transforms.loading import LoadAnnotations as _LoadAnnotations
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class LoadAnnotations(_LoadAnnotations):
    """Load annotations with optional label mapping.

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Defaults to None.
        backend_args (dict): Arguments to instantiate a file backend.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type.
            Defaults to 'pillow'.
        label_mapping (dict, optional): Mapping from old label ids to new
            label ids, e.g. ``{255: 1}``. Applied after reduce_zero_label.
            Defaults to None.
    """

    def __init__(self,
                 reduce_zero_label=None,
                 backend_args=None,
                 imdecode_backend='pillow',
                 label_mapping=None) -> None:
        super().__init__(
            reduce_zero_label=reduce_zero_label,
            backend_args=backend_args,
            imdecode_backend=imdecode_backend)
        self.label_mapping = label_mapping

    def _load_seg_map(self, results):
        """Load semantic segmentation map and apply label mapping."""
        super()._load_seg_map(results)
        if self.label_mapping is not None and 'gt_seg_map' in results:
            gt = results['gt_seg_map'].copy()
            gt_copy = gt.copy()
            for src, dst in self.label_mapping.items():
                gt[gt_copy == src] = dst
            results['gt_seg_map'] = gt
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'label_mapping={self.label_mapping})'
        return repr_str
