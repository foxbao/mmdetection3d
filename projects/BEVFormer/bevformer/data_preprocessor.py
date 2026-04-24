"""Project-local data preprocessor for LiDAR BEVFormer.

This subclasses the standard mmdet3d 3D data preprocessor with two changes:

1. Pass through ``history_points`` from the dataset to the detector — Stage 2's
   ``KlBEVFormerDataset`` packs the queue's older frames there.
2. Skip the auto-voxelization that the parent runs in ``simple_process``. The
   detector calls ``self.data_preprocessor.voxelize`` on demand for both the
   current frame and each history step, so pre-computing voxels for the
   current frame here would just be discarded work. ``voxel=True`` is still
   required in the config so ``voxel_layer`` is constructed.
"""

from __future__ import annotations

from typing import List, Union

from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet3d.models import Det3DDataPreprocessor
from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVFormerDataPreprocessor(Det3DDataPreprocessor):
    """Det3D preprocessor with temporal point-cloud pass-through."""

    def simple_process(self,
                       data: dict,
                       training: bool = False) -> Union[dict, List[dict]]:
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

        if 'history_points' in inputs:
            batch_inputs['history_points'] = inputs['history_points']

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}
