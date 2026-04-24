"""LiDAR-only BEVFormer detector with a temporal BEV encoder."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS

from ..modules import warp_prev_bev


@MODELS.register_module()
class BEVFormerLidar(MVXTwoStageDetector):
    """LiDAR BEVFormer that fuses history BEV before CenterHead."""

    def __init__(self,
                 *args,
                 temporal_encoder: Optional[dict] = None,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temporal_encoder = (MODELS.build(temporal_encoder)
                                 if temporal_encoder is not None else None)
        self.point_cloud_range = point_cloud_range

    def extract_pts_bev_from_points(
            self, points: List[Tensor],
            batch_data_samples) -> List[Tensor]:
        """Extract single-frame LiDAR BEV features from raw points."""
        voxel_dict = self.data_preprocessor.voxelize(points, batch_data_samples)
        batch_input_metas = [sample.metainfo for sample in batch_data_samples]
        return self.extract_pts_feat(
            voxel_dict,
            points=points,
            batch_input_metas=batch_input_metas)

    @staticmethod
    def _unwrap_single_bev(pts_feats: Sequence[Tensor]) -> Tensor:
        if not isinstance(pts_feats, (list, tuple)) or len(pts_feats) != 1:
            raise ValueError('Stage 3 BEVFormerLidar expects a single-level '
                             f'BEV feature list, got {type(pts_feats)} '
                             f'with len={len(pts_feats)}.')
        return pts_feats[0]

    @staticmethod
    def _wrap_single_bev(bev: Tensor) -> List[Tensor]:
        return [bev]

    @staticmethod
    def _normalize_history_points(history_points, batch_size: int):
        if history_points is None:
            return [[] for _ in range(batch_size)]
        if len(history_points) == 0:
            return [[] for _ in range(batch_size)]

        if batch_size == 1 and isinstance(history_points[0], Tensor):
            return [list(history_points)]

        if len(history_points) == batch_size and isinstance(history_points[0],
                                                            (list, tuple)):
            return [list(sample_history) for sample_history in history_points]

        if isinstance(history_points[0], (list, tuple)):
            per_sample = [[] for _ in range(batch_size)]
            for step_points in history_points:
                if len(step_points) != batch_size:
                    raise ValueError('history_points collate shape mismatch.')
                for batch_idx, points in enumerate(step_points):
                    per_sample[batch_idx].append(points)
            return per_sample

        raise TypeError('Unsupported history_points structure: '
                        f'{type(history_points)}')

    def _current_queue_meta(self, batch_data_samples) -> List[dict]:
        current = []
        for sample in batch_data_samples:
            queue_metas = sample.metainfo.get('queue_metas')
            if queue_metas is None:
                current.append(None)
                continue
            last_idx = max(queue_metas.keys())
            current.append(queue_metas[last_idx])
        return current

    def _fuse_bev(self,
                  current_bev: Tensor,
                  prev_bev: Optional[Tensor] = None) -> Tensor:
        if self.temporal_encoder is None:
            return current_bev
        return self.temporal_encoder(current_bev, prev_bev)

    def _warp_prev_bev_if_needed(self, prev_bev: Optional[Tensor],
                                 queue_meta: Sequence[dict]) -> Optional[Tensor]:
        if prev_bev is None or self.point_cloud_range is None:
            return prev_bev
        if any(meta is None or not meta.get('prev_bev_exists', False)
               for meta in queue_meta):
            return None
        deltas = [meta['ego_motion_delta'] for meta in queue_meta]
        return warp_prev_bev(prev_bev, deltas, self.point_cloud_range)

    def obtain_history_bev(self, history_points,
                           batch_data_samples) -> Optional[Tensor]:
        if self.temporal_encoder is None or history_points is None:
            return None

        batch_size = len(batch_data_samples)
        history_by_sample = self._normalize_history_points(
            history_points, batch_size)
        if not history_by_sample or len(history_by_sample[0]) == 0:
            return None

        num_history = len(history_by_sample[0])
        if any(len(sample_history) != num_history
               for sample_history in history_by_sample):
            raise ValueError('All samples must share the same history length.')

        queue_metas = [sample.metainfo['queue_metas'] for sample in batch_data_samples]
        prev_bev = None
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for step in range(num_history):
                    step_points = [
                        sample_history[step] for sample_history in history_by_sample
                    ]
                    step_bev = self._unwrap_single_bev(
                        self.extract_pts_bev_from_points(step_points,
                                                         batch_data_samples))
                    if prev_bev is not None:
                        step_meta = [sample_queue_metas[step]
                                     for sample_queue_metas in queue_metas]
                        prev_bev = self._warp_prev_bev_if_needed(
                            prev_bev, step_meta)
                    prev_bev = self._fuse_bev(step_bev, prev_bev)
        finally:
            if was_training:
                self.train()
        return prev_bev

    def loss(self, batch_inputs_dict, batch_data_samples,
             **kwargs) -> dict:
        current_bev = self._unwrap_single_bev(
            self.extract_pts_bev_from_points(batch_inputs_dict['points'],
                                             batch_data_samples))
        prev_bev = self.obtain_history_bev(
            batch_inputs_dict.get('history_points'), batch_data_samples)
        prev_bev = self._warp_prev_bev_if_needed(
            prev_bev, self._current_queue_meta(batch_data_samples))
        fused_bev = self._fuse_bev(current_bev, prev_bev)
        return self.pts_bbox_head.loss(
            self._wrap_single_bev(fused_bev), batch_data_samples, **kwargs)

    def predict(self, batch_inputs_dict, batch_data_samples,
                **kwargs):
        current_bev = self._unwrap_single_bev(
            self.extract_pts_bev_from_points(batch_inputs_dict['points'],
                                             batch_data_samples))
        prev_bev = None
        if 'history_points' in batch_inputs_dict:
            prev_bev = self.obtain_history_bev(batch_inputs_dict['history_points'],
                                               batch_data_samples)
            prev_bev = self._warp_prev_bev_if_needed(
                prev_bev, self._current_queue_meta(batch_data_samples))
        fused_bev = self._fuse_bev(current_bev, prev_bev)
        results_list_3d = self.pts_bbox_head.predict(
            self._wrap_single_bev(fused_bev), batch_data_samples, **kwargs)
        return self.add_pred_to_datasample(batch_data_samples, results_list_3d,
                                           None)
