"""LiDAR-only BEVFormer detector with a temporal BEV encoder."""

from __future__ import annotations

import inspect
from typing import Iterable, List, Optional, Sequence

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData

from ..modules import warp_prev_bev


@MODELS.register_module()
class BEVFormerLidar(MVXTwoStageDetector):
    """LiDAR BEVFormer that fuses history BEV before the point head."""

    def __init__(self,
                 *args,
                 temporal_encoder: Optional[dict] = None,
                 map_head: Optional[dict] = None,
                 occ_head: Optional[dict] = None,
                 forecasting_head: Optional[dict] = None,
                 voxel_coord_order: str = 'zyx',
                 bev_feature_layout: str = 'yx',
                 point_cloud_range: Optional[Sequence[float]] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temporal_encoder = (MODELS.build(temporal_encoder)
                                 if temporal_encoder is not None else None)
        self.map_head = (MODELS.build(map_head)
                         if map_head is not None else None)
        self.forecasting_head = (MODELS.build(forecasting_head)
                                 if forecasting_head is not None else None)
        if voxel_coord_order not in ('zyx', 'xyz'):
            raise ValueError('voxel_coord_order must be "zyx" or "xyz", '
                             f'got {voxel_coord_order}.')
        if bev_feature_layout not in ('yx', 'xy'):
            raise ValueError('bev_feature_layout must be "yx" or "xy", '
                             f'got {bev_feature_layout}.')
        self.voxel_coord_order = voxel_coord_order
        self.bev_feature_layout = bev_feature_layout
        self.point_cloud_range = point_cloud_range
        if occ_head is not None:
            occ_head = dict(occ_head)
            occ_head.setdefault('bev_feature_layout',
                                self.bev_feature_layout)
            self.occ_head = MODELS.build(occ_head)
        else:
            self.occ_head = None

    def _maybe_reorder_voxel_dict(self, voxel_dict: dict) -> dict:
        """Align voxel coordinates with the sparse encoder's axis convention.

        Standard mmdet3d hard voxelization emits coordinates as ``[b, z, y, x]``.
        SparseEncoderXYZ expects ``[b, x, y, z]`` because BEVFusion-style
        voxelization keeps the original xyz index order.
        """
        if self.voxel_coord_order == 'zyx':
            return voxel_dict

        reordered = dict(voxel_dict)
        reordered['coors'] = voxel_dict['coors'][:, [0, 3, 2, 1]].contiguous()
        return reordered

    def extract_pts_bev_from_points(
            self, points: List[Tensor],
            batch_data_samples) -> List[Tensor]:
        """Extract single-frame LiDAR BEV features from raw points."""
        voxel_dict = self.data_preprocessor.voxelize(points, batch_data_samples)
        voxel_dict = self._maybe_reorder_voxel_dict(voxel_dict)
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

    def _to_temporal_bev_layout(self, bev: Optional[Tensor]) -> Optional[Tensor]:
        """Convert detector BEV layout into the temporal module's [Y, X] view."""
        if bev is None or self.bev_feature_layout == 'yx':
            return bev
        return bev.transpose(-1, -2).contiguous()

    def _from_temporal_bev_layout(self,
                                  bev: Optional[Tensor]) -> Optional[Tensor]:
        """Convert temporal-module BEV output back into detector layout."""
        if bev is None or self.bev_feature_layout == 'yx':
            return bev
        return bev.transpose(-1, -2).contiguous()

    def _bbox_head_predict_inputs(self, batch_data_samples):
        """Adapt predict() inputs to whichever bbox head is attached.

        mmdet3d heads are inconsistent here: some expect
        ``batch_data_samples`` while TransFusion-style heads expect a plain
        ``batch_input_metas`` list.
        """
        predict_params = tuple(
            inspect.signature(self.pts_bbox_head.predict).parameters)
        if len(predict_params) >= 2 and predict_params[1] == 'batch_input_metas':
            return [sample.metainfo for sample in batch_data_samples]
        return batch_data_samples

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
        temporal_current = self._to_temporal_bev_layout(current_bev)
        temporal_prev = self._to_temporal_bev_layout(prev_bev)
        fused = self.temporal_encoder(temporal_current, temporal_prev)
        return self._from_temporal_bev_layout(fused)

    def _warp_prev_bev_if_needed(self, prev_bev: Optional[Tensor],
                                 queue_meta: Sequence[dict]) -> Optional[Tensor]:
        if prev_bev is None or self.point_cloud_range is None:
            return prev_bev
        if any(meta is None or not meta.get('prev_bev_exists', False)
               for meta in queue_meta):
            return None
        deltas = [meta['ego_motion_delta'] for meta in queue_meta]
        temporal_prev = self._to_temporal_bev_layout(prev_bev)
        temporal_prev = warp_prev_bev(temporal_prev, deltas,
                                      self.point_cloud_range)
        return self._from_temporal_bev_layout(temporal_prev)

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
        losses = self.pts_bbox_head.loss(
            self._wrap_single_bev(fused_bev), batch_data_samples, **kwargs)
        if self.map_head is not None:
            losses.update(self.map_head.loss(fused_bev, batch_data_samples))
        if self.occ_head is not None:
            losses.update(self.occ_head.loss(fused_bev, batch_data_samples))
        if self.forecasting_head is not None:
            losses.update(self._forecasting_loss(fused_bev,
                                                 batch_data_samples))
        return losses

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
        head_predict_inputs = self._bbox_head_predict_inputs(batch_data_samples)
        results_list_3d = self.pts_bbox_head.predict(
            self._wrap_single_bev(fused_bev), head_predict_inputs, **kwargs)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list_3d, None)
        if self.map_head is not None:
            batch_data_samples = self.map_head.predict(fused_bev,
                                                       batch_data_samples)
        else:
            for data_sample in batch_data_samples:
                if 'pred_pts_seg' not in data_sample:
                    data_sample.pred_pts_seg = PointData()
        if self.occ_head is not None:
            batch_data_samples = self.occ_head.predict(fused_bev,
                                                       batch_data_samples)
        if self.forecasting_head is not None:
            self._forecasting_predict(fused_bev, batch_data_samples)
        return batch_data_samples

    # ----------------------- forecasting head plumbing ----------------------

    @staticmethod
    def _box_centers_and_state(boxes_3d, labels_3d):
        """Pull (cx, cy), (vx, vy), labels from a LiDARInstance3DBoxes obj.

        Velocity falls back to zeros when ``box_dim < 9`` (i.e. dataset was
        built with ``with_velocity=False``); the head can still train on
        the geometric sample even without a velocity prior.
        """
        centers = boxes_3d.gravity_center[:, :2]
        n = boxes_3d.tensor.shape[0]
        if boxes_3d.tensor.shape[1] >= 9:
            vels = boxes_3d.tensor[:, 7:9]
        else:
            vels = boxes_3d.tensor.new_zeros(n, 2)
        return centers, vels, labels_3d

    @staticmethod
    def _language_meta_lists(batch_data_samples):
        token_list, token_mask_list = [], []
        has_language = False
        for sample in batch_data_samples:
            tokens = sample.metainfo.get('language_tokens')
            token_mask = sample.metainfo.get('language_token_mask')
            if tokens is not None:
                has_language = True
                tokens = torch.as_tensor(tokens, dtype=torch.long)
            if token_mask is not None:
                token_mask = torch.as_tensor(token_mask, dtype=torch.bool)
            token_list.append(tokens)
            token_mask_list.append(token_mask)
        if not has_language:
            return None, None
        return token_list, token_mask_list

    @staticmethod
    def _language_target_masks(batch_data_samples):
        target_masks = []
        has_target = False
        for sample in batch_data_samples:
            gi = sample.gt_instances_3d
            target = getattr(gi, 'language_target_mask', None)
            if target is not None:
                has_target = True
            target_masks.append(target)
        if not has_target:
            return None
        return target_masks

    def _forecasting_loss(self, bev_feat: Tensor,
                          batch_data_samples) -> dict:
        """Run forecasting head against GT centers + GT trajectories.

        Returns ``{}`` when no sample carries forecasting GT (e.g. running
        against a config that didn't add ``gt_forecasting_locs/mask`` to
        ``Pack3DDetInputs.keys``) — head is then a no-op for that batch.
        """
        # Forecasting heads currently interpret BEV as [B, C, Y, X], matching
        # the original BEVFormer / CenterHead path. TransFusion stage3 keeps
        # detector-internal BEV as [B, C, X, Y], so adapt at this boundary.
        bev_feat = self._to_temporal_bev_layout(bev_feat)
        centers_list, vels_list, labels_list = [], [], []
        gt_locs_list, gt_mask_list = [], []
        for sample in batch_data_samples:
            gi = sample.gt_instances_3d
            if not (hasattr(gi, 'forecasting_locs')
                    and hasattr(gi, 'forecasting_mask')):
                return {}
            c, v, l = self._box_centers_and_state(gi.bboxes_3d, gi.labels_3d)
            centers_list.append(c)
            vels_list.append(v)
            labels_list.append(l)
            gt_locs_list.append(gi.forecasting_locs)
            gt_mask_list.append(gi.forecasting_mask)

        extra_kwargs = {}
        loss_params = inspect.signature(self.forecasting_head.loss).parameters
        if 'language_tokens_list' in loss_params:
            tokens, token_masks = self._language_meta_lists(batch_data_samples)
            target_masks = self._language_target_masks(batch_data_samples)
            extra_kwargs.update(
                language_tokens_list=tokens,
                language_token_mask_list=token_masks,
                language_target_mask_list=target_masks)
        return self.forecasting_head.loss(
            bev_feat, centers_list, vels_list, labels_list,
            gt_locs_list, gt_mask_list, **extra_kwargs)

    def _forecasting_predict(self, bev_feat: Tensor,
                             batch_data_samples) -> None:
        """Predict trajectories at detection centers, attach to data_samples."""
        bev_feat = self._to_temporal_bev_layout(bev_feat)
        centers_list, vels_list, labels_list = [], [], []
        for sample in batch_data_samples:
            pi = sample.pred_instances_3d
            c, v, l = self._box_centers_and_state(pi.bboxes_3d, pi.labels_3d)
            centers_list.append(c)
            vels_list.append(v)
            labels_list.append(l)
        tokens, token_masks = self._language_meta_lists(batch_data_samples)
        if (tokens is not None and
                hasattr(self.forecasting_head, 'forward_with_selection')):
            forecasts, selected_logits = (
                self.forecasting_head.forward_with_selection(
                    bev_feat, centers_list, vels_list, labels_list,
                    tokens, token_masks))
        else:
            forecasts = self.forecasting_head(
                bev_feat, centers_list, vels_list, labels_list)
            selected_logits = None
        for sample, traj in zip(batch_data_samples, forecasts):
            sample.pred_instances_3d.forecasting_3d = traj
        if selected_logits is not None:
            for sample, logits in zip(batch_data_samples, selected_logits):
                sample.pred_instances_3d.language_selected_score = (
                    logits.sigmoid())
