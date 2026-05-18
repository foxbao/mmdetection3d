"""Standalone UniAD-like LiDAR BEV detector.

This detector intentionally does not inherit ``BEVFormerLidar``. The old
content-driven temporal BEV path remains available for ablation, while this
class uses explicit names for the new query-driven BEV encoder path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from mmengine.dist import get_world_size
from torch import Tensor, nn

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData


@MODELS.register_module()
class BEVFormerLidarUniAD(MVXTwoStageDetector):
    """LiDAR-only detector with a UniAD-like query-driven BEV encoder.

    Object queries and reference points are owned by the detector, matching
    UniAD's track-head boundary. References are kept in inverse-sigmoid/logit
    space before they are passed into the BEVDETR decoder.
    """

    def __init__(self,
                 *args,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 num_query: int = 600,
                 embed_dims: int = 256,
                 use_prev_bev: bool = True,
                 video_test_mode: bool = True,
                 eval_prev_bev_mode: str = 'auto',
                 forecasting_head: Optional[dict] = None,
                 train_bbox_head: bool = True,
                 forecasting_detach_bev: bool = False,
                 freeze_detector_for_forecasting: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        valid_eval_modes = {'auto', 'online', 'history'}
        if eval_prev_bev_mode not in valid_eval_modes:
            raise ValueError(
                'eval_prev_bev_mode must be one of '
                f'{sorted(valid_eval_modes)}, got {eval_prev_bev_mode!r}.')
        self.point_cloud_range = point_cloud_range
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.use_prev_bev = use_prev_bev
        self.video_test_mode = video_test_mode
        self.eval_prev_bev_mode = eval_prev_bev_mode
        self.train_bbox_head = bool(train_bbox_head)
        self.forecasting_detach_bev = bool(forecasting_detach_bev)
        self.freeze_detector_for_forecasting = bool(
            freeze_detector_for_forecasting)
        self.forecasting_head = (MODELS.build(forecasting_head)
                                 if forecasting_head is not None else None)
        self.query_embedding = nn.Embedding(num_query, embed_dims * 2)
        self.reference_points = nn.Linear(embed_dims, 3)
        self._test_prev_bev: Optional[Tensor] = None
        self._test_scene_token: Optional[str] = None
        self._init_detector_queries()
        if self.freeze_detector_for_forecasting:
            self._freeze_detector_modules()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_detector_for_forecasting:
            for name in self._detector_module_names():
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()
            if self.forecasting_head is not None:
                self.forecasting_head.train(mode)
        return self

    @staticmethod
    def _detector_module_names() -> Sequence[str]:
        return (
            'pts_voxel_encoder',
            'pts_middle_encoder',
            'pts_backbone',
            'pts_neck',
            'pts_bbox_head',
            'query_embedding',
            'reference_points',
        )

    def _freeze_detector_modules(self) -> None:
        for name in self._detector_module_names():
            module = getattr(self, name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(False)

    def extract_lidar_bev_from_points(self, points: List[Tensor],
                                      batch_data_samples) -> Tensor:
        """Extract single-frame LiDAR BEV features from raw points.

        Voxel coords come out as ``[b, z, y, x]`` and the dense BEV is laid
        out as ``[B, C, Y, X]`` — both standard mmdet3d conventions matching
        UniAD's BEV encoder boundary.
        """
        voxel_dict = self.data_preprocessor.voxelize(points, batch_data_samples)
        batch_input_metas = [sample.metainfo for sample in batch_data_samples]
        pts_feats = self.extract_pts_feat(
            voxel_dict,
            points=points,
            batch_input_metas=batch_input_metas)
        return self._unwrap_single_bev(pts_feats)

    @staticmethod
    def _unwrap_single_bev(pts_feats) -> Tensor:
        if not isinstance(pts_feats, (list, tuple)) or len(pts_feats) != 1:
            raise ValueError('BEVFormerLidarUniAD expects a single-level BEV '
                             f'feature list, got {type(pts_feats)} with '
                             f'len={len(pts_feats)}.')
        return pts_feats[0]

    @staticmethod
    def _wrap_single_bev(bev: Tensor) -> List[Tensor]:
        return [bev]

    def encode_bev(self,
                   lidar_bev: Tensor,
                   prev_bev: Optional[Tensor] = None,
                   queue_meta: Optional[Sequence[dict]] = None) -> Tensor:
        if not self.use_prev_bev:
            prev_bev = None
        return self.pts_bbox_head.get_bev_features(
            lidar_bev, prev_bev=prev_bev, queue_meta=queue_meta)

    def valid_prev_bev(self, prev_bev: Optional[Tensor],
                       queue_meta: Optional[Sequence[dict]]
                       ) -> Optional[Tensor]:
        if prev_bev is None:
            return None
        if not self.use_prev_bev:
            return None
        if queue_meta is None:
            return prev_bev
        if any(meta is None or not meta.get('prev_bev_exists', False)
               for meta in queue_meta):
            return None
        return prev_bev

    @staticmethod
    def _normalize_history_points(history_points, batch_size: int):
        if history_points is None or len(history_points) == 0:
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

    def current_queue_meta(self, batch_data_samples) -> List[dict]:
        current = []
        for sample in batch_data_samples:
            queue_metas = sample.metainfo.get('queue_metas')
            if queue_metas is None:
                current.append(None)
                continue
            last_idx = max(queue_metas.keys())
            current.append(queue_metas[last_idx])
        return current

    def _init_detector_queries(self) -> None:
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0)

    def generate_init_query_embeds(self) -> Tensor:
        return self.query_embedding.weight

    def generate_init_ref_points(self,
                                 query_embeds: Optional[Tensor] = None
                                 ) -> Tensor:
        if query_embeds is None:
            query_embeds = self.generate_init_query_embeds()
        dim = query_embeds.shape[-1] // 2
        return self.reference_points(query_embeds[:, :dim])

    def _detector_query_inputs(self, batch_size: int, device: torch.device,
                               dtype: torch.dtype):
        query_embeds = self.generate_init_query_embeds().to(device=device)
        ref_points = self.generate_init_ref_points(query_embeds)
        query_embeds = query_embeds.to(dtype=dtype)
        ref_points = ref_points.to(dtype=dtype)
        ref_points = ref_points[None].expand(batch_size, -1, -1)
        return query_embeds, ref_points

    def obtain_history_bev(self, history_points,
                           batch_data_samples) -> Optional[Tensor]:
        if not self.use_prev_bev:
            return None
        if history_points is None:
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
                    step_lidar_bev = self.extract_lidar_bev_from_points(
                        step_points, batch_data_samples)
                    step_meta = None
                    if prev_bev is not None:
                        step_meta = [sample_queue_metas[step]
                                     for sample_queue_metas in queue_metas]
                        prev_bev = self.valid_prev_bev(prev_bev, step_meta)
                        if prev_bev is None:
                            step_meta = None
                    prev_bev = self.encode_bev(
                        step_lidar_bev, prev_bev, queue_meta=step_meta)
        finally:
            if was_training:
                self.train()
        return prev_bev

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs) -> dict:
        freeze_bev_graph = (
            self.forecasting_head is not None and
            self.forecasting_detach_bev and not self.train_bbox_head)
        if freeze_bev_graph:
            with torch.no_grad():
                bev_embed, _ = self._extract_current_bev_embed(
                    batch_inputs_dict, batch_data_samples)
        else:
            bev_embed, _ = self._extract_current_bev_embed(
                batch_inputs_dict, batch_data_samples)

        losses = {}
        if self.train_bbox_head:
            query_embeds, ref_points = self._detector_query_inputs(
                bev_embed.size(0), bev_embed.device, bev_embed.dtype)
            preds = self.pts_bbox_head.get_detections(
                self._wrap_single_bev(bev_embed),
                object_query_embeds=query_embeds,
                ref_points=ref_points)
            batch_gt_instances = [
                sample.gt_instances_3d for sample in batch_data_samples
            ]
            losses.update(
                self.pts_bbox_head.loss_by_feat(preds, batch_gt_instances))

        if self.forecasting_head is not None:
            forecasting_bev = (
                bev_embed.detach() if self.forecasting_detach_bev else
                bev_embed)
            losses.update(
                self._forecasting_loss(forecasting_bev, batch_data_samples))
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        lidar_bev = self.extract_lidar_bev_from_points(
            batch_inputs_dict['points'], batch_data_samples)
        current_meta = self.current_queue_meta(batch_data_samples)
        prev_bev = self._predict_prev_bev(
            batch_inputs_dict, batch_data_samples, current_meta)
        bev_embed = self.encode_bev(
            lidar_bev, prev_bev, queue_meta=current_meta)
        query_embeds, ref_points = self._detector_query_inputs(
            bev_embed.size(0), bev_embed.device, bev_embed.dtype)
        preds = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed),
            object_query_embeds=query_embeds,
            ref_points=ref_points)
        batch_metas = [sample.metainfo for sample in batch_data_samples]
        results_list_3d = self.pts_bbox_head.predict_by_feat(
            preds, batch_metas)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list_3d, None)
        for data_sample in batch_data_samples:
            if 'pred_pts_seg' not in data_sample:
                data_sample.pred_pts_seg = PointData()
        if self.forecasting_head is not None:
            self._forecasting_predict(bev_embed, batch_data_samples)
        if self.video_test_mode and len(batch_data_samples) == 1:
            self._test_prev_bev = bev_embed.detach()
        return batch_data_samples

    def _extract_current_bev_embed(self, batch_inputs_dict, batch_data_samples):
        lidar_bev = self.extract_lidar_bev_from_points(
            batch_inputs_dict['points'], batch_data_samples)
        prev_bev = self.obtain_history_bev(
            batch_inputs_dict.get('history_points'), batch_data_samples)
        current_meta = self.current_queue_meta(batch_data_samples)
        prev_bev = self.valid_prev_bev(prev_bev, current_meta)
        bev_embed = self.encode_bev(
            lidar_bev, prev_bev, queue_meta=current_meta)
        return bev_embed, current_meta

    @staticmethod
    def _box_centers_and_state(boxes_3d, labels_3d):
        centers = boxes_3d.gravity_center[:, :2]
        num_boxes = boxes_3d.tensor.shape[0]
        if boxes_3d.tensor.shape[1] >= 9:
            velocities = boxes_3d.tensor[:, 7:9]
        else:
            velocities = boxes_3d.tensor.new_zeros(num_boxes, 2)
        return centers, velocities, labels_3d

    def _forecasting_loss(self, bev_feat: Tensor,
                          batch_data_samples) -> dict:
        centers_list, velocities_list, labels_list = [], [], []
        gt_locs_list, gt_mask_list = [], []
        for sample in batch_data_samples:
            gt_instances = sample.gt_instances_3d
            if not (hasattr(gt_instances, 'forecasting_locs')
                    and hasattr(gt_instances, 'forecasting_mask')):
                return {}
            centers, velocities, labels = self._box_centers_and_state(
                gt_instances.bboxes_3d, gt_instances.labels_3d)
            centers_list.append(centers)
            velocities_list.append(velocities)
            labels_list.append(labels)
            gt_locs_list.append(gt_instances.forecasting_locs)
            gt_mask_list.append(gt_instances.forecasting_mask)
        return self.forecasting_head.loss(
            bev_feat, centers_list, velocities_list, labels_list,
            gt_locs_list, gt_mask_list)

    def _forecasting_predict(self, bev_feat: Tensor,
                             batch_data_samples) -> None:
        centers_list, velocities_list, labels_list = [], [], []
        for sample in batch_data_samples:
            pred_instances = sample.pred_instances_3d
            centers, velocities, labels = self._box_centers_and_state(
                pred_instances.bboxes_3d, pred_instances.labels_3d)
            centers_list.append(centers)
            velocities_list.append(velocities)
            labels_list.append(labels)
        forecasts = self.forecasting_head(
            bev_feat, centers_list, velocities_list, labels_list)
        for sample, forecast in zip(batch_data_samples, forecasts):
            sample.pred_instances_3d.forecasting_3d = forecast

    def _predict_prev_bev(self, batch_inputs_dict, batch_data_samples,
                          current_meta: List[dict]) -> Optional[Tensor]:
        """Return prev_bev for evaluation.

        ``online`` mirrors UniAD's scene-token gated cache and is the intended
        single-card sequential video-test path. In DDP evaluation the default
        sampler shards indices as ``rank::world_size``, so each rank no longer
        sees consecutive frames. ``history`` rebuilds prev_bev from the sample's
        own queue and is DDP-safe at the cost of extra BEV encoder work.
        """
        if not self.use_prev_bev:
            return None
        mode = self.eval_prev_bev_mode
        if mode == 'auto':
            mode = 'history' if get_world_size() > 1 else 'online'
        if mode == 'history':
            prev_bev = self.obtain_history_bev(
                batch_inputs_dict.get('history_points'), batch_data_samples)
            return self.valid_prev_bev(prev_bev, current_meta)
        if not (self.video_test_mode and len(batch_data_samples) == 1):
            return None
        current_scene = None
        if current_meta and current_meta[0] is not None:
            current_scene = current_meta[0].get('scene_token')
        if current_scene != self._test_scene_token:
            self._test_prev_bev = None
            self._test_scene_token = current_scene
        return self.valid_prev_bev(self._test_prev_bev, current_meta)
