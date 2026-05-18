"""Standalone UniAD-like LiDAR BEV detector with MOTR-style tracking."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData

from ..dense_heads.bev_detr_head import denormalize_bbox, inverse_sigmoid
from ..dense_heads.track_head_plugin import (Instances, MemoryBank,
                                              QueryInteractionModule,
                                              RuntimeTrackerBase)
from .bevformer_lidar_uniad import BEVFormerLidarUniAD


@MODELS.register_module()
class UniADTrackLiDAR(BEVFormerLidarUniAD):
    """MOTR-style tracking on top of ``BEVFormerLidarUniAD``.

    Track references follow UniAD's logit-space convention. They are converted
    to normalized coordinates only when applying velocity/ego-motion updates or
    when the detection decoder samples BEV memory.
    """

    def __init__(self,
                 *args,
                 track_loss_cfg: Optional[dict] = None,
                 motion_head: Optional[dict] = None,
                 freeze_lidar_backbone: bool = False,
                 freeze_lidar_neck: bool = False,
                 freeze_bev_encoder: bool = False,
                 qim_args: Optional[dict] = None,
                 score_thresh: float = 0.4,
                 filter_score_thresh: float = 0.3,
                 miss_tolerance: int = 3,
                 num_query: int = 300,
                 embed_dims: int = 256,
                 num_classes: int = 15,
                 reset_track_query_each_frame: bool = False,
                 use_velocity_ref_update: bool = True,
                 debug_track: bool = False,
                 debug_track_interval: int = 1,
                 debug_track_max_frames: int = 50,
                 debug_track_class_stats: bool = False,
                 class_names: Optional[List[str]] = None,
                 **kwargs) -> None:
        super().__init__(
            *args, num_query=num_query, embed_dims=embed_dims, **kwargs)
        if track_loss_cfg is None:
            raise ValueError('track_loss_cfg is required.')
        self.criterion = MODELS.build(track_loss_cfg)
        self.motion_head = (MODELS.build(motion_head)
                            if motion_head is not None else None)
        self.freeze_lidar_backbone = bool(freeze_lidar_backbone)
        self.freeze_lidar_neck = bool(freeze_lidar_neck)
        self.freeze_bev_encoder = bool(freeze_bev_encoder)
        qim_args = qim_args or {}
        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims)
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance)
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        # Diagnostic switch: keep temporal BEV, but disable object-query/QIM
        # carry-over at inference. This isolates whether t-1 track queries are
        # degrading current-frame detection.
        self.reset_track_query_each_frame = reset_track_query_each_frame
        # Keep ego-motion ref warping available while allowing velocity-only
        # propagation to be disabled for diagnostics.
        self.use_velocity_ref_update = use_velocity_ref_update
        self.debug_track = debug_track
        self.debug_track_interval = max(int(debug_track_interval), 1)
        self.debug_track_max_frames = int(debug_track_max_frames)
        self.debug_track_class_stats = bool(debug_track_class_stats)
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        self.class_names = list(class_names)

        self._test_track_instances: Optional[Instances] = None
        self._test_scene_token: Optional[str] = None
        self._test_prev_bev: Optional[Tensor] = None
        self._debug_track_frame = 0
        self._debug_prev_centers = {}
        if self.freeze_lidar_backbone:
            self._freeze_lidar_backbone_modules()
        if self.freeze_lidar_neck:
            self._freeze_lidar_neck_modules()
        if self.freeze_bev_encoder:
            self._freeze_bev_encoder_modules()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.freeze_lidar_backbone:
                self._set_lidar_backbone_eval()
            if self.freeze_lidar_neck:
                self._set_lidar_neck_eval()
            if self.freeze_bev_encoder:
                self._set_bev_encoder_eval()
        return self

    @staticmethod
    def _set_module_requires_grad(module, requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _lidar_backbone_modules(self):
        return [
            getattr(self, name, None)
            for name in ('pts_voxel_encoder', 'pts_middle_encoder',
                         'pts_backbone')
        ]

    def _lidar_neck_modules(self):
        return [getattr(self, 'pts_neck', None)]

    def _bev_encoder_modules(self):
        head = getattr(self, 'pts_bbox_head', None)
        if head is None:
            return []
        return [
            getattr(head, name, None)
            for name in ('lidar_input_proj', 'bev_embedding',
                         'positional_encoding', 'transformer')
        ]

    def _freeze_lidar_backbone_modules(self) -> None:
        for module in self._lidar_backbone_modules():
            self._set_module_requires_grad(module, False)
        self._set_lidar_backbone_eval()

    def _freeze_lidar_neck_modules(self) -> None:
        for module in self._lidar_neck_modules():
            self._set_module_requires_grad(module, False)
        self._set_lidar_neck_eval()

    def _freeze_bev_encoder_modules(self) -> None:
        for module in self._bev_encoder_modules():
            self._set_module_requires_grad(module, False)
        self._set_bev_encoder_eval()

    def _set_lidar_backbone_eval(self) -> None:
        for module in self._lidar_backbone_modules():
            if module is not None:
                module.eval()

    def _set_lidar_neck_eval(self) -> None:
        for module in self._lidar_neck_modules():
            if module is not None:
                module.eval()

    def _set_bev_encoder_eval(self) -> None:
        for module in self._bev_encoder_modules():
            if module is not None:
                module.eval()

    def _extract_lidar_bev_for_track(self, points, batch_data_samples) -> Tensor:
        if self.freeze_lidar_backbone and self.freeze_lidar_neck:
            with torch.no_grad():
                return self.extract_lidar_bev_from_points(
                    points, batch_data_samples).detach()
        return self.extract_lidar_bev_from_points(points, batch_data_samples)

    def _encode_bev_for_track(self, lidar_bev: Tensor,
                              prev_bev: Optional[Tensor],
                              queue_meta: Optional[Sequence[dict]]) -> Tensor:
        if self.freeze_bev_encoder:
            with torch.no_grad():
                return self.encode_bev(
                    lidar_bev, prev_bev, queue_meta=queue_meta).detach()
        return self.encode_bev(lidar_bev, prev_bev, queue_meta=queue_meta)

    @staticmethod
    def _track_centers_and_state(instances: Instances):
        boxes = instances.pred_boxes.detach()
        labels = instances.pred_logits.detach().sigmoid().argmax(dim=-1)
        centers = boxes[:, :2] if boxes.numel() else boxes.new_zeros((0, 2))
        if boxes.shape[-1] >= 10:
            velocities = boxes[:, 8:10]
        else:
            velocities = boxes.new_zeros((boxes.shape[0], 2))
        return centers, velocities, labels

    def _empty_motion_inputs(self) -> dict:
        return dict(
            query_embeddings_list=[],
            centers_list=[],
            velocities_list=[],
            labels_list=[],
            gt_locs_list=[],
            gt_mask_list=[])

    def _motion_head_uses_track_query(self) -> bool:
        return bool(getattr(self.motion_head, 'uses_track_query', False))

    def _motion_targets_from_tracks(self, track_instances: Instances,
                                    gt_instances):
        if not (hasattr(gt_instances, 'forecasting_locs')
                and hasattr(gt_instances, 'forecasting_mask')):
            return self._empty_motion_inputs()
        if len(track_instances) == 0:
            return self._empty_motion_inputs()
        active = track_instances.obj_idxes >= 0
        if not bool(active.any()):
            return self._empty_motion_inputs()
        matched = track_instances.matched_gt_idxes[active].long()
        valid = matched >= 0
        if not bool(valid.any()):
            return self._empty_motion_inputs()
        active_tracks = track_instances[active][valid]
        matched = matched[valid]
        centers, velocities, labels = self._track_centers_and_state(
            active_tracks)
        gt_locs = gt_instances.forecasting_locs[matched]
        gt_mask = gt_instances.forecasting_mask[matched]
        query_embeddings = active_tracks.output_embedding.detach()
        return dict(
            query_embeddings_list=[query_embeddings],
            centers_list=[centers],
            velocities_list=[velocities],
            labels_list=[labels],
            gt_locs_list=[gt_locs],
            gt_mask_list=[gt_mask])

    def _motion_targets_from_outs_track(self, outs_track: dict, gt_instances):
        track_instances = outs_track.get('track_instances', None)
        if track_instances is None:
            return self._empty_motion_inputs()
        return self._motion_targets_from_tracks(track_instances, gt_instances)

    def _motion_predict_from_outs_track(self, outs_track: dict,
                                        data_sample) -> None:
        if self.motion_head is None:
            return
        track_instances = outs_track.get('active_track_instances', None)
        if track_instances is None or len(track_instances) == 0:
            return
        active_tracks = track_instances
        keep = outs_track.get('track_bbox_results', {}).get('keep', None)
        if keep is None:
            _, _, _, keep = self._track_output_boxes_and_keep(active_tracks)
        query_embeddings = active_tracks.output_embedding.detach()
        active_tracks = active_tracks[keep]
        query_embeddings = query_embeddings[keep]
        if len(active_tracks) == 0:
            return
        centers, velocities, labels = self._track_centers_and_state(
            active_tracks)
        if self._motion_head_uses_track_query():
            forecasts = self.motion_head(
                [query_embeddings], [centers], [velocities], [labels])[0]
        else:
            bev_embed = outs_track['bev_embed']
            forecasts = self.motion_head(
                bev_embed, [centers], [velocities], [labels])[0]
        if not hasattr(data_sample, 'pred_track_instances_3d'):
            return
        data_sample.pred_track_instances_3d.forecasting_3d = forecasts

    def _track_output_boxes_and_keep(self, active: Instances):
        boxes = denormalize_bbox(active.pred_boxes)
        scores = active.pred_logits.sigmoid().max(dim=-1).values
        labels = active.pred_logits.sigmoid().argmax(dim=-1)
        keep = torch.ones_like(scores, dtype=torch.bool)
        post_center_range = self.pts_bbox_head.test_cfg.get(
            'post_center_range', None)
        if post_center_range is not None:
            pcr = boxes.new_tensor(post_center_range)
            keep &= (boxes[:, :3] >= pcr[:3]).all(dim=1)
            keep &= (boxes[:, :3] <= pcr[3:]).all(dim=1)
        return boxes, scores, labels, keep

    def _track_instances_to_results(self, active: Instances) -> dict:
        if len(active) == 0:
            boxes = active.pred_boxes.new_zeros((0, 7))
            scores = active.scores.new_zeros(0)
            labels = active.obj_idxes.new_zeros(0, dtype=torch.long)
            keep = active.obj_idxes.new_zeros(0, dtype=torch.bool)
            track_ids = active.obj_idxes.new_zeros(0, dtype=torch.long)
            return dict(
                boxes_3d=boxes,
                scores_3d=scores,
                labels_3d=labels,
                track_scores=scores,
                track_ids=track_ids,
                keep=keep,
                raw_boxes_3d=boxes,
                raw_scores_3d=scores,
                raw_labels_3d=labels,
                raw_track_ids=track_ids)

        boxes, scores, labels, keep = self._track_output_boxes_and_keep(active)
        return dict(
            boxes_3d=boxes[keep],
            scores_3d=scores[keep],
            labels_3d=labels[keep],
            track_scores=scores[keep],
            track_ids=active.obj_idxes[keep],
            keep=keep,
            raw_boxes_3d=boxes,
            raw_scores_3d=scores,
            raw_labels_3d=labels,
            raw_track_ids=active.obj_idxes)

    def _build_outs_track(self,
                          bev_embed: Tensor,
                          track_instances: Instances,
                          active_track_instances: Optional[Instances] = None,
                          det_out: Optional[dict] = None) -> dict:
        track_bbox_results = None
        if active_track_instances is not None:
            track_bbox_results = self._track_instances_to_results(
                active_track_instances)

        outs_track = dict(
            bev_embed=bev_embed,
            bev_pos=None,
            track_instances=track_instances,
            active_track_instances=active_track_instances,
            track_query_embeddings=getattr(
                track_instances, 'output_embedding', None),
            track_query_matched_idxes=getattr(
                track_instances, 'matched_gt_idxes', None),
            track_bbox_results=track_bbox_results)
        if det_out is not None:
            outs_track.update(
                all_cls_scores=det_out.get('all_cls_scores', None),
                all_bbox_preds=det_out.get('all_bbox_preds', None),
                query_feats=det_out.get('query_feats', None),
                reference_points=det_out.get('reference_points', None),
                det_out=det_out)
        return outs_track

    def _generate_empty_tracks(self, device) -> Instances:
        init_embeds = self.generate_init_query_embeds().to(device)
        num_queries = init_embeds.shape[0]
        embed_dims = init_embeds.shape[1] // 2

        ti = Instances((1, 1))
        ti.query = init_embeds
        ti.ref_pts = self.generate_init_ref_points(init_embeds)
        ti.obj_idxes = torch.full(
            (num_queries, ), -1, dtype=torch.long, device=device)
        ti.matched_gt_idxes = torch.full(
            (num_queries, ), -1, dtype=torch.long, device=device)
        ti.scores = torch.zeros(num_queries, device=device)
        ti.iou = torch.zeros(num_queries, device=device)
        ti.disappear_time = torch.zeros(
            num_queries, dtype=torch.long, device=device)
        ti.output_embedding = torch.zeros(num_queries, embed_dims, device=device)
        ti.pred_boxes = torch.zeros(num_queries, 10, device=device)
        ti.pred_logits = torch.zeros(num_queries, self.num_classes, device=device)
        return ti

    def _copy_tracks_for_loss(self, track_instances: Instances) -> Instances:
        device = track_instances.obj_idxes.device
        ti = Instances(track_instances.image_size)
        ti.obj_idxes = copy.deepcopy(track_instances.obj_idxes)
        ti.matched_gt_idxes = copy.deepcopy(track_instances.matched_gt_idxes)
        ti.disappear_time = copy.deepcopy(track_instances.disappear_time)
        ti.scores = torch.zeros(len(ti), device=device)
        ti.iou = torch.zeros(len(ti), device=device)
        ti.pred_boxes = torch.zeros(
            len(ti), 10, dtype=torch.float, device=device)
        ti.pred_logits = torch.zeros(
            len(ti), self.num_classes, dtype=torch.float, device=device)
        return ti

    def velo_update_ref_pts(self, ref_pts: Tensor, vel_xy: Tensor,
                            ego_motion_delta: Tensor,
                            time_delta: float) -> Tensor:
        if self.point_cloud_range is None:
            return ref_pts
        pc = torch.as_tensor(
            self.point_cloud_range, device=ref_pts.device, dtype=ref_pts.dtype)
        xyz_prev = ref_pts.sigmoid().clone()
        xyz_prev[..., 0] = xyz_prev[..., 0] * (pc[3] - pc[0]) + pc[0]
        xyz_prev[..., 1] = xyz_prev[..., 1] * (pc[4] - pc[1]) + pc[1]
        xyz_prev[..., 2] = xyz_prev[..., 2] * (pc[5] - pc[2]) + pc[2]

        xyz_prev[..., 0:2] = xyz_prev[..., 0:2] + vel_xy * float(time_delta)

        delta = ego_motion_delta.to(ref_pts.device).to(ref_pts.dtype)
        xyz_cur = xyz_prev @ delta[:3, :3].t() + delta[:3, 3]

        xyz_cur[..., 0] = (xyz_cur[..., 0] - pc[0]) / (pc[3] - pc[0])
        xyz_cur[..., 1] = (xyz_cur[..., 1] - pc[1]) / (pc[4] - pc[1])
        xyz_cur[..., 2] = (xyz_cur[..., 2] - pc[2]) / (pc[5] - pc[2])
        xyz_cur = xyz_cur.clamp(min=1e-5, max=1 - 1e-5)
        return inverse_sigmoid(xyz_cur)

    def _reference_from_query(self, ti: Instances) -> Tensor:
        """Regenerate logit-space reference points from query_pos.

        UniAD refreshes ref_pts from the current query positional embedding
        before the next decoder call, then only uses velocity/ego-motion to
        carry the active tracks' xy anchors forward. Keeping query_pos and
        ref_pts aligned is important after QIM updates query_pos.
        """
        return self.generate_init_ref_points(ti.query)

    def _prepare_ref_pts_for_next_frame(self, ti: Instances,
                                        meta: dict) -> Instances:
        base_ref = self._reference_from_query(ti)
        active_mask = ti.obj_idxes >= 0
        ego_delta = meta.get('ego_motion_delta')
        time_delta = meta.get('time_delta', 0.0)
        if ego_delta is not None and time_delta > 0 and active_mask.any():
            if self.use_velocity_ref_update:
                vel = ti.pred_boxes[active_mask, 8:10].detach()
            else:
                vel = ti.pred_boxes.new_zeros(
                    (int(active_mask.sum().item()), 2))
            ego_motion = torch.as_tensor(
                ego_delta, device=ti.query.device, dtype=torch.float32)
            moved_ref = self.velo_update_ref_pts(
                ti.ref_pts[active_mask], vel, ego_motion, time_delta)
            active_inds = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            # Avoid in-place writes on sigmoid outputs; autograd needs their
            # original values when backpropagating through query_pos -> ref_pts.
            xy_ref = base_ref[:, :2].index_copy(
                0, active_inds, moved_ref[:, :2])
            base_ref = torch.cat([xy_ref, base_ref[:, 2:3]], dim=-1)
        ti.ref_pts = base_ref
        return ti

    @staticmethod
    def _queue_points(batch_inputs_dict, batch_data_samples, t: int, T: int
                      ) -> List[Tensor]:
        points = batch_inputs_dict.get('points', [])
        history_points = batch_inputs_dict.get('history_points', None)
        if t == T - 1:
            return points
        batch_size = len(points)
        per_sample = BEVFormerLidarUniAD._normalize_history_points(
            history_points, batch_size)
        return [per_sample[b][t] for b in range(batch_size)]

    def loss(self, batch_inputs_dict: dict, batch_data_samples: List,
             **kwargs) -> dict:
        assert len(batch_data_samples) == 1, (
            'UniADTrackLiDAR currently requires batch_size=1.')
        sample = batch_data_samples[0]
        queue_metas = sample.metainfo['queue_metas']
        queue_gt = sample.metainfo['queue_gt_instances_3d']
        T = len(queue_metas)
        assert T == len(queue_gt), (
            f'queue_metas ({T}) and queue_gt_instances_3d ({len(queue_gt)}) '
            f'length mismatch.')

        device = self.query_embedding.weight.device
        self.criterion.initialize_for_single_clip(queue_gt)
        track_instances = self._generate_empty_tracks(device)
        prev_bev: Optional[Tensor] = None
        final_outs_track: Optional[dict] = None

        for t in range(T):
            outs_track = self._forward_single_frame_train(
                batch_inputs_dict, batch_data_samples, queue_metas,
                t, T, prev_bev, track_instances)
            bev_embed = outs_track['bev_embed']
            track_instances = outs_track['track_instances']
            if t < T - 1:
                track_instances = self._advance_to_next_frame(
                    track_instances, queue_metas[t + 1], device)
            # Keep the same bounded graph contract as UniAD history BEV:
            # every frame has its own loss, but the next frame cannot backprop
            # through the previous frame's BEV encoder graph.
            prev_bev = bev_embed.detach()
            final_outs_track = outs_track

        losses = self.criterion.forward()
        if self.motion_head is not None and final_outs_track is not None:
            motion_gt = self._motion_targets_from_outs_track(
                final_outs_track, queue_gt[-1])
            if motion_gt['centers_list']:
                if self._motion_head_uses_track_query():
                    losses_motion = self.motion_head.loss(
                        motion_gt['query_embeddings_list'],
                        motion_gt['centers_list'],
                        motion_gt['velocities_list'],
                        motion_gt['labels_list'],
                        motion_gt['gt_locs_list'],
                        motion_gt['gt_mask_list'])
                else:
                    bev_embed = final_outs_track['bev_embed']
                    losses_motion = self.motion_head.loss(
                        bev_embed.detach(),
                        motion_gt['centers_list'],
                        motion_gt['velocities_list'],
                        motion_gt['labels_list'],
                        motion_gt['gt_locs_list'],
                        motion_gt['gt_mask_list'])
                losses.update({
                    f'motion.{k}': torch.nan_to_num(v)
                    for k, v in losses_motion.items()
                })

        return losses

    def _forward_single_frame_train(self, batch_inputs_dict, batch_data_samples,
                                    queue_metas, t: int, T: int,
                                    prev_bev: Optional[Tensor],
                                    track_instances: Instances):
        """UniAD-name-compatible wrapper for one training frame."""
        return self._forward_frame(batch_inputs_dict, batch_data_samples,
                                   queue_metas, t, T, prev_bev,
                                   track_instances)

    def _forward_frame(self, batch_inputs_dict, batch_data_samples,
                       queue_metas, t: int, T: int,
                       prev_bev: Optional[Tensor],
                       track_instances: Instances):
        step_points = self._queue_points(
            batch_inputs_dict, batch_data_samples, t, T)
        lidar_bev = self._extract_lidar_bev_for_track(
            step_points, batch_data_samples)
        if t > 0 and prev_bev is not None:
            prev_bev = self.valid_prev_bev(prev_bev, [queue_metas[t]])
        bev_embed = self._encode_bev_for_track(
            lidar_bev, prev_bev, queue_meta=[queue_metas[t]])

        det_out = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed),
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts[None])
        all_cls = det_out['all_cls_scores']
        all_box = det_out['all_bbox_preds']
        query_feats = det_out['query_feats']
        num_layers = all_cls.shape[0]

        layer_track_instances = [
            self._copy_tracks_for_loss(track_instances)
            for _ in range(num_layers - 1)
        ]
        track_instances.output_embedding = query_feats[-1, 0]
        last_ref = det_out['reference_points']
        if last_ref.dim() == 3:
            last_ref = last_ref[0]
        track_instances.ref_pts = last_ref
        layer_track_instances.append(track_instances)

        for layer_id, layer_ti in enumerate(layer_track_instances):
            layer_ti.pred_logits = all_cls[layer_id, 0]
            layer_ti.pred_boxes = all_box[layer_id, 0]
            layer_ti.scores = all_cls[layer_id, 0].sigmoid().max(
                dim=-1).values.detach()
            layer_ti, _ = self.criterion.match_for_single_frame(
                {'track_instances': layer_ti},
                dec_lvl=layer_id,
                if_step=(layer_id == num_layers - 1))
            if layer_id == num_layers - 1:
                track_instances = layer_ti

        return self._build_outs_track(
            bev_embed, track_instances, det_out=det_out)

    def _advance_to_next_frame(self, ti: Instances, next_meta: dict,
                               device) -> Instances:
        ti = self._prepare_ref_pts_for_next_frame(ti, next_meta)
        return self._qim_step(ti, device)

    def _qim_step(self, ti: Instances, device) -> Instances:
        init_ti = self._generate_empty_tracks(device)
        return self.query_interact(
            {'track_instances': ti, 'init_track_instances': init_ti})

    def predict(self, batch_inputs_dict: dict, batch_data_samples: List,
                **kwargs):
        return self.simple_test_track(batch_inputs_dict, batch_data_samples,
                                      **kwargs)

    def simple_test_track(self, batch_inputs_dict: dict,
                          batch_data_samples: List, **kwargs):
        """UniAD-name-compatible inference entry for tracking."""
        assert len(batch_data_samples) == 1, (
            'UniADTrackLiDAR predict requires batch_size=1.')
        sample = batch_data_samples[0]
        queue_metas = sample.metainfo.get('queue_metas', {})
        T = len(queue_metas) if queue_metas else 1
        current_meta = queue_metas.get(T - 1, {}) if queue_metas else {}
        current_scene = current_meta.get('scene_token')
        device = self.query_embedding.weight.device

        scene_changed = (
            self._test_track_instances is None or
            current_scene != self._test_scene_token)
        if scene_changed:
            self._test_track_instances = self._generate_empty_tracks(device)
            self._test_scene_token = current_scene
            self._test_prev_bev = None
            self._debug_prev_centers = {}
            self.track_base.clear()
        elif self.reset_track_query_each_frame:
            self._test_track_instances = self._generate_empty_tracks(device)
        else:
            ti = self._test_track_instances
            self._test_track_instances = self._prepare_ref_pts_for_next_frame(
                ti, current_meta)

        lidar_bev = self._extract_lidar_bev_for_track(
            batch_inputs_dict['points'], batch_data_samples)
        prev_bev = self._test_prev_bev
        if prev_bev is not None:
            prev_bev = self.valid_prev_bev(prev_bev, [current_meta])
        elif 'history_points' in batch_inputs_dict:
            prev_bev = self.obtain_history_bev(
                batch_inputs_dict['history_points'], batch_data_samples)
            if prev_bev is not None:
                prev_bev = self.valid_prev_bev(prev_bev, [current_meta])
        bev_embed = self._encode_bev_for_track(
            lidar_bev, prev_bev, queue_meta=[current_meta])

        ti = self._test_track_instances
        num_queries_before = len(ti)
        det_out = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed),
            object_query_embeds=ti.query,
            ref_points=ti.ref_pts[None])
        ti.pred_logits = det_out['all_cls_scores'][-1, 0]
        ti.pred_boxes = det_out['all_bbox_preds'][-1, 0]
        ti.output_embedding = det_out['query_feats'][-1, 0]
        ti.scores = ti.pred_logits.sigmoid().max(dim=-1).values
        last_ref = det_out['reference_points']
        if last_ref.dim() == 3:
            last_ref = last_ref[0]
        ti.ref_pts = last_ref

        obj_idxes_before_update = ti.obj_idxes.clone()
        self.track_base.update(ti)
        debug_runtime = self._collect_debug_runtime_stats(
            ti, obj_idxes_before_update, num_queries_before)
        # Follow UniAD inference: keep low-score tracks alive internally until
        # miss_tolerance expires, but do not emit them as current-frame results.
        active = ti[(ti.obj_idxes >= 0) &
                    (ti.scores >= self.track_base.filter_score_thresh)]
        outs_track = self._build_outs_track(
            bev_embed, ti, active_track_instances=active, det_out=det_out)
        det_results = self.pts_bbox_head.predict_by_feat(
            det_out, [sample.metainfo])
        sample.pred_instances_3d = det_results[0]
        sample.pred_instances = InstanceData()
        self._fill_track_data_sample(sample, outs_track)
        if self.motion_head is not None:
            self._motion_predict_from_outs_track(outs_track, sample)
        debug_output = self._collect_debug_output_stats(sample, scene_changed)

        if self.reset_track_query_each_frame:
            self._test_track_instances = self._generate_empty_tracks(device)
        else:
            self._test_track_instances = self._qim_step(ti, device)
        debug_next = self._collect_debug_next_stats(self._test_track_instances)
        self._maybe_log_debug_track(
            current_meta, scene_changed, debug_runtime, debug_output,
            debug_next)
        self._test_prev_bev = bev_embed
        for ds in batch_data_samples:
            if 'pred_pts_seg' not in ds:
                ds.pred_pts_seg = PointData()
        return batch_data_samples

    def _debug_should_log(self) -> bool:
        if not self.debug_track:
            return False
        if self.debug_track_max_frames >= 0 and (
                self._debug_track_frame >= self.debug_track_max_frames):
            return False
        return self._debug_track_frame % self.debug_track_interval == 0

    @staticmethod
    def _tensor_stat(tensor: Tensor) -> dict:
        if tensor.numel() == 0:
            return dict(mean=0.0, max=0.0, p50=0.0, p90=0.0)
        values = tensor.detach().float()
        return dict(
            mean=float(values.mean().item()),
            max=float(values.max().item()),
            p50=float(values.quantile(0.5).item()),
            p90=float(values.quantile(0.9).item()))

    def _count_by_class(self, labels: Tensor) -> Dict[int, int]:
        if labels.numel() == 0:
            return {}
        labels = labels.detach().long().cpu()
        valid = (labels >= 0) & (labels < self.num_classes)
        if not bool(valid.any()):
            return {}
        counts = torch.bincount(
            labels[valid], minlength=self.num_classes).tolist()
        return {i: int(v) for i, v in enumerate(counts) if v}

    def _format_class_counts(self, counts: Dict[int, int]) -> str:
        parts = []
        for i in range(self.num_classes):
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            parts.append(f'{name}:{int(counts.get(i, 0))}')
        return '{' + ', '.join(parts) + '}'

    def _collect_debug_runtime_stats(self, ti: Instances,
                                     obj_idxes_before_update: Tensor,
                                     num_queries_before: int) -> dict:
        scores = ti.scores.detach()
        labels = ti.pred_logits.detach().sigmoid().argmax(dim=-1)
        assigned_before = obj_idxes_before_update >= 0
        assigned_after = ti.obj_idxes >= 0
        promoted = (obj_idxes_before_update < 0) & assigned_after
        retired = assigned_before & (ti.obj_idxes < 0)
        alive_low = assigned_after & (
            scores < self.track_base.filter_score_thresh)
        emit_mask = assigned_after & (
            scores >= self.track_base.filter_score_thresh)
        high_unassigned = (obj_idxes_before_update < 0) & (
            scores >= self.track_base.score_thresh)
        ref_raw = ti.ref_pts.detach()
        ref_for_range = ref_raw
        if ref_raw.numel() > 0 and (
                ref_raw.min().item() < 0.0 or ref_raw.max().item() > 1.0):
            ref_for_range = ref_raw.sigmoid()
        ref_out = ((ref_for_range < 0) | (ref_for_range > 1)).any(
            dim=-1).sum().item()
        stats = dict(
            q_before=num_queries_before,
            assigned_before=int(assigned_before.sum().item()),
            assigned_after=int(assigned_after.sum().item()),
            promoted=int(promoted.sum().item()),
            retired=int(retired.sum().item()),
            alive_low=int(alive_low.sum().item()),
            emit_pre_center=int(emit_mask.sum().item()),
            high_unassigned=int(high_unassigned.sum().item()),
            max_obj_id=int(self.track_base.max_obj_id),
            score=self._tensor_stat(scores),
            ref_raw_min=float(ref_raw.min().item()) if ref_raw.numel() else 0.0,
            ref_raw_max=float(ref_raw.max().item()) if ref_raw.numel() else 0.0,
            ref_out=int(ref_out))
        if self.debug_track_class_stats:
            stats['class_counts'] = dict(
                assigned_after=self._count_by_class(labels[assigned_after]),
                promoted=self._count_by_class(labels[promoted]),
                retired=self._count_by_class(labels[retired]),
                alive_low=self._count_by_class(labels[alive_low]),
                emit_pre=self._count_by_class(labels[emit_mask]))
        return stats

    def _collect_debug_output_stats(self, sample, scene_changed: bool) -> dict:
        det_count = 0
        det_by_cls = {}
        if hasattr(sample, 'pred_instances_3d') and hasattr(
                sample.pred_instances_3d, 'scores_3d'):
            det_count = len(sample.pred_instances_3d.scores_3d)
            if self.debug_track_class_stats and hasattr(
                    sample.pred_instances_3d, 'labels_3d'):
                det_by_cls = self._count_by_class(
                    sample.pred_instances_3d.labels_3d)

        track_count = 0
        track_by_cls = {}
        unique_ids = 0
        repeated_ids = 0
        matched_prev = 0
        center_jump = torch.empty(0)
        if hasattr(sample, 'pred_track_instances_3d'):
            pred = sample.pred_track_instances_3d
            if self.debug_track_class_stats and hasattr(pred, 'labels_3d'):
                track_by_cls = self._count_by_class(pred.labels_3d)
            if hasattr(pred, 'instance_id'):
                ids = pred.instance_id.detach().cpu().long()
                track_count = int(ids.numel())
                unique_ids = int(torch.unique(ids).numel()) if ids.numel() else 0
                repeated_ids = track_count - unique_ids
                boxes = getattr(pred, 'bboxes_3d', None)
                if boxes is not None and ids.numel():
                    if hasattr(boxes, 'gravity_center'):
                        centers = boxes.gravity_center.detach().cpu()
                    else:
                        centers = boxes.tensor[:, :3].detach().cpu()
                    jumps = []
                    next_centers = {}
                    for obj_id, center in zip(ids.tolist(), centers):
                        if obj_id in self._debug_prev_centers:
                            jumps.append(torch.norm(
                                center[:2] -
                                self._debug_prev_centers[obj_id][:2]))
                        next_centers[obj_id] = center
                    matched_prev = len(jumps)
                    if jumps:
                        center_jump = torch.stack(jumps)
                    self._debug_prev_centers = next_centers
                elif scene_changed:
                    self._debug_prev_centers = {}
        return dict(
            det_count=det_count,
            track_count=track_count,
            unique_ids=unique_ids,
            repeated_ids=repeated_ids,
            matched_prev=matched_prev,
            center_jump=self._tensor_stat(center_jump),
            det_by_cls=det_by_cls,
            track_by_cls=track_by_cls)

    def _collect_debug_next_stats(self, ti: Instances) -> dict:
        assigned = ti.obj_idxes >= 0
        stats = dict(
            q_next=len(ti),
            carry_next=int(assigned.sum().item()),
            init_next=int((ti.obj_idxes < 0).sum().item()))
        if self.debug_track_class_stats and hasattr(ti, 'pred_logits'):
            labels = ti.pred_logits.detach().sigmoid().argmax(dim=-1)
            stats['carry_next_by_cls'] = self._count_by_class(
                labels[assigned])
        return stats

    def _maybe_log_debug_track(self, meta: dict, scene_changed: bool,
                               runtime: dict, output: dict,
                               next_stats: dict) -> None:
        if not self._debug_should_log():
            self._debug_track_frame += 1
            return
        try:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            log = logger.info
        except Exception:
            log = print

        score = runtime['score']
        jump = output['center_jump']
        token = str(meta.get('token', ''))[:8]
        scene = str(meta.get('scene_token', ''))[-24:]
        log(
            '[track-debug] '
            f'frame={self._debug_track_frame} token={token} '
            f'scene_tail={scene} scene_changed={scene_changed} '
            f'q={runtime["q_before"]}->{next_stats["q_next"]} '
            f'assigned={runtime["assigned_before"]}->{runtime["assigned_after"]} '
            f'promote={runtime["promoted"]} retire={runtime["retired"]} '
            f'alive_low={runtime["alive_low"]} '
            f'emit_pre={runtime["emit_pre_center"]} '
            f'det={output["det_count"]} track={output["track_count"]} '
            f'unique={output["unique_ids"]} repeat_id={output["repeated_ids"]} '
            f'carry_next={next_stats["carry_next"]} '
            f'max_id={runtime["max_obj_id"]} '
            f'score(mean/p50/p90/max)='
            f'{score["mean"]:.3f}/{score["p50"]:.3f}/'
            f'{score["p90"]:.3f}/{score["max"]:.3f} '
            f'ref_raw=({runtime["ref_raw_min"]:.2f},'
            f'{runtime["ref_raw_max"]:.2f}) ref_out={runtime["ref_out"]} '
            f'same_id_jump(n/mean/p90/max)='
            f'{output["matched_prev"]}/{jump["mean"]:.2f}/'
            f'{jump["p90"]:.2f}/{jump["max"]:.2f}')
        if self.debug_track_class_stats:
            cls = runtime.get('class_counts', {})
            log(
                '[track-debug-class] '
                f'frame={self._debug_track_frame} token={token} '
                f'det={self._format_class_counts(output["det_by_cls"])} '
                f'track={self._format_class_counts(output["track_by_cls"])} '
                f'promote={self._format_class_counts(cls.get("promoted", {}))} '
                f'emit_pre={self._format_class_counts(cls.get("emit_pre", {}))} '
                f'alive_low={self._format_class_counts(cls.get("alive_low", {}))} '
                f'retired={self._format_class_counts(cls.get("retired", {}))} '
                f'carry_next={self._format_class_counts(next_stats.get("carry_next_by_cls", {}))}')
        self._debug_track_frame += 1

    def _fill_track_data_sample(self, data_sample, outs_track: dict) -> None:
        results = outs_track['track_bbox_results']
        active = outs_track['active_track_instances']
        if active is None or len(active) == 0:
            inst = InstanceData()
            box_type = data_sample.metainfo['box_type_3d']
            empty_boxes = results['boxes_3d']
            inst.bboxes_3d = box_type(empty_boxes, box_dim=7)
            inst.scores_3d = results['scores_3d']
            inst.labels_3d = results['labels_3d']
            inst.instance_id = results['track_ids']
            inst.track_scores = results['track_scores']
            inst.track_ids = results['track_ids']
            data_sample.pred_track_instances_3d = inst
            return

        inst = InstanceData()
        box_type = data_sample.metainfo['box_type_3d']
        boxes = results['boxes_3d']
        inst.bboxes_3d = box_type(
            boxes, box_dim=boxes.size(-1), origin=(0.5, 0.5, 0.5))
        inst.scores_3d = results['scores_3d']
        inst.labels_3d = results['labels_3d']
        inst.instance_id = results['track_ids']
        inst.track_scores = results['track_scores']
        inst.track_ids = results['track_ids']
        data_sample.pred_track_instances_3d = inst


@MODELS.register_module()
class UniADTrackLiDARMemory(UniADTrackLiDAR):
    """MemoryBank ablation of ``UniADTrackLiDAR``.

    This class is intentionally separate so the baseline track config and any
    currently running training process keep the old query-update path.
    """

    def __init__(self,
                 *args,
                 mem_args: Optional[dict] = None,
                 embed_dims: int = 256,
                 **kwargs) -> None:
        super().__init__(*args, embed_dims=embed_dims, **kwargs)
        mem_args = mem_args or {}
        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims)
        self.mem_bank_len = self.memory_bank.max_his_length

    def _generate_empty_tracks(self, device) -> Instances:
        ti = super()._generate_empty_tracks(device)
        embed_dims = ti.output_embedding.shape[-1]
        ti.mem_bank = torch.zeros(
            (len(ti), self.mem_bank_len, embed_dims),
            dtype=torch.float32,
            device=device)
        ti.mem_padding_mask = torch.ones(
            (len(ti), self.mem_bank_len), dtype=torch.bool, device=device)
        ti.save_period = torch.zeros(
            (len(ti), ), dtype=torch.float32, device=device)
        return ti

    def _qim_step(self, ti: Instances, device) -> Instances:
        ti = self.memory_bank(ti)
        init_ti = self._generate_empty_tracks(device)
        return self.query_interact(
            {'track_instances': ti, 'init_track_instances': init_ti})
