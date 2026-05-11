"""LiDAR BEVFormer with MOTR-style clip-level tracking.

Extends ``BEVFormerLidar`` so a queue of frames is consumed as a single
MOTR-style clip: query embeddings persist across frames, Hungarian matches
only the unmatched queries against untracked GT, and per-frame losses are
aggregated by ``BEVDETRClipMatcher``.

The BEV extraction path (voxelization, sparse encoder, SECOND/FPN, temporal
fusion) is inherited unchanged. What is rewritten here is:

  * ``loss()``     — clip loop with gradients through every frame
  * ``predict()``  — scene-aware online track state, RuntimeTrackerBase
  * ``_generate_empty_tracks()`` — init track queries from the head
  * ``velo_update_ref_pts()``    — ego-motion warp of ref_pts between frames

This detector expects the dataset to attach ``queue_gt_instances_3d`` and
``queue_metas`` to ``data_samples.metainfo`` (see ``KlBEVFormerDataset``).
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData

from ..dense_heads.track_head_plugin import (Instances, QueryInteractionModule,
                                              RuntimeTrackerBase)
from .bevformer_lidar import BEVFormerLidar


@MODELS.register_module()
class BEVFormerLidarTrack(BEVFormerLidar):
    """MOTR-style tracking detector on top of BEVFormerLidar."""

    def __init__(self,
                 *args,
                 track_loss_cfg: Optional[dict] = None,
                 qim_args: Optional[dict] = None,
                 score_thresh: float = 0.4,
                 filter_score_thresh: float = 0.3,
                 miss_tolerance: int = 3,
                 num_query: int = 300,
                 embed_dims: int = 256,
                 num_classes: int = 15,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if track_loss_cfg is None:
            raise ValueError('track_loss_cfg is required.')
        self.criterion = MODELS.build(track_loss_cfg)
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

        # Online tracking state (inference only).
        self._test_track_instances: Optional[Instances] = None
        self._test_scene_token: Optional[str] = None
        self._test_prev_bev: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    # Track instance bookkeeping
    # ------------------------------------------------------------------ #

    def _generate_empty_tracks(self, device) -> Instances:
        """Build a fresh ``Instances`` bag seeded from the head's init queries.

        Shape contract:
          query            [N, 2*D]  — [query_pos || query_feat]
          ref_pts          [N, 3]    — sigmoid-space (normalized to pc_range)
          obj_idxes        [N]       — -1 (unassigned)
          matched_gt_idxes [N]       — -1
          scores           [N]       — 0
          iou              [N]       — 0
          disappear_time   [N]       — 0
          output_embedding [N, D]    — 0 (updated after first decoder pass)
          pred_boxes       [N, 10]   — 0
          pred_logits      [N, K]    — 0
        """
        head = self.pts_bbox_head
        init_embeds = head.generate_init_query_embeds().to(device)
        N = init_embeds.shape[0]
        D = init_embeds.shape[1] // 2

        ti = Instances((1, 1))
        ti.query = init_embeds
        # Reference points derived from the learnable query_pos embedding
        # (same as detection-mode forward when object_query_embeds is None).
        ti.ref_pts = head.reference_points(
            head.query_pos_embed.weight.detach().to(device)).sigmoid()
        ti.obj_idxes = torch.full((N, ), -1, dtype=torch.long, device=device)
        ti.matched_gt_idxes = torch.full((N, ), -1, dtype=torch.long,
                                         device=device)
        ti.scores = torch.zeros(N, device=device)
        ti.iou = torch.zeros(N, device=device)
        ti.disappear_time = torch.zeros(N, dtype=torch.long, device=device)
        ti.output_embedding = torch.zeros(N, D, device=device)
        ti.pred_boxes = torch.zeros(N, 10, device=device)
        ti.pred_logits = torch.zeros(N, self.num_classes, device=device)
        return ti

    # ------------------------------------------------------------------ #
    # Ego-motion warp of ref_pts between frames
    # ------------------------------------------------------------------ #

    def velo_update_ref_pts(self, ref_pts: Tensor, vel_xy: Tensor,
                            ego_motion_delta: Tensor,
                            time_delta: float) -> Tensor:
        """Warp reference points from frame t (prev) to frame t+1 (current).

        Args:
            ref_pts: [N, 3] in sigmoid space, normalized to pc_range
            vel_xy:  [N, 2] per-track xy velocity in m/s (frame-t LiDAR frame)
            ego_motion_delta: [4, 4] homogeneous transform mapping a point in
                the previous ego/LiDAR frame to the current one
                (``queue_metas[i].ego_motion_delta``).
            time_delta: float seconds between frames
        Returns:
            [N, 3] ref_pts in sigmoid space, valid in the new frame.
        """
        if self.point_cloud_range is None:
            return ref_pts
        pc = torch.as_tensor(
            self.point_cloud_range, device=ref_pts.device, dtype=ref_pts.dtype)
        # ref_pts is already in sigmoid space [0, 1]; convert to metric.
        xyz_prev = ref_pts.clone()
        xyz_prev[..., 0] = xyz_prev[..., 0] * (pc[3] - pc[0]) + pc[0]
        xyz_prev[..., 1] = xyz_prev[..., 1] * (pc[4] - pc[1]) + pc[1]
        xyz_prev[..., 2] = xyz_prev[..., 2] * (pc[5] - pc[2]) + pc[2]

        # constant-velocity update in prev-frame LiDAR (xy only; dz=0)
        dt = float(time_delta)
        xyz_prev = xyz_prev.clone()
        xyz_prev[..., 0:2] = xyz_prev[..., 0:2] + vel_xy * dt

        # transform into current frame: p_cur = R @ p_prev + t
        delta = ego_motion_delta.to(ref_pts.device).to(ref_pts.dtype)
        R = delta[:3, :3]
        t = delta[:3, 3]
        xyz_cur = xyz_prev @ R.t() + t

        # back to sigmoid space
        xyz_cur[..., 0] = (xyz_cur[..., 0] - pc[0]) / (pc[3] - pc[0])
        xyz_cur[..., 1] = (xyz_cur[..., 1] - pc[1]) / (pc[4] - pc[1])
        xyz_cur[..., 2] = (xyz_cur[..., 2] - pc[2]) / (pc[5] - pc[2])
        xyz_cur = xyz_cur.clamp(min=1e-5, max=1 - 1e-5)
        return xyz_cur

    # ------------------------------------------------------------------ #
    # Clip training loop
    # ------------------------------------------------------------------ #

    @staticmethod
    def _queue_points(batch_inputs_dict, batch_data_samples, t: int, T: int
                      ) -> List[Tensor]:
        """Return per-sample point cloud for queue step t (0..T-1).

        For batch=1 (which clip-mode training uses), this is a 1-element list.
        Step T-1 is the current frame (``points``); earlier steps come from
        ``history_points`` in oldest→newest order.
        """
        points = batch_inputs_dict.get('points', [])
        hp = batch_inputs_dict.get('history_points', None)
        if t == T - 1:
            return points
        # history stored per-sample as a list of tensors oldest→newest,
        # length = T - 1; normalize using existing helper from parent.
        batch_size = len(points)
        per_sample = BEVFormerLidar._normalize_history_points(hp, batch_size)
        return [per_sample[b][t] for b in range(batch_size)]

    def loss(self, batch_inputs_dict: dict, batch_data_samples: List,
             **kwargs) -> dict:
        """Run clip-level MOTR training. Assumes batch_size == 1."""
        assert len(batch_data_samples) == 1, (
            'BEVFormerLidarTrack currently requires batch_size=1.')
        sample = batch_data_samples[0]
        queue_metas = sample.metainfo['queue_metas']
        queue_gt = sample.metainfo['queue_gt_instances_3d']
        T = len(queue_metas)
        assert T == len(queue_gt), (
            f'queue_metas ({T}) and queue_gt_instances_3d ({len(queue_gt)}) '
            f'length mismatch.')

        device = self.pts_bbox_head.query_embed.weight.device
        self.criterion.initialize_for_single_clip(queue_gt)

        track_instances = self._generate_empty_tracks(device)
        prev_bev: Optional[Tensor] = None

        for t in range(T):
            step_points = self._queue_points(
                batch_inputs_dict, batch_data_samples, t, T)
            current_bev = self._unwrap_single_bev(
                self.extract_pts_bev_from_points(step_points,
                                                 batch_data_samples))
            if t > 0:
                meta = queue_metas[t]
                prev_bev = self._warp_prev_bev_if_needed(prev_bev, [meta])
            fused_bev = self._fuse_bev(current_bev, prev_bev)

            # Head forward with external queries (Stage A contract).
            det_out = self.pts_bbox_head.get_detections(
                self._wrap_single_bev(fused_bev),
                object_query_embeds=track_instances.query,
                ref_points=track_instances.ref_pts[None])  # [1, N, 3]
            all_cls = det_out['all_cls_scores']     # [L, B=1, N, K]
            all_box = det_out['all_bbox_preds']     # [L, B=1, N, 10]
            query_feats = det_out['query_feats']    # [L, B=1, N, D]
            L = all_cls.shape[0]

            # Supervise every decoder layer on current frame.
            for lyr in range(L):
                # Stash per-layer predictions into track_instances before
                # calling the matcher.
                track_instances.pred_logits = all_cls[lyr, 0]
                track_instances.pred_boxes = all_box[lyr, 0]
                track_instances.scores = all_cls[lyr, 0].sigmoid().max(
                    dim=-1).values.detach()
                track_instances, _ = self.criterion.match_for_single_frame(
                    {'track_instances': track_instances},
                    dec_lvl=lyr,
                    if_step=(lyr == L - 1))

            # Carry output embedding from the last decoder layer forward.
            track_instances.output_embedding = query_feats[-1, 0]
            # Pick up the last (refined) reference points so the next-frame
            # warp uses the head's iterative-refine output rather than the
            # input ref.
            last_ref = det_out['reference_points']
            if last_ref.dim() == 3:  # [B, N, 3]
                last_ref = last_ref[0]
            track_instances.ref_pts = last_ref

            # Advance for next frame: QIM + ego-motion warp of ref_pts.
            if t < T - 1:
                track_instances = self._advance_to_next_frame(
                    track_instances, queue_metas[t + 1], device)
            prev_bev = fused_bev

        return self.criterion.forward()

    def _advance_to_next_frame(self, ti: Instances, next_meta: dict,
                               device) -> Instances:
        """QIM self-update + ref_pts warp before consuming the next frame."""
        # Velocity from last-layer box pred (code_size=10; vx,vy are last 2).
        vel = ti.pred_boxes[:, 8:10].detach()
        ego_motion = torch.as_tensor(
            next_meta['ego_motion_delta'], device=device, dtype=torch.float32)
        ti.ref_pts = self.velo_update_ref_pts(
            ti.ref_pts, vel, ego_motion, next_meta.get('time_delta', 0.0))
        return self._qim_step(ti, device)

    def _qim_step(self, ti: Instances, device) -> Instances:
        """Run QIM to merge active tracks with fresh init queries."""
        init_ti = self._generate_empty_tracks(device)
        return self.query_interact(
            {'track_instances': ti, 'init_track_instances': init_ti})

    # ------------------------------------------------------------------ #
    # Online inference
    # ------------------------------------------------------------------ #

    def predict(self, batch_inputs_dict: dict, batch_data_samples: List,
                **kwargs):
        assert len(batch_data_samples) == 1, (
            'BEVFormerLidarTrack predict requires batch_size=1.')
        sample = batch_data_samples[0]
        queue_metas = sample.metainfo.get('queue_metas', {})
        T = len(queue_metas) if queue_metas else 1
        current_meta = queue_metas.get(T - 1, {}) if queue_metas else {}
        current_scene = current_meta.get('scene_token')
        device = self.pts_bbox_head.query_embed.weight.device

        # Reset on scene change.
        if (self._test_track_instances is None or
                current_scene != self._test_scene_token):
            self._test_track_instances = self._generate_empty_tracks(device)
            self._test_scene_token = current_scene
            self._test_prev_bev = None
            self.track_base.clear()
        else:
            # Warp persisted ref_pts from prev frame into current frame using
            # the ego_motion_delta that the current frame provides.
            ti = self._test_track_instances
            ego_delta = current_meta.get('ego_motion_delta')
            time_delta = current_meta.get('time_delta', 0.0)
            if ego_delta is not None and time_delta > 0:
                vel = ti.pred_boxes[:, 8:10].detach()
                ego_motion = torch.as_tensor(
                    ego_delta, device=device, dtype=torch.float32)
                ti.ref_pts = self.velo_update_ref_pts(
                    ti.ref_pts, vel, ego_motion, time_delta)

        # BEV extraction: only run the current frame's encoder.
        # Use cached prev_bev from the last predict() call (Bug 5 fix).
        current_bev = self._unwrap_single_bev(
            self.extract_pts_bev_from_points(batch_inputs_dict['points'],
                                             batch_data_samples))
        prev_bev = self._test_prev_bev
        if prev_bev is not None:
            prev_bev = self._warp_prev_bev_if_needed(prev_bev, [current_meta])
        elif 'history_points' in batch_inputs_dict:
            # First frame of scene: bootstrap from history if available.
            prev_bev = self.obtain_history_bev(
                batch_inputs_dict['history_points'], batch_data_samples)
            if prev_bev is not None:
                prev_bev = self._warp_prev_bev_if_needed(
                    prev_bev, [current_meta])
        fused_bev = self._fuse_bev(current_bev, prev_bev)

        # Run detection head with track queries.
        ti = self._test_track_instances
        det_out = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(fused_bev),
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

        # Online id allocation / retirement.
        self.track_base.update(ti)

        # Build prediction for current frame: only active tracks.
        active_mask = ti.obj_idxes >= 0
        active = ti[active_mask]
        self._fill_data_sample(sample, active)

        # Advance track state for next frame (Bug 6 fix): run QIM to refresh
        # query pool. Ego-motion warp is deferred to the start of the next
        # predict() call because we don't know the next frame's transform yet.
        self._test_track_instances = self._qim_step(ti, device)
        self._test_prev_bev = fused_bev

        # Placeholder pts_seg key to keep mmdet3d infra happy.
        for ds in batch_data_samples:
            if 'pred_pts_seg' not in ds:
                ds.pred_pts_seg = PointData()
        return batch_data_samples

    def _fill_data_sample(self, data_sample, active: Instances) -> None:
        """Convert active track instances into pred_instances_3d w/ instance_id."""
        from mmengine.structures import InstanceData
        from ..dense_heads.bev_detr_head import denormalize_bbox
        head = self.pts_bbox_head
        n = len(active)
        if n == 0:
            inst = InstanceData()
            empty_boxes = active.pred_boxes.new_zeros((0, 7))
            box_type = data_sample.metainfo['box_type_3d']
            inst.bboxes_3d = box_type(empty_boxes, box_dim=7)
            inst.scores_3d = active.scores.new_zeros(0)
            inst.labels_3d = active.obj_idxes.new_zeros(0, dtype=torch.long)
            inst.instance_id = active.obj_idxes.new_zeros(0, dtype=torch.long)
            data_sample.pred_instances_3d = inst
            return
        boxes = denormalize_bbox(active.pred_boxes)
        scores = active.pred_logits.sigmoid().max(dim=-1).values
        labels = active.pred_logits.sigmoid().argmax(dim=-1)

        # post_center_range filter (same contract as detection head).
        pcr = head.test_cfg.get('post_center_range', None)
        keep = torch.ones_like(scores, dtype=torch.bool)
        if pcr is not None:
            pcr_t = boxes.new_tensor(pcr)
            keep &= (boxes[:, :3] >= pcr_t[:3]).all(dim=1)
            keep &= (boxes[:, :3] <= pcr_t[3:]).all(dim=1)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        ids = active.obj_idxes[keep]

        inst = InstanceData()
        box_type = data_sample.metainfo['box_type_3d']
        inst.bboxes_3d = box_type(boxes, box_dim=boxes.size(-1))
        inst.scores_3d = scores
        inst.labels_3d = labels
        inst.instance_id = ids
        data_sample.pred_instances_3d = inst
