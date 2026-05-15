"""MOTR-style clip matcher for the BEVDETR track head.

This is a stripped UniAD ClipMatcher: no SDC query, no past_trajs, and
delegates Hungarian matching to ``BEVDETRHungarianAssigner3D`` so KL-specific
pi-symmetric orientation handling flows into new-track association. The matcher
still supports optional IoU matching/loss for ablations, but the UniAD-aligned
track config keeps tracking supervision to classification + L1 box loss.

GT contract per frame (entry from ``Instances``-style bag built by the
detector):
  * ``bboxes_3d``  — Nx7 LiDAR boxes (cx,cy,cz,dx,dy,dz,yaw), plus optional vx,vy
  * ``labels_3d``  — N int64 class labels
  * ``track_ids``  — N int64 per-instance track ids, unique within a scene

The detector is expected to call:
  1. ``initialize_for_single_clip(gt_instances_per_frame)`` once per clip
  2. ``match_for_single_frame(out, dec_lvl)`` once per (frame, decoder layer)
     where ``out`` carries the current frame's ``track_instances`` plus the
     last-layer ``pred_logits`` / ``pred_boxes``.

Returns the ``losses_dict`` keyed ``frame_{i}_{name}_{dec_lvl}``.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from mmdet.utils import reduce_mean
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS, TASK_UTILS

from ..dense_heads.bev_detr_head import denormalize_bbox, normalize_bbox
from ..dense_heads.track_head_plugin import Instances


@MODELS.register_module()
class BEVDETRClipMatcher(nn.Module):
    """Clip-level MOTR matcher driving frame-by-frame track loss."""

    def __init__(
        self,
        num_classes: int,
        weight_dict: dict,
        code_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0,
                                         1.0, 1.0, 1.0, 0.2, 0.2),
        assigner: dict = None,
        loss_cls: dict = None,
        loss_bbox: dict = None,
        loss_iou: dict = None,
        iou_calculator: dict = None,
        pc_range: Sequence[float] = None,
        pi_symmetric_class_indices: Sequence[int] = (),
        debug_match: bool = False,
        debug_match_interval: int = 1,
        debug_match_max_steps: int = 50,
    ) -> None:
        super().__init__()
        if pc_range is None:
            raise ValueError('pc_range is required for BEVDETRClipMatcher.')
        self.num_classes = num_classes
        self.weight_dict = dict(weight_dict)
        self.pc_range = list(pc_range)
        self.pi_symmetric_class_indices = list(pi_symmetric_class_indices)
        self.matcher = TASK_UTILS.build(assigner)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou) if loss_iou else None
        self.iou_calculator = (TASK_UTILS.build(iou_calculator)
                               if iou_calculator else None)
        self.debug_match = bool(debug_match)
        self.debug_match_interval = max(int(debug_match_interval), 1)
        self.debug_match_max_steps = int(debug_match_max_steps)
        self._debug_match_step = 0
        self.register_buffer(
            'code_weights',
            torch.tensor(list(code_weights), dtype=torch.float32),
            persistent=False)
        self.losses_dict: dict = {}
        self._current_frame_idx = 0
        self.gt_instances: List[InstanceData] = []

    # ------------------------------------------------------------------ #
    # Clip lifecycle
    # ------------------------------------------------------------------ #

    def initialize_for_single_clip(
            self, gt_instances: List[InstanceData]) -> None:
        """Reset state at the start of a clip."""
        self.gt_instances = gt_instances
        self.losses_dict = {}
        self._current_frame_idx = 0

    def step(self) -> None:
        self._current_frame_idx += 1

    # ------------------------------------------------------------------ #
    # Per-frame matching + loss
    # ------------------------------------------------------------------ #

    def match_for_single_frame(self,
                               outputs: dict,
                               dec_lvl: int,
                               if_step: bool = False) -> tuple:
        """Match track queries to GT at frame ``self._current_frame_idx``.

        Updates ``track_instances.obj_idxes`` / ``matched_gt_idxes`` for
        next-frame carry-over, then computes labels + boxes loss for this
        frame at this decoder layer and stores them in ``losses_dict``.
        """
        track_instances: Instances = outputs['track_instances']
        gt_inst = self.gt_instances[self._current_frame_idx]
        device = track_instances.pred_logits.device

        # Pull GT fields out of InstanceData. track_ids is required for clip.
        # Pack3DDetInputs' to_tensor whitelist doesn't cover gt_track_ids_3d,
        # so it can arrive as a numpy array — coerce defensively.
        gt_track_ids = torch.as_tensor(
            gt_inst.track_ids_3d, device=device, dtype=torch.long)
        gt_labels = gt_inst.labels_3d.to(device).long()
        gt_bboxes = self._gt_to_tensor(gt_inst).to(device)

        # ---- 1. Inherit previous-frame matches via track id continuity ----
        obj_id_to_gt = {int(t): i for i, t in enumerate(gt_track_ids.tolist())}
        for j in range(len(track_instances)):
            tid = int(track_instances.obj_idxes[j].item())
            if tid >= 0:
                track_instances.matched_gt_idxes[j] = obj_id_to_gt.get(tid, -1)
            else:
                track_instances.matched_gt_idxes[j] = -1

        # ---- 2. Identify untracked GT (potential new tracks) -------------
        full_idxes = torch.arange(
            len(track_instances), dtype=torch.long, device=device)
        unmatched_track_idxes = full_idxes[track_instances.obj_idxes == -1]

        matched_state = torch.zeros(
            len(gt_track_ids), dtype=torch.bool, device=device)
        already_matched = track_instances.matched_gt_idxes[
            track_instances.matched_gt_idxes >= 0]
        if already_matched.numel() > 0:
            matched_state[already_matched] = True
        untracked_gt = torch.nonzero(
            ~matched_state, as_tuple=False).squeeze(-1)

        # ---- 3. Hungarian on the leftover (unmatched query, untracked GT)
        prev_matched = self._collect_prev_matched(track_instances, device)
        if len(unmatched_track_idxes) > 0 and len(untracked_gt) > 0:
            new_pred_logits = track_instances.pred_logits[
                unmatched_track_idxes]
            new_pred_boxes = track_instances.pred_boxes[unmatched_track_idxes]
            new_gt_bboxes = gt_bboxes[untracked_gt]
            new_gt_labels = gt_labels[untracked_gt]

            assign_result = self.matcher.assign(
                new_pred_boxes, new_pred_logits,
                new_gt_bboxes, new_gt_labels, gt_bboxes_ignore=None)
            gt_inds = assign_result.gt_inds  # 0=bg, 1..N=fg
            pos = torch.nonzero(gt_inds > 0, as_tuple=False).squeeze(-1)
            if pos.numel() > 0:
                src = unmatched_track_idxes[pos]
                tgt = untracked_gt[gt_inds[pos] - 1]
                # Promote new tracks: copy GT track id, write matched_gt_idxes.
                track_instances.obj_idxes[src] = gt_track_ids[tgt]
                track_instances.matched_gt_idxes[src] = tgt
                new_matched = torch.stack([src, tgt], dim=1)
            else:
                new_matched = torch.empty((0, 2), dtype=torch.long,
                                          device=device)
        else:
            new_matched = torch.empty((0, 2), dtype=torch.long, device=device)

        matched_indices = torch.cat([new_matched, prev_matched], dim=0)

        # ---- 4. Frame loss ------------------------------------------------
        frame_losses = self._compute_frame_losses(
            track_instances, gt_bboxes, gt_labels, gt_track_ids,
            matched_indices)
        for k, v in frame_losses.items():
            self.losses_dict[
                f'frame_{self._current_frame_idx}_{k}_{dec_lvl}'] = v

        if if_step:
            # Compute approximate BEV IoU for QIM's active-track filter.
            track_instances.iou = torch.zeros(
                len(track_instances), device=device)
            if matched_indices.numel() > 0:
                src_idx = matched_indices[:, 0]
                tgt_idx = matched_indices[:, 1]
                with torch.no_grad():
                    pred_dec = denormalize_bbox(
                        track_instances.pred_boxes[src_idx][:, :8])
                    gt_norm = normalize_bbox(gt_bboxes[tgt_idx], self.pc_range)
                    gt_dec = denormalize_bbox(gt_norm[:, :8])
                    # Axis-aligned BEV IoU: [x1,y1,x2,y2] from (cx,cy,l,w)
                    p_x1 = pred_dec[:, 0] - pred_dec[:, 3] / 2
                    p_y1 = pred_dec[:, 1] - pred_dec[:, 4] / 2
                    p_x2 = pred_dec[:, 0] + pred_dec[:, 3] / 2
                    p_y2 = pred_dec[:, 1] + pred_dec[:, 4] / 2
                    g_x1 = gt_dec[:, 0] - gt_dec[:, 3] / 2
                    g_y1 = gt_dec[:, 1] - gt_dec[:, 4] / 2
                    g_x2 = gt_dec[:, 0] + gt_dec[:, 3] / 2
                    g_y2 = gt_dec[:, 1] + gt_dec[:, 4] / 2
                    inter_x1 = torch.max(p_x1, g_x1)
                    inter_y1 = torch.max(p_y1, g_y1)
                    inter_x2 = torch.min(p_x2, g_x2)
                    inter_y2 = torch.min(p_y2, g_y2)
                    inter = (inter_x2 - inter_x1).clamp(min=0) * \
                            (inter_y2 - inter_y1).clamp(min=0)
                    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
                    area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
                    union = area_p + area_g - inter
                    iou = inter / union.clamp(min=1e-6)
                    track_instances.iou[src_idx] = iou
            self._maybe_log_debug_match(
                dec_lvl=dec_lvl,
                num_queries=len(track_instances),
                num_gt=len(gt_track_ids),
                inherited=int(prev_matched.size(0)),
                untracked_gt=int(untracked_gt.numel()),
                unmatched_queries=int(unmatched_track_idxes.numel()),
                new_matched=int(new_matched.size(0)),
                total_matched=int(matched_indices.size(0)),
                active_after=int((track_instances.obj_idxes >= 0).sum()),
                qim_active=int((track_instances.iou > 0.5).sum()))
            self.step()
        return track_instances, matched_indices

    def _debug_should_log_match(self) -> bool:
        if not self.debug_match:
            return False
        if self.debug_match_max_steps >= 0 and (
                self._debug_match_step >= self.debug_match_max_steps):
            return False
        return self._debug_match_step % self.debug_match_interval == 0

    def _maybe_log_debug_match(self, **stats) -> None:
        if not self._debug_should_log_match():
            self._debug_match_step += 1
            return
        try:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            log = logger.info
        except Exception:
            log = print
        log(
            '[match-debug] '
            f'step={self._debug_match_step} '
            f'frame={self._current_frame_idx} dec={stats["dec_lvl"]} '
            f'q={stats["num_queries"]} gt={stats["num_gt"]} '
            f'inherited={stats["inherited"]} '
            f'untracked_gt={stats["untracked_gt"]} '
            f'unmatched_q={stats["unmatched_queries"]} '
            f'new={stats["new_matched"]} '
            f'total={stats["total_matched"]} '
            f'active_after={stats["active_after"]} '
            f'qim_iou>0.5={stats["qim_active"]}')
        self._debug_match_step += 1

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gt_to_tensor(gt_inst: InstanceData) -> torch.Tensor:
        """Get N x (7+vel?) tensor in LiDAR convention (gravity_center + dims)."""
        boxes_3d = gt_inst.bboxes_3d
        return torch.cat(
            [boxes_3d.gravity_center, boxes_3d.tensor[:, 3:]], dim=1)

    @staticmethod
    def _collect_prev_matched(track_instances: Instances,
                              device) -> torch.Tensor:
        """Pairs (query_idx, gt_idx) for queries that already have an obj_id
        AND still see their GT in this frame."""
        full_idxes = torch.arange(
            len(track_instances), dtype=torch.long, device=device)
        mask = (track_instances.obj_idxes >= 0) & (
            track_instances.matched_gt_idxes >= 0)
        if not mask.any():
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.stack(
            [full_idxes[mask], track_instances.matched_gt_idxes[mask]], dim=1)

    def _compute_frame_losses(self,
                              track_instances: Instances,
                              gt_bboxes: torch.Tensor,
                              gt_labels: torch.Tensor,
                              gt_track_ids: torch.Tensor,
                              matched_indices: torch.Tensor) -> dict:
        device = track_instances.pred_logits.device
        N_q = len(track_instances)

        # ---- Classification target: bg by default, matched queries get GT label
        target_classes = torch.full(
            (N_q, ), self.num_classes, dtype=torch.long, device=device)
        if matched_indices.numel() > 0:
            src, tgt = matched_indices[:, 0], matched_indices[:, 1]
            target_classes[src] = gt_labels[tgt]
        label_weights = track_instances.pred_logits.new_ones(N_q)
        avg_factor = float(matched_indices.size(0))
        avg_factor = reduce_mean(
            track_instances.pred_logits.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)
        loss_cls = self.loss_cls(
            track_instances.pred_logits, target_classes,
            label_weights, avg_factor=avg_factor)

        # ---- Box regression on matched queries only ---------------------
        if matched_indices.numel() > 0:
            src, tgt = matched_indices[:, 0], matched_indices[:, 1]
            src_boxes = track_instances.pred_boxes[src]
            tgt_boxes = normalize_bbox(gt_bboxes[tgt], self.pc_range)
            code_size = src_boxes.size(-1)
            real_dims = min(tgt_boxes.size(-1), code_size)
            if tgt_boxes.size(-1) < code_size:
                pad = src_boxes.new_zeros((tgt_boxes.size(0),
                                           code_size - tgt_boxes.size(-1)))
                tgt_boxes = torch.cat([tgt_boxes, pad], dim=-1)
            # Pi-symmetric yaw target flip: pick the closer orientation.
            if self.pi_symmetric_class_indices:
                matched_labels = gt_labels[tgt]
                sym_mask = torch.zeros_like(matched_labels, dtype=torch.bool)
                for cls_idx in self.pi_symmetric_class_indices:
                    sym_mask |= (matched_labels == cls_idx)
                if sym_mask.any() and tgt_boxes.size(-1) > 7:
                    pred_sin = src_boxes[..., 6].detach()
                    pred_cos = src_boxes[..., 7].detach()
                    gt_sin = tgt_boxes[..., 6]
                    gt_cos = tgt_boxes[..., 7]
                    dist_orig = (pred_sin - gt_sin).abs() + \
                        (pred_cos - gt_cos).abs()
                    dist_flip = (pred_sin + gt_sin).abs() + \
                        (pred_cos + gt_cos).abs()
                    flip_mask = sym_mask & (dist_flip < dist_orig)
                    if flip_mask.any():
                        tgt_boxes = tgt_boxes.clone()
                        tgt_boxes[..., 6] = torch.where(
                            flip_mask, -tgt_boxes[..., 6], tgt_boxes[..., 6])
                        tgt_boxes[..., 7] = torch.where(
                            flip_mask, -tgt_boxes[..., 7], tgt_boxes[..., 7])
            bbox_weights = src_boxes.new_zeros(src_boxes.shape)
            bbox_weights[:, :real_dims] = 1.0
            bbox_weights = bbox_weights * self.code_weights
            valid = torch.isfinite(tgt_boxes).all(dim=-1)
            loss_bbox = self.loss_bbox(
                src_boxes[valid], tgt_boxes[valid], bbox_weights[valid],
                avg_factor=avg_factor)
        else:
            loss_bbox = track_instances.pred_boxes.sum() * 0.0

        # ---- BEV GIoU on matched queries (axis-aligned, same as det head) --
        loss_iou = None
        if self.loss_iou is not None and matched_indices.numel() > 0:
            src, tgt = matched_indices[:, 0], matched_indices[:, 1]
            src_boxes = track_instances.pred_boxes[src]
            tgt_boxes_norm = normalize_bbox(gt_bboxes[tgt], self.pc_range)
            code_size = src_boxes.size(-1)
            if tgt_boxes_norm.size(-1) < code_size:
                pad = src_boxes.new_zeros(
                    (tgt_boxes_norm.size(0), code_size - tgt_boxes_norm.size(-1)))
                tgt_boxes_norm = torch.cat([tgt_boxes_norm, pad], dim=-1)
            valid = torch.isfinite(tgt_boxes_norm).all(dim=-1)
            if valid.any():
                pred_dec = denormalize_bbox(src_boxes[valid][:, :8])
                tgt_dec = denormalize_bbox(tgt_boxes_norm[valid][:, :8])
                pred_bev = torch.cat([
                    pred_dec[:, 0:2] - pred_dec[:, 3:5] / 2,
                    pred_dec[:, 0:2] + pred_dec[:, 3:5] / 2,
                ], dim=-1)
                tgt_bev = torch.cat([
                    tgt_dec[:, 0:2] - tgt_dec[:, 3:5] / 2,
                    tgt_dec[:, 0:2] + tgt_dec[:, 3:5] / 2,
                ], dim=-1)
                loss_iou = self.loss_iou(
                    pred_bev, tgt_bev, avg_factor=avg_factor)
            else:
                loss_iou = src_boxes.new_zeros(())

        losses = dict(
            loss_cls=torch.nan_to_num(loss_cls),
            loss_bbox=torch.nan_to_num(loss_bbox),
        )
        if loss_iou is not None:
            losses['loss_iou'] = torch.nan_to_num(loss_iou)
        return losses

    # ------------------------------------------------------------------ #
    # Forward — return weighted clip loss dict
    # ------------------------------------------------------------------ #

    def forward(self) -> dict:
        """Apply per-loss weights and return the final loss dict."""
        out = {}
        for k, v in self.losses_dict.items():
            # Key shape: frame_{i}_{name}_{dec_lvl}; weight_dict keys map by
            # 'name' (e.g. 'loss_cls', 'loss_bbox').
            for wk, wv in self.weight_dict.items():
                if f'_{wk}_' in k:
                    out[k] = v * wv
                    break
            else:
                out[k] = v
        return out
