"""UniAD-aligned LiDAR BEV detection head.

This head is intentionally independent from ``BEVDETRHead``. It assumes the
detector owns object queries and reference points (UniAD convention) and
keeps the reference-point carry between decoder layers entirely in
inverse-sigmoid / logit space. The standalone ``BEVDETRHead`` keeps its own
sigmoid-space refine path for the legacy temporal configs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.task_modules import AssignResult
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.registry import MODELS, TASK_UTILS

from .bev_detr_head import denormalize_bbox, inverse_sigmoid, normalize_bbox
from ..modules.lidar_bevformer_encoder import LearnedBEVPositionalEncoding


@MODELS.register_module()
class BEVFormerLiDARHead(BaseModule):
    """UniAD-aligned DETR head over BEVFormer-style BEV features.

    The detector supplies object queries and inverse-sigmoid reference points
    each call. The head owns:
      * the BEV query / positional encoding consumed by the BEV encoder, plus
        the LiDAR-feature projection;
      * the ``transformer`` with its BEV encoder and detection decoder;
      * cls/reg branches, losses, and assignment + prediction utilities.

    Box refinement is done entirely in logit space — ``reference_points`` in
    the returned dict is logit-space, matching UniAD's ``last_ref_points``.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 transformer: dict,
                 bev_h: int,
                 bev_w: int,
                 embed_dims: int = 256,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 num_points: int = 4,
                 ffn_channels: int = 1024,
                 dropout: float = 0.1,
                 num_reg_fcs: int = 2,
                 code_size: int = 10,
                 code_weights: Optional[Sequence[float]] = None,
                 pc_range: Optional[Sequence[float]] = None,
                 with_box_refine: bool = True,
                 as_two_stage: bool = False,
                 sync_cls_avg_factor: bool = True,
                 bbox_coder: Optional[dict] = None,
                 positional_encoding: Optional[dict] = None,
                 lidar_in_channels: Optional[int] = None,
                 bev_embed_dims: Optional[int] = None,
                 loss_cls: dict = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0,
                     reduction='mean'),
                 loss_bbox: dict = dict(
                     type='mmdet.L1Loss',
                     loss_weight=0.25,
                     reduction='mean'),
                 loss_iou: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 # Accepted for config-symmetry with BEVDETRHead; this head
                 # does not own queries, so num_query is informational only.
                 num_query: Optional[int] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if as_two_stage:
            raise ValueError('BEVFormerLiDARHead does not support '
                             'as_two_stage=True.')
        if bbox_coder is not None and pc_range is None:
            pc_range = bbox_coder.get('pc_range', None)
        if pc_range is None:
            raise ValueError('pc_range is required for BEVFormerLiDARHead.')

        self.in_channels = int(in_channels)
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_decoder_layers = num_decoder_layers
        self.code_size = code_size
        self.pc_range = list(pc_range)
        self.with_box_refine = with_box_refine
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg or {}
        if bbox_coder is not None:
            self.test_cfg.setdefault('max_num', bbox_coder.get('max_num'))
            self.test_cfg.setdefault(
                'post_center_range', bbox_coder.get('post_center_range'))
        self.test_cfg = {
            key: value
            for key, value in self.test_cfg.items()
            if value is not None
        }
        self.pi_symmetric_class_indices = list(
            (train_cfg or {}).get('pi_symmetric_class_indices', []) or [])
        self.num_query = num_query
        self.as_two_stage = as_two_stage
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bbox_coder_cfg = bbox_coder
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.lidar_in_channels = int(lidar_in_channels or self.in_channels)

        if code_weights is None:
            code_weights = [1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 0.2, 0.2]
        if len(code_weights) != code_size:
            raise ValueError('code_weights length must match code_size: '
                             f'{len(code_weights)} vs {code_size}.')
        self.register_buffer(
            'code_weights',
            torch.tensor(code_weights, dtype=torch.float32),
            persistent=False)

        # ---- BEV encoder boundary (matches UniAD BEVFormer head) ----------
        transformer = dict(transformer)
        configured_embed_dims = transformer.get('embed_dims')
        self.bev_embed_dims = int(bev_embed_dims or configured_embed_dims
                                  or self.in_channels)
        if (configured_embed_dims is not None
                and configured_embed_dims != self.bev_embed_dims):
            raise ValueError('transformer.embed_dims must match '
                             'bev_embed_dims: '
                             f'{configured_embed_dims} vs '
                             f'{self.bev_embed_dims}.')
        if self.bev_embed_dims != self.in_channels:
            raise ValueError('bev_embed_dims must match in_channels because '
                             'the encoded BEV is fed directly into the '
                             'decoder: '
                             f'{self.bev_embed_dims} vs {self.in_channels}.')

        self.lidar_input_proj = (
            nn.Identity()
            if self.lidar_in_channels == self.bev_embed_dims else nn.Conv2d(
                self.lidar_in_channels, self.bev_embed_dims, kernel_size=1))
        self.bev_embedding = nn.Embedding(bev_h * bev_w,
                                          self.bev_embed_dims)
        self._validate_positional_encoding(
            positional_encoding, bev_h, bev_w, self.bev_embed_dims)
        self.positional_encoding = LearnedBEVPositionalEncoding(
            bev_h, bev_w, self.bev_embed_dims)
        transformer.setdefault('bev_h', bev_h)
        transformer.setdefault('bev_w', bev_w)
        transformer.setdefault('embed_dims', self.bev_embed_dims)
        self.transformer = MODELS.build(transformer)
        if getattr(self.transformer, 'decoder', None) is None:
            raise ValueError('transformer.decoder is required for '
                             'BEVFormerLiDARHead.')
        self.num_decoder_layers = self.transformer.decoder.num_layers

        # ---- DETR decoder over BEV memory ----------------------------------
        self.input_proj = nn.Conv2d(in_channels, embed_dims, kernel_size=1)

        self.cls_branches = nn.ModuleList(
            [self._make_cls_branch(num_reg_fcs)
             for _ in range(self.num_decoder_layers)])
        self.reg_branches = nn.ModuleList(
            [self._make_reg_branch(num_reg_fcs)
             for _ in range(self.num_decoder_layers)])

        # ---- losses / assigner --------------------------------------------
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou) if loss_iou else None
        self.bg_cls_weight = 0.0

        if train_cfg is not None:
            self.assigner = TASK_UTILS.build(train_cfg['assigner'])
        else:
            self.assigner = None

    @staticmethod
    def _validate_positional_encoding(positional_encoding: Optional[dict],
                                      bev_h: int, bev_w: int,
                                      embed_dims: int) -> None:
        if positional_encoding is None:
            return
        cfg = dict(positional_encoding)
        if cfg.get('type') != 'LearnedPositionalEncoding':
            raise ValueError('BEVFormerLiDARHead only supports '
                             'LearnedPositionalEncoding-style BEV position '
                             f'config, got {cfg.get("type")}.')
        expected = dict(
            num_feats=embed_dims // 2,
            row_num_embed=bev_h,
            col_num_embed=bev_w)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError(f'positional_encoding.{key} must be {value}, '
                                 f'got {cfg.get(key)}.')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Map old head-owned decoder checkpoints to transformer.decoder."""
        old_prefix = prefix + 'decoder_layers.'
        new_prefix = prefix + 'transformer.decoder.layers.'
        for key in list(state_dict.keys()):
            if key.startswith(old_prefix):
                mapped = new_prefix + key[len(old_prefix):]
                state_dict.setdefault(mapped, state_dict[key])
                state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def _make_cls_branch(self, num_fcs: int) -> nn.Sequential:
        layers = []
        for _ in range(num_fcs):
            layers.extend([
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Linear(self.embed_dims, self.num_classes))
        return nn.Sequential(*layers)

    def _make_reg_branch(self, num_fcs: int) -> nn.Sequential:
        layers = []
        for _ in range(num_fcs):
            layers.extend([
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Linear(self.embed_dims, self.code_size))
        return nn.Sequential(*layers)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        if getattr(self.loss_cls, 'use_sigmoid', False):
            bias_init = bias_init_with_prob(0.01)
            for cls_branch in self.cls_branches:
                nn.init.constant_(cls_branch[-1].bias, bias_init)

    # ----------------------- BEV encoder front-door ------------------------

    def get_bev_features(self,
                         lidar_bev: Tensor,
                         prev_bev: Optional[Tensor] = None,
                         queue_meta: Optional[Sequence[dict]] = None
                         ) -> Tensor:
        batch_size, channels, bev_h, bev_w = lidar_bev.shape
        expected_shape = (self.lidar_in_channels, self.bev_h, self.bev_w)
        if (channels, bev_h, bev_w) != expected_shape:
            raise ValueError('lidar_bev shape mismatch for '
                             'BEVFormerLiDARHead: expected '
                             f'(B, {expected_shape[0]}, {expected_shape[1]}, '
                             f'{expected_shape[2]}), got '
                             f'{tuple(lidar_bev.shape)}.')
        if prev_bev is not None:
            expected_prev_shape = (
                batch_size, self.bev_embed_dims, self.bev_h, self.bev_w)
            if tuple(prev_bev.shape) != expected_prev_shape:
                raise ValueError('prev_bev shape mismatch for '
                                 'BEVFormerLiDARHead: expected '
                                 f'{expected_prev_shape}, got '
                                 f'{tuple(prev_bev.shape)}.')

        encoder_lidar_bev = self.lidar_input_proj(lidar_bev)
        bev_queries = self.bev_embedding.weight.to(
            dtype=lidar_bev.dtype, device=lidar_bev.device)
        bev_pos = self.positional_encoding(batch_size, lidar_bev.device,
                                           lidar_bev.dtype)
        return self.transformer.get_bev_features(
            encoder_lidar_bev, bev_queries=bev_queries, bev_pos=bev_pos,
            prev_bev=prev_bev, queue_meta=queue_meta)

    # ------------------------------ decoder --------------------------------

    @staticmethod
    def _unwrap_feats(feats) -> Tensor:
        if not isinstance(feats, (list, tuple)) or len(feats) != 1:
            raise ValueError('BEVFormerLiDARHead expects a single BEV feature '
                             f'map, got {type(feats)} with len={len(feats)}.')
        return feats[0]

    def forward(self, feats, object_query_embeds: Tensor,
                ref_points: Tensor) -> dict:
        """Run decoder. ``object_query_embeds`` and ``ref_points`` are
        mandatory and supplied by the detector (UniAD convention).

        Args:
            feats: Single-element list containing BEV tensor [B, C, H, W].
            object_query_embeds: [N, 2*D] where first half is query_pos,
                second half is query_feat.
            ref_points: [B, N, 3] logit-space reference points
                (inverse_sigmoid of normalized (cx, cy, cz)).
        """
        if object_query_embeds is None or ref_points is None:
            raise ValueError('BEVFormerLiDARHead.forward requires '
                             'object_query_embeds and ref_points; the '
                             'detector owns these tensors.')

        bev = self._unwrap_feats(feats)
        batch_size, _, bev_h, bev_w = bev.shape

        memory = self.input_proj(bev)
        memory = memory.flatten(2).transpose(1, 2).contiguous()
        inter_states, init_reference, inter_references = (
            self.transformer.get_states_and_refs(
                memory,
                object_query_embeds,
                bev_h,
                bev_w,
                reference_points=ref_points,
                reg_branches=self.reg_branches
                if self.with_box_refine else None))
        hs = inter_states.permute(0, 2, 1, 3).contiguous()

        all_cls_scores = []
        all_bbox_preds = []
        query_feats = []
        last_ref_points = ref_points
        for layer_id in range(hs.shape[0]):
            reference = (
                init_reference if layer_id == 0 else
                inter_references[layer_id - 1])
            ref_logit = inverse_sigmoid(reference)
            query_b = hs[layer_id]
            cls_score = self.cls_branches[layer_id](query_b)
            reg_raw = self.reg_branches[layer_id](query_b)
            bbox_pred = self._decode_regression(reg_raw, ref_logit)
            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)
            query_feats.append(query_b)

            # UniAD carry: refined xy/z stay in logit space; next layer
            # re-sigmoids for sampling.
            refined_xy = reg_raw[..., 0:2] + ref_logit[..., 0:2]
            refined_z = reg_raw[..., 4:5] + ref_logit[..., 2:3]
            last_ref_points = torch.cat([refined_xy, refined_z], dim=-1)

        return dict(
            all_cls_scores=torch.stack(all_cls_scores),
            all_bbox_preds=torch.stack(all_bbox_preds),
            query_feats=torch.stack(query_feats),
            reference_points=last_ref_points.detach(),
            last_ref_points=last_ref_points.detach(),
        )

    def _decode_regression(self, raw: Tensor, reference_logit: Tensor) -> Tensor:
        raw = raw.clone()
        raw[..., 0:2] = (raw[..., 0:2] + reference_logit[..., 0:2]).sigmoid()
        raw[..., 4:5] = (
            raw[..., 4:5] + reference_logit[..., 2:3]).sigmoid()

        raw[..., 0:1] = raw[..., 0:1] * (
            self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        raw[..., 1:2] = raw[..., 1:2] * (
            self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        raw[..., 4:5] = raw[..., 4:5] * (
            self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        return raw

    def get_detections(self, feats, object_query_embeds: Tensor,
                       ref_points: Tensor) -> dict:
        """UniAD-compatible entry point — alias for ``forward``."""
        return self.forward(feats, object_query_embeds=object_query_embeds,
                            ref_points=ref_points)

    # ------------------------- target / loss ------------------------------

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances_3d: InstanceData
                            ) -> Tuple[Tensor, ...]:
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_bboxes = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1)
        gt_labels = gt_instances_3d.labels_3d

        assign_result: AssignResult = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore=None)
        gt_inds = assign_result.gt_inds
        pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).squeeze(-1)
        neg_inds = torch.nonzero(gt_inds == 0, as_tuple=False).squeeze(-1)

        num_bboxes = bbox_pred.size(0)
        labels = bbox_pred.new_full(
            (num_bboxes, ), self.num_classes, dtype=torch.long)
        label_weights = bbox_pred.new_ones(num_bboxes)
        bbox_targets = bbox_pred.new_zeros((num_bboxes, self.code_size))
        bbox_weights = bbox_pred.new_zeros((num_bboxes, self.code_size))

        if len(pos_inds) > 0:
            assigned_gt = gt_inds[pos_inds] - 1
            labels[pos_inds] = gt_labels[assigned_gt]
            target = normalize_bbox(gt_bboxes[assigned_gt], self.pc_range)
            real_dims = min(target.size(-1), self.code_size)
            if target.size(-1) < self.code_size:
                target = F.pad(target, (0, self.code_size - target.size(-1)))
            bbox_targets[pos_inds] = target[:, :self.code_size]
            bbox_weights[pos_inds, :real_dims] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, \
            neg_inds

    def get_targets(self, batch_cls_scores: Tensor, batch_bbox_preds: Tensor,
                    batch_gt_instances: List[InstanceData]) -> Tuple:
        results = [
            self._get_targets_single(batch_cls_scores[i], batch_bbox_preds[i],
                                     batch_gt_instances[i])
            for i in range(batch_cls_scores.size(0))
        ]
        labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = \
            zip(*results)
        num_total_pos = sum(inds.numel() for inds in pos_inds)
        num_total_neg = sum(inds.numel() for inds in neg_inds)
        return (list(labels), list(label_weights), list(bbox_targets),
                list(bbox_weights), num_total_pos, num_total_neg)

    def loss_by_feat_single(self, batch_cls_scores: Tensor,
                            batch_bbox_preds: Tensor,
                            batch_gt_instances: List[InstanceData]) -> Tuple:
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = self.get_targets(
             batch_cls_scores, batch_bbox_preds, batch_gt_instances)

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores = batch_cls_scores.reshape(-1, self.num_classes)
        cls_avg_factor = num_total_pos + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos),
                                    min=1).item()
        bbox_preds = batch_bbox_preds.reshape(-1, self.code_size)
        if self.pi_symmetric_class_indices:
            sym_mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls_idx in self.pi_symmetric_class_indices:
                sym_mask |= (labels == cls_idx)
            if sym_mask.any():
                pred_sin = bbox_preds[..., 6].detach()
                pred_cos = bbox_preds[..., 7].detach()
                gt_sin = bbox_targets[..., 6]
                gt_cos = bbox_targets[..., 7]
                dist_orig = (pred_sin - gt_sin).abs() + \
                    (pred_cos - gt_cos).abs()
                dist_flip = (pred_sin + gt_sin).abs() + \
                    (pred_cos + gt_cos).abs()
                flip_mask = sym_mask & (dist_flip < dist_orig)
                if flip_mask.any():
                    bbox_targets = bbox_targets.clone()
                    bbox_targets[..., 6] = torch.where(
                        flip_mask, -bbox_targets[..., 6], bbox_targets[..., 6])
                    bbox_targets[..., 7] = torch.where(
                        flip_mask, -bbox_targets[..., 7], bbox_targets[..., 7])
        bbox_weights = bbox_weights * self.code_weights
        valid = torch.isfinite(bbox_targets).all(dim=-1)
        loss_bbox = self.loss_bbox(
            bbox_preds[valid],
            bbox_targets[valid],
            bbox_weights[valid],
            avg_factor=num_total_pos)

        with torch.no_grad():
            vel_mask = valid & (bbox_weights[:, 8:10].sum(dim=-1) > 0)
            if vel_mask.any():
                vel_diff = bbox_preds[vel_mask, 8:10] - \
                    bbox_targets[vel_mask, 8:10]
                vel_l1 = vel_diff.abs().mean()
                vel_l2 = vel_diff.norm(dim=-1).mean()
            else:
                vel_l1 = bbox_preds.new_zeros(())
                vel_l2 = bbox_preds.new_zeros(())

        loss_iou = None
        if self.loss_iou is not None:
            pos_bbox_weights = bbox_weights.sum(dim=-1) > 0
            pos_mask = pos_bbox_weights & valid
            if pos_mask.any():
                pred_dec = denormalize_bbox(bbox_preds[pos_mask][:, :8])
                tgt_dec = denormalize_bbox(bbox_targets[pos_mask][:, :8])
                pred_bev = torch.cat([
                    pred_dec[:, 0:2] - pred_dec[:, 3:5] / 2,
                    pred_dec[:, 0:2] + pred_dec[:, 3:5] / 2,
                ], dim=-1)
                tgt_bev = torch.cat([
                    tgt_dec[:, 0:2] - tgt_dec[:, 3:5] / 2,
                    tgt_dec[:, 0:2] + tgt_dec[:, 3:5] / 2,
                ], dim=-1)
                loss_iou = self.loss_iou(
                    pred_bev, tgt_bev, avg_factor=num_total_pos)
            else:
                loss_iou = bbox_preds.new_zeros(())
        return (torch.nan_to_num(loss_cls), torch.nan_to_num(loss_bbox),
                torch.nan_to_num(loss_iou) if loss_iou is not None else None,
                torch.nan_to_num(vel_l1).detach(),
                torch.nan_to_num(vel_l2).detach())

    def loss_by_feat(self, preds: dict,
                     batch_gt_instances: List[InstanceData]) -> dict:
        all_cls_scores = preds['all_cls_scores']
        all_bbox_preds = preds['all_bbox_preds']
        losses_cls = []
        losses_bbox = []
        losses_iou = []
        vel_l1_list = []
        vel_l2_list = []
        for cls_scores, bbox_preds in zip(all_cls_scores, all_bbox_preds):
            loss_cls, loss_bbox, loss_iou, vel_l1, vel_l2 = \
                self.loss_by_feat_single(
                cls_scores, bbox_preds, batch_gt_instances)
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)
            losses_iou.append(loss_iou)
            vel_l1_list.append(vel_l1)
            vel_l2_list.append(vel_l2)

        loss_dict = dict(
            loss_cls=losses_cls[-1],
            loss_bbox=losses_bbox[-1],
            vel_l1=vel_l1_list[-1],
            vel_l2=vel_l2_list[-1],
        )
        if losses_iou[-1] is not None:
            loss_dict['loss_iou'] = losses_iou[-1]
        for i in range(len(losses_cls) - 1):
            loss_dict[f'd{i}.loss_cls'] = losses_cls[i]
            loss_dict[f'd{i}.loss_bbox'] = losses_bbox[i]
            if losses_iou[i] is not None:
                loss_dict[f'd{i}.loss_iou'] = losses_iou[i]
        return loss_dict

    # ----------------------------- predict --------------------------------

    def predict_by_feat(self, preds: dict,
                        batch_metas: List[dict]) -> List[InstanceData]:
        cls_scores = preds['all_cls_scores'][-1].sigmoid()
        bbox_preds = preds['all_bbox_preds'][-1]
        max_num = int(self.test_cfg.get('max_num', cls_scores.size(1)))
        score_threshold = self.test_cfg.get('score_threshold', None)
        post_center_range = self.test_cfg.get('post_center_range', None)

        ret_list = []
        for sample_idx in range(cls_scores.size(0)):
            scores, indexes = cls_scores[sample_idx].reshape(-1).topk(
                min(max_num, cls_scores[sample_idx].numel()))
            labels = indexes % self.num_classes
            bbox_index = torch.div(
                indexes, self.num_classes, rounding_mode='trunc')
            boxes = denormalize_bbox(bbox_preds[sample_idx][bbox_index])

            keep = torch.ones_like(scores, dtype=torch.bool)
            if score_threshold is not None:
                keep &= scores > float(score_threshold)
            if post_center_range is not None:
                pcr = boxes.new_tensor(post_center_range)
                keep &= (boxes[:, :3] >= pcr[:3]).all(dim=1)
                keep &= (boxes[:, :3] <= pcr[3:]).all(dim=1)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            result = InstanceData()
            box_type_3d = batch_metas[sample_idx]['box_type_3d']
            result.bboxes_3d = box_type_3d(
                boxes, box_dim=boxes.size(-1), origin=(0.5, 0.5, 0.5))
            result.scores_3d = scores
            result.labels_3d = labels
            result.query_feats = preds['query_feats'][-1, sample_idx][
                bbox_index[keep]]
            ret_list.append(result)
        return ret_list
