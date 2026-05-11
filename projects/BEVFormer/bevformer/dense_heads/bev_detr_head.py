"""DETR-style detection head on top of BEV features.

This is intentionally smaller than the camera BEVFormer/DETR3D head: the
upstream detector already provides a fused BEV tensor, so the head only needs a
query decoder over BEV memory plus set-prediction losses.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules.assigners import BaseAssigner
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmcv.ops import MultiScaleDeformableAttention

from mmdet3d.registry import MODELS, TASK_UTILS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def normalize_bbox(bboxes: Tensor, pc_range: Sequence[float]) -> Tensor:
    """Convert LiDAR boxes to the DETR3D regression target layout.

    Input layout is ``(cx, cy, cz, l, w, h, yaw[, vx, vy])``.  Output layout is
    ``(cx, cy, log(l), log(w), cz, log(h), sin(yaw), cos(yaw)[, vx, vy])``.
    The center coordinates remain metric, matching the DETR3D project code in
    this repository.
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    length = bboxes[..., 3:4].clamp(min=1e-5).log()
    width = bboxes[..., 4:5].clamp(min=1e-5).log()
    height = bboxes[..., 5:6].clamp(min=1e-5).log()
    yaw = bboxes[..., 6:7]
    parts = [cx, cy, length, width, cz, height, yaw.sin(), yaw.cos()]
    if bboxes.size(-1) > 7:
        parts.extend([bboxes[..., 7:8], bboxes[..., 8:9]])
    return torch.cat(parts, dim=-1)


def denormalize_bbox(preds: Tensor) -> Tensor:
    """Convert predicted DETR layout to LiDAR box tensor layout."""
    yaw = torch.atan2(preds[..., 6:7], preds[..., 7:8])
    length = preds[..., 2:3].exp()
    width = preds[..., 3:4].exp()
    height = preds[..., 5:6].exp()
    parts = [
        preds[..., 0:1], preds[..., 1:2], preds[..., 4:5], length, width,
        height, yaw
    ]
    if preds.size(-1) > 8:
        parts.extend([preds[..., 8:9], preds[..., 9:10]])
    return torch.cat(parts, dim=-1)


@TASK_UTILS.register_module()
class BEVDETRBBox3DL1Cost:
    """L1 matching cost for DETR-layout 3D boxes."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def __call__(self, bbox_pred: Tensor, gt_bboxes: Tensor) -> Tensor:
        return torch.cdist(bbox_pred, gt_bboxes, p=1) * self.weight


@TASK_UTILS.register_module()
class BEVDETRHungarianAssigner3D(BaseAssigner):
    """Hungarian assigner for BEVDETRHead.

    This keeps a unique registry name so configs can import BEVFusion and this
    head together without clashing with other project-local assigners.
    """

    def __init__(self,
                 cls_cost: dict,
                 reg_cost: dict,
                 pc_range: Sequence[float],
                 pi_symmetric_class_indices: Optional[Sequence[int]] = None,
                 iou_cost: Optional[dict] = None,
                 iou_calculator: Optional[dict] = None) -> None:
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.pc_range = pc_range
        self.pi_symmetric_class_indices = (
            list(pi_symmetric_class_indices)
            if pi_symmetric_class_indices else [])
        self.iou_cost = TASK_UTILS.build(iou_cost) if iou_cost else None
        self.iou_calculator = (TASK_UTILS.build(iou_calculator)
                               if iou_calculator else None)
        if (self.iou_cost is None) != (self.iou_calculator is None):
            raise ValueError('iou_cost and iou_calculator must be both '
                             'set or both omitted.')

    def assign(self,
               bbox_pred: Tensor,
               cls_pred: Tensor,
               gt_bboxes: Tensor,
               gt_labels: Tensor,
               gt_bboxes_ignore=None,
               eps: float = 1e-7) -> AssignResult:
        assert gt_bboxes_ignore is None, \
            'BEVDETRHungarianAssigner3D does not support ignored boxes.'
        num_gts = gt_bboxes.size(0)
        num_bboxes = bbox_pred.size(0)
        assigned_gt_inds = bbox_pred.new_full(
            (num_bboxes, ), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full(
            (num_bboxes, ), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        pred_instances = InstanceData(scores=cls_pred)
        gt_instances = InstanceData(labels=gt_labels)
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        if normalized_gt_bboxes.size(-1) < bbox_pred.size(-1):
            normalized_gt_bboxes = F.pad(
                normalized_gt_bboxes,
                (0, bbox_pred.size(-1) - normalized_gt_bboxes.size(-1)))
        reg_cost = self.reg_cost(
            bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # For pi-symmetric classes (e.g. IGV, WheelCrane in port scenes), yaw
        # and yaw + pi are physically identical, so the matching cost should
        # take the cheaper of the two orientations per-GT to avoid penalising
        # ambiguous head/tail labels.
        if self.pi_symmetric_class_indices:
            sym_gt_mask = torch.zeros_like(gt_labels, dtype=torch.bool)
            for cls_idx in self.pi_symmetric_class_indices:
                sym_gt_mask |= (gt_labels == cls_idx)
            if sym_gt_mask.any():
                flipped_gt = normalized_gt_bboxes.clone()
                flipped_gt[:, 6] = -flipped_gt[:, 6]
                flipped_gt[:, 7] = -flipped_gt[:, 7]
                reg_cost_flip = self.reg_cost(
                    bbox_pred[:, :8], flipped_gt[:, :8])
                reg_cost = torch.where(
                    sym_gt_mask[None, :], torch.minimum(reg_cost, reg_cost_flip),
                    reg_cost)

        cost = cls_cost + reg_cost

        if self.iou_cost is not None:
            # IoU is computed on decoded 7-dim LiDAR boxes (no grad through
            # matching). BEV nearest IoU is yaw mod pi by construction, so
            # symmetric classes need no special handling here.
            pred_dec = denormalize_bbox(bbox_pred[:, :8])[:, :7]
            gt_dec = gt_bboxes[:, :7]
            iou = self.iou_calculator(pred_dec, gt_dec)
            cost = cost + self.iou_cost(iou)

        cost = cost.detach().cpu()
        if torch.isnan(cost).any() or torch.isinf(cost).any():
            cost = torch.nan_to_num(
                cost, nan=100.0, posinf=100.0, neginf=-100.0)

        if linear_sum_assignment is None:
            raise ImportError('Please install scipy for Hungarian matching.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


class _BEVDetrDecoderLayer(BaseModule):
    """A DETR decoder layer with deformable BEV cross-attention.

    Self-attention is the standard nn.MultiheadAttention; cross-attention is
    MultiScaleDeformableAttention, sampling K points per query around the
    query's reference point on the BEV memory. This matches BEVFormer's
    DetectionTransformerDecoder.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_points: int = 4,
                 num_levels: int = 1,
                 ffn_channels: int = 1024,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_channels, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query: Tensor, query_pos: Tensor, value: Tensor,
                reference_points: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor) -> Tensor:
        """Args:
            query: (N_q, B, D) — sequence-first to match self-attn
            query_pos: (N_q, B, D)
            value: (B, N_kv, D) — flattened BEV memory, batch-first
            reference_points: (B, N_q, num_levels, 2), normalized [0, 1]
            spatial_shapes: (num_levels, 2), each row is (H, W)
            level_start_index: (num_levels,)
        """
        # Self-attn (sequence-first)
        q = k = query + query_pos
        query2 = self.self_attn(q, k, value=query)[0]
        query = self.norm1(query + self.dropout1(query2))

        # Deformable cross-attn (batch-first). Internally adds query_pos to
        # query and applies its own residual + dropout.
        query_b = query.permute(1, 0, 2).contiguous()
        query_pos_b = query_pos.permute(1, 0, 2).contiguous()
        query_b = self.cross_attn(
            query=query_b,
            value=value,
            query_pos=query_pos_b,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = query_b.permute(1, 0, 2).contiguous()
        query = self.norm2(query)

        query2 = self.ffn(query)
        query = self.norm3(query + self.dropout3(query2))
        return query


@MODELS.register_module()
class BEVDETRHead(BaseModule):
    """DETR-style 3D detection head over a single BEV feature map."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_query: int = 300,
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
                 bev_feature_layout: str = 'xy',
                 with_box_refine: bool = True,
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
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if bev_feature_layout not in ('xy', 'yx'):
            raise ValueError('bev_feature_layout must be "xy" or "yx", '
                             f'got {bev_feature_layout}.')
        if pc_range is None:
            raise ValueError('pc_range is required for BEVDETRHead.')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.num_decoder_layers = num_decoder_layers
        self.num_points = num_points
        self.code_size = code_size
        self.pc_range = list(pc_range)
        self.bev_feature_layout = bev_feature_layout
        self.with_box_refine = with_box_refine
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg or {}
        self.pi_symmetric_class_indices = list(
            (train_cfg or {}).get('pi_symmetric_class_indices', []) or [])

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

        self.input_proj = nn.Conv2d(in_channels, embed_dims, kernel_size=1)
        self.query_embed = nn.Embedding(num_query, embed_dims)
        self.query_pos_embed = nn.Embedding(num_query, embed_dims)
        self.reference_points = nn.Linear(embed_dims, 3)

        self.decoder_layers = nn.ModuleList([
            _BEVDetrDecoderLayer(
                embed_dims=embed_dims, num_heads=num_heads,
                num_points=num_points, num_levels=1,
                ffn_channels=ffn_channels, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        # Independent branches per layer (required for iterative box refine).
        self.cls_branches = nn.ModuleList(
            [self._make_cls_branch(num_reg_fcs)
             for _ in range(num_decoder_layers)])
        self.reg_branches = nn.ModuleList(
            [self._make_reg_branch(num_reg_fcs)
             for _ in range(num_decoder_layers)])

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou) if loss_iou else None
        self.bg_cls_weight = 0.0
        self.sync_cls_avg_factor = True

        if train_cfg is not None:
            self.assigner = TASK_UTILS.build(train_cfg['assigner'])
        else:
            self.assigner = None

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
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0)
        if getattr(self.loss_cls, 'use_sigmoid', False):
            bias_init = bias_init_with_prob(0.01)
            for cls_branch in self.cls_branches:
                nn.init.constant_(cls_branch[-1].bias, bias_init)

    def generate_init_query_embeds(self) -> Tensor:
        """Return the learnable queries in UniAD-style [N, 2*D] format.

        First half is query_pos, second half is query_feat. This is the
        starting point for track instances at the beginning of a clip.
        """
        return torch.cat([self.query_pos_embed.weight,
                          self.query_embed.weight], dim=-1)

    @staticmethod
    def _unwrap_feats(feats: List[Tensor]) -> Tensor:
        if not isinstance(feats, (list, tuple)) or len(feats) != 1:
            raise ValueError('BEVDETRHead expects a single BEV feature map, '
                             f'got {type(feats)} with len={len(feats)}.')
        return feats[0]

    def _to_msda_reference(self, reference_xy: Tensor) -> Tensor:
        """Reorder (cx, cy) -> MSDA's (W_frac, H_frac) layout.

        Our `reference[..., 0:2]` is (cx_norm, cy_norm) where cx is along the
        ego X axis and cy along Y. MSDA interprets the last dim as
        (x_in_W, y_in_H). For BEV layout 'xy' the tensor is [B, C, H=X, W=Y],
        so the X axis corresponds to the H dimension — we swap. For 'yx'
        the mapping is already aligned.
        """
        if self.bev_feature_layout == 'xy':
            # (cx, cy) -> (cy, cx)   because W corresponds to Y, H to X.
            return reference_xy[..., [1, 0]]
        return reference_xy

    def forward(self, feats: List[Tensor],
                object_query_embeds: Optional[Tensor] = None,
                ref_points: Optional[Tensor] = None) -> dict:
        """Run the deformable DETR decoder over BEV features.

        Args:
            feats: Single-element list containing BEV tensor [B, C, H, W].
            object_query_embeds: Optional external queries [N, 2*D] where the
                first half is query_pos and the second half is query_feat.
                When None, uses the internal learned embeddings (detection
                mode). This is the entry point for track-query injection.
            ref_points: Optional reference points [B, N, 3] in sigmoid space
                with semantics (cx_norm, cy_norm, cz_norm). When None,
                derived from query_pos via ``self.reference_points``.
        """
        bev = self._unwrap_feats(feats)
        batch_size, _, bev_h, bev_w = bev.shape

        # MSDA wants value in batch-first layout [B, HW, D] plus
        # spatial_shapes [(H, W)] and level_start_index [(0,)].
        memory = self.input_proj(bev)  # [B, D, H, W]
        memory = memory.flatten(2).transpose(1, 2).contiguous()  # [B, HW, D]
        spatial_shapes = torch.as_tensor(
            [[bev_h, bev_w]], dtype=torch.long, device=bev.device)
        level_start_index = torch.as_tensor(
            [0], dtype=torch.long, device=bev.device)

        if object_query_embeds is not None:
            # UniAD-style: [N, 2*D] -> split into pos and feat
            dim = object_query_embeds.shape[-1] // 2
            query_pos_w = object_query_embeds[:, :dim]
            query_feat_w = object_query_embeds[:, dim:]
            query = query_feat_w[:, None, :].expand(-1, batch_size, -1)
            query_pos = query_pos_w[:, None, :].expand(-1, batch_size, -1)
        else:
            query = self.query_embed.weight[:, None, :].repeat(
                1, batch_size, 1)
            query_pos = self.query_pos_embed.weight[:, None, :].repeat(
                1, batch_size, 1)

        if ref_points is not None:
            reference = ref_points
        else:
            if object_query_embeds is not None:
                reference = self.reference_points(query_pos_w).sigmoid()
            else:
                reference = self.reference_points(
                    self.query_pos_embed.weight).sigmoid()
            reference = reference[None].expand(batch_size, -1, -1)

        all_cls_scores = []
        all_bbox_preds = []
        query_feats = []
        for layer_id, layer in enumerate(self.decoder_layers):
            # MSDA reference: [B, N, num_levels=1, 2] in (W_frac, H_frac).
            msda_ref = self._to_msda_reference(reference[..., :2])
            msda_ref = msda_ref.unsqueeze(2).contiguous()

            query = layer(query, query_pos, memory, msda_ref,
                          spatial_shapes, level_start_index)

            query_b = query.permute(1, 0, 2).contiguous()
            cls_score = self.cls_branches[layer_id](query_b)
            reg_raw = self.reg_branches[layer_id](query_b)
            bbox_pred = self._decode_regression(reg_raw, reference)
            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)
            query_feats.append(query_b)

            if self.with_box_refine and layer_id < len(self.decoder_layers) - 1:
                # Iterative refine: add reg delta to current ref in sigmoid
                # space, detach to bound backprop depth (BEVFormer convention).
                new_ref = reference.new_zeros(reference.shape)
                ref_logit = inverse_sigmoid(reference)
                new_ref[..., 0:2] = (
                    reg_raw[..., 0:2] + ref_logit[..., 0:2]).sigmoid()
                new_ref[..., 2:3] = (
                    reg_raw[..., 4:5] + ref_logit[..., 2:3]).sigmoid()
                reference = new_ref.detach()

        return dict(
            all_cls_scores=torch.stack(all_cls_scores),
            all_bbox_preds=torch.stack(all_bbox_preds),
            query_feats=torch.stack(query_feats),
            reference_points=reference,
        )

    def _decode_regression(self, raw: Tensor, reference: Tensor) -> Tensor:
        raw = raw.clone()
        ref = inverse_sigmoid(reference)
        raw[..., 0:2] = (raw[..., 0:2] + ref[..., 0:2]).sigmoid()
        raw[..., 4:5] = (raw[..., 4:5] + ref[..., 2:3]).sigmoid()

        raw[..., 0:1] = raw[..., 0:1] * (
            self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        raw[..., 1:2] = raw[..., 1:2] * (
            self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        raw[..., 4:5] = raw[..., 4:5] * (
            self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        return raw

    def get_detections(self, feats: List[Tensor],
                       object_query_embeds: Optional[Tensor] = None,
                       ref_points: Optional[Tensor] = None) -> dict:
        """UniAD-compatible entry: run decoder with external queries.

        This is the interface that the tracking detector will call per-frame.
        It is functionally identical to forward() but named explicitly to
        match UniAD's convention and make the call site self-documenting.
        """
        return self.forward(feats, object_query_embeds=object_query_embeds,
                            ref_points=ref_points)

    def loss(self, feats: List[Tensor], batch_data_samples: List,
             **kwargs) -> dict:
        preds = self(feats)
        batch_gt_instances = [
            sample.gt_instances_3d for sample in batch_data_samples
        ]
        return self.loss_by_feat(preds, batch_gt_instances)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances_3d: InstanceData) -> Tuple[Tensor, ...]:
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
            # Only supervise the dims the GT actually carries; padding the rest
            # with zero would otherwise pull velocity (and any other absent
            # channel) toward zero under L1.
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
        # For pi-symmetric classes, theta and theta + pi are equivalent. Flip
        # the GT (sin, cos) target to whichever branch is closer to the
        # current prediction so we don't fight ambiguous head/tail labels.
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

        loss_iou = None
        if self.loss_iou is not None:
            # Axis-aligned BEV GIoU on decoded boxes. Rotated 3D IoU is not
            # differentiable in mmdet3d, so we fall back to the tight BEV
            # bbox [x1, y1, x2, y2] around (cx, cy, dx, dy) — this still
            # correlates strongly with rotated IoU because port actors have
            # moderate aspect ratios.
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
                torch.nan_to_num(loss_iou) if loss_iou is not None else None)

    def loss_by_feat(self, preds: dict,
                     batch_gt_instances: List[InstanceData]) -> dict:
        all_cls_scores = preds['all_cls_scores']
        all_bbox_preds = preds['all_bbox_preds']
        losses_cls = []
        losses_bbox = []
        losses_iou = []
        for cls_scores, bbox_preds in zip(all_cls_scores, all_bbox_preds):
            loss_cls, loss_bbox, loss_iou = self.loss_by_feat_single(
                cls_scores, bbox_preds, batch_gt_instances)
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)
            losses_iou.append(loss_iou)

        loss_dict = dict(
            loss_cls=losses_cls[-1],
            loss_bbox=losses_bbox[-1],
        )
        if losses_iou[-1] is not None:
            loss_dict['loss_iou'] = losses_iou[-1]
        for i in range(len(losses_cls) - 1):
            loss_dict[f'd{i}.loss_cls'] = losses_cls[i]
            loss_dict[f'd{i}.loss_bbox'] = losses_bbox[i]
            if losses_iou[i] is not None:
                loss_dict[f'd{i}.loss_iou'] = losses_iou[i]
        return loss_dict

    def predict(self, feats: List[Tensor], batch_data_samples: List,
                rescale: bool = False, **kwargs) -> List[InstanceData]:
        preds = self(feats)
        if len(batch_data_samples) > 0 and hasattr(batch_data_samples[0],
                                                   'metainfo'):
            batch_metas = [sample.metainfo for sample in batch_data_samples]
        else:
            batch_metas = batch_data_samples
        return self.predict_by_feat(preds, batch_metas)

    def predict_by_feat(self, preds: dict,
                        batch_metas: List[dict]) -> List[InstanceData]:
        cls_scores = preds['all_cls_scores'][-1].sigmoid()
        bbox_preds = preds['all_bbox_preds'][-1]
        max_num = int(self.test_cfg.get('max_num', self.num_query))
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
