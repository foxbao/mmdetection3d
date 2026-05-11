"""Lightweight BEV occupancy head for box-derived KL supervision."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData


@MODELS.register_module()
class BEVOccHead2D(BaseModule):
    """Predict an ``[X, Y, Z]`` occupancy grid from BEV features.

    The first bootstrap target is generated from 3D boxes, so this head keeps
    the model deliberately simple: a 2D BEV convolution predicts
    ``num_classes * num_z`` channels, then reshapes them into per-voxel class
    logits.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_classes: int = 16,
                 num_z: int = 10,
                 bev_feature_layout: str = 'xy',
                 empty_idx: int = 0,
                 empty_weight: float = 0.05,
                 class_weight: Optional[Sequence[float]] = None,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        if bev_feature_layout not in ('xy', 'yx'):
            raise ValueError('bev_feature_layout must be "xy" or "yx", '
                             f'got {bev_feature_layout}.')
        self.num_classes = int(num_classes)
        self.num_z = int(num_z)
        self.bev_feature_layout = bev_feature_layout
        self.empty_idx = int(empty_idx)
        self.ignore_index = int(ignore_index)
        self.loss_weight = float(loss_weight)

        self.conv = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.pred = nn.Conv2d(
            hidden_channels, self.num_classes * self.num_z, kernel_size=1)

        if class_weight is None:
            weight = torch.ones(self.num_classes, dtype=torch.float32)
            if 0 <= self.empty_idx < self.num_classes:
                weight[self.empty_idx] = float(empty_weight)
        else:
            if len(class_weight) != self.num_classes:
                raise ValueError(
                    f'class_weight len={len(class_weight)} does not match '
                    f'num_classes={self.num_classes}.')
            weight = torch.as_tensor(class_weight, dtype=torch.float32)
        # Loss weights are configuration, not learned state. Keep them out of
        # checkpoints so changing a config takes effect when fine-tuning.
        self.register_buffer('class_weight', weight, persistent=False)

    def forward(self, bev_feat: Tensor) -> Tensor:
        """Return logits in canonical ``[B, C, X, Y, Z]`` layout."""
        logits_2d = self.pred(self.conv(bev_feat))
        B, _, H, W = logits_2d.shape
        logits = logits_2d.view(
            B, self.num_classes, self.num_z, H, W)
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        if self.bev_feature_layout == 'yx':
            logits = logits.transpose(2, 3).contiguous()
        return logits

    @staticmethod
    def _stack_gt_occ(batch_data_samples: List, device) -> Tensor:
        targets = []
        for sample in batch_data_samples:
            if 'gt_pts_seg' not in sample or 'occ' not in sample.gt_pts_seg:
                raise KeyError('BEVOccHead2D expects `data_sample.gt_pts_seg.'
                               'occ` in training.')
            targets.append(torch.as_tensor(
                sample.gt_pts_seg.occ, device=device, dtype=torch.long))
        return torch.stack(targets, dim=0)

    def loss(self, bev_feat: Tensor, batch_data_samples: List) -> dict:
        logits = self.forward(bev_feat)
        targets = self._stack_gt_occ(batch_data_samples, logits.device)
        if targets.shape != logits.shape[0:1] + logits.shape[2:]:
            raise ValueError('BEVOccHead2D target shape mismatch: '
                             f'logits={tuple(logits.shape)}, '
                             f'targets={tuple(targets.shape)}')

        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weight.to(device=logits.device,
                                        dtype=logits.dtype),
            ignore_index=self.ignore_index)
        return dict(loss_occ_ce=loss * self.loss_weight)

    def predict(self, bev_feat: Tensor, batch_data_samples: List):
        logits = self.forward(bev_feat)
        preds = logits.argmax(dim=1).to(torch.uint8)
        for data_sample, pred in zip(batch_data_samples, preds):
            if 'pred_pts_seg' not in data_sample:
                data_sample.pred_pts_seg = PointData()
            data_sample.pred_pts_seg.occ = pred.detach()
        return batch_data_samples
