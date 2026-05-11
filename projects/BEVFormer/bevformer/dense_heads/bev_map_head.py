"""Lightweight BEV map head for rasterized local-map supervision."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from torch import Tensor, nn


@MODELS.register_module()
class BEVMapHead(BaseModule):
    """Predict a binary raster map from fused BEV features."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, bev_feat: Tensor) -> Tensor:
        x = self.conv1(bev_feat)
        return self.conv2(x)

    @staticmethod
    def _stack_gt_masks(batch_data_samples: List, device, dtype) -> Tensor:
        masks = []
        for sample in batch_data_samples:
            if 'gt_pts_seg' not in sample or 'seg_map' not in sample.gt_pts_seg:
                raise KeyError('BEVMapHead expects `data_sample.gt_pts_seg.'
                               'seg_map` in training.')
            seg_map = sample.gt_pts_seg.seg_map
            mask = torch.as_tensor(seg_map, device=device)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            masks.append(mask.to(dtype=dtype))
        return torch.stack(masks, dim=0)

    def loss(self, bev_feat: Tensor, batch_data_samples: List) -> dict:
        logits = self.forward(bev_feat)
        targets = self._stack_gt_masks(batch_data_samples, logits.device,
                                       logits.dtype)
        if targets.shape != logits.shape:
            raise ValueError('BEVMapHead target shape mismatch: '
                             f'logits={tuple(logits.shape)}, '
                             f'targets={tuple(targets.shape)}')
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return dict(loss_map=loss * self.loss_weight)

    def predict(self, bev_feat: Tensor, batch_data_samples: List):
        logits = self.forward(bev_feat)
        probs = torch.sigmoid(logits)
        pred_masks = probs > 0.5
        for data_sample, prob, mask in zip(batch_data_samples, probs, pred_masks):
            data_sample.pred_pts_seg = PointData(
                seg_map=prob.detach(),
                seg_map_mask=mask.detach())
        return batch_data_samples
