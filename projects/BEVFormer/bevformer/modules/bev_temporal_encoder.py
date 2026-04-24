"""LiDAR BEV temporal encoder for Stage 3."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from .temporal_self_attention import TemporalSelfAttention


def warp_prev_bev(prev_bev: torch.Tensor,
                  ego_motion_delta: Sequence,
                  point_cloud_range: Sequence[float]) -> torch.Tensor:
    """Warp ``prev_bev`` from the previous ego frame into the current one."""
    if prev_bev is None:
        return None

    batch_size, _, bev_h, bev_w = prev_bev.shape
    if isinstance(ego_motion_delta, torch.Tensor):
        delta = ego_motion_delta.to(device=prev_bev.device, dtype=prev_bev.dtype)
    else:
        delta = np.asarray(ego_motion_delta, dtype=np.float32)
        delta = torch.as_tensor(
            delta, device=prev_bev.device, dtype=prev_bev.dtype)
    if delta.ndim == 2:
        delta = delta.unsqueeze(0)
    if delta.shape[0] != batch_size:
        raise ValueError('ego_motion_delta batch size mismatch: '
                         f'expected {batch_size}, got {delta.shape[0]}.')

    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    x_max, y_max = point_cloud_range[3], point_cloud_range[4]

    xs = torch.linspace(
        x_min + (x_max - x_min) / (2 * bev_w),
        x_max - (x_max - x_min) / (2 * bev_w),
        bev_w,
        device=prev_bev.device,
        dtype=prev_bev.dtype)
    ys = torch.linspace(
        y_min + (y_max - y_min) / (2 * bev_h),
        y_max - (y_max - y_min) / (2 * bev_h),
        bev_h,
        device=prev_bev.device,
        dtype=prev_bev.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    zeros = torch.zeros_like(grid_x)
    ones = torch.ones_like(grid_x)
    coords = torch.stack([grid_x, grid_y, zeros, ones], dim=0)
    coords = coords.reshape(4, bev_h * bev_w).unsqueeze(0).repeat(
        batch_size, 1, 1)

    source_coords = torch.bmm(torch.inverse(delta), coords)
    src_x = source_coords[:, 0].reshape(batch_size, bev_h, bev_w)
    src_y = source_coords[:, 1].reshape(batch_size, bev_h, bev_w)

    norm_x = 2 * (src_x - x_min) / max(x_max - x_min, 1e-6) - 1
    norm_y = 2 * (src_y - y_min) / max(y_max - y_min, 1e-6) - 1
    sample_grid = torch.stack([norm_x, norm_y], dim=-1)

    return torch.nn.functional.grid_sample(
        prev_bev,
        sample_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)


class _BEVFeedForward(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 hidden_dims: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dims, embed_dims)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class _BEVTemporalEncoderLayer(BaseModule):

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 num_points: int,
                 ffn_channels: int,
                 dropout: float = 0.1,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.temporal_attn = TemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = _BEVFeedForward(embed_dims, ffn_channels, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                prev_bev: torch.Tensor = None) -> torch.Tensor:
        if prev_bev is not None:
            value = torch.stack([prev_bev, query], dim=1).reshape(
                query.size(0) * 2, query.size(1), query.size(2))
        else:
            value = None

        hybrid_ref = torch.stack([reference_points, reference_points], dim=1)
        hybrid_ref = hybrid_ref.reshape(query.size(0) * 2, query.size(1), 1, 2)

        query = self.temporal_attn(
            query,
            value=value,
            identity=query,
            reference_points=hybrid_ref,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = self.norm1(query)
        query = self.norm2(query + self.ffn(query))
        return query


@MODELS.register_module()
class BEVTemporalEncoder(BaseModule):
    """Stacked LiDAR BEV temporal encoder."""

    def __init__(self,
                 embed_dims: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 num_points: int = 4,
                 ffn_channels: int = 1024,
                 dropout: float = 0.1,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList([
            _BEVTemporalEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_points=num_points,
                ffn_channels=ffn_channels,
                dropout=dropout)
            for _ in range(num_layers)
        ])

    @staticmethod
    def _get_reference_points(bev_h: int, bev_w: int, batch_size: int,
                              device: torch.device,
                              dtype: torch.dtype) -> torch.Tensor:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, bev_h - 0.5, bev_h, device=device, dtype=dtype),
            torch.linspace(0.5, bev_w - 0.5, bev_w, device=device, dtype=dtype),
            indexing='ij')
        ref_y = ref_y.reshape(-1)[None] / bev_h
        ref_x = ref_x.reshape(-1)[None] / bev_w
        return torch.stack((ref_x, ref_y), dim=-1).repeat(batch_size, 1,
                                                           1)

    def forward(self,
                current_bev: torch.Tensor,
                prev_bev: torch.Tensor = None) -> torch.Tensor:
        batch_size, channels, bev_h, bev_w = current_bev.shape
        if channels != self.embed_dims:
            raise ValueError('current_bev channel mismatch: '
                             f'expected {self.embed_dims}, got {channels}.')
        if prev_bev is not None and prev_bev.shape != current_bev.shape:
            raise ValueError('prev_bev shape mismatch: '
                             f'{prev_bev.shape} vs {current_bev.shape}.')

        query = current_bev.flatten(2).transpose(1, 2).contiguous()
        prev_flat = None
        if prev_bev is not None:
            prev_flat = prev_bev.flatten(2).transpose(1, 2).contiguous()

        ref_points = self._get_reference_points(
            bev_h, bev_w, batch_size, current_bev.device, current_bev.dtype)
        spatial_shapes = current_bev.new_tensor([[bev_h, bev_w]], dtype=torch.long)
        level_start_index = current_bev.new_tensor([0], dtype=torch.long)

        for layer in self.layers:
            query = layer(
                query,
                reference_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_flat)

        return query.transpose(1, 2).reshape(batch_size, channels, bev_h, bev_w)
