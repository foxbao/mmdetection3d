"""Temporal self-attention vendored from BEVFormer for single-level LiDAR BEV.

This is the BEVFormer TSA block with the minimum adaptation needed for the
current codebase:

* ``MODELS`` registry instead of the original mmcv attention registry.
* ``mmengine.model.BaseModule`` instead of the mmcv runner base module.
* Use mmcv's deformable-attention op directly; fall back to the PyTorch
  reference implementation on CPU.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmengine.model import BaseModule, constant_init, xavier_init

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TemporalSelfAttention(BaseModule):
    """BEVFormer temporal self-attention for a 2-slot BEV queue."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 1,
                 num_points: int = 4,
                 num_bev_queue: int = 2,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads, '
                             f'got {embed_dims} and {num_heads}.')

        dim_per_head = embed_dims // num_heads
        if dim_per_head & (dim_per_head - 1) != 0:
            warnings.warn(
                'Using a non power-of-two dim per head makes the deformable '
                'attention CUDA kernel less efficient.')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi /
                                                    self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(
            -1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(
                1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.reshape(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query: torch.Tensor,
                value: torch.Tensor = None,
                identity: torch.Tensor = None,
                query_pos: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                reference_points: torch.Tensor = None,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            if value is not None:
                value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape
        if value is None:
            value = torch.stack([query, query], 1).reshape(
                bs * self.num_bev_queue, num_query, embed_dims)

        _, num_value, _ = value.shape
        if spatial_shapes is None or reference_points is None:
            raise ValueError('TemporalSelfAttention requires spatial_shapes '
                             'and reference_points.')
        expected_num_value = int((spatial_shapes[:, 0] *
                                  spatial_shapes[:, 1]).sum().item())
        if expected_num_value != num_value:
            raise ValueError('reference shape mismatch: spatial_shapes imply '
                             f'{expected_num_value} values, got {num_value}.')
        if self.num_bev_queue != 2:
            raise ValueError('TemporalSelfAttention currently assumes a '
                             '2-slot BEV queue.')

        query = torch.cat([value[:bs], query], dim=-1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs * self.num_bev_queue, num_value,
                              self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels * self.num_points).softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)
        attention_weights = attention_weights.reshape(
            bs * self.num_bev_queue, num_query, self.num_heads,
            self.num_levels, self.num_points).contiguous()

        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)
        sampling_offsets = sampling_offsets.reshape(
            bs * self.num_bev_queue, num_query, self.num_heads,
            self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] != 2:
            raise ValueError('TemporalSelfAttention expects 2D reference '
                             f'points, got shape {reference_points.shape}.')

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :] +
            sampling_offsets / offset_normalizer[None, None, None, :, None, :])

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1).permute(2, 0, 1)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
