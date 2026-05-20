"""UniAD-like query-driven BEV encoder for LiDAR BEV features."""

from __future__ import annotations

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from .lidar_spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention


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


class LearnedBEVPositionalEncoding(nn.Module):
    """Learned row/column BEV position encoding, matching BEVFormer style."""

    def __init__(self, bev_h: int, bev_w: int, embed_dims: int) -> None:
        super().__init__()
        if embed_dims % 2 != 0:
            raise ValueError('embed_dims must be even for learned 2D BEV '
                             f'position encoding, got {embed_dims}.')
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.row_embed = nn.Embedding(bev_h, embed_dims // 2)
        self.col_embed = nn.Embedding(bev_w, embed_dims // 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, batch_size: int, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
        rows = torch.arange(self.bev_h, device=device)
        cols = torch.arange(self.bev_w, device=device)
        row_pos = self.row_embed(rows).to(dtype)
        col_pos = self.col_embed(cols).to(dtype)
        pos = torch.cat([
            col_pos.unsqueeze(0).expand(self.bev_h, -1, -1),
            row_pos.unsqueeze(1).expand(-1, self.bev_w, -1)
        ],
                        dim=-1)
        return pos.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1)


class BEVFormerLayer(BaseModule):

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 temporal_num_points: int,
                 spatial_num_points: int,
                 ffn_channels: int,
                 dropout: float = 0.1,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.temporal_attn = TemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=temporal_num_points,
            dropout=dropout,
            batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.spatial_attn = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=spatial_num_points,
            dropout=dropout,
            batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = _BEVFeedForward(embed_dims, ffn_channels, dropout=dropout)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self,
                query: torch.Tensor,
                lidar_value: torch.Tensor,
                bev_pos: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                prev_bev: torch.Tensor = None,
                shift: torch.Tensor = None) -> torch.Tensor:
        if prev_bev is not None:
            temporal_value = torch.stack([prev_bev, query], dim=1).reshape(
                query.size(0) * 2, query.size(1), query.size(2))
        else:
            temporal_value = None

        shift_ref_2d = reference_points
        if prev_bev is not None and shift is not None:
            shift_ref_2d = reference_points + shift[:, None, :]
        hybrid_ref_2d = torch.stack(
            [shift_ref_2d, reference_points], dim=1)
        hybrid_ref_2d = hybrid_ref_2d.reshape(query.size(0) * 2,
                                              query.size(1), 1, 2)
        query = self.temporal_attn(
            query,
            value=temporal_value,
            identity=query,
            query_pos=bev_pos,
            reference_points=hybrid_ref_2d,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = self.norm1(query)

        query = self.spatial_attn(
            query,
            value=lidar_value,
            identity=query,
            query_pos=bev_pos,
            reference_points=reference_points.unsqueeze(2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = self.norm2(query)

        query = self.norm3(query + self.ffn(query))
        return query


@MODELS.register_module()
class BEVFormerEncoder(BaseModule):
    """LiDAR BEV encoder layers with temporal and spatial attention.

    The learned BEV query and positional encoding live in the head, matching
    UniAD-style ``BEVFormerLiDARHead.get_bev_features`` boundary. This
    module only consumes those tensors and applies the BEVFormer-style
    encoder layers.
    """

    def __init__(self,
                 embed_dims: int,
                 bev_h: int,
                 bev_w: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 temporal_num_points: int = 4,
                 spatial_num_points: int = 4,
                 ffn_channels: int = 1024,
                 dropout: float = 0.1,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.layers = nn.ModuleList([
            BEVFormerLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                temporal_num_points=temporal_num_points,
                spatial_num_points=spatial_num_points,
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
        return torch.stack((ref_x, ref_y), dim=-1).repeat(batch_size, 1, 1)

    def forward(self,
                lidar_bev: torch.Tensor,
                bev_queries: torch.Tensor,
                bev_pos: torch.Tensor,
                prev_bev: torch.Tensor = None,
                shift: torch.Tensor = None) -> torch.Tensor:
        batch_size, channels, bev_h, bev_w = lidar_bev.shape
        if channels != self.embed_dims:
            raise ValueError('lidar_bev channel mismatch: '
                             f'expected {self.embed_dims}, got {channels}.')
        if (bev_h, bev_w) != (self.bev_h, self.bev_w):
            raise ValueError('lidar_bev spatial shape mismatch: expected '
                             f'{(self.bev_h, self.bev_w)}, got '
                             f'{(bev_h, bev_w)}.')
        if prev_bev is not None and prev_bev.shape != lidar_bev.shape:
            raise ValueError('prev_bev shape mismatch: '
                             f'{prev_bev.shape} vs {lidar_bev.shape}.')
        if shift is not None and shift.shape != (batch_size, 2):
            raise ValueError('shift shape mismatch: expected '
                             f'{(batch_size, 2)}, got '
                             f'{tuple(shift.shape)}.')
        if bev_queries.shape != (bev_h * bev_w, self.embed_dims):
            raise ValueError('bev_queries shape mismatch: expected '
                             f'{(bev_h * bev_w, self.embed_dims)}, got '
                             f'{tuple(bev_queries.shape)}.')
        if bev_pos.shape != (batch_size, bev_h * bev_w, self.embed_dims):
            raise ValueError('bev_pos shape mismatch: expected '
                             f'{(batch_size, bev_h * bev_w, self.embed_dims)}, '
                             f'got {tuple(bev_pos.shape)}.')

        lidar_value = lidar_bev.flatten(2).transpose(1, 2).contiguous()
        query = bev_queries.to(
            dtype=lidar_bev.dtype, device=lidar_bev.device)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        bev_pos = bev_pos.to(dtype=lidar_bev.dtype, device=lidar_bev.device)
        if shift is not None:
            shift = shift.to(
                dtype=lidar_bev.dtype, device=lidar_bev.device)

        prev_flat = None
        if prev_bev is not None:
            prev_flat = prev_bev.flatten(2).transpose(1, 2).contiguous()

        ref_points = self._get_reference_points(
            bev_h, bev_w, batch_size, lidar_bev.device, lidar_bev.dtype)
        spatial_shapes = lidar_bev.new_tensor([[bev_h, bev_w]],
                                              dtype=torch.long)
        level_start_index = lidar_bev.new_tensor([0], dtype=torch.long)

        for layer in self.layers:
            query = layer(
                query,
                lidar_value=lidar_value,
                bev_pos=bev_pos,
                reference_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_flat,
                shift=shift)

        return query.transpose(1, 2).reshape(batch_size, channels, bev_h,
                                             bev_w)
