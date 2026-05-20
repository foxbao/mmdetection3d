"""LiDAR-only PerceptionTransformer wrapper for UniAD-style BEV encoding."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet3d.registry import MODELS


def _inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class _BEVFormerDetrDecoderLayer(BaseModule):
    """DETR decoder layer with deformable BEV cross-attention."""

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
        q = k = query + query_pos
        query2 = self.self_attn(q, k, value=query)[0]
        query = self.norm1(query + self.dropout1(query2))

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


class _BEVFormerDetectionDecoder(BaseModule):
    """Small BEVFormer decoder wrapper mirroring UniAD's transformer boundary.

    The config parser accepts both a compact LiDAR-native decoder dict and the
    UniAD-style ``DetectionTransformerDecoder`` / ``transformerlayers`` shape.
    """

    def __init__(self,
                 num_layers: int = 6,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_points: int = 4,
                 num_levels: int = 1,
                 ffn_channels: int = 1024,
                 dropout: float = 0.1,
                 return_intermediate: bool = True,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_layers = int(num_layers)
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList([
            _BEVFormerDetrDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_points=num_points,
                num_levels=num_levels,
                ffn_channels=ffn_channels,
                dropout=dropout)
            for _ in range(self.num_layers)
        ])

    @classmethod
    def from_config(cls, decoder: dict, embed_dims: int):
        cfg = dict(decoder)
        cfg.pop('type', None)
        layer_cfg = cfg.pop('transformerlayers', None)
        num_layers = cfg.pop('num_layers', 6)
        return_intermediate = cfg.pop('return_intermediate', True)

        num_heads = cfg.pop('num_heads', 8)
        num_points = cfg.pop('num_points', 4)
        num_levels = cfg.pop('num_levels', 1)
        ffn_channels = cfg.pop('ffn_channels', None)
        dropout = cfg.pop('dropout', None)

        if layer_cfg is not None:
            layer_cfg = dict(layer_cfg)
            ffn_channels = layer_cfg.get(
                'feedforward_channels',
                ffn_channels if ffn_channels is not None else embed_dims * 2)
            dropout = layer_cfg.get(
                'ffn_dropout', dropout if dropout is not None else 0.1)
            attn_cfgs = layer_cfg.get('attn_cfgs', [])
            if len(attn_cfgs) > 0:
                self_attn_cfg = dict(attn_cfgs[0])
                num_heads = self_attn_cfg.get('num_heads', num_heads)
            if len(attn_cfgs) > 1:
                cross_attn_cfg = dict(attn_cfgs[1])
                num_points = cross_attn_cfg.get('num_points', num_points)
                num_levels = cross_attn_cfg.get('num_levels', num_levels)
        else:
            ffn_channels = (
                ffn_channels if ffn_channels is not None else embed_dims * 2)
            dropout = dropout if dropout is not None else 0.1

        return cls(
            num_layers=num_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
            num_levels=num_levels,
            ffn_channels=ffn_channels,
            dropout=dropout,
            return_intermediate=return_intermediate)

    def forward(self,
                query: Tensor,
                value: Tensor,
                query_pos: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                reg_branches: Optional[nn.ModuleList] = None):
        output = query
        reference = reference_points
        intermediate = []
        intermediate_refs = []

        for layer_id, layer in enumerate(self.layers):
            # BEV memory is [B, C, H=Y, W=X]; MSDA reference order is
            # (W_frac, H_frac) = (cx_norm, cy_norm).
            msda_ref = reference[..., :2].unsqueeze(2).contiguous()
            output = layer(output, query_pos, value, msda_ref, spatial_shapes,
                           level_start_index)

            if reg_branches is not None:
                reg_raw = reg_branches[layer_id](
                    output.permute(1, 0, 2).contiguous())
                reference_logit = _inverse_sigmoid(reference)
                refined_xy = reg_raw[..., 0:2] + reference_logit[..., 0:2]
                refined_z = reg_raw[..., 4:5] + reference_logit[..., 2:3]
                reference = torch.cat(
                    [refined_xy, refined_z], dim=-1).sigmoid().detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_refs.append(reference)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_refs)
        return output.unsqueeze(0), reference.unsqueeze(0)


@MODELS.register_module()
class PerceptionTransformer(BaseModule):
    """Thin UniAD-style wrapper around the LiDAR BEV encoder.

    UniAD routes BEV query construction through ``PerceptionTransformer``
    before the encoder. This LiDAR version keeps the same structural boundary
    and can pass ego-motion shifts into temporal self-attention, matching
    UniAD's BEV encoder alignment more closely than whole-feature external
    warping.
    """

    def __init__(self,
                 encoder: dict,
                 embed_dims: int,
                 bev_h: int,
                 bev_w: int,
                 decoder: Optional[dict] = None,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 use_shift: bool = False,
                 rotate_prev_bev: bool = False,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.point_cloud_range = point_cloud_range
        self.use_shift = use_shift
        self.rotate_prev_bev = rotate_prev_bev
        encoder = dict(encoder)
        encoder.setdefault('embed_dims', embed_dims)
        encoder.setdefault('bev_h', bev_h)
        encoder.setdefault('bev_w', bev_w)
        self.encoder = MODELS.build(encoder)
        self.decoder = (
            _BEVFormerDetectionDecoder.from_config(decoder, embed_dims)
            if decoder is not None else None)

    def shift_from_queue_meta(self,
                              queue_meta: Optional[Sequence[dict]],
                              device: torch.device,
                              dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Build UniAD-style shifted reference offsets for prev BEV sampling.

        ``ego_motion_delta`` maps previous-frame ego coordinates into the
        current ego frame. If ``prev_bev`` is pre-rotated, translation remains
        as a normalized reference-point offset. Otherwise the full inverse
        translation is used for legacy compatibility.
        """
        if (not self.use_shift or queue_meta is None or
                self.point_cloud_range is None):
            return None
        x_extent = max(float(self.point_cloud_range[3] -
                             self.point_cloud_range[0]), 1e-6)
        y_extent = max(float(self.point_cloud_range[4] -
                             self.point_cloud_range[1]), 1e-6)
        extent = torch.tensor([x_extent, y_extent], device=device, dtype=dtype)
        shifts = []
        for meta in queue_meta:
            if meta is None or not meta.get('prev_bev_exists', False):
                shifts.append(torch.zeros(2, device=device, dtype=dtype))
                continue
            delta = torch.as_tensor(
                meta['ego_motion_delta'], device=device, dtype=torch.float32)
            if delta.shape != (4, 4):
                raise ValueError('ego_motion_delta must have shape (4, 4), '
                                 f'got {tuple(delta.shape)}.')
            if self.rotate_prev_bev:
                shifts.append(-delta[:2, 3].to(dtype=dtype) / extent)
            else:
                prev_from_curr = torch.inverse(delta)
                shifts.append(prev_from_curr[:2, 3].to(dtype=dtype) / extent)
        return torch.stack(shifts, dim=0)

    def rotate_prev_bev_if_needed(
            self, prev_bev: Optional[torch.Tensor],
            queue_meta: Optional[Sequence[dict]]) -> Optional[torch.Tensor]:
        """Rotate previous BEV features into the current ego orientation.

        Translation is intentionally not applied here. UniAD-style temporal
        self-attention handles translation with shifted reference points.
        """
        if (prev_bev is None or not self.rotate_prev_bev or
                queue_meta is None or self.point_cloud_range is None):
            return prev_bev

        batch_size, _, bev_h, bev_w = prev_bev.shape
        if len(queue_meta) != batch_size:
            raise ValueError('queue_meta batch size mismatch: expected '
                             f'{batch_size}, got {len(queue_meta)}.')

        rotations = []
        for meta in queue_meta:
            if meta is None or not meta.get('prev_bev_exists', False):
                rotations.append(
                    torch.eye(
                        2, device=prev_bev.device, dtype=prev_bev.dtype))
                continue
            delta = torch.as_tensor(
                meta['ego_motion_delta'],
                device=prev_bev.device,
                dtype=torch.float32)
            if delta.shape != (4, 4):
                raise ValueError('ego_motion_delta must have shape (4, 4), '
                                 f'got {tuple(delta.shape)}.')
            prev_from_curr = torch.inverse(delta)
            rotations.append(prev_from_curr[:2, :2].to(dtype=prev_bev.dtype))
        rotations = torch.stack(rotations, dim=0)

        x_min = float(self.point_cloud_range[0])
        y_min = float(self.point_cloud_range[1])
        x_max = float(self.point_cloud_range[3])
        y_max = float(self.point_cloud_range[4])
        x_extent = max(x_max - x_min, 1e-6)
        y_extent = max(y_max - y_min, 1e-6)

        xs = torch.linspace(
            x_min + x_extent / (2 * bev_w),
            x_max - x_extent / (2 * bev_w),
            bev_w,
            device=prev_bev.device,
            dtype=prev_bev.dtype)
        ys = torch.linspace(
            y_min + y_extent / (2 * bev_h),
            y_max - y_extent / (2 * bev_h),
            bev_h,
            device=prev_bev.device,
            dtype=prev_bev.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)

        source_coords = torch.bmm(rotations, coords)
        src_x = source_coords[:, 0].reshape(batch_size, bev_h, bev_w)
        src_y = source_coords[:, 1].reshape(batch_size, bev_h, bev_w)

        norm_x = 2 * (src_x - x_min) / x_extent - 1
        norm_y = 2 * (src_y - y_min) / y_extent - 1
        sample_grid = torch.stack([norm_x, norm_y], dim=-1)

        return F.grid_sample(
            prev_bev,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)

    def get_bev_features(self,
                         lidar_bev: torch.Tensor,
                         bev_queries: torch.Tensor,
                         bev_pos: torch.Tensor,
                         prev_bev: Optional[torch.Tensor] = None,
                         queue_meta: Optional[Sequence[dict]] = None
                         ) -> torch.Tensor:
        """Encode LiDAR BEV memory with learned BEV queries."""
        prev_bev = self.rotate_prev_bev_if_needed(prev_bev, queue_meta)
        shift = self.shift_from_queue_meta(
            queue_meta, lidar_bev.device, lidar_bev.dtype)
        return self.encoder(
            lidar_bev,
            bev_queries=bev_queries,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            shift=shift)

    def get_states_and_refs(self,
                            bev_embed: torch.Tensor,
                            object_query_embed: torch.Tensor,
                            bev_h: int,
                            bev_w: int,
                            reference_points: torch.Tensor,
                            reg_branches: Optional[nn.ModuleList] = None):
        """Run the detection decoder over encoded BEV memory.

        Args:
            bev_embed: Flattened BEV memory in shape ``[B, H*W, C]``.
            object_query_embed: Query tensor in shape ``[N, 2*C]``.
            reference_points: Logit-space reference points in shape
                ``[B, N, 3]`` or ``[N, 3]``.
        """
        if self.decoder is None:
            raise ValueError('PerceptionTransformer.decoder is required for '
                             'detection states.')
        if bev_embed.dim() != 3:
            raise ValueError('bev_embed must have shape [B, H*W, C], got '
                             f'{tuple(bev_embed.shape)}.')

        batch_size = bev_embed.shape[0]
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos[:, None, :].expand(-1, batch_size, -1)
        query = query[:, None, :].expand(-1, batch_size, -1)

        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).expand(
                batch_size, -1, -1)
        elif reference_points.dim() == 3:
            if reference_points.size(0) == 1 and batch_size != 1:
                reference_points = reference_points.expand(
                    batch_size, -1, -1)
            elif reference_points.size(0) != batch_size:
                raise ValueError('reference_points batch size mismatch: '
                                 f'expected {batch_size}, got '
                                 f'{reference_points.size(0)}.')
        else:
            raise ValueError('reference_points must have shape [N, 3] or '
                             f'[B, N, 3], got '
                             f'{tuple(reference_points.shape)}.')
        init_reference = reference_points.sigmoid()

        spatial_shapes = torch.as_tensor(
            [[bev_h, bev_w]], dtype=torch.long, device=bev_embed.device)
        level_start_index = torch.as_tensor(
            [0], dtype=torch.long, device=bev_embed.device)
        inter_states, inter_references = self.decoder(
            query=query,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=init_reference,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=reg_branches)
        return inter_states, init_reference, inter_references
