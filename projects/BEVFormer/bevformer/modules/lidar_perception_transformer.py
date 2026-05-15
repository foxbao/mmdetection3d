"""LiDAR-only PerceptionTransformer wrapper for UniAD-style BEV encoding."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


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
