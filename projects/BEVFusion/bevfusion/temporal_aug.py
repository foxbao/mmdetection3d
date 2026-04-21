"""Sync current-frame global augmentation onto temporal (adj) fields.

`BEVFusionGlobalRotScaleTrans` and `BEVFusionRandomFlip3D` accumulate the
cumulative 4x4 LiDAR-frame augmentation into ``results['lidar_aug_matrix']``.
This transform re-applies the same matrix to every historical point cloud in
``adj_points`` and conjugates every ``adj_ego_motions`` entry so that
``TemporalBEVFuser.warp_bev`` keeps aligning history to the augmented current
frame.

Math: with ``A`` the LiDAR-frame augmentation and
``M_lidar = ego2lidar @ ego_motion @ lidar2ego`` the per-frame hist->curr
mapping in LiDAR space, the augmented version is ``A @ M_lidar @ A^-1``.
Because the fuser always applies the same ``ego2lidar``/``lidar2ego``
conjugation internally, writing ``A_ego @ ego_motion @ A_ego^-1`` into
``adj_ego_motions`` (with ``A_ego = lidar2ego @ A @ ego2lidar``) yields the
desired effect.
"""
from typing import Any, Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SyncTemporalAug(BaseTransform):
    """Apply the accumulated ``lidar_aug_matrix`` to adj_points / adj_ego_motions.

    Must run AFTER ``BEVFusionGlobalRotScaleTrans`` and
    ``BEVFusionRandomFlip3D`` (which build ``lidar_aug_matrix``) and BEFORE
    ``PointsRangeFilter`` (so out-of-range augmented points are pruned).

    Args:
        default_lidar_coord_frame (str): Fallback ``lidar_coord_frame`` when
            ``results`` lacks the metainfo key. One of {'FLU', 'RFU'}.
    """

    def __init__(self, default_lidar_coord_frame: str = 'FLU') -> None:
        if default_lidar_coord_frame not in ('FLU', 'RFU'):
            raise ValueError(
                f'default_lidar_coord_frame must be FLU or RFU, '
                f'got {default_lidar_coord_frame!r}')
        self.default_frame = default_lidar_coord_frame

    @staticmethod
    def _lidar2ego(frame: str) -> np.ndarray:
        if frame == 'RFU':
            # LiDAR RFU -> ego FLU: col0 Right, col1 Forward, col2 Up
            return np.array(
                [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float64)
        return np.eye(4, dtype=np.float64)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'adj_points' not in results or 'adj_ego_motions' not in results:
            return results

        A = results.get('lidar_aug_matrix', None)
        if A is None:
            return results
        A = np.asarray(A, dtype=np.float64)
        if A.shape != (4, 4) or np.allclose(A, np.eye(4)):
            return results

        frame = results.get('lidar_coord_frame', self.default_frame)
        lidar2ego = self._lidar2ego(frame)
        ego2lidar = np.linalg.inv(lidar2ego)
        A_ego = lidar2ego @ A @ ego2lidar
        A_ego_inv = np.linalg.inv(A_ego)

        A_t = torch.from_numpy(A.astype(np.float32))
        A_ego_t = torch.from_numpy(A_ego.astype(np.float32))
        A_ego_inv_t = torch.from_numpy(A_ego_inv.astype(np.float32))

        new_adj_points = []
        for pts in results['adj_points']:
            if pts is None:
                new_adj_points.append(None)
                continue
            pts = pts.clone()
            xyz = pts[:, :3]
            ones = torch.ones_like(xyz[:, :1])
            xyz_h = torch.cat([xyz, ones], dim=1)
            pts[:, :3] = (xyz_h @ A_t.T)[:, :3]
            new_adj_points.append(pts)
        results['adj_points'] = new_adj_points

        new_motions = []
        for M in results['adj_ego_motions']:
            if not torch.is_tensor(M):
                M = torch.as_tensor(M, dtype=torch.float32)
            else:
                M = M.to(torch.float32)
            new_motions.append(A_ego_t @ M @ A_ego_inv_t)
        results['adj_ego_motions'] = new_motions

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'default_lidar_coord_frame={self.default_frame!r})')
