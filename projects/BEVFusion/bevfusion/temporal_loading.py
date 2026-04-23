# Copyright (c) OpenMMLab. All rights reserved.
"""Pipeline transforms for loading temporal LiDAR data."""

import numpy as np
import torch

from mmdet3d.datasets.transforms import LoadPointsFromFile
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadTemporalData(object):
    """Load historical frames' point clouds and compute ego-motion matrices.

    Reads ``results['adj_infos']`` (set by KlDataset when num_adj_frames > 0)
    and fills:

    - ``results['adj_points']``     : list of (N_i, 4) float32 arrays, one per
                                      historical frame (None if unavailable).
    - ``results['adj_ego_motions']``: list of (4, 4) float32 arrays.
                                      Each matrix transforms a point from the
                                      historical frame's ego coordinate into the
                                      *current* frame's ego coordinate.
                                      Identity when the historical frame is
                                      missing (scene boundary).

    If ``results`` has no ``adj_infos`` key (i.e. num_adj_frames == 0), this
    transform is a no-op so it is safe to include unconditionally in the
    pipeline.

    Args:
        load_dim (int): Dimension of loaded point cloud. Default: 5.
        use_dim (int or list[int]): Dimensions to keep after loading.
            Default: 4 (x, y, z, intensity).
        min_time_diff (float): Minimum allowed current-adjacent timestamp
            gap in seconds. Defaults to 0.0.
        max_time_diff (float): Maximum allowed current-adjacent timestamp
            gap in seconds. Defaults to 1.2.
        reject_identity_pose (bool): Whether to treat identity ego poses as
            invalid. Defaults to True.
        strict_temporal (bool): If True, return ``None`` (which signals
            mmengine to re-sample another index) whenever any adjacent
            slot in the queue is invalid — at the cost of dropping early
            frames in each scene. Use for train, where you want the
            temporal fuser to only see real history (UniAD does the
            same). Leave False for val/test so every sample is scored.
            Defaults to False.
    """

    def __init__(self,
                 load_dim: int = 5,
                 use_dim=4,
                 min_time_diff: float = 0.0,
                 max_time_diff: float = 1.2,
                 reject_identity_pose: bool = True,
                 strict_temporal: bool = False):
        self.load_dim = load_dim
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.reject_identity_pose = reject_identity_pose
        self.strict_temporal = strict_temporal

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_points(lidar_path: str, load_dim: int,
                     use_dim: list) -> np.ndarray:
        """Load a single .bin point cloud file."""
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, load_dim)
        return pts[:, use_dim]

    @staticmethod
    def _ego_motion(ego2global_curr: list,
                    ego2global_adj: list) -> np.ndarray:
        """Compute T such that  p_curr = T @ p_adj  (4×4 float32).

        T = inv(ego2global_curr) @ ego2global_adj
        """
        T_curr = np.array(ego2global_curr, dtype=np.float64)
        T_adj  = np.array(ego2global_adj,  dtype=np.float64)
        return (np.linalg.inv(T_curr) @ T_adj).astype(np.float32)

    def _valid_pose(self, ego2global) -> bool:
        T = np.array(ego2global, dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            return False
        if self.reject_identity_pose and np.allclose(T, np.eye(4)):
            return False
        return True

    def _valid_time_gap(self, curr_ts, adj_ts) -> bool:
        if curr_ts is None or adj_ts is None:
            return True
        dt = float(curr_ts) - float(adj_ts)
        return self.min_time_diff <= dt <= self.max_time_diff

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def __call__(self, results: dict) -> dict:
        adj_infos = results.get('adj_infos', None)
        if not adj_infos:
            # num_adj_frames == 0 or key missing — no-op
            return results

        ego2global_curr = results.get('ego2global', np.eye(4).tolist())
        curr_ts = results.get('timestamp', None)

        adj_points     = []
        adj_ego_motions = []

        curr_pose_valid = self._valid_pose(ego2global_curr)

        for adj in adj_infos:
            adj_pose = None if adj is None else adj.get('ego2global')
            adj_ts = None if adj is None else adj.get('timestamp', None)
            if (adj is None or not adj.get('lidar_path', '')
                    or not curr_pose_valid
                    or not self._valid_pose(adj_pose)
                    or not self._valid_time_gap(curr_ts, adj_ts)):
                if self.strict_temporal:
                    # UniAD-style: drop the whole sample, mmengine will
                    # _rand_another. Forces the fuser to only ever see
                    # real history during training.
                    return None
                # Scene boundary — fill with empty / identity
                adj_points.append(None)
                adj_ego_motions.append(torch.eye(4, dtype=torch.float32))
                continue

            pts = self._load_points(adj['lidar_path'],
                                    self.load_dim, self.use_dim)
            motion = self._ego_motion(ego2global_curr, adj['ego2global'])

            adj_points.append(torch.from_numpy(pts))
            adj_ego_motions.append(torch.from_numpy(motion))

        results['adj_points']      = adj_points
        results['adj_ego_motions'] = adj_ego_motions
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'load_dim={self.load_dim}, use_dim={self.use_dim}, '
                f'min_time_diff={self.min_time_diff}, '
                f'max_time_diff={self.max_time_diff})')


@TRANSFORMS.register_module()
class LoadPrevFramePoints(LoadPointsFromFile):
    """Load the immediate previous-frame point cloud for BEVFormer-style
    `prev_bev` training.

    Reads ``results['prev_info']`` (set by ``KlDataset(load_prev_frame=True)``)
    and fills:

    - ``results['prev_points']``: previous-frame points as ``BasePoints``,
      or ``None`` on cold start / invalid history.
    - ``results['prev_ego2global']``: previous-frame ego pose, or ``None``.
    - ``results['prev_bev_exists']``: whether valid temporal history exists.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim=4,
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 min_time_diff: float = 0.0,
                 max_time_diff: float = 1.2,
                 backend_args=None) -> None:
        super().__init__(
            coord_type=coord_type,
            load_dim=load_dim,
            use_dim=use_dim,
            shift_height=shift_height,
            use_color=use_color,
            norm_intensity=norm_intensity,
            norm_elongation=norm_elongation,
            backend_args=backend_args)
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff

    def _valid_time_gap(self, curr_ts, prev_ts) -> bool:
        if curr_ts is None or prev_ts is None:
            return True
        dt = float(curr_ts) - float(prev_ts)
        return self.min_time_diff <= dt <= self.max_time_diff

    def transform(self, results: dict) -> dict:
        prev_info = results.get('prev_info')
        curr_ts = results.get('timestamp', None)
        if (prev_info is None or not prev_info.get('lidar_path', '')
                or not self._valid_time_gap(curr_ts,
                                            prev_info.get('timestamp', None))):
            results['prev_points'] = None
            results['prev_ego2global'] = None
            results['prev_bev_exists'] = False
            return results

        prev_results = dict(
            lidar_points=dict(lidar_path=prev_info['lidar_path']))
        prev_results = super().transform(prev_results)
        results['prev_points'] = prev_results['points']
        results['prev_ego2global'] = prev_info.get('ego2global', None)
        results['prev_bev_exists'] = True
        return results


@TRANSFORMS.register_module()
class LoadPrevFrameQueuePoints(LoadPointsFromFile):
    """Load a short oldest-to-newest queue of previous-frame point clouds."""

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim=4,
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 min_time_diff: float = 0.0,
                 max_time_diff: float = 1.2,
                 backend_args=None) -> None:
        super().__init__(
            coord_type=coord_type,
            load_dim=load_dim,
            use_dim=use_dim,
            shift_height=shift_height,
            use_color=use_color,
            norm_intensity=norm_intensity,
            norm_elongation=norm_elongation,
            backend_args=backend_args)
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff

    def _valid_time_gap(self, curr_ts, prev_ts) -> bool:
        if curr_ts is None or prev_ts is None:
            return True
        dt = float(curr_ts) - float(prev_ts)
        return self.min_time_diff <= dt <= self.max_time_diff

    def transform(self, results: dict) -> dict:
        prev_infos = results.get('prev_infos', None)
        if not prev_infos:
            return results

        curr_ts = results.get('timestamp', None)
        prev_points_queue = []
        prev_ego2global_queue = []
        prev_bev_exists_queue = []

        for prev_info in prev_infos:
            if (prev_info is None or not prev_info.get('lidar_path', '')
                    or not self._valid_time_gap(
                        curr_ts, prev_info.get('timestamp', None))):
                prev_points_queue.append(None)
                prev_ego2global_queue.append(None)
                prev_bev_exists_queue.append(False)
                continue

            prev_results = dict(
                lidar_points=dict(lidar_path=prev_info['lidar_path']))
            prev_results = super().transform(prev_results)
            prev_points_queue.append(prev_results['points'])
            prev_ego2global_queue.append(prev_info.get('ego2global', None))
            prev_bev_exists_queue.append(True)

        results['prev_points_queue'] = prev_points_queue
        results['prev_ego2global_queue'] = prev_ego2global_queue
        results['prev_bev_exists_queue'] = prev_bev_exists_queue
        return results
