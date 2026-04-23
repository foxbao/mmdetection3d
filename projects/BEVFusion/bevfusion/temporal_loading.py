# Copyright (c) OpenMMLab. All rights reserved.
"""Pipeline transforms for loading previous-frame LiDAR data."""

from mmdet3d.datasets.transforms import LoadPointsFromFile
from mmdet3d.registry import TRANSFORMS


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
