"""KL-specific data transforms."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import PointData

from .map_utils import (load_kl_base_map, rasterize_local_map,
                        read_map_origin, select_local_map_geometries)


KL_CLASS_NAMES = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]


def _normalize_language_token(text: str) -> str:
    return text.lower().replace('-', '_').replace('/', '_')


KL_LANGUAGE_VOCAB = [
    '<pad>', '<unk>', 'forecast', 'predict', 'all', 'objects', 'near',
    'front', 'back', 'left', 'right', 'within', 'meters', 'risk',
    'future', 'corridor', 'entering', 'moving', 'static',
] + [_normalize_language_token(name) for name in KL_CLASS_NAMES]


KL_LANGUAGE_TOKEN_TO_ID = {
    token: idx for idx, token in enumerate(KL_LANGUAGE_VOCAB)
}


@TRANSFORMS.register_module()
class GenerateKLLanguageQuery(BaseTransform):
    """Generate a template language query and per-instance target mask.

    This is a bootstrap transform for language-conditioned forecasting. It
    intentionally avoids external tokenizers: prompts are generated from a
    small fixed vocabulary and stored as token ids in metainfo. The selected
    instances are stored as ``gt_language_target_mask`` so forecasting losses
    can be applied only to the objects requested by the prompt.
    """

    def __init__(self,
                 class_names: Optional[Sequence[str]] = None,
                 query_types: Sequence[str] = ('class', 'front', 'risk'),
                 max_tokens: int = 16,
                 distance: float = 40.0,
                 corridor_x: Sequence[float] = (0.0, 40.0),
                 corridor_y: Sequence[float] = (-3.0, 3.0),
                 fallback_to_all: bool = True,
                 seed: Optional[int] = None) -> None:
        self.class_names = list(class_names or KL_CLASS_NAMES)
        self.query_types = tuple(query_types)
        if len(self.query_types) == 0:
            raise ValueError('query_types must not be empty.')
        allowed = {'class', 'front', 'left', 'right', 'risk', 'all'}
        unknown = set(self.query_types) - allowed
        if unknown:
            raise ValueError(f'Unsupported query_types: {sorted(unknown)}.')
        if max_tokens <= 0:
            raise ValueError(f'max_tokens must be positive, got {max_tokens}.')
        if len(corridor_x) != 2 or len(corridor_y) != 2:
            raise ValueError('corridor_x and corridor_y must have two values.')
        self.max_tokens = int(max_tokens)
        self.distance = float(distance)
        self.corridor_x = (float(corridor_x[0]), float(corridor_x[1]))
        self.corridor_y = (float(corridor_y[0]), float(corridor_y[1]))
        self.fallback_to_all = bool(fallback_to_all)
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def _choice(self, values: Sequence):
        if self.rng is None:
            return values[int(np.random.randint(len(values)))]
        return values[int(self.rng.randint(len(values)))]

    @staticmethod
    def _boxes_xy(boxes) -> np.ndarray:
        if boxes is None:
            return np.zeros((0, 2), dtype=np.float32)
        centers = boxes.gravity_center[:, :2]
        return centers.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _to_numpy(data, dtype=None) -> np.ndarray:
        if data is None:
            return np.asarray([], dtype=dtype)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        arr = np.asarray(data)
        return arr.astype(dtype) if dtype is not None else arr

    def _tokenize(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        raw_tokens = prompt.lower().replace(',', ' ').replace('.', ' ').split()
        ids: List[int] = []
        for token in raw_tokens[:self.max_tokens]:
            norm = _normalize_language_token(token)
            ids.append(KL_LANGUAGE_TOKEN_TO_ID.get(
                norm, KL_LANGUAGE_TOKEN_TO_ID['<unk>']))
        token_ids = np.zeros(self.max_tokens, dtype=np.int64)
        token_mask = np.zeros(self.max_tokens, dtype=np.bool_)
        if ids:
            token_ids[:len(ids)] = np.asarray(ids, dtype=np.int64)
            token_mask[:len(ids)] = True
        return token_ids, token_mask

    def _select_class_query(self, labels: np.ndarray) -> Tuple[str, np.ndarray]:
        if labels.size == 0:
            class_idx = 0
        else:
            present = sorted(set(labels.astype(np.int64).tolist()))
            class_idx = int(self._choice(present))
        class_idx = int(np.clip(class_idx, 0, len(self.class_names) - 1))
        class_name = self.class_names[class_idx]
        prompt = f'forecast all {_normalize_language_token(class_name)} objects'
        return prompt, labels == class_idx

    def _select_spatial_query(self, query_type: str,
                              xy: np.ndarray) -> Tuple[str, np.ndarray]:
        if query_type == 'front':
            mask = (xy[:, 0] > 0.0) & (xy[:, 0] <= self.distance)
            prompt = f'forecast all front objects within {int(self.distance)} meters'
        elif query_type == 'left':
            mask = (xy[:, 1] > 0.0) & (np.abs(xy[:, 1]) <= self.distance)
            prompt = f'forecast all left objects within {int(self.distance)} meters'
        elif query_type == 'right':
            mask = (xy[:, 1] < 0.0) & (np.abs(xy[:, 1]) <= self.distance)
            prompt = f'forecast all right objects within {int(self.distance)} meters'
        else:
            raise ValueError(f'Unsupported spatial query {query_type}.')
        return prompt, mask

    def _select_risk_query(self, xy: np.ndarray,
                           forecasting_locs) -> Tuple[str, np.ndarray]:
        prompt = 'predict risk objects entering future corridor'
        locs = self._to_numpy(forecasting_locs, dtype=np.float32)
        if locs.ndim != 3 or locs.shape[-1] != 2 or locs.shape[0] != xy.shape[0]:
            return prompt, np.zeros(xy.shape[0], dtype=bool)
        future_xy = xy[:, None, :] + locs
        in_x = ((future_xy[..., 0] >= self.corridor_x[0]) &
                (future_xy[..., 0] <= self.corridor_x[1]))
        in_y = ((future_xy[..., 1] >= self.corridor_y[0]) &
                (future_xy[..., 1] <= self.corridor_y[1]))
        return prompt, (in_x & in_y).any(axis=1)

    def transform(self, results: dict) -> dict:
        boxes = results.get('gt_bboxes_3d')
        labels = self._to_numpy(results.get('gt_labels_3d'), dtype=np.int64)
        xy = self._boxes_xy(boxes)
        num_instances = xy.shape[0]
        if labels.shape[0] != num_instances:
            raise ValueError('GenerateKLLanguageQuery expects labels and boxes '
                             f'to be aligned, got {labels.shape[0]} labels '
                             f'and {num_instances} boxes.')

        query_type = self._choice(self.query_types)
        if query_type == 'class':
            prompt, target_mask = self._select_class_query(labels)
        elif query_type in ('front', 'left', 'right'):
            prompt, target_mask = self._select_spatial_query(query_type, xy)
        elif query_type == 'risk':
            prompt, target_mask = self._select_risk_query(
                xy, results.get('gt_forecasting_locs'))
        elif query_type == 'all':
            prompt = 'forecast all objects'
            target_mask = np.ones(num_instances, dtype=bool)
        else:
            raise AssertionError(f'Unhandled query_type {query_type}.')

        target_mask = np.asarray(target_mask, dtype=np.bool_)
        if (self.fallback_to_all and num_instances > 0 and
                not bool(target_mask.any())):
            prompt = 'forecast all objects'
            target_mask = np.ones(num_instances, dtype=np.bool_)

        token_ids, token_mask = self._tokenize(prompt)
        results['gt_language_target_mask'] = target_mask
        results['language_prompt'] = prompt
        results['language_query_type'] = query_type
        results['language_tokens'] = token_ids
        results['language_token_mask'] = token_mask
        return results

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(query_types={self.query_types}, '
                f'max_tokens={self.max_tokens}, distance={self.distance})')


@TRANSFORMS.register_module()
class ComputeVelocityFromForecasting(BaseTransform):
    """Overwrite gt_bboxes_3d velocity columns with forecasting-derived vel.

    KL converter stamps ``instance['velocity']`` as zeros — the real motion
    signal lives in ``gt_forecasting_locs``. With ``with_velocity=True``
    (KlDataset default) ``gt_bboxes_3d`` already has 9 dims, the last two
    being zero velocity slots; this transform fills them with
    ``locs[:, 0, :] / dt``, gated by ``mask[:, 0]`` (False rows keep 0).

    Required keys: ``gt_bboxes_3d``, ``gt_forecasting_locs``,
    ``gt_forecasting_mask``.
    Modified keys: ``gt_bboxes_3d`` (expanded to 9-dim if input is 7-dim).

    Limitation: boxes whose step-0 mask is False (~5%, mostly scene
    boundaries / newborn tracks) receive velocity=0 as target. This biases
    the vel head slightly toward stationary; fine for a sanity head.
    """

    def __init__(self, dt: float = 0.5) -> None:
        assert dt > 0, f'dt must be >0, got {dt}'
        self.dt = float(dt)

    def transform(self, results: dict) -> dict:
        boxes = results.get('gt_bboxes_3d')
        locs = results.get('gt_forecasting_locs')
        mask = results.get('gt_forecasting_mask')
        if boxes is None or locs is None or mask is None:
            return results

        locs_np = np.asarray(locs)
        mask_np = np.asarray(mask)
        n_box = boxes.tensor.shape[0]
        if locs_np.shape[0] != n_box:
            raise ValueError(
                f'forecasting_locs N={locs_np.shape[0]} != '
                f'gt_bboxes_3d N={n_box}; pipeline lost row alignment '
                'before ComputeVelocityFromForecasting.')

        if n_box == 0:
            if boxes.box_dim < 9:
                empty9 = torch.zeros((0, 9), dtype=boxes.tensor.dtype)
                results['gt_bboxes_3d'] = type(boxes)(
                    empty9, box_dim=9, with_yaw=boxes.with_yaw)
            return results

        vel = locs_np[:, 0, :].astype(np.float32) / self.dt
        vel *= mask_np[:, 0:1].astype(vel.dtype)
        vel_t = torch.as_tensor(
            vel, dtype=boxes.tensor.dtype, device=boxes.tensor.device)

        if boxes.box_dim >= 9:
            new_tensor = boxes.tensor.clone()
            new_tensor[:, 7:9] = vel_t
            results['gt_bboxes_3d'] = boxes.new_box(new_tensor)
        else:
            new_tensor = torch.cat([boxes.tensor[:, :7], vel_t], dim=-1)
            results['gt_bboxes_3d'] = type(boxes)(
                new_tensor, box_dim=9, with_yaw=boxes.with_yaw)
        return results

    def __repr__(self) -> str:
        return f'{type(self).__name__}(dt={self.dt})'


@TRANSFORMS.register_module()
class GenerateKLOccFromBoxes(BaseTransform):
    """Generate a coarse KL occupancy target from 3D boxes.

    Two target modes are supported:

    - ``bbox_fill``: every voxel whose center falls inside a GT box receives
      that box class.
    - ``points_in_boxes``: only voxels with LiDAR points inside a GT box are
      labeled; box-interior voxels without observed points can be marked as
      ``ignore_idx`` so they do not become false empty supervision.
    - ``raycast``: initialize all voxels as unknown, ray-cast observed free
      space from the LiDAR origin, then label hit voxels as semantic objects,
      ground, or other obstacles.

    Labels are shifted by ``label_offset`` so detector labels ``0..K-1``
    become occupancy labels ``1..K``.
    """

    def __init__(self,
                 point_cloud_range: Sequence[float],
                 occ_size: Sequence[int],
                 empty_idx: int = 0,
                 ignore_idx: int = 255,
                 label_offset: int = 1,
                 mode: str = 'bbox_fill',
                 min_points_per_voxel: int = 1,
                 dilation_xy: int = 0,
                 mark_unobserved_box_ignore: bool = True,
                 ray_origin: Sequence[float] = (0.0, 0.0, 0.0),
                 ground_label: int = 16,
                 obstacle_label: int = 17,
                 label_scene: bool = True,
                 ground_height_threshold: float = 0.55,
                 ground_smooth_radius: int = 3,
                 fill_ground: bool = True,
                 ground_fill_radius: int = 2,
                 ground_fill_min_neighbors: int = 5,
                 remove_ground_under_obstacle: bool = True,
                 obstacle_min_points_per_voxel: int = 1,
                 obstacle_min_component_voxels: int = 1,
                 obstacle_small_component_keep_min_points: int = 0,
                 obstacle_thin_component_min_major_span: float = 0.0,
                 obstacle_thin_component_max_minor_span: float = 0.0,
                 obstacle_thin_component_max_z_span: float = 0.0,
                 obstacle_thin_component_keep_min_points: int = 0,
                 obstacle_box_ignore_margin: float = 0.0,
                 ego_ignore_range: Optional[Sequence[float]] = None,
                 current_frame_only: bool = False) -> None:
        if len(point_cloud_range) != 6:
            raise ValueError('point_cloud_range must have 6 values, got '
                             f'{point_cloud_range}.')
        if len(occ_size) != 3:
            raise ValueError(f'occ_size must have 3 values, got {occ_size}.')
        if mode not in ('bbox_fill', 'points_in_boxes', 'raycast'):
            raise ValueError('mode must be "bbox_fill", "points_in_boxes", '
                             f'or "raycast", got {mode}.')
        if min_points_per_voxel <= 0:
            raise ValueError('min_points_per_voxel must be positive, got '
                             f'{min_points_per_voxel}.')
        if dilation_xy < 0:
            raise ValueError(f'dilation_xy must be >= 0, got {dilation_xy}.')
        if not (0 <= empty_idx <= 255 and 0 <= ignore_idx <= 255):
            raise ValueError('empty_idx and ignore_idx must fit uint8, got '
                             f'{empty_idx}, {ignore_idx}.')
        if not (0 <= ground_label <= 255 and 0 <= obstacle_label <= 255):
            raise ValueError('ground_label and obstacle_label must fit uint8, '
                             f'got {ground_label}, {obstacle_label}.')
        if len(ray_origin) != 3:
            raise ValueError(f'ray_origin must have 3 values, got {ray_origin}.')
        if ground_smooth_radius < 0:
            raise ValueError('ground_smooth_radius must be >= 0, got '
                             f'{ground_smooth_radius}.')
        if ground_fill_radius < 0:
            raise ValueError('ground_fill_radius must be >= 0, got '
                             f'{ground_fill_radius}.')
        if ground_fill_min_neighbors <= 0:
            raise ValueError('ground_fill_min_neighbors must be positive, got '
                             f'{ground_fill_min_neighbors}.')
        if obstacle_min_points_per_voxel <= 0:
            raise ValueError('obstacle_min_points_per_voxel must be positive, '
                             f'got {obstacle_min_points_per_voxel}.')
        if obstacle_min_component_voxels <= 0:
            raise ValueError('obstacle_min_component_voxels must be positive, '
                             f'got {obstacle_min_component_voxels}.')
        if obstacle_small_component_keep_min_points < 0:
            raise ValueError(
                'obstacle_small_component_keep_min_points must be >= 0, '
                f'got {obstacle_small_component_keep_min_points}.')
        if obstacle_thin_component_min_major_span < 0:
            raise ValueError(
                'obstacle_thin_component_min_major_span must be >= 0, '
                f'got {obstacle_thin_component_min_major_span}.')
        if obstacle_thin_component_max_minor_span < 0:
            raise ValueError(
                'obstacle_thin_component_max_minor_span must be >= 0, '
                f'got {obstacle_thin_component_max_minor_span}.')
        if obstacle_thin_component_max_z_span < 0:
            raise ValueError(
                'obstacle_thin_component_max_z_span must be >= 0, '
                f'got {obstacle_thin_component_max_z_span}.')
        if obstacle_thin_component_keep_min_points < 0:
            raise ValueError(
                'obstacle_thin_component_keep_min_points must be >= 0, '
                f'got {obstacle_thin_component_keep_min_points}.')
        if obstacle_box_ignore_margin < 0:
            raise ValueError('obstacle_box_ignore_margin must be >= 0, got '
                             f'{obstacle_box_ignore_margin}.')
        if ego_ignore_range is not None and len(ego_ignore_range) != 6:
            raise ValueError('ego_ignore_range must have 6 values '
                             '[x_min, y_min, z_min, x_max, y_max, z_max], '
                             f'got {ego_ignore_range}.')

        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.occ_size = np.asarray(occ_size, dtype=np.int64)
        if np.any(self.occ_size <= 0):
            raise ValueError(f'occ_size values must be positive, got {occ_size}.')
        self.empty_idx = int(empty_idx)
        self.ignore_idx = int(ignore_idx)
        self.label_offset = int(label_offset)
        self.mode = mode
        self.min_points_per_voxel = int(min_points_per_voxel)
        self.dilation_xy = int(dilation_xy)
        self.mark_unobserved_box_ignore = bool(mark_unobserved_box_ignore)
        self.ray_origin = np.asarray(ray_origin, dtype=np.float32)
        self.ground_label = int(ground_label)
        self.obstacle_label = int(obstacle_label)
        self.label_scene = bool(label_scene)
        self.ground_height_threshold = float(ground_height_threshold)
        self.ground_smooth_radius = int(ground_smooth_radius)
        self.fill_ground = bool(fill_ground)
        self.ground_fill_radius = int(ground_fill_radius)
        self.ground_fill_min_neighbors = int(ground_fill_min_neighbors)
        self.remove_ground_under_obstacle = bool(remove_ground_under_obstacle)
        self.obstacle_min_points_per_voxel = int(
            obstacle_min_points_per_voxel)
        self.obstacle_min_component_voxels = int(
            obstacle_min_component_voxels)
        self.obstacle_small_component_keep_min_points = int(
            obstacle_small_component_keep_min_points)
        self.obstacle_thin_component_min_major_span = float(
            obstacle_thin_component_min_major_span)
        self.obstacle_thin_component_max_minor_span = float(
            obstacle_thin_component_max_minor_span)
        self.obstacle_thin_component_max_z_span = float(
            obstacle_thin_component_max_z_span)
        self.obstacle_thin_component_keep_min_points = int(
            obstacle_thin_component_keep_min_points)
        self.filter_thin_obstacle_components = (
            self.obstacle_thin_component_min_major_span > 0 and
            self.obstacle_thin_component_max_minor_span > 0 and
            self.obstacle_thin_component_max_z_span > 0)
        self.obstacle_box_ignore_margin = float(obstacle_box_ignore_margin)
        self.ego_ignore_range = (
            None if ego_ignore_range is None else
            np.asarray(ego_ignore_range, dtype=np.float32))
        if self.ego_ignore_range is not None:
            if np.any(self.ego_ignore_range[3:] <= self.ego_ignore_range[:3]):
                raise ValueError('ego_ignore_range max values must be larger '
                                 'than min values, got '
                                 f'{ego_ignore_range}.')
        self.current_frame_only = bool(current_frame_only)
        self.voxel_size = (
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) /
            self.occ_size.astype(np.float32))

    def _coord_to_index_floor(self, xyz: np.ndarray) -> np.ndarray:
        return np.floor(
            (xyz - self.point_cloud_range[:3]) / self.voxel_size).astype(
                np.int64)

    def _coord_to_index_ceil(self, xyz: np.ndarray) -> np.ndarray:
        return np.ceil(
            (xyz - self.point_cloud_range[:3]) / self.voxel_size).astype(
                np.int64)

    def _voxel_centers(self, axis: int, indices: np.ndarray) -> np.ndarray:
        return (self.point_cloud_range[axis] +
                (indices.astype(np.float32) + 0.5) * self.voxel_size[axis])

    def _box_voxel_indices(self, box: np.ndarray):
        corners_min = box[:3].copy()
        corners_min[2] = box[2]
        corners_max = box[:3].copy()
        corners_max[2] = box[2] + box[5]

        # XY extents must account for yaw; use corners for a tight candidate
        # window, then exact-test voxel centers in the rotated box frame.
        length, width, height = box[3:6]
        if length <= 0 or width <= 0 or height <= 0:
            return None

        yaw = box[6]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        dx = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float32) * length
        dy = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32) * width
        x_corners = box[0] + dx * cos_yaw - dy * sin_yaw
        y_corners = box[1] + dx * sin_yaw + dy * cos_yaw
        corners_min[:2] = [x_corners.min(), y_corners.min()]
        corners_max[:2] = [x_corners.max(), y_corners.max()]

        lo = self._coord_to_index_floor(corners_min)
        hi = self._coord_to_index_ceil(corners_max)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.occ_size)
        if np.any(hi <= lo):
            return None

        x_idx = np.arange(lo[0], hi[0], dtype=np.int64)
        y_idx = np.arange(lo[1], hi[1], dtype=np.int64)
        z_idx = np.arange(lo[2], hi[2], dtype=np.int64)
        if x_idx.size == 0 or y_idx.size == 0 or z_idx.size == 0:
            return None

        xs = self._voxel_centers(0, x_idx)
        ys = self._voxel_centers(1, y_idx)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        rel_x = xx - box[0]
        rel_y = yy - box[1]
        local_x = rel_x * cos_yaw + rel_y * sin_yaw
        local_y = -rel_x * sin_yaw + rel_y * cos_yaw
        inside_xy = (
            (np.abs(local_x) <= length * 0.5) &
            (np.abs(local_y) <= width * 0.5))
        if not np.any(inside_xy):
            return None

        grid_x, grid_y = np.meshgrid(x_idx, y_idx, indexing='ij')
        fill_x = grid_x[inside_xy]
        fill_y = grid_y[inside_xy]
        return fill_x, fill_y, z_idx

    def _fill_box(self, occ: np.ndarray, box: np.ndarray, label: int) -> None:
        indices = self._box_voxel_indices(box)
        if indices is None:
            return
        fill_x, fill_y, z_idx = indices
        occ[fill_x[:, None], fill_y[:, None], z_idx[None, :]] = label

    @staticmethod
    def _extract_points_xyz(points) -> np.ndarray:
        if hasattr(points, 'tensor'):
            points = points.tensor
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError('GenerateKLOccFromBoxes expects points with '
                             f'shape [N, >=3], got {points.shape}.')
        return points[:, :3].astype(np.float32, copy=False)

    def _points_in_box_voxels(self, points_xyz: np.ndarray,
                              box: np.ndarray):
        length, width, height = box[3:6]
        if length <= 0 or width <= 0 or height <= 0:
            return (np.empty((0, 3), dtype=np.int64),
                    np.empty((0, ), dtype=np.int64))

        yaw = box[6]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        dx = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float32) * length
        dy = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32) * width
        x_corners = box[0] + dx * cos_yaw - dy * sin_yaw
        y_corners = box[1] + dx * sin_yaw + dy * cos_yaw

        # Pre-filter by the yawed box's axis-aligned bounds before the exact
        # rotated-frame test; this keeps online point-conditioned targets cheap.
        in_aabb = (
            (points_xyz[:, 0] >= x_corners.min()) &
            (points_xyz[:, 0] <= x_corners.max()) &
            (points_xyz[:, 1] >= y_corners.min()) &
            (points_xyz[:, 1] <= y_corners.max()) &
            (points_xyz[:, 2] >= box[2]) &
            (points_xyz[:, 2] <= box[2] + height))
        pts = points_xyz[in_aabb]
        if pts.shape[0] == 0:
            return (np.empty((0, 3), dtype=np.int64),
                    np.empty((0, ), dtype=np.int64))

        rel_x = pts[:, 0] - box[0]
        rel_y = pts[:, 1] - box[1]
        local_x = rel_x * cos_yaw + rel_y * sin_yaw
        local_y = -rel_x * sin_yaw + rel_y * cos_yaw
        local_z = pts[:, 2] - box[2]
        inside = (
            (np.abs(local_x) <= length * 0.5) &
            (np.abs(local_y) <= width * 0.5) &
            (local_z >= 0.0) & (local_z <= height))
        pts = pts[inside]
        if pts.shape[0] == 0:
            return (np.empty((0, 3), dtype=np.int64),
                    np.empty((0, ), dtype=np.int64))

        voxels = self._coord_to_index_floor(pts)
        valid = np.all((voxels >= 0) & (voxels < self.occ_size), axis=1)
        voxels = voxels[valid]
        if voxels.shape[0] == 0:
            return (np.empty((0, 3), dtype=np.int64),
                    np.empty((0, ), dtype=np.int64))
        return np.unique(voxels, axis=0, return_counts=True)

    @staticmethod
    def _points_in_box_mask(points_xyz: np.ndarray,
                            box: np.ndarray) -> np.ndarray:
        mask = np.zeros((points_xyz.shape[0], ), dtype=bool)
        if points_xyz.shape[0] == 0:
            return mask
        length, width, height = box[3:6]
        if length <= 0 or width <= 0 or height <= 0:
            return mask
        yaw = box[6]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rel_x = points_xyz[:, 0] - box[0]
        rel_y = points_xyz[:, 1] - box[1]
        local_x = rel_x * cos_yaw + rel_y * sin_yaw
        local_y = -rel_x * sin_yaw + rel_y * cos_yaw
        local_z = points_xyz[:, 2] - box[2]
        return (
            (np.abs(local_x) <= length * 0.5) &
            (np.abs(local_y) <= width * 0.5) &
            (local_z >= 0.0) & (local_z <= height))

    @classmethod
    def _points_in_any_box_mask(cls, points_xyz: np.ndarray,
                                box_tensor: np.ndarray) -> np.ndarray:
        mask = np.zeros((points_xyz.shape[0], ), dtype=bool)
        if points_xyz.shape[0] == 0 or box_tensor.shape[0] == 0:
            return mask

        for box in box_tensor:
            mask |= cls._points_in_box_mask(points_xyz, box)
        return mask

    def _dilate_observed_voxels(self, voxels: np.ndarray, box: np.ndarray):
        if self.dilation_xy == 0 or voxels.shape[0] == 0:
            return voxels

        indices = self._box_voxel_indices(box)
        if indices is None:
            return voxels
        fill_x, fill_y, z_idx = indices
        inside_box = np.zeros(tuple(self.occ_size.tolist()), dtype=bool)
        inside_box[fill_x[:, None], fill_y[:, None], z_idx[None, :]] = True

        expanded = set(map(tuple, voxels.tolist()))
        radius = self.dilation_xy
        for i, j, k in voxels:
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = int(i + di), int(j + dj)
                    nk = int(k)
                    if (0 <= ni < self.occ_size[0] and
                            0 <= nj < self.occ_size[1] and
                            inside_box[ni, nj, nk]):
                        expanded.add((ni, nj, nk))
        return np.asarray(sorted(expanded), dtype=np.int64)

    def _fill_points_in_box(self, occ: np.ndarray, points_xyz: np.ndarray,
                            box: np.ndarray, label: int) -> np.ndarray:
        indices = self._box_voxel_indices(box)
        if indices is not None and self.mark_unobserved_box_ignore:
            fill_x, fill_y, z_idx = indices
            occ[fill_x[:, None], fill_y[:, None], z_idx[None, :]] = (
                self.ignore_idx)

        voxels, counts = self._points_in_box_voxels(points_xyz, box)
        if voxels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)
        voxels = voxels[counts >= self.min_points_per_voxel]
        if voxels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)
        voxels = self._dilate_observed_voxels(voxels, box)
        occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = label
        return voxels

    def _point_voxels(self, points_xyz: np.ndarray) -> np.ndarray:
        voxels = self._coord_to_index_floor(points_xyz)
        valid = np.all((voxels >= 0) & (voxels < self.occ_size), axis=1)
        return voxels[valid]

    def _raycast_free_voxels(self, hit_voxels: np.ndarray) -> np.ndarray:
        if hit_voxels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)

        free_chunks = []
        origin = self.ray_origin
        for hit in hit_voxels:
            hit_center = (
                self.point_cloud_range[:3] +
                (hit.astype(np.float32) + 0.5) * self.voxel_size)
            delta = hit_center - origin
            max_grid_dist = np.max(np.abs(delta / self.voxel_size))
            num_steps = int(np.ceil(max_grid_dist))
            if num_steps <= 0:
                continue

            t = (np.arange(num_steps, dtype=np.float32) /
                 float(num_steps))[:, None]
            samples = origin[None, :] + t * delta[None, :]
            voxels = self._coord_to_index_floor(samples)
            valid = np.all((voxels >= 0) & (voxels < self.occ_size), axis=1)
            voxels = voxels[valid]
            if voxels.shape[0] == 0:
                continue
            voxels = np.unique(voxels, axis=0)
            voxels = voxels[np.any(voxels != hit[None, :], axis=1)]
            if voxels.shape[0] > 0:
                free_chunks.append(voxels)

        if not free_chunks:
            return np.empty((0, 3), dtype=np.int64)
        return np.unique(np.concatenate(free_chunks, axis=0), axis=0)

    def _box_interior_mask(self, box_tensor: np.ndarray) -> np.ndarray:
        mask = np.zeros(tuple(self.occ_size.tolist()), dtype=bool)
        for box in box_tensor:
            indices = self._box_voxel_indices(box)
            if indices is None:
                continue
            fill_x, fill_y, z_idx = indices
            mask[fill_x[:, None], fill_y[:, None], z_idx[None, :]] = True
        return mask

    def _ego_ignore_mask(self) -> np.ndarray:
        mask = np.zeros(tuple(self.occ_size.tolist()), dtype=bool)
        if self.ego_ignore_range is None:
            return mask

        lo = np.maximum(self.ego_ignore_range[:3],
                        self.point_cloud_range[:3])
        hi = np.minimum(self.ego_ignore_range[3:],
                        self.point_cloud_range[3:])
        if np.any(hi <= lo):
            return mask

        x_idx = np.arange(self.occ_size[0], dtype=np.int64)
        y_idx = np.arange(self.occ_size[1], dtype=np.int64)
        z_idx = np.arange(self.occ_size[2], dtype=np.int64)
        xs = self._voxel_centers(0, x_idx)
        ys = self._voxel_centers(1, y_idx)
        zs = self._voxel_centers(2, z_idx)
        x_sel = (xs >= lo[0]) & (xs <= hi[0])
        y_sel = (ys >= lo[1]) & (ys <= hi[1])
        z_sel = (zs >= lo[2]) & (zs <= hi[2])
        x_keep = np.flatnonzero(x_sel)
        y_keep = np.flatnonzero(y_sel)
        z_keep = np.flatnonzero(z_sel)
        if x_keep.size == 0 or y_keep.size == 0 or z_keep.size == 0:
            return mask
        mask[np.ix_(x_keep, y_keep, z_keep)] = True
        return mask

    def _apply_ego_ignore(self, occ: np.ndarray) -> int:
        mask = self._ego_ignore_mask()
        if not np.any(mask):
            return 0
        occ[mask] = self.ignore_idx
        return int(mask.sum())

    def _estimate_ground_height(self, points_xyz: np.ndarray,
                                point_voxels: np.ndarray):
        ground_raw = np.full(
            (self.occ_size[0], self.occ_size[1]), np.nan, dtype=np.float32)
        raw_ground_xy = np.zeros(
            (self.occ_size[0], self.occ_size[1]), dtype=bool)
        buckets = [[[] for _ in range(self.occ_size[1])]
                   for _ in range(self.occ_size[0])]
        for voxel, point in zip(point_voxels, points_xyz):
            buckets[int(voxel[0])][int(voxel[1])].append(float(point[2]))

        for i in range(self.occ_size[0]):
            for j in range(self.occ_size[1]):
                if buckets[i][j]:
                    raw_ground_xy[i, j] = True
                    ground_raw[i, j] = np.percentile(buckets[i][j], 10)

        ground_est = np.full_like(ground_raw, np.nan)
        radius = self.ground_smooth_radius
        for i in range(self.occ_size[0]):
            x0, x1 = max(0, i - radius), min(self.occ_size[0], i + radius + 1)
            for j in range(self.occ_size[1]):
                y0 = max(0, j - radius)
                y1 = min(self.occ_size[1], j + radius + 1)
                vals = ground_raw[x0:x1, y0:y1]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ground_est[i, j] = np.percentile(vals, 25)
        return ground_est, raw_ground_xy

    def _split_scene_ground_obstacle(
            self, scene_points_xyz: np.ndarray,
            scene_point_voxels: np.ndarray,
            scene_hit_voxels: np.ndarray,
            semantic_voxels: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
        if scene_hit_voxels.shape[0] == 0 or scene_points_xyz.shape[0] == 0:
            empty = np.empty((0, 3), dtype=np.int64)
            return empty, empty, empty

        ground_est, raw_ground_xy = self._estimate_ground_height(
            scene_points_xyz, scene_point_voxels)

        center_z = (self.point_cloud_range[2] +
                    (scene_hit_voxels[:, 2].astype(np.float32) + 0.5) *
                    self.voxel_size[2])
        cell_ground = ground_est[scene_hit_voxels[:, 0],
                                 scene_hit_voxels[:, 1]]
        is_ground = (
            np.isfinite(cell_ground) &
            (center_z <= cell_ground + self.ground_height_threshold))
        observed_ground = scene_hit_voxels[is_ground]
        raw_obstacle = scene_hit_voxels[~is_ground]
        obstacle = self._filter_obstacle_voxels(
            raw_obstacle, scene_point_voxels)

        if self.fill_ground:
            fill_xy = np.zeros_like(raw_ground_xy)
            radius = self.ground_fill_radius
            for i in range(self.occ_size[0]):
                x0 = max(0, i - radius)
                x1 = min(self.occ_size[0], i + radius + 1)
                for j in range(self.occ_size[1]):
                    y0 = max(0, j - radius)
                    y1 = min(self.occ_size[1], j + radius + 1)
                    if (raw_ground_xy[x0:x1, y0:y1].sum() >=
                            self.ground_fill_min_neighbors and
                            np.isfinite(ground_est[i, j])):
                        fill_xy[i, j] = True
            xy = np.argwhere(fill_xy)
            z_idx = np.floor(
                (ground_est[fill_xy] - self.point_cloud_range[2]) /
                self.voxel_size[2]).astype(np.int64)
            valid = (z_idx >= 0) & (z_idx < self.occ_size[2])
            ground = np.column_stack([xy[valid], z_idx[valid]])
        else:
            ground = observed_ground

        if self.remove_ground_under_obstacle and ground.shape[0] > 0:
            blocked_xy = set(map(tuple, obstacle[:, :2].tolist()))
            if semantic_voxels.shape[0] > 0:
                blocked_xy.update(map(tuple, semantic_voxels[:, :2].tolist()))
            keep = np.array(
                [tuple(voxel[:2]) not in blocked_xy for voxel in ground],
                dtype=bool)
            ground = ground[keep]
        return ground, obstacle, raw_obstacle

    def _filter_obstacle_voxels(self, obstacle_voxels: np.ndarray,
                                scene_point_voxels: np.ndarray) -> np.ndarray:
        """Drop weak obstacle hits before they become supervised labels."""
        if obstacle_voxels.shape[0] == 0:
            return obstacle_voxels

        filtered = obstacle_voxels
        counts = None
        if (self.obstacle_min_points_per_voxel > 1 or
                self.obstacle_small_component_keep_min_points > 0 or
                self.obstacle_thin_component_keep_min_points > 0):
            counts = np.zeros(tuple(self.occ_size.tolist()), dtype=np.uint16)
            np.add.at(counts, tuple(scene_point_voxels.T), 1)
        if self.obstacle_min_points_per_voxel > 1:
            keep = (
                counts[filtered[:, 0], filtered[:, 1], filtered[:, 2]] >=
                self.obstacle_min_points_per_voxel)
            filtered = filtered[keep]
            if filtered.shape[0] == 0:
                return filtered

        if (self.obstacle_min_component_voxels <= 1 and
                not self.filter_thin_obstacle_components):
            return filtered

        obstacle_mask = np.zeros(tuple(self.occ_size.tolist()), dtype=bool)
        obstacle_mask[filtered[:, 0], filtered[:, 1], filtered[:, 2]] = True
        visited = np.zeros_like(obstacle_mask)
        keep_mask = np.zeros_like(obstacle_mask)
        neighbors = (
            (1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1))

        for start in filtered:
            start_tuple = tuple(int(v) for v in start)
            if visited[start_tuple] or not obstacle_mask[start_tuple]:
                continue
            stack = [start_tuple]
            visited[start_tuple] = True
            component = []
            while stack:
                voxel = stack.pop()
                component.append(voxel)
                for offset in neighbors:
                    nxt = (
                        voxel[0] + offset[0],
                        voxel[1] + offset[1],
                        voxel[2] + offset[2])
                    if (0 <= nxt[0] < self.occ_size[0] and
                            0 <= nxt[1] < self.occ_size[1] and
                            0 <= nxt[2] < self.occ_size[2] and
                            obstacle_mask[nxt] and not visited[nxt]):
                        visited[nxt] = True
                        stack.append(nxt)
            if len(component) < self.obstacle_min_component_voxels:
                if not self._should_keep_dense_small_component(
                        component, counts):
                    continue
            if self._is_thin_obstacle_component(component, counts):
                continue
            for voxel in component:
                keep_mask[voxel] = True

        keep = keep_mask[filtered[:, 0], filtered[:, 1], filtered[:, 2]]
        return filtered[keep]

    def _component_point_count(
            self,
            component: Sequence[Tuple[int, int, int]],
            point_counts: Optional[np.ndarray]) -> int:
        if point_counts is None:
            return 0
        component_idx = np.asarray(component, dtype=np.int64)
        return int(point_counts[component_idx[:, 0], component_idx[:, 1],
                                component_idx[:, 2]].sum())

    def _should_keep_dense_small_component(
            self,
            component: Sequence[Tuple[int, int, int]],
            point_counts: Optional[np.ndarray]) -> bool:
        return (
            self.obstacle_small_component_keep_min_points > 0 and
            self._component_point_count(component, point_counts) >=
            self.obstacle_small_component_keep_min_points)

    def _is_thin_obstacle_component(
            self,
            component: Sequence[Tuple[int, int, int]],
            point_counts: Optional[np.ndarray] = None) -> bool:
        """Identify low, strip-like obstacle components as ambiguous noise."""
        if not self.filter_thin_obstacle_components:
            return False

        component_idx = np.asarray(component, dtype=np.int64)
        if (self.obstacle_thin_component_keep_min_points > 0 and
                point_counts is not None):
            num_points = self._component_point_count(component_idx,
                                                     point_counts)
            if num_points >= self.obstacle_thin_component_keep_min_points:
                return False

        component_arr = component_idx.astype(np.float32)
        centers = self.point_cloud_range[:3] + (
            component_arr + 0.5) * self.voxel_size
        span = centers.max(axis=0) - centers.min(axis=0)
        major_xy_span = float(max(span[0], span[1]))
        minor_xy_span = float(min(span[0], span[1]))
        z_span = float(span[2])
        eps = 1e-4
        return (
            major_xy_span + eps >=
            self.obstacle_thin_component_min_major_span and
            minor_xy_span <=
            self.obstacle_thin_component_max_minor_span + eps and
            z_span <= self.obstacle_thin_component_max_z_span + eps)

    def _obstacle_near_box_mask(self, obstacle_voxels: np.ndarray,
                                box_tensor: np.ndarray) -> np.ndarray:
        """Find obstacle voxels inside an expanded annotated-box envelope."""
        mask = np.zeros((obstacle_voxels.shape[0], ), dtype=bool)
        if (obstacle_voxels.shape[0] == 0 or box_tensor.shape[0] == 0 or
                self.obstacle_box_ignore_margin <= 0):
            return mask

        margin = self.obstacle_box_ignore_margin
        centers = (
            self.point_cloud_range[:3] +
            (obstacle_voxels.astype(np.float32) + 0.5) * self.voxel_size)
        for box in box_tensor:
            length, width, height = box[3:6]
            if length <= 0 or width <= 0 or height <= 0:
                continue
            yaw = box[6]
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rel_x = centers[:, 0] - box[0]
            rel_y = centers[:, 1] - box[1]
            local_x = rel_x * cos_yaw + rel_y * sin_yaw
            local_y = -rel_x * sin_yaw + rel_y * cos_yaw
            inside = (
                (np.abs(local_x) <= length * 0.5 + margin) &
                (np.abs(local_y) <= width * 0.5 + margin) &
                (centers[:, 2] >= box[2] - margin) &
                (centers[:, 2] <= box[2] + height + margin))
            mask |= inside
        return mask

    def _transform_raycast(self, results: dict, box_tensor: np.ndarray,
                           label_array: np.ndarray,
                           volumes: np.ndarray) -> dict:
        if 'points' not in results:
            raise KeyError('GenerateKLOccFromBoxes(mode="raycast") requires '
                           '`points` in results.')

        points_xyz = self._extract_points_xyz(results['points'])
        point_voxels = self._coord_to_index_floor(points_xyz)
        valid_points = np.all(
            (point_voxels >= 0) & (point_voxels < self.occ_size), axis=1)
        points_xyz = points_xyz[valid_points]
        point_voxels = point_voxels[valid_points]
        if point_voxels.shape[0] == 0:
            occ = np.full(
                tuple(self.occ_size.tolist()), self.ignore_idx, dtype=np.uint8)
            num_ego_ignore = self._apply_ego_ignore(occ)
            results['gt_occ'] = occ
            results['gt_occ_meta'] = dict(
                point_cloud_range=self.point_cloud_range.tolist(),
                occ_size=self.occ_size.tolist(),
                voxel_size=self.voxel_size.tolist(),
                source='points_raycast',
                mode=self.mode,
                ignore_idx=self.ignore_idx,
                ray_origin=self.ray_origin.tolist(),
                ego_ignore_range=(
                    None if self.ego_ignore_range is None else
                    self.ego_ignore_range.tolist()),
                num_ego_ignore_voxels=num_ego_ignore)
            return results

        hit_voxels = np.unique(point_voxels, axis=0)
        occ = np.full(
            tuple(self.occ_size.tolist()), self.ignore_idx, dtype=np.uint8)
        free_voxels = self._raycast_free_voxels(hit_voxels)
        if free_voxels.shape[0] > 0:
            occ[free_voxels[:, 0], free_voxels[:, 1],
                free_voxels[:, 2]] = self.empty_idx

        box_mask = self._box_interior_mask(box_tensor)
        if self.mark_unobserved_box_ignore:
            occ[box_mask] = self.ignore_idx

        semantic_chunks = []
        for idx in np.argsort(-volumes):
            label = int(label_array[idx])
            if label < 0:
                continue
            semantic_voxels = self._fill_points_in_box(
                occ, points_xyz, box_tensor[idx], label + self.label_offset)
            if semantic_voxels.shape[0] > 0:
                semantic_chunks.append(semantic_voxels)

        if semantic_chunks:
            semantic_voxels = np.unique(
                np.concatenate(semantic_chunks, axis=0), axis=0)
        else:
            semantic_voxels = np.empty((0, 3), dtype=np.int64)

        if self.label_scene:
            point_in_box = box_mask[
                point_voxels[:, 0], point_voxels[:, 1], point_voxels[:, 2]]
            scene_points = points_xyz[~point_in_box]
            scene_point_voxels = point_voxels[~point_in_box]
            hit_in_box = box_mask[
                hit_voxels[:, 0], hit_voxels[:, 1], hit_voxels[:, 2]]
            scene_hit_voxels = hit_voxels[~hit_in_box]
            ground_voxels, obstacle_voxels, raw_obstacle_voxels = (
                self._split_scene_ground_obstacle(
                    scene_points, scene_point_voxels, scene_hit_voxels,
                    semantic_voxels))
            raw_obstacle_count = int(raw_obstacle_voxels.shape[0])
            box_margin_obstacle_count = 0
            if obstacle_voxels.shape[0] > 0:
                near_box = self._obstacle_near_box_mask(
                    obstacle_voxels, box_tensor)
                box_margin_obstacle_count = int(near_box.sum())
                obstacle_voxels = obstacle_voxels[~near_box]
                if obstacle_voxels.shape[0] > 0:
                    obstacle_voxels = self._filter_obstacle_voxels(
                        obstacle_voxels, scene_point_voxels)
            ignored_obstacle_mask = np.zeros(
                tuple(self.occ_size.tolist()), dtype=bool)
            if raw_obstacle_voxels.shape[0] > 0:
                ignored_obstacle_mask[
                    raw_obstacle_voxels[:, 0], raw_obstacle_voxels[:, 1],
                    raw_obstacle_voxels[:, 2]] = True
            if obstacle_voxels.shape[0] > 0:
                ignored_obstacle_mask[
                    obstacle_voxels[:, 0], obstacle_voxels[:, 1],
                    obstacle_voxels[:, 2]] = False
            ignored_obstacle_voxels = np.argwhere(ignored_obstacle_mask)
            if obstacle_voxels.shape[0] > 0:
                occ[obstacle_voxels[:, 0], obstacle_voxels[:, 1],
                    obstacle_voxels[:, 2]] = self.obstacle_label
            if ground_voxels.shape[0] > 0:
                occ[ground_voxels[:, 0], ground_voxels[:, 1],
                    ground_voxels[:, 2]] = self.ground_label
            if ignored_obstacle_voxels.shape[0] > 0:
                occ[ignored_obstacle_voxels[:, 0],
                    ignored_obstacle_voxels[:, 1],
                    ignored_obstacle_voxels[:, 2]] = self.ignore_idx
        else:
            ground_voxels = np.empty((0, 3), dtype=np.int64)
            obstacle_voxels = np.empty((0, 3), dtype=np.int64)
            raw_obstacle_count = 0
            box_margin_obstacle_count = 0
            ignored_obstacle_voxels = np.empty((0, 3), dtype=np.int64)

        num_ego_ignore = self._apply_ego_ignore(occ)
        results['gt_occ'] = occ
        results['gt_occ_meta'] = dict(
            point_cloud_range=self.point_cloud_range.tolist(),
            occ_size=self.occ_size.tolist(),
            voxel_size=self.voxel_size.tolist(),
            source='points_raycast',
            mode=self.mode,
            free_idx=self.empty_idx,
            ignore_idx=self.ignore_idx,
            ground_label=self.ground_label,
            obstacle_label=self.obstacle_label,
            ray_origin=self.ray_origin.tolist(),
            ego_ignore_range=(
                None if self.ego_ignore_range is None else
                self.ego_ignore_range.tolist()),
            current_frame_only=self.current_frame_only,
            num_hit_voxels=int(hit_voxels.shape[0]),
            num_free_voxels=int(free_voxels.shape[0]),
            num_semantic_voxels=int(semantic_voxels.shape[0]),
            num_ground_voxels=int(ground_voxels.shape[0]),
            num_raw_obstacle_voxels=int(raw_obstacle_count),
            num_obstacle_voxels=int(obstacle_voxels.shape[0]),
            num_box_margin_obstacle_voxels=int(box_margin_obstacle_count),
            num_filtered_obstacle_voxels=int(
                ignored_obstacle_voxels.shape[0]),
            obstacle_min_points_per_voxel=(
                self.obstacle_min_points_per_voxel),
            obstacle_min_component_voxels=(
                self.obstacle_min_component_voxels),
            obstacle_small_component_keep_min_points=(
                self.obstacle_small_component_keep_min_points),
            obstacle_thin_component_min_major_span=(
                self.obstacle_thin_component_min_major_span),
            obstacle_thin_component_max_minor_span=(
                self.obstacle_thin_component_max_minor_span),
            obstacle_thin_component_max_z_span=(
                self.obstacle_thin_component_max_z_span),
            obstacle_thin_component_keep_min_points=(
                self.obstacle_thin_component_keep_min_points),
            obstacle_box_ignore_margin=self.obstacle_box_ignore_margin,
            num_ego_ignore_voxels=num_ego_ignore)
        return results

    def transform(self, results: dict) -> dict:
        if (self.current_frame_only and
                not results.get('_kl_is_current_frame', True)):
            return results

        boxes = results.get('gt_bboxes_3d')
        labels = results.get('gt_labels_3d')
        init_idx = self.ignore_idx if self.mode == 'raycast' else self.empty_idx
        occ = np.full(tuple(self.occ_size.tolist()), init_idx, dtype=np.uint8)

        if boxes is None or labels is None:
            if self.mode == 'raycast' and 'points' in results:
                return self._transform_raycast(
                    results,
                    np.zeros((0, 7), dtype=np.float32),
                np.zeros((0, ), dtype=np.int64),
                np.zeros((0, ), dtype=np.float32))
            num_ego_ignore = self._apply_ego_ignore(occ)
            results['gt_occ'] = occ
            results['gt_occ_meta'] = dict(
                point_cloud_range=self.point_cloud_range.tolist(),
                occ_size=self.occ_size.tolist(),
                voxel_size=self.voxel_size.tolist(),
                source='boxes',
                mode=self.mode,
                ego_ignore_range=(
                    None if self.ego_ignore_range is None else
                    self.ego_ignore_range.tolist()),
                num_ego_ignore_voxels=num_ego_ignore)
            return results

        box_tensor = boxes.tensor.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            label_array = labels.detach().cpu().numpy()
        else:
            label_array = np.asarray(labels)

        if box_tensor.shape[0] != label_array.shape[0]:
            raise ValueError(
                f'gt_bboxes_3d N={box_tensor.shape[0]} != '
                f'gt_labels_3d N={label_array.shape[0]}; cannot generate '
                'aligned occupancy labels.')

        volumes = box_tensor[:, 3] * box_tensor[:, 4] * box_tensor[:, 5]
        if self.mode == 'raycast':
            return self._transform_raycast(
                results, box_tensor, label_array, volumes)

        if self.mode == 'points_in_boxes':
            if 'points' not in results:
                raise KeyError('GenerateKLOccFromBoxes(mode="points_in_boxes") '
                               'requires `points` in results.')
            points_xyz = self._extract_points_xyz(results['points'])
        else:
            points_xyz = None

        # Fill larger boxes first so small boxes keep their labels if a rare
        # overlap occurs.
        for idx in np.argsort(-volumes):
            label = int(label_array[idx])
            if label < 0:
                continue
            occ_label = label + self.label_offset
            if self.mode == 'bbox_fill':
                self._fill_box(occ, box_tensor[idx], occ_label)
            else:
                self._fill_points_in_box(occ, points_xyz, box_tensor[idx],
                                         occ_label)

        num_ego_ignore = self._apply_ego_ignore(occ)
        results['gt_occ'] = occ
        results['gt_occ_meta'] = dict(
            point_cloud_range=self.point_cloud_range.tolist(),
            occ_size=self.occ_size.tolist(),
            voxel_size=self.voxel_size.tolist(),
            source='boxes',
            mode=self.mode,
            min_points_per_voxel=self.min_points_per_voxel,
            dilation_xy=self.dilation_xy,
            mark_unobserved_box_ignore=self.mark_unobserved_box_ignore,
            ground_label=self.ground_label,
            obstacle_label=self.obstacle_label,
            ego_ignore_range=(
                None if self.ego_ignore_range is None else
                self.ego_ignore_range.tolist()),
            num_ego_ignore_voxels=num_ego_ignore,
            current_frame_only=self.current_frame_only)
        return results

    def __repr__(self) -> str:
        return (f'{type(self).__name__}('
                f'point_cloud_range={self.point_cloud_range.tolist()}, '
                f'occ_size={self.occ_size.tolist()}, '
                f'empty_idx={self.empty_idx}, ignore_idx={self.ignore_idx}, '
                f'label_offset={self.label_offset}, mode={self.mode}, '
                f'min_points_per_voxel={self.min_points_per_voxel}, '
                f'dilation_xy={self.dilation_xy}, '
                f'mark_unobserved_box_ignore='
                f'{self.mark_unobserved_box_ignore}, '
                f'obstacle_min_component_voxels='
                f'{self.obstacle_min_component_voxels}, '
                f'obstacle_small_component_keep_min_points='
                f'{self.obstacle_small_component_keep_min_points}, '
                f'obstacle_thin_component_min_major_span='
                f'{self.obstacle_thin_component_min_major_span}, '
                f'obstacle_thin_component_max_minor_span='
                f'{self.obstacle_thin_component_max_minor_span}, '
                f'obstacle_thin_component_max_z_span='
                f'{self.obstacle_thin_component_max_z_span}, '
                f'obstacle_thin_component_keep_min_points='
                f'{self.obstacle_thin_component_keep_min_points}, '
                f'obstacle_box_ignore_margin='
                f'{self.obstacle_box_ignore_margin}, '
                f'ego_ignore_range='
                f'{None if self.ego_ignore_range is None else self.ego_ignore_range.tolist()}, '
                f'current_frame_only={self.current_frame_only})')


@TRANSFORMS.register_module()
class GenerateKLMultiFrameOccFromQueue(GenerateKLOccFromBoxes):
    """Generate current-frame OCC after BEVFormer queue assembly.

    This is a conservative first multi-frame target: current-frame points are
    still the only source of free-space rays and semantic object hits, while
    history frames contribute scene/background hit points after ego-motion
    alignment. Points inside annotated boxes in their own history frame are
    removed so old dynamic objects do not leak into the static scene target.
    """

    def __init__(self,
                 *args,
                 aggregate_history_scene: bool = True,
                 history_scene_only: bool = True,
                 aggregate_dynamic_instances: bool = False,
                 dynamic_instance_min_points: int = 1,
                 **kwargs) -> None:
        kwargs.setdefault('mode', 'raycast')
        super().__init__(*args, **kwargs)
        if self.mode != 'raycast':
            raise ValueError(
                'GenerateKLMultiFrameOccFromQueue only supports raycast '
                f'mode, got {self.mode}.')
        if dynamic_instance_min_points <= 0:
            raise ValueError('dynamic_instance_min_points must be positive, '
                             f'got {dynamic_instance_min_points}.')
        self.aggregate_history_scene = bool(aggregate_history_scene)
        self.history_scene_only = bool(history_scene_only)
        self.aggregate_dynamic_instances = bool(aggregate_dynamic_instances)
        self.dynamic_instance_min_points = int(dynamic_instance_min_points)

    @staticmethod
    def _matrix4x4(matrix) -> np.ndarray:
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape == (4, 4):
            return mat
        if mat.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :4] = mat
            return out
        raise ValueError(f'ego pose must have shape (4, 4) or (3, 4), '
                         f'got {mat.shape}.')

    @classmethod
    def _transform_ego_points(cls, points_xyz: np.ndarray, src_ego2global,
                              dst_ego2global) -> np.ndarray:
        if points_xyz.shape[0] == 0:
            return points_xyz
        src = cls._matrix4x4(src_ego2global)
        dst = cls._matrix4x4(dst_ego2global)
        src2dst = np.linalg.inv(dst) @ src
        homo = np.concatenate(
            [points_xyz.astype(np.float64),
             np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
            axis=1)
        return (homo @ src2dst.T)[:, :3].astype(np.float32)

    @staticmethod
    def _get_queue_meta(queue_metas: Optional[dict], index: int):
        if queue_metas is None:
            return None
        if index in queue_metas:
            return queue_metas[index]
        return queue_metas.get(str(index))

    @staticmethod
    def _extract_sample_boxes_labels(data_sample) -> Tuple[np.ndarray,
                                                           np.ndarray]:
        if data_sample is None or not hasattr(data_sample, 'gt_instances_3d'):
            return (np.zeros((0, 7), dtype=np.float32),
                    np.zeros((0, ), dtype=np.int64))

        instances = data_sample.gt_instances_3d
        boxes = getattr(instances, 'bboxes_3d', None)
        labels = getattr(instances, 'labels_3d', None)
        if boxes is None or labels is None:
            return (np.zeros((0, 7), dtype=np.float32),
                    np.zeros((0, ), dtype=np.int64))

        if hasattr(boxes, 'tensor'):
            box_tensor = boxes.tensor
        else:
            box_tensor = boxes
        if isinstance(box_tensor, torch.Tensor):
            box_tensor = box_tensor.detach().cpu().numpy()
        else:
            box_tensor = np.asarray(box_tensor)
        box_tensor = box_tensor.astype(np.float32, copy=False)
        if box_tensor.ndim == 1:
            box_tensor = box_tensor.reshape(1, -1)
        if box_tensor.shape[0] == 0:
            box_tensor = np.zeros((0, 7), dtype=np.float32)

        if isinstance(labels, torch.Tensor):
            label_array = labels.detach().cpu().numpy()
        else:
            label_array = np.asarray(labels)
        label_array = label_array.astype(np.int64, copy=False)

        if box_tensor.shape[0] != label_array.shape[0]:
            raise ValueError(
                f'gt_bboxes_3d N={box_tensor.shape[0]} != '
                f'gt_labels_3d N={label_array.shape[0]}; cannot generate '
                'aligned multi-frame occupancy labels.')
        return box_tensor, label_array

    @staticmethod
    def _extract_sample_track_ids(data_sample, expected_len: int) -> np.ndarray:
        default = np.full((expected_len, ), -1, dtype=np.int64)
        if data_sample is None or not hasattr(data_sample, 'gt_instances_3d'):
            return default

        instances = data_sample.gt_instances_3d
        track_ids = getattr(instances, 'track_ids_3d', None)
        if track_ids is None:
            return default
        if isinstance(track_ids, torch.Tensor):
            track_ids = track_ids.detach().cpu().numpy()
        else:
            track_ids = np.asarray(track_ids)
        track_ids = track_ids.astype(np.int64, copy=False).reshape(-1)
        if track_ids.shape[0] != expected_len:
            raise ValueError(
                f'gt_track_ids_3d N={track_ids.shape[0]} != '
                f'gt_bboxes_3d N={expected_len}; cannot align dynamic '
                'occupancy instances.')
        return track_ids

    @staticmethod
    def _points_to_box_local(points_xyz: np.ndarray,
                             box: np.ndarray) -> np.ndarray:
        yaw = box[6]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rel_x = points_xyz[:, 0] - box[0]
        rel_y = points_xyz[:, 1] - box[1]
        local_x = rel_x * cos_yaw + rel_y * sin_yaw
        local_y = -rel_x * sin_yaw + rel_y * cos_yaw
        local_z = points_xyz[:, 2] - box[2]
        return np.stack([local_x, local_y, local_z], axis=1).astype(
            np.float32)

    @staticmethod
    def _box_local_to_points(local_xyz: np.ndarray,
                             box: np.ndarray) -> np.ndarray:
        yaw = box[6]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        x = box[0] + local_xyz[:, 0] * cos_yaw - local_xyz[:, 1] * sin_yaw
        y = box[1] + local_xyz[:, 0] * sin_yaw + local_xyz[:, 1] * cos_yaw
        z = box[2] + local_xyz[:, 2]
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _collect_dynamic_instance_points(
            self, history_points: Sequence,
            history_samples: Sequence,
            box_tensor: np.ndarray,
            label_array: np.ndarray,
            track_ids: np.ndarray) -> Tuple[np.ndarray, int, int]:
        if (not self.aggregate_dynamic_instances or box_tensor.shape[0] == 0
                or history_points is None or len(history_points) == 0):
            return np.empty((0, 3), dtype=np.float32), 0, 0

        current_by_track: Dict[int, int] = {}
        duplicate_tracks = set()
        for idx, track_id in enumerate(track_ids.tolist()):
            if track_id < 0 or label_array[idx] < 0:
                continue
            if track_id in current_by_track:
                duplicate_tracks.add(track_id)
                continue
            current_by_track[track_id] = idx
        for track_id in duplicate_tracks:
            current_by_track.pop(track_id, None)
        if not current_by_track:
            return np.empty((0, 3), dtype=np.float32), 0, 0

        chunks = []
        matched_tracks = set()
        num_history_box_points = 0
        for history_index, points in enumerate(history_points):
            if history_index >= len(history_samples):
                continue
            hist_sample = history_samples[history_index]
            hist_boxes, hist_labels = self._extract_sample_boxes_labels(
                hist_sample)
            hist_track_ids = self._extract_sample_track_ids(
                hist_sample, hist_boxes.shape[0])
            hist_points = self._extract_points_xyz(points)
            if hist_points.shape[0] == 0:
                continue

            for hist_idx, track_id in enumerate(hist_track_ids.tolist()):
                cur_idx = current_by_track.get(track_id)
                if cur_idx is None:
                    continue
                if int(hist_labels[hist_idx]) != int(label_array[cur_idx]):
                    continue
                point_mask = self._points_in_box_mask(
                    hist_points, hist_boxes[hist_idx])
                if int(point_mask.sum()) < self.dynamic_instance_min_points:
                    continue
                hist_box_points = hist_points[point_mask]
                local_points = self._points_to_box_local(
                    hist_box_points, hist_boxes[hist_idx])
                current_points = self._box_local_to_points(
                    local_points, box_tensor[cur_idx])
                current_mask = self._points_in_box_mask(
                    current_points, box_tensor[cur_idx])
                current_points = current_points[current_mask]
                if current_points.shape[0] == 0:
                    continue
                num_history_box_points += int(hist_box_points.shape[0])
                matched_tracks.add(track_id)
                chunks.append(current_points)

        if not chunks:
            return np.empty((0, 3), dtype=np.float32), 0, 0
        return (np.concatenate(chunks, axis=0),
                int(len(matched_tracks)),
                int(num_history_box_points))

    def _valid_points_and_voxels(
            self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if points_xyz.shape[0] == 0:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.int64))
        voxels = self._coord_to_index_floor(points_xyz)
        valid = np.all((voxels >= 0) & (voxels < self.occ_size), axis=1)
        return points_xyz[valid], voxels[valid]

    def _build_multiframe_raycast_occ(
            self, current_points_xyz: np.ndarray,
            scene_points_xyz: np.ndarray,
            box_tensor: np.ndarray,
            label_array: np.ndarray,
            volumes: np.ndarray,
            semantic_points_xyz: Optional[np.ndarray] = None) -> Tuple[
                np.ndarray, dict]:
        current_points_xyz, current_point_voxels = (
            self._valid_points_and_voxels(current_points_xyz))
        if semantic_points_xyz is None:
            semantic_points_xyz = current_points_xyz
        semantic_points_xyz, _ = self._valid_points_and_voxels(
            semantic_points_xyz)
        scene_points_xyz, scene_point_voxels = self._valid_points_and_voxels(
            scene_points_xyz)

        occ = np.full(
            tuple(self.occ_size.tolist()), self.ignore_idx, dtype=np.uint8)

        if current_point_voxels.shape[0] > 0:
            current_hit_voxels = np.unique(current_point_voxels, axis=0)
        else:
            current_hit_voxels = np.empty((0, 3), dtype=np.int64)
        free_voxels = self._raycast_free_voxels(current_hit_voxels)
        if free_voxels.shape[0] > 0:
            occ[free_voxels[:, 0], free_voxels[:, 1],
                free_voxels[:, 2]] = self.empty_idx

        box_mask = self._box_interior_mask(box_tensor)
        if self.mark_unobserved_box_ignore:
            occ[box_mask] = self.ignore_idx

        semantic_chunks = []
        for idx in np.argsort(-volumes):
            label = int(label_array[idx])
            if label < 0:
                continue
            semantic_voxels = self._fill_points_in_box(
                occ, semantic_points_xyz, box_tensor[idx],
                label + self.label_offset)
            if semantic_voxels.shape[0] > 0:
                semantic_chunks.append(semantic_voxels)

        if semantic_chunks:
            semantic_voxels = np.unique(
                np.concatenate(semantic_chunks, axis=0), axis=0)
        else:
            semantic_voxels = np.empty((0, 3), dtype=np.int64)

        if self.label_scene and scene_point_voxels.shape[0] > 0:
            scene_hit_voxels = np.unique(scene_point_voxels, axis=0)
            hit_in_box = box_mask[
                scene_hit_voxels[:, 0], scene_hit_voxels[:, 1],
                scene_hit_voxels[:, 2]]
            scene_hit_voxels = scene_hit_voxels[~hit_in_box]
            point_in_box = box_mask[
                scene_point_voxels[:, 0], scene_point_voxels[:, 1],
                scene_point_voxels[:, 2]]
            scene_only_points_xyz = scene_points_xyz[~point_in_box]
            scene_only_point_voxels = scene_point_voxels[~point_in_box]
            ground_voxels, obstacle_voxels, raw_obstacle_voxels = (
                self._split_scene_ground_obstacle(
                    scene_only_points_xyz,
                    scene_only_point_voxels, scene_hit_voxels,
                    semantic_voxels))
            raw_obstacle_count = int(raw_obstacle_voxels.shape[0])
            box_margin_obstacle_count = 0
            if obstacle_voxels.shape[0] > 0:
                near_box = self._obstacle_near_box_mask(
                    obstacle_voxels, box_tensor)
                box_margin_obstacle_count = int(near_box.sum())
                obstacle_voxels = obstacle_voxels[~near_box]
                if obstacle_voxels.shape[0] > 0:
                    obstacle_voxels = self._filter_obstacle_voxels(
                        obstacle_voxels, scene_only_point_voxels)
            ignored_obstacle_mask = np.zeros(
                tuple(self.occ_size.tolist()), dtype=bool)
            if raw_obstacle_voxels.shape[0] > 0:
                ignored_obstacle_mask[
                    raw_obstacle_voxels[:, 0], raw_obstacle_voxels[:, 1],
                    raw_obstacle_voxels[:, 2]] = True
            if obstacle_voxels.shape[0] > 0:
                ignored_obstacle_mask[
                    obstacle_voxels[:, 0], obstacle_voxels[:, 1],
                    obstacle_voxels[:, 2]] = False
            ignored_obstacle_voxels = np.argwhere(ignored_obstacle_mask)
            if obstacle_voxels.shape[0] > 0:
                occ[obstacle_voxels[:, 0], obstacle_voxels[:, 1],
                    obstacle_voxels[:, 2]] = self.obstacle_label
            if ground_voxels.shape[0] > 0:
                occ[ground_voxels[:, 0], ground_voxels[:, 1],
                    ground_voxels[:, 2]] = self.ground_label
            if ignored_obstacle_voxels.shape[0] > 0:
                occ[ignored_obstacle_voxels[:, 0],
                    ignored_obstacle_voxels[:, 1],
                    ignored_obstacle_voxels[:, 2]] = self.ignore_idx
        else:
            scene_hit_voxels = np.empty((0, 3), dtype=np.int64)
            ground_voxels = np.empty((0, 3), dtype=np.int64)
            obstacle_voxels = np.empty((0, 3), dtype=np.int64)
            raw_obstacle_count = 0
            box_margin_obstacle_count = 0
            ignored_obstacle_voxels = np.empty((0, 3), dtype=np.int64)

        num_ego_ignore = self._apply_ego_ignore(occ)
        meta = dict(
            point_cloud_range=self.point_cloud_range.tolist(),
            occ_size=self.occ_size.tolist(),
            voxel_size=self.voxel_size.tolist(),
            source='queue_static_multiframe_points_raycast',
            mode=self.mode,
            free_idx=self.empty_idx,
            ignore_idx=self.ignore_idx,
            ground_label=self.ground_label,
            obstacle_label=self.obstacle_label,
            ray_origin=self.ray_origin.tolist(),
            ego_ignore_range=(
                None if self.ego_ignore_range is None else
                self.ego_ignore_range.tolist()),
            aggregate_history_scene=self.aggregate_history_scene,
            history_scene_only=self.history_scene_only,
            use_history_free_space=False,
            num_current_hit_voxels=int(current_hit_voxels.shape[0]),
            num_scene_hit_voxels=int(scene_hit_voxels.shape[0]),
            num_free_voxels=int(free_voxels.shape[0]),
            num_semantic_voxels=int(semantic_voxels.shape[0]),
            num_ground_voxels=int(ground_voxels.shape[0]),
            num_raw_obstacle_voxels=int(raw_obstacle_count),
            num_obstacle_voxels=int(obstacle_voxels.shape[0]),
            num_box_margin_obstacle_voxels=int(box_margin_obstacle_count),
            num_filtered_obstacle_voxels=int(
                ignored_obstacle_voxels.shape[0]),
            obstacle_min_points_per_voxel=(
                self.obstacle_min_points_per_voxel),
            obstacle_min_component_voxels=(
                self.obstacle_min_component_voxels),
            obstacle_small_component_keep_min_points=(
                self.obstacle_small_component_keep_min_points),
            obstacle_thin_component_min_major_span=(
                self.obstacle_thin_component_min_major_span),
            obstacle_thin_component_max_minor_span=(
                self.obstacle_thin_component_max_minor_span),
            obstacle_thin_component_max_z_span=(
                self.obstacle_thin_component_max_z_span),
            obstacle_thin_component_keep_min_points=(
                self.obstacle_thin_component_keep_min_points),
            obstacle_box_ignore_margin=self.obstacle_box_ignore_margin,
            num_ego_ignore_voxels=num_ego_ignore)
        return occ, meta

    def transform(self, results: dict) -> dict:
        if 'inputs' not in results or 'data_samples' not in results:
            raise KeyError('GenerateKLMultiFrameOccFromQueue expects the '
                           'packed queue sample with `inputs` and '
                           '`data_samples`.')
        inputs = results['inputs']
        data_sample = results['data_samples']
        if 'points' not in inputs:
            raise KeyError('GenerateKLMultiFrameOccFromQueue requires '
                           '`inputs["points"]`.')

        current_points = self._extract_points_xyz(inputs['points'])
        box_tensor, label_array = self._extract_sample_boxes_labels(data_sample)
        track_ids = self._extract_sample_track_ids(
            data_sample, box_tensor.shape[0])
        history_points = inputs.get('history_points', [])
        history_samples = results.get('_kl_history_data_samples', [])
        dynamic_points, num_dynamic_instances, num_dynamic_history_points = (
            self._collect_dynamic_instance_points(
                history_points, history_samples, box_tensor, label_array,
                track_ids))
        if dynamic_points.shape[0] > 0:
            semantic_points = np.concatenate(
                [current_points, dynamic_points], axis=0)
        else:
            semantic_points = current_points
        current_scene_mask = ~self._points_in_any_box_mask(
            current_points, box_tensor)
        scene_chunks = [current_points[current_scene_mask]]

        queue_metas = data_sample.metainfo.get('queue_metas', {})
        current_index = len(history_points)
        current_meta = self._get_queue_meta(queue_metas, current_index)
        current_pose = (
            np.eye(4, dtype=np.float64) if current_meta is None else
            current_meta.get('ego2global', np.eye(4, dtype=np.float64)))

        num_history_points = 0
        num_history_scene_points = 0
        if self.aggregate_history_scene:
            for history_index, points in enumerate(history_points):
                hist_points = self._extract_points_xyz(points)
                num_history_points += int(hist_points.shape[0])
                hist_sample = (
                    history_samples[history_index]
                    if history_index < len(history_samples) else None)
                hist_boxes, _ = self._extract_sample_boxes_labels(hist_sample)
                if self.history_scene_only:
                    hist_scene = hist_points[
                        ~self._points_in_any_box_mask(hist_points, hist_boxes)]
                else:
                    hist_scene = hist_points

                history_meta = self._get_queue_meta(queue_metas, history_index)
                if history_meta is not None:
                    hist_scene = self._transform_ego_points(
                        hist_scene,
                        history_meta.get(
                            'ego2global', np.eye(4, dtype=np.float64)),
                        current_pose)

                if box_tensor.shape[0] > 0:
                    hist_scene = hist_scene[
                        ~self._points_in_any_box_mask(hist_scene, box_tensor)]
                num_history_scene_points += int(hist_scene.shape[0])
                if hist_scene.shape[0] > 0:
                    scene_chunks.append(hist_scene)

        if scene_chunks:
            scene_points = np.concatenate(scene_chunks, axis=0)
        else:
            scene_points = np.empty((0, 3), dtype=np.float32)

        volumes = box_tensor[:, 3] * box_tensor[:, 4] * box_tensor[:, 5]
        occ, meta = self._build_multiframe_raycast_occ(
            current_points, scene_points, box_tensor, label_array, volumes,
            semantic_points)
        meta.update(
            current_frame_only=False,
            num_history_frames=int(len(history_points)),
            num_history_points=int(num_history_points),
            num_history_scene_points=int(num_history_scene_points),
            num_current_points=int(current_points.shape[0]),
            num_scene_points=int(scene_points.shape[0]),
            aggregate_dynamic_instances=self.aggregate_dynamic_instances,
            dynamic_instance_min_points=self.dynamic_instance_min_points,
            num_dynamic_instances=int(num_dynamic_instances),
            num_dynamic_history_points=int(num_dynamic_history_points),
            num_dynamic_points=int(dynamic_points.shape[0]))

        if not hasattr(data_sample, 'gt_pts_seg') or data_sample.gt_pts_seg is None:
            data_sample.gt_pts_seg = PointData()
        data_sample.gt_pts_seg.occ = torch.from_numpy(occ)
        data_sample.set_metainfo(dict(gt_occ_meta=meta))
        return results

    def __repr__(self) -> str:
        return (f'{type(self).__name__}('
                f'point_cloud_range={self.point_cloud_range.tolist()}, '
                f'occ_size={self.occ_size.tolist()}, '
                f'aggregate_history_scene={self.aggregate_history_scene}, '
                f'history_scene_only={self.history_scene_only}, '
                f'aggregate_dynamic_instances='
                f'{self.aggregate_dynamic_instances}, '
                f'obstacle_box_ignore_margin='
                f'{self.obstacle_box_ignore_margin})')


@TRANSFORMS.register_module()
class GenerateKLMapMask(BaseTransform):
    """Generate a local rasterized map target from the KL base map."""

    _map_cache: ClassVar[Dict[str, dict]] = {}
    _origin_cache: ClassVar[Dict[str, dict]] = {}

    def __init__(self,
                 map_file: str,
                 point_cloud_range,
                 mask_shape,
                 target: str = 'drivable',
                 map_origin: str | None = None) -> None:
        self.map_file = map_file
        self.map_origin = map_origin
        self.point_cloud_range = point_cloud_range
        self.mask_shape = mask_shape
        self.target = target

    @classmethod
    def _load_map_cached(cls, path: str) -> dict:
        key = str(Path(path).resolve())
        if key not in cls._map_cache:
            cls._map_cache[key] = load_kl_base_map(key)
        return cls._map_cache[key]

    @classmethod
    def _load_origin_cached(cls, path: str) -> dict:
        key = str(Path(path).resolve())
        if key not in cls._origin_cache:
            cls._origin_cache[key] = read_map_origin(key)
        return cls._origin_cache[key]

    def transform(self, results: dict) -> dict:
        ego2global = results.get('ego2global')
        if ego2global is None:
            raise KeyError('GenerateKLMapMask requires `ego2global` in '
                           'pipeline results.')

        map_data = self._load_map_cached(self.map_file)
        local_map = select_local_map_geometries(
            map_data=map_data,
            ego2global=ego2global,
            point_cloud_range=self.point_cloud_range)
        masks = rasterize_local_map(
            local_map=local_map,
            point_cloud_range=self.point_cloud_range,
            mask_shape=self.mask_shape)
        if self.target not in masks:
            raise KeyError(f'Unknown KL map target {self.target!r}. '
                           f'Available targets: {sorted(masks.keys())}')

        results['gt_seg_map'] = masks[self.target]
        results['kl_local_map'] = local_map
        results['kl_map_target'] = self.target
        if self.map_origin is not None:
            results['kl_map_origin'] = self._load_origin_cached(self.map_origin)
        return results
