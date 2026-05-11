"""Utilities for parsing and rasterizing the KL base map."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import yaml


def read_map_origin(path: str | Path) -> dict:
    """Load the map origin metadata YAML."""
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _find_blocks(text: str, entity: str) -> List[str]:
    """Extract top-level ``entity { ... }`` blocks from a textproto-like file."""
    lines = text.splitlines()
    blocks: List[str] = []
    start = None
    depth = 0
    token = f'{entity} {{'
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start is None and stripped.startswith(token):
            start = i
            depth = line.count('{') - line.count('}')
            continue
        if start is not None:
            depth += line.count('{') - line.count('}')
            if depth <= 0:
                blocks.append('\n'.join(lines[start:i + 1]))
                start = None
                depth = 0
    return blocks


def _parse_points(block: str) -> np.ndarray:
    """Parse ``point { x: ... y: ... }`` entries from a text block."""
    matches = re.findall(
        r'point\s*\{\s*x:\s*([-+0-9.eE]+)\s*y:\s*([-+0-9.eE]+)',
        block,
        flags=re.S)
    if not matches:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.asarray(matches, dtype=np.float64)
    return pts.reshape(-1, 2)


def _parse_id(block: str) -> str:
    match = re.search(r'id:\s*"([^"]+)"', block)
    return match.group(1) if match else ''


def _close_polygon(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    if np.allclose(points[0], points[-1]):
        return points
    return np.concatenate([points, points[:1]], axis=0)


def _dedupe_consecutive(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.abs(points[1:] - points[:-1]) > 1e-6, axis=1)
    return points[keep]


def _make_ribbon_polygon(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _dedupe_consecutive(left)
    right = _dedupe_consecutive(right)
    if len(left) < 2 or len(right) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    polygon = np.concatenate([left, right[::-1]], axis=0)
    polygon = _close_polygon(_dedupe_consecutive(polygon))
    if len(polygon) < 4:
        return np.zeros((0, 2), dtype=np.float64)
    return polygon


def load_kl_base_map(path: str | Path) -> Dict[str, List[dict]]:
    """Parse the KL base map file into lane/road/junction geometry."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')

    lanes = []
    for block in _find_blocks(text, 'lane'):
        central_blocks = _find_blocks(block, 'central_curve')
        left_blocks = _find_blocks(block, 'left_boundary')
        right_blocks = _find_blocks(block, 'right_boundary')
        if not central_blocks or not left_blocks or not right_blocks:
            continue
        centerline = _dedupe_consecutive(_parse_points(central_blocks[0]))
        left_boundary = _dedupe_consecutive(_parse_points(left_blocks[0]))
        right_boundary = _dedupe_consecutive(_parse_points(right_blocks[0]))
        polygon = _make_ribbon_polygon(left_boundary, right_boundary)
        if len(centerline) < 2 or len(polygon) < 4:
            continue
        lanes.append(
            dict(
                id=_parse_id(block),
                centerline=centerline,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                polygon=polygon))

    junctions = []
    for block in _find_blocks(text, 'junction'):
        polygon_blocks = _find_blocks(block, 'polygon')
        if not polygon_blocks:
            continue
        points = _close_polygon(_dedupe_consecutive(_parse_points(polygon_blocks[0])))
        if len(points) < 4:
            continue
        junctions.append(dict(id=_parse_id(block), polygon=points))

    roads = []
    for block in _find_blocks(text, 'road'):
        section_blocks = _find_blocks(block, 'section')
        if not section_blocks:
            continue
        outer_blocks = _find_blocks(section_blocks[0], 'outer_polygon')
        if not outer_blocks:
            continue
        edge_blocks = _find_blocks(outer_blocks[0], 'edge')
        edges = []
        for edge_block in edge_blocks:
            edge_points = _dedupe_consecutive(_parse_points(edge_block))
            if len(edge_points) >= 2:
                edges.append(edge_points)
        if len(edges) < 2:
            continue
        polygon = np.concatenate([edges[0], edges[1][::-1]], axis=0)
        polygon = _close_polygon(_dedupe_consecutive(polygon))
        if len(polygon) < 4:
            continue
        roads.append(dict(id=_parse_id(block), polygon=polygon))

    return dict(
        lanes=lanes,
        roads=roads,
        junctions=junctions,
    )


def transform_global_points_to_ego(points_xy: np.ndarray,
                                   ego2global: Sequence[Sequence[float]]
                                   ) -> np.ndarray:
    """Transform 2D global XY points into the current ego frame."""
    if len(points_xy) == 0:
        return points_xy.reshape(0, 2)
    ego2global = np.asarray(ego2global, dtype=np.float64)
    global2ego = np.linalg.inv(ego2global)
    zeros = np.zeros((len(points_xy), 1), dtype=np.float64)
    ones = np.ones((len(points_xy), 1), dtype=np.float64)
    homo = np.concatenate([points_xy, zeros, ones], axis=1)
    ego = homo @ global2ego.T
    return ego[:, :2]


def geometry_intersects_range(points_xy: np.ndarray,
                              point_cloud_range: Sequence[float],
                              margin: float = 5.0) -> bool:
    """Check whether the geometry intersects a local BEV crop."""
    if len(points_xy) == 0:
        return False
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    return not (
        points_xy[:, 0].max() < x_min - margin or
        points_xy[:, 0].min() > x_max + margin or
        points_xy[:, 1].max() < y_min - margin or
        points_xy[:, 1].min() > y_max + margin
    )


def select_local_map_geometries(map_data: Dict[str, List[dict]],
                                ego2global: Sequence[Sequence[float]],
                                point_cloud_range: Sequence[float]) -> dict:
    """Transform map geometries into ego frame and keep local ones only."""
    out = dict(lanes=[], roads=[], junctions=[])

    for lane in map_data['lanes']:
        centerline = transform_global_points_to_ego(lane['centerline'],
                                                    ego2global)
        polygon = transform_global_points_to_ego(lane['polygon'], ego2global)
        if geometry_intersects_range(centerline, point_cloud_range) or (
                geometry_intersects_range(polygon, point_cloud_range)):
            out['lanes'].append(
                dict(
                    id=lane['id'],
                    centerline=centerline,
                    polygon=polygon,
                    left_boundary=transform_global_points_to_ego(
                        lane['left_boundary'], ego2global),
                    right_boundary=transform_global_points_to_ego(
                        lane['right_boundary'], ego2global)))

    for road in map_data['roads']:
        pts = transform_global_points_to_ego(road['polygon'], ego2global)
        if geometry_intersects_range(pts, point_cloud_range):
            out['roads'].append(dict(id=road['id'], polygon=pts))

    for junction in map_data['junctions']:
        pts = transform_global_points_to_ego(junction['polygon'], ego2global)
        if geometry_intersects_range(pts, point_cloud_range):
            out['junctions'].append(dict(id=junction['id'], polygon=pts))

    return out


def _ego_points_to_mask(points_xy: np.ndarray,
                        point_cloud_range: Sequence[float],
                        mask_shape: Sequence[int]) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    h, w = int(mask_shape[0]), int(mask_shape[1])
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    scale_x = (w - 1) / max(x_max - x_min, 1e-6)
    scale_y = (h - 1) / max(y_max - y_min, 1e-6)
    cols = (points_xy[:, 0] - x_min) * scale_x
    rows = (y_max - points_xy[:, 1]) * scale_y
    coords = np.stack([cols, rows], axis=1)
    return np.round(coords).astype(np.int32)


def rasterize_polygon_mask(polygons: Sequence[np.ndarray],
                           point_cloud_range: Sequence[float],
                           mask_shape: Sequence[int]) -> np.ndarray:
    mask = np.zeros(tuple(mask_shape), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 4:
            continue
        poly_ij = _ego_points_to_mask(polygon, point_cloud_range, mask_shape)
        cv2.fillPoly(mask, [poly_ij], color=1)
    return mask


def rasterize_local_map(local_map: dict,
                        point_cloud_range: Sequence[float],
                        mask_shape: Sequence[int]) -> dict:
    road_mask = rasterize_polygon_mask(
        [road['polygon'] for road in local_map['roads']],
        point_cloud_range=point_cloud_range,
        mask_shape=mask_shape)
    junction_mask = rasterize_polygon_mask(
        [junction['polygon'] for junction in local_map['junctions']],
        point_cloud_range=point_cloud_range,
        mask_shape=mask_shape)
    lane_mask = rasterize_polygon_mask(
        [lane['polygon'] for lane in local_map['lanes']],
        point_cloud_range=point_cloud_range,
        mask_shape=mask_shape)
    drivable_mask = np.clip(road_mask + junction_mask, 0, 1).astype(np.uint8)
    return dict(
        road=road_mask,
        junction=junction_mask,
        lane=lane_mask,
        drivable=drivable_mask)
