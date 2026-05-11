"""Visualize the KL base map over local BEV point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.font_manager import FontProperties

from projects.KL8.map_utils import (
    load_kl_base_map,
    rasterize_local_map,
    read_map_origin,
    select_local_map_geometries,
)


ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT = FontProperties(fname=FONT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize KL base map overlay in local BEV.')
    parser.add_argument(
        '--ann-file',
        default='data/kl_8/kl_infos_train.pkl',
        help='Path to KL info pkl.')
    parser.add_argument(
        '--map-file',
        default='data/kl_8/map/base_map.txt',
        help='Path to the KL base map file.')
    parser.add_argument(
        '--map-origin',
        default='data/kl_8/map/map_origin.yaml',
        help='Path to the map origin YAML.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/kl_map_overlay_check',
        help='Output directory.')
    parser.add_argument(
        '--indices',
        nargs='*',
        type=int,
        default=None,
        help='Specific sample indices to visualize.')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='How many evenly spaced samples to visualize when indices are '
        'not specified.')
    return parser.parse_args()


def load_merged_lidar_xyz(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f'Unexpected merged lidar shape: {path}')
    points = points.reshape(-1, 5)
    return points[:, :3]


def resolve_lidar_path(data_root: Path, lidar_rel: str) -> Path:
    path = Path(lidar_rel)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            ROOT / path,
            data_root / path,
            data_root / 'v1.0-trainval' / 'samples' / path.name,
            data_root / 'v1.0-mini' / 'samples' / path.name,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'Cannot resolve lidar path {lidar_rel!r}. Tried: '
        + ', '.join(str(p) for p in candidates))


def choose_indices(total: int, num_samples: int) -> list[int]:
    if total <= 0:
        return []
    if num_samples <= 1:
        return [0]
    return sorted(set(np.linspace(0, total - 1, num=num_samples, dtype=int).tolist()))


def draw_sample(ax,
                points: np.ndarray,
                local_map: dict,
                point_cloud_range: list[float],
                title: str) -> None:
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    ax.set_facecolor('black')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#666666')
    ax.set_title(title, fontproperties=FONT, fontsize=18)

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    )
    local_points = points[mask]
    if len(local_points) > 0:
        ax.scatter(local_points[:, 0], local_points[:, 1],
                   s=0.12, c='white', alpha=0.28, linewidths=0,
                   rasterized=True)

    for road in local_map['roads']:
        poly = road['polygon']
        ax.fill(poly[:, 0], poly[:, 1], color='#2f72ff', alpha=0.18)

    for junction in local_map['junctions']:
        poly = junction['polygon']
        ax.fill(poly[:, 0], poly[:, 1], color='#ff8c3b', alpha=0.22)

    for lane in local_map['lanes']:
        lane_poly = lane['polygon']
        ax.fill(lane_poly[:, 0], lane_poly[:, 1], color='#fff176', alpha=0.10)
        pts = lane['centerline']
        ax.plot(pts[:, 0], pts[:, 1], color='#ffe24a', linewidth=1.0,
                alpha=0.95)
        left = lane['left_boundary']
        right = lane['right_boundary']
        ax.plot(left[:, 0], left[:, 1], color='#ffd54f', linewidth=0.7,
                alpha=0.55)
        ax.plot(right[:, 0], right[:, 1], color='#ffd54f', linewidth=0.7,
                alpha=0.55)


def draw_mask(ax, mask: np.ndarray, title: str) -> None:
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#666666')
    ax.set_title(title, fontproperties=FONT, fontsize=18)


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = (ROOT / args.ann_file).resolve().parent

    point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]

    with (ROOT / args.ann_file).open('rb') as f:
        infos = pickle.load(f)['data_list']
    indices = args.indices or choose_indices(len(infos), args.num_samples)

    map_data = load_kl_base_map(ROOT / args.map_file)
    map_origin = read_map_origin(ROOT / args.map_origin)

    summary = {
        'ann_file': args.ann_file,
        'map_file': args.map_file,
        'map_origin': map_origin,
        'map_counts': {
            'lanes': len(map_data['lanes']),
            'roads': len(map_data['roads']),
            'junctions': len(map_data['junctions']),
        },
        'samples': [],
    }

    panels = []
    for idx in indices:
        info = infos[idx]
        lidar_rel = info['lidar_points']['lidar_path']
        points = load_merged_lidar_xyz(resolve_lidar_path(data_root, lidar_rel))
        ego2global = info['ego2global']
        local_map = select_local_map_geometries(
            map_data=map_data,
            ego2global=ego2global,
            point_cloud_range=point_cloud_range)
        masks = rasterize_local_map(
            local_map=local_map,
            point_cloud_range=point_cloud_range,
            mask_shape=(512, 512))
        title = f'idx={idx}  scene={info["scene_token"]}'
        panels.append((title, points, local_map, masks))
        summary['samples'].append({
            'index': idx,
            'scene_token': info['scene_token'],
            'timestamp': float(info['timestamp']),
            'token': info['token'],
            'visible_map_counts': {
                'lanes': len(local_map['lanes']),
                'roads': len(local_map['roads']),
                'junctions': len(local_map['junctions']),
            },
            'mask_pixels': {
                key: int(mask.sum()) for key, mask in masks.items()
            },
        })

    for i, (title, points, local_map, masks) in enumerate(panels):
        fig, ax = plt.subplots(figsize=(8.0, 6.8), dpi=180)
        draw_sample(ax, points, local_map, point_cloud_range, title)
        fig.tight_layout()
        fig.savefig(out_dir / f'sample_{i:02d}_idx_{indices[i]}.png',
                    bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=180)
        draw_mask(ax, masks['drivable'],
                  f'idx={indices[i]} drivable mask')
        fig.tight_layout()
        fig.savefig(out_dir / f'sample_{i:02d}_idx_{indices[i]}_drivable.png',
                    bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, len(panels), figsize=(8.0 * len(panels), 6.8),
                             dpi=180)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, points, local_map, _) in zip(axes, panels):
        draw_sample(ax, points, local_map, point_cloud_range, title)
    fig.suptitle('KL 地图与局部 BEV 点云叠加检查', fontproperties=FONT,
                 fontsize=24, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / 'kl_map_overlay_triptych.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.2),
                             dpi=180)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, _, _, masks) in zip(axes, panels):
        draw_mask(ax, masks['drivable'], title)
    fig.suptitle('KL 局部 Drivable Mask 检查', fontproperties=FONT,
                 fontsize=24, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / 'kl_map_drivable_triptych.png', bbox_inches='tight')
    plt.close(fig)

    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
