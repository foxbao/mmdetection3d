#!/usr/bin/env python
"""Visualize UniAD-style prev-BEV rotation and shifted-reference alignment.

This tool uses raw queued point clouds to build BEV occupancy maps, then
focuses on two useful checks for the immediate previous frame:

1. Direct point transform with ``ego_motion_delta``.
2. The current UniAD-like split: rotate ``prev_bev`` first, then sample it with
   the shifted reference offset used by temporal self-attention.

The direct point transform is the main visual reference because it is closest
to the metadata semantics. A legacy full-BEV warp comparison is still available
with ``--with-full-warp`` when you want an extra implementation sanity check.

Example:
    python tools/analysis_tools/visualize_uniad_bev_alignment.py \
        --config projects/BEVFormer/configs/bevformer_lidar_kl_uniad_det.py \
        --split val \
        --index 0 \
        --num-samples 6 \
        --index-stride 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules
from projects.BEVFormer.bevformer.modules.lidar_perception_transformer import (
    PerceptionTransformer,
)
from projects.BEVFormer.bevformer.modules import warp_prev_bev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize UniAD-like BEV rotate+shift alignment.')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument(
        '--split',
        default='val',
        choices=['train', 'val', 'test'],
        help='which dataloader config to inspect')
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='dataset index after queue filtering')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='number of samples to visualize starting from --index')
    parser.add_argument(
        '--index-stride',
        type=int,
        default=1,
        help='dataset index stride when --num-samples > 1')
    parser.add_argument(
        '--indices',
        default=None,
        help='comma-separated dataset indices; overrides --index/--num-samples')
    parser.add_argument(
        '--token',
        default=None,
        help='sample token to inspect; overrides --index')
    parser.add_argument(
        '--out-dir',
        default='visualizations/uniad_bev_alignment',
        help='directory to save figures and metrics')
    parser.add_argument(
        '--point-stride',
        type=int,
        default=8,
        help='subsample factor for point scatter plotting')
    parser.add_argument(
        '--occupancy-thr',
        type=float,
        default=0.25,
        help='threshold applied after bilinear sampling for occupancy IoU')
    parser.add_argument(
        '--with-full-warp',
        action='store_true',
        help='also compute legacy full-BEV warp metrics for t-1')
    return parser.parse_args()


def import_cfg_modules(cfg: Config) -> None:
    register_all_modules(init_default_scope=True)
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)


def get_dataset_cfg(cfg: Config, split: str):
    if split == 'train':
        return cfg.train_dataloader.dataset
    if split == 'val':
        return cfg.val_dataloader.dataset
    return cfg.test_dataloader.dataset


def points_to_numpy(points) -> np.ndarray:
    tensor = points.tensor if hasattr(points, 'tensor') else points
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz1 = np.concatenate(
        [points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1)
    out = (transform.astype(np.float32) @ xyz1.T).T
    aligned = points.copy()
    aligned[:, :3] = out[:, :3]
    return aligned


def compose_queue_delta(queue_metas: dict, start_idx: int,
                        end_idx: int) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    for step in range(start_idx + 1, end_idx + 1):
        transform = (
            np.asarray(queue_metas[step]['ego_motion_delta'], dtype=np.float64)
            @ transform)
    return transform


def direct_queue_delta(queue_metas: dict, start_idx: int,
                       end_idx: int) -> np.ndarray:
    start_ego2global = np.asarray(
        queue_metas[start_idx]['ego2global'], dtype=np.float64)
    end_ego2global = np.asarray(
        queue_metas[end_idx]['ego2global'], dtype=np.float64)
    return np.linalg.inv(end_ego2global) @ start_ego2global


def get_bev_shape(cfg: Config) -> Tuple[int, int]:
    head = cfg.model.get('pts_bbox_head', {})
    if 'bev_h' in head and 'bev_w' in head:
        return int(head.bev_h), int(head.bev_w)
    raise KeyError('Cannot infer BEV shape. Please use a config whose '
                   'model.pts_bbox_head defines bev_h and bev_w.')


def rasterize_points_yx(points: np.ndarray,
                        point_cloud_range: Sequence[float],
                        bev_h: int,
                        bev_w: int) -> np.ndarray:
    occ = np.zeros((bev_h, bev_w), dtype=np.float32)
    x_min, y_min = float(point_cloud_range[0]), float(point_cloud_range[1])
    x_max, y_max = float(point_cloud_range[3]), float(point_cloud_range[4])
    x_extent = max(x_max - x_min, 1e-6)
    y_extent = max(y_max - y_min, 1e-6)

    ix = np.floor((points[:, 0] - x_min) / x_extent * bev_w).astype(np.int64)
    iy = np.floor((points[:, 1] - y_min) / y_extent * bev_h).astype(np.int64)
    valid = ((ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h))
    np.add.at(occ, (iy[valid], ix[valid]), 1.0)
    return np.clip(occ, 0.0, 1.0)


def sample_with_shift(bev_yx: torch.Tensor,
                      shift_xy: torch.Tensor,
                      point_cloud_range: Sequence[float]) -> torch.Tensor:
    """Sample ``bev_yx`` at reference points shifted in normalized [0, 1]."""
    batch_size, _, bev_h, bev_w = bev_yx.shape
    x_min, y_min = float(point_cloud_range[0]), float(point_cloud_range[1])
    x_max, y_max = float(point_cloud_range[3]), float(point_cloud_range[4])
    x_extent = max(x_max - x_min, 1e-6)
    y_extent = max(y_max - y_min, 1e-6)

    xs = torch.linspace(
        x_min + x_extent / (2 * bev_w),
        x_max - x_extent / (2 * bev_w),
        bev_w,
        device=bev_yx.device,
        dtype=bev_yx.dtype)
    ys = torch.linspace(
        y_min + y_extent / (2 * bev_h),
        y_max - y_extent / (2 * bev_h),
        bev_h,
        device=bev_yx.device,
        dtype=bev_yx.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    src_x = grid_x.unsqueeze(0) + shift_xy[:, 0, None, None] * x_extent
    src_y = grid_y.unsqueeze(0) + shift_xy[:, 1, None, None] * y_extent
    norm_x = 2 * (src_x - x_min) / x_extent - 1
    norm_y = 2 * (src_y - y_min) / y_extent - 1
    sample_grid = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(
        bev_yx,
        sample_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)


def rotate_then_shift_bev(prev_occ_yx: np.ndarray,
                          current_meta: dict,
                          point_cloud_range: Sequence[float]) -> np.ndarray:
    prev = torch.from_numpy(prev_occ_yx)[None, None].to(torch.float32)
    transformer = object.__new__(PerceptionTransformer)
    transformer.use_shift = True
    transformer.rotate_prev_bev = True
    transformer.point_cloud_range = point_cloud_range
    queue_meta = [current_meta]

    rotated = transformer.rotate_prev_bev_if_needed(prev, queue_meta)
    shift = transformer.shift_from_queue_meta(
        queue_meta, prev.device, prev.dtype)
    shifted = sample_with_shift(rotated, shift, point_cloud_range)
    return shifted[0, 0].cpu().numpy()


def full_warp_bev(prev_occ_yx: np.ndarray,
                  current_meta: dict,
                  point_cloud_range: Sequence[float]) -> np.ndarray:
    prev = torch.from_numpy(prev_occ_yx)[None, None].to(torch.float32)
    delta = np.asarray(current_meta['ego_motion_delta'], dtype=np.float32)
    warped = warp_prev_bev(prev, delta, point_cloud_range)
    return warped[0, 0].cpu().numpy()


def occ_iou(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    a_bin = a > thr
    b_bin = b > thr
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_bin, b_bin).sum() / union)


def occ_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def occ_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def overlay_image(base: np.ndarray,
                  other: np.ndarray,
                  color_base=(0.2, 0.9, 0.2),
                  color_other=(0.95, 0.2, 0.2)) -> np.ndarray:
    img = np.zeros((*base.shape, 3), dtype=np.float32)
    img[base > 0] += np.asarray(color_base, dtype=np.float32)
    img[other > 0] += np.asarray(color_other, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def draw_occ_overlay(ax,
                     base: np.ndarray,
                     other: np.ndarray,
                     point_cloud_range: Sequence[float],
                     title: str,
                     color_base=(0.2, 0.9, 0.2),
                     color_other=(0.95, 0.2, 0.2)) -> None:
    ax.imshow(
        overlay_image(base, other, color_base, color_other),
        origin='lower',
        extent=[
            point_cloud_range[0], point_cloud_range[3],
            point_cloud_range[1], point_cloud_range[4]
        ])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def draw_point_overlay(ax,
                       current_points: np.ndarray,
                       prev_points: np.ndarray,
                       point_cloud_range: Sequence[float],
                       point_stride: int,
                       title: str) -> None:
    cur = current_points[::max(point_stride, 1)]
    prev = prev_points[::max(point_stride, 1)]
    ax.scatter(prev[:, 0], prev[:, 1], s=0.2, c='#377eb8', alpha=0.55,
               label='prev')
    ax.scatter(cur[:, 0], cur[:, 1], s=0.2, c='#4daf4a', alpha=0.55,
               label='current')
    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)


def draw_queue_point_overlay(ax,
                             frames: Sequence[np.ndarray],
                             labels: Sequence[str],
                             colors: Sequence[str],
                             point_cloud_range: Sequence[float],
                             point_stride: int,
                             title: str) -> None:
    for points, label, color in zip(frames, labels, colors):
        if points.shape[0] == 0:
            continue
        sub = points[::max(point_stride, 1)]
        ax.scatter(
            sub[:, 0], sub[:, 1], s=0.2, c=color, alpha=0.55, label=label)
    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)


def overlay_multi_occ(occs: Sequence[np.ndarray],
                      colors: Sequence[Sequence[float]]) -> np.ndarray:
    if len(occs) == 0:
        raise ValueError('Need at least one occupancy map.')
    img = np.zeros((*occs[0].shape, 3), dtype=np.float32)
    for occ, color in zip(occs, colors):
        img[occ > 0] += np.asarray(color, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def draw_multi_occ_overlay(ax,
                           occs: Sequence[np.ndarray],
                           colors: Sequence[Sequence[float]],
                           point_cloud_range: Sequence[float],
                           title: str) -> None:
    ax.imshow(
        overlay_multi_occ(occs, colors),
        origin='lower',
        extent=[
            point_cloud_range[0], point_cloud_range[3],
            point_cloud_range[1], point_cloud_range[4]
        ])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def rotate_shift_to_current(prev_occ_yx: np.ndarray,
                            queue_metas: dict,
                            start_step: int,
                            current_step: int,
                            point_cloud_range: Sequence[float]) -> np.ndarray:
    aligned = prev_occ_yx
    for step in range(start_step + 1, current_step + 1):
        aligned = rotate_then_shift_bev(
            aligned, queue_metas[step], point_cloud_range)
    return aligned


def yaw_deg(delta: np.ndarray) -> float:
    return math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))


def resolve_sample_index(dataset, token: Optional[str], index: int) -> int:
    if token is None:
        return index
    dataset.full_init()
    raw_idx = dataset.token2index[token]
    if dataset.valid_data_indices is None:
        return raw_idx
    return dataset.valid_data_indices.index(raw_idx)


def parse_indices(indices: str) -> List[int]:
    parsed = []
    for item in indices.split(','):
        item = item.strip()
        if item:
            parsed.append(int(item))
    if not parsed:
        raise ValueError('--indices was provided but no valid index was found.')
    return parsed


def selected_sample_indices(dataset, args: argparse.Namespace) -> List[int]:
    if args.token is not None:
        return [resolve_sample_index(dataset, args.token, args.index)]
    if args.indices is not None:
        return parse_indices(args.indices)
    if args.num_samples < 1:
        raise ValueError('--num-samples must be >= 1.')
    if args.index_stride < 1:
        raise ValueError('--index-stride must be >= 1.')
    return [
        args.index + sample_offset * args.index_stride
        for sample_offset in range(args.num_samples)
    ]


def visualize_sample(args: argparse.Namespace, cfg: Config, dataset,
                     sample_idx: int, out_dir: Path) -> dict:
    sample = dataset[sample_idx]

    point_cloud_range = cfg.model.point_cloud_range
    bev_h, bev_w = get_bev_shape(cfg)
    queue_metas = sample['data_samples'].metainfo['queue_metas']
    queue_keys = sorted(queue_metas.keys())
    if len(queue_keys) < 2:
        raise RuntimeError('Need at least two frames in the BEV queue.')

    current_step = queue_keys[-1]
    current_meta = queue_metas[current_step]
    immediate_delta = np.asarray(
        current_meta['ego_motion_delta'], dtype=np.float64)

    current_points = points_to_numpy(sample['inputs']['points'])
    history_points = [
        points_to_numpy(points)
        for points in sample['inputs'].get('history_points', [])
    ]
    queue_points = history_points + [current_points]
    if len(queue_points) != len(queue_keys):
        raise RuntimeError('queue_points and queue_metas length mismatch: '
                           f'{len(queue_points)} vs {len(queue_keys)}.')

    queue_labels = []
    for step in queue_keys:
        lag = current_step - step
        queue_labels.append('t' if lag == 0 else f't-{lag}')

    raw_occs = []
    direct_occs = []
    split_occs = []
    aligned_frames = []
    compose_errors = {}
    for step, points in zip(queue_keys, queue_points):
        raw_occ = rasterize_points_yx(
            points, point_cloud_range, bev_h, bev_w)
        if step == current_step:
            aligned_points = points
            direct_occ = raw_occ
            split_occ = raw_occ
        else:
            direct_delta = direct_queue_delta(queue_metas, step, current_step)
            chained_delta = compose_queue_delta(queue_metas, step,
                                                current_step)
            compose_errors[str(step)] = float(
                np.max(np.abs(chained_delta - direct_delta)))
            aligned_points = apply_transform(points, direct_delta)
            direct_occ = rasterize_points_yx(
                aligned_points, point_cloud_range, bev_h, bev_w)
            split_occ = rotate_shift_to_current(
                raw_occ, queue_metas, step, current_step, point_cloud_range)
        raw_occs.append(raw_occ)
        direct_occs.append(direct_occ)
        split_occs.append(split_occ)
        aligned_frames.append(aligned_points)

    current_occ = direct_occs[-1]
    history_checks = []
    for step, label, direct_occ, split_occ in zip(
            queue_keys[:-1], queue_labels[:-1], direct_occs[:-1],
            split_occs[:-1]):
        history_checks.append({
            'step': int(step),
            'label': label,
            'token': queue_metas[step].get('token'),
            'direct_points_vs_rotate_shift_iou': occ_iou(
                direct_occ, split_occ, args.occupancy_thr),
            'direct_points_vs_rotate_shift_l1': occ_l1(direct_occ,
                                                       split_occ),
            'direct_points_vs_rotate_shift_max_abs': occ_max_abs(
                direct_occ, split_occ),
            'current_vs_direct_iou': occ_iou(
                current_occ, direct_occ, args.occupancy_thr),
            'current_vs_rotate_shift_iou': occ_iou(
                current_occ, split_occ, args.occupancy_thr),
        })

    immediate_check = history_checks[-1]

    metrics = {
        'config': args.config,
        'split': args.split,
        'sample_index': int(sample_idx),
        'current_token': current_meta.get('token'),
        'queue_tokens': [
            queue_metas[step].get('token') for step in queue_keys
        ],
        'queue_labels': queue_labels,
        'bev_h': int(bev_h),
        'bev_w': int(bev_w),
        'time_delta': float(current_meta.get('time_delta', 0.0)),
        'delta_translation_xy': [
            float(immediate_delta[0, 3]),
            float(immediate_delta[1, 3]),
        ],
        'delta_yaw_deg': yaw_deg(immediate_delta),
        'compose_to_current_max_abs': compose_errors,
        'history_checks': history_checks,
        'direct_points_vs_rotate_shift_iou':
        immediate_check['direct_points_vs_rotate_shift_iou'],
        'direct_points_vs_rotate_shift_l1':
        immediate_check['direct_points_vs_rotate_shift_l1'],
        'direct_points_vs_rotate_shift_max_abs':
        immediate_check['direct_points_vs_rotate_shift_max_abs'],
        'current_vs_direct_prev_iou': immediate_check['current_vs_direct_iou'],
        'current_vs_rotate_shift_prev_iou':
        immediate_check['current_vs_rotate_shift_iou'],
    }
    if args.with_full_warp:
        prev_full_warp_occ = full_warp_bev(
            raw_occs[-2], current_meta, point_cloud_range)
        metrics.update({
            'full_warp_vs_rotate_shift_iou': occ_iou(
                prev_full_warp_occ, split_occs[-2],
                args.occupancy_thr),
            'full_warp_vs_rotate_shift_l1': occ_l1(
                prev_full_warp_occ, split_occs[-2]),
            'full_warp_vs_rotate_shift_max_abs': occ_max_abs(
                prev_full_warp_occ, split_occs[-2]),
            'direct_points_vs_full_warp_iou': occ_iou(
                direct_occs[-2], prev_full_warp_occ, args.occupancy_thr),
            'current_vs_full_warp_prev_iou': occ_iou(
                current_occ, prev_full_warp_occ, args.occupancy_thr),
        })

    stem = f'{args.split}_{sample_idx}'
    fig_path = out_dir / f'{stem}_bev_alignment.png'
    metrics_path = out_dir / f'{stem}_metrics.json'
    metrics['figure'] = str(fig_path)
    metrics['metrics_path'] = str(metrics_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=170)
    point_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3']
    point_colors = point_colors[:len(queue_points)]
    occ_colors = [
        (0.2, 0.45, 1.0),
        (1.0, 0.50, 0.05),
        (0.2, 0.9, 0.2),
        (0.7, 0.3, 1.0),
    ]
    occ_colors = occ_colors[:len(queue_points)]

    draw_queue_point_overlay(
        axes[0, 0], queue_points, queue_labels, point_colors,
        point_cloud_range, args.point_stride,
        'Raw queue points\n(each frame in its own ego frame)')
    draw_queue_point_overlay(
        axes[0, 1], aligned_frames, queue_labels, point_colors,
        point_cloud_range, args.point_stride,
        'Queue points transformed into current ego frame')
    draw_multi_occ_overlay(
        axes[0, 2], direct_occs, occ_colors, point_cloud_range,
        'Direct-aligned occupancy queue\n'
        f'{", ".join(queue_labels)}')

    history_to_plot = history_checks[-2:]
    history_start_col = 0
    for ax, check in zip(axes[1, :2], history_to_plot):
        step = check['step']
        draw_occ_overlay(
            ax, direct_occs[queue_keys.index(step)],
            split_occs[queue_keys.index(step)], point_cloud_range,
            f'{check["label"]}: direct-aligned vs rotate+shift\n'
            f'IoU={check["direct_points_vs_rotate_shift_iou"]:.4f}',
            color_base=(0.2, 0.45, 1.0),
            color_other=(0.95, 0.2, 0.2))
        history_start_col += 1
    for ax in axes[1, history_start_col:2]:
        ax.axis('off')

    text_ax = axes[1, 2]

    text_lines = [
        f'config: {Path(args.config).name}',
        f'split/index: {args.split}/{sample_idx}',
        f'current token: {metrics["current_token"]}',
        'queue:',
        *[
            f'  {label}: {token}'
            for label, token in zip(metrics['queue_labels'],
                                    metrics['queue_tokens'])
        ],
        f'BEV: {bev_h} x {bev_w}',
        '',
        f'dt: {metrics["time_delta"]:.3f}s',
        'delta translation xy: '
        f'{metrics["delta_translation_xy"][0]:.3f}, '
        f'{metrics["delta_translation_xy"][1]:.3f} m',
        f'delta yaw: {metrics["delta_yaw_deg"]:.3f} deg',
        'compose errors:',
        *[
            f'  step {step}: {err:.3e}'
            for step, err in metrics['compose_to_current_max_abs'].items()
        ],
        '',
        'Key checks:',
        *[
            f'  {check["label"]}: direct/split IoU '
            f'{check["direct_points_vs_rotate_shift_iou"]:.4f}, '
            f'current/split {check["current_vs_rotate_shift_iou"]:.4f}'
            for check in metrics['history_checks']
        ],
    ]
    if args.with_full_warp:
        text_lines.extend([
            '',
            'optional full-warp check',
            f'  full vs split IoU: '
            f'{metrics["full_warp_vs_rotate_shift_iou"]:.6f}',
            f'  full vs split L1:  '
            f'{metrics["full_warp_vs_rotate_shift_l1"]:.8f}',
            f'  current full warp: '
            f'{metrics["current_vs_full_warp_prev_iou"]:.4f}',
        ])
    text_lines.extend(['', f'metrics json: {metrics_path.name}'])

    text_ax.axis('off')
    text_ax.text(
        0.02,
        0.98,
        '\n'.join(text_lines),
        va='top',
        ha='left',
        family='monospace',
        fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f'Saved figure: {fig_path}')
    print(f'Saved metrics: {metrics_path}')
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_cfg_modules(cfg)

    dataset = DATASETS.build(get_dataset_cfg(cfg, args.split))
    sample_indices = selected_sample_indices(dataset, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    for sample_idx in sample_indices:
        metrics_list.append(visualize_sample(args, cfg, dataset, sample_idx,
                                             out_dir))

    summary = {
        'config': args.config,
        'split': args.split,
        'sample_indices': sample_indices,
        'num_samples': len(metrics_list),
        'items': metrics_list,
    }
    summary_path = out_dir / 'summary_metrics.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Saved summary: {summary_path}')
    print('index  yaw_deg  trans_m  direct_vs_split  current_direct  '
          'current_split')
    for item in metrics_list:
        trans_xy = item['delta_translation_xy']
        trans_m = math.hypot(trans_xy[0], trans_xy[1])
        print(f"{item['sample_index']:5d}  "
              f"{item['delta_yaw_deg']:7.3f}  "
              f"{trans_m:7.3f}  "
              f"{item['direct_points_vs_rotate_shift_iou']:15.4f}  "
              f"{item['current_vs_direct_prev_iou']:14.4f}  "
              f"{item['current_vs_rotate_shift_prev_iou']:13.4f}")


if __name__ == '__main__':
    main()
