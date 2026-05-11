#!/usr/bin/env python
"""Audit BEVFormer temporal warping and BEV layout adaptation.

This tool visualizes and measures whether the current temporal path is
consistent:

1. The queue metadata should compose correctly:
   ``ego_motion_delta`` chained across the queue should match the direct
   transform from a historical frame into the current frame.
2. ``warp_prev_bev()`` should agree with explicit point-cloud alignment when
   we rasterize points into BEV occupancy maps.
3. For TransFusion-style ``[X, Y]`` detector BEV layout, the extra transpose
   around the temporal boundary should outperform the intentionally wrong
   "no transpose" baseline by a wide margin.

Example:
    python tools/analysis_tools/audit_bevformer_temporal_warp.py \
        --config projects/BEVFormer/configs/bevformer_lidar_kl_temporal_transfusion.py \
        --split val \
        --index 0 \
        --out-dir work_dirs/temporal_warp_audit
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules
from projects.BEVFormer.bevformer.modules import warp_prev_bev


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit BEVFormer temporal warp / layout behavior.')
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
        '--token',
        default=None,
        help='sample token to inspect; overrides --index')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='directory to save figures / metrics')
    parser.add_argument(
        '--point-stride',
        type=int,
        default=8,
        help='subsample factor for point scatter plotting')
    parser.add_argument(
        '--occupancy-thr',
        type=float,
        default=0.25,
        help='threshold applied after grid_sample for occupancy IoU')
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


def apply_transform(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz1 = np.concatenate(
        [points_xyz[:, :3], np.ones((points_xyz.shape[0], 1), dtype=np.float32)],
        axis=1)
    out = (transform.astype(np.float32) @ xyz1.T).T
    aligned = points_xyz.copy()
    aligned[:, :3] = out[:, :3]
    return aligned


def compose_queue_delta(queue_metas: Dict[int, dict], start_idx: int,
                        end_idx: int) -> np.ndarray:
    assert start_idx <= end_idx
    transform = np.eye(4, dtype=np.float64)
    for step in range(start_idx + 1, end_idx + 1):
        transform = np.asarray(
            queue_metas[step]['ego_motion_delta'], dtype=np.float64) @ transform
    return transform


def direct_queue_delta(queue_metas: Dict[int, dict], start_idx: int,
                       end_idx: int) -> np.ndarray:
    start_ego2global = np.asarray(
        queue_metas[start_idx]['ego2global'], dtype=np.float64)
    end_ego2global = np.asarray(
        queue_metas[end_idx]['ego2global'], dtype=np.float64)
    return np.linalg.inv(end_ego2global) @ start_ego2global


def detector_bev_shape(point_cloud_range: Sequence[float],
                       voxel_size_xy: Sequence[float],
                       out_size_factor: int) -> Tuple[int, int]:
    x_bins = int(round((point_cloud_range[3] - point_cloud_range[0]) /
                       voxel_size_xy[0] / out_size_factor))
    y_bins = int(round((point_cloud_range[4] - point_cloud_range[1]) /
                       voxel_size_xy[1] / out_size_factor))
    return x_bins, y_bins


def rasterize_points_xy(points_xyz: np.ndarray,
                        point_cloud_range: Sequence[float],
                        voxel_size_xy: Sequence[float],
                        out_size_factor: int) -> np.ndarray:
    x_bins, y_bins = detector_bev_shape(
        point_cloud_range, voxel_size_xy, out_size_factor)
    occ = np.zeros((x_bins, y_bins), dtype=np.float32)

    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    step_x = voxel_size_xy[0] * out_size_factor
    step_y = voxel_size_xy[1] * out_size_factor

    xs = points_xyz[:, 0]
    ys = points_xyz[:, 1]
    ix = np.floor((xs - x_min) / step_x).astype(np.int64)
    iy = np.floor((ys - y_min) / step_y).astype(np.int64)
    valid = ((ix >= 0) & (ix < x_bins) & (iy >= 0) & (iy < y_bins))
    occ[ix[valid], iy[valid]] = 1.0
    return occ


def overlay_image(current_occ_xy: np.ndarray,
                  other_occ_xy: np.ndarray,
                  color_current=(0.2, 0.9, 0.2),
                  color_other=(0.95, 0.2, 0.2)) -> np.ndarray:
    h_y, w_x = current_occ_xy.shape[1], current_occ_xy.shape[0]
    img = np.zeros((h_y, w_x, 3), dtype=np.float32)
    cur = current_occ_xy.T > 0
    oth = other_occ_xy.T > 0
    img[cur] += np.asarray(color_current, dtype=np.float32)
    img[oth] += np.asarray(color_other, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def occ_iou(a_xy: np.ndarray, b_xy: np.ndarray, thr: float) -> float:
    a = a_xy > thr
    b = b_xy > thr
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def occ_l1(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    return float(np.mean(np.abs(a_xy - b_xy)))


def warp_detector_xy(prev_occ_xy: np.ndarray,
                     delta: np.ndarray,
                     point_cloud_range: Sequence[float],
                     bev_feature_layout: str,
                     correct_layout: bool) -> np.ndarray:
    prev = torch.from_numpy(prev_occ_xy).unsqueeze(0).unsqueeze(0)
    prev = prev.to(dtype=torch.float32)
    if bev_feature_layout == 'xy' and correct_layout:
        prev = prev.transpose(-1, -2).contiguous()
    warped = warp_prev_bev(prev, delta, point_cloud_range)
    if bev_feature_layout == 'xy' and correct_layout:
        warped = warped.transpose(-1, -2).contiguous()
    return warped.squeeze(0).squeeze(0).cpu().numpy()


def draw_point_overlay(ax,
                       frames_xyz: List[np.ndarray],
                       colors: List[str],
                       labels: List[str],
                       point_cloud_range: Sequence[float],
                       point_stride: int,
                       title: str) -> None:
    for xyz, color, label in zip(frames_xyz, colors, labels):
        if xyz.shape[0] == 0:
            continue
        sub = xyz[::point_stride]
        ax.scatter(sub[:, 0], sub[:, 1], s=0.25, c=color, alpha=0.55, label=label)
    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, frameon=True)


def draw_occ_overlay(ax,
                     current_occ_xy: np.ndarray,
                     other_occ_xy: np.ndarray,
                     point_cloud_range: Sequence[float],
                     title: str,
                     color_current=(0.2, 0.9, 0.2),
                     color_other=(0.95, 0.2, 0.2)) -> None:
    img = overlay_image(
        current_occ_xy, other_occ_xy, color_current=color_current,
        color_other=color_other)
    ax.imshow(
        img,
        origin='lower',
        extent=[
            point_cloud_range[0], point_cloud_range[3], point_cloud_range[1],
            point_cloud_range[4]
        ])
    ax.set_aspect('equal')
    ax.set_title(title)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    import_cfg_modules(cfg)

    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset = DATASETS.build(dataset_cfg)

    if args.token is not None:
        dataset.full_init()
        raw_idx = dataset.token2index[args.token]
        if dataset.valid_data_indices is None:
            sample_idx = raw_idx
        else:
            sample_idx = dataset.valid_data_indices.index(raw_idx)
    else:
        sample_idx = args.index

    sample = dataset[sample_idx]
    current_points = points_to_numpy(sample['inputs']['points'])
    history_points = [points_to_numpy(points)
                      for points in sample['inputs'].get('history_points', [])]
    queue_metas = sample['data_samples'].metainfo['queue_metas']
    queue_keys = sorted(queue_metas.keys())
    current_step = queue_keys[-1]
    prev_step = queue_keys[-2]

    point_cloud_range = cfg.model['point_cloud_range']
    train_cfg = cfg.model['train_cfg']['pts']
    voxel_size_xy = train_cfg['voxel_size'][:2]
    out_size_factor = int(train_cfg['out_size_factor'])
    bev_feature_layout = cfg.model.get('bev_feature_layout', 'yx')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------- queue metadata consistency -----------------------
    compose_errors = {}
    aligned_history = []
    for hist_pos, step in enumerate(queue_keys[:-1]):
        chained = compose_queue_delta(queue_metas, step, current_step)
        direct = direct_queue_delta(queue_metas, step, current_step)
        compose_errors[str(step)] = float(np.max(np.abs(chained - direct)))
        aligned_history.append(apply_transform(history_points[hist_pos], direct))

    # ------------------------ occupancy warp consistency ----------------------
    current_occ_xy = rasterize_points_xy(
        current_points, point_cloud_range, voxel_size_xy, out_size_factor)

    prev_points = history_points[-1]
    prev_to_current = np.asarray(
        queue_metas[current_step]['ego_motion_delta'], dtype=np.float64)
    prev_occ_xy = rasterize_points_xy(
        prev_points, point_cloud_range, voxel_size_xy, out_size_factor)
    prev_direct_occ_xy = rasterize_points_xy(
        apply_transform(prev_points, prev_to_current),
        point_cloud_range, voxel_size_xy, out_size_factor)
    prev_warp_correct_xy = warp_detector_xy(
        prev_occ_xy, prev_to_current, point_cloud_range, bev_feature_layout,
        correct_layout=True)
    prev_warp_wrong_xy = warp_detector_xy(
        prev_occ_xy, prev_to_current, point_cloud_range, bev_feature_layout,
        correct_layout=False)

    oldest_points = history_points[0]
    oldest_direct = direct_queue_delta(queue_metas, queue_keys[0], current_step)
    oldest_direct_occ_xy = rasterize_points_xy(
        apply_transform(oldest_points, oldest_direct),
        point_cloud_range, voxel_size_xy, out_size_factor)
    oldest_occ_xy = rasterize_points_xy(
        oldest_points, point_cloud_range, voxel_size_xy, out_size_factor)
    oldest_warp_correct_xy = oldest_occ_xy
    oldest_warp_wrong_xy = oldest_occ_xy
    for step in queue_keys[1:]:
        delta = np.asarray(queue_metas[step]['ego_motion_delta'], dtype=np.float64)
        oldest_warp_correct_xy = warp_detector_xy(
            oldest_warp_correct_xy, delta, point_cloud_range,
            bev_feature_layout, correct_layout=True)
        oldest_warp_wrong_xy = warp_detector_xy(
            oldest_warp_wrong_xy, delta, point_cloud_range,
            bev_feature_layout, correct_layout=False)

    metrics = {
        'config': args.config,
        'split': args.split,
        'sample_index': int(sample_idx),
        'token': queue_metas[current_step]['token'],
        'queue_steps': len(queue_keys),
        'history_steps': len(history_points),
        'compose_error_max_abs': compose_errors,
        'prev_iou_correct': occ_iou(
            prev_warp_correct_xy, prev_direct_occ_xy, args.occupancy_thr),
        'prev_iou_wrong': occ_iou(
            prev_warp_wrong_xy, prev_direct_occ_xy, args.occupancy_thr),
        'prev_l1_correct': occ_l1(prev_warp_correct_xy, prev_direct_occ_xy),
        'prev_l1_wrong': occ_l1(prev_warp_wrong_xy, prev_direct_occ_xy),
        'current_iou_direct_prev': occ_iou(
            current_occ_xy, prev_direct_occ_xy, args.occupancy_thr),
        'current_iou_warp_correct_prev': occ_iou(
            current_occ_xy, prev_warp_correct_xy, args.occupancy_thr),
        'current_iou_warp_wrong_prev': occ_iou(
            current_occ_xy, prev_warp_wrong_xy, args.occupancy_thr),
        'oldest_iou_correct': occ_iou(
            oldest_warp_correct_xy, oldest_direct_occ_xy, args.occupancy_thr),
        'oldest_iou_wrong': occ_iou(
            oldest_warp_wrong_xy, oldest_direct_occ_xy, args.occupancy_thr),
        'oldest_l1_correct': occ_l1(
            oldest_warp_correct_xy, oldest_direct_occ_xy),
        'oldest_l1_wrong': occ_l1(
            oldest_warp_wrong_xy, oldest_direct_occ_xy),
    }

    # ------------------------------ visualization ----------------------------
    colors = ['#377eb8', '#984ea3', '#ff7f00', '#4daf4a']
    raw_frames = history_points + [current_points]
    aligned_frames = aligned_history + [current_points]
    labels = [f'hist_{i}' for i in range(len(history_points))] + ['current']
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))

    draw_point_overlay(
        axes[0, 0], raw_frames, colors[:len(raw_frames)], labels,
        point_cloud_range, args.point_stride,
        'Raw queue points (own ego frames)')
    draw_point_overlay(
        axes[0, 1], aligned_frames, colors[:len(aligned_frames)], labels,
        point_cloud_range, args.point_stride,
        'Queue points aligned into current ego frame')
    draw_occ_overlay(
        axes[0, 2], current_occ_xy, prev_direct_occ_xy, point_cloud_range,
        'Current vs direct-aligned prev\n'
        '(green=current, blue=reference prev)',
        color_current=(0.2, 0.9, 0.2),
        color_other=(0.2, 0.45, 1.0))

    axes[0, 3].axis('off')
    axes[0, 3].text(
        0.02,
        0.98,
        '\n'.join([
            f"token: {metrics['token']}",
            f"queue steps: {metrics['queue_steps']}",
            f"history steps: {metrics['history_steps']}",
            '',
            'compose max |direct - chained|:',
            *[
                f'  step {step}: {err:.3e}'
                for step, err in metrics['compose_error_max_abs'].items()
            ],
            '',
            f"prev IoU  correct/wrong: "
            f"{metrics['prev_iou_correct']:.4f} / {metrics['prev_iou_wrong']:.4f}",
            f"prev L1   correct/wrong: "
            f"{metrics['prev_l1_correct']:.4f} / {metrics['prev_l1_wrong']:.4f}",
            '',
            f"current IoU direct/correct/wrong:",
            f"  {metrics['current_iou_direct_prev']:.4f} / "
            f"{metrics['current_iou_warp_correct_prev']:.4f} / "
            f"{metrics['current_iou_warp_wrong_prev']:.4f}",
            '',
            f"oldest IoU correct/wrong: "
            f"{metrics['oldest_iou_correct']:.4f} / {metrics['oldest_iou_wrong']:.4f}",
            f"oldest L1  correct/wrong: "
            f"{metrics['oldest_l1_correct']:.4f} / {metrics['oldest_l1_wrong']:.4f}",
        ]),
        va='top',
        ha='left',
        family='monospace',
        fontsize=10)

    draw_occ_overlay(
        axes[1, 0], prev_direct_occ_xy, prev_warp_correct_xy,
        point_cloud_range,
        f'Direct-aligned prev vs warped prev (correct)\n'
        f'IoU={metrics["prev_iou_correct"]:.4f}',
        color_current=(0.2, 0.45, 1.0),
        color_other=(0.95, 0.2, 0.2))
    draw_occ_overlay(
        axes[1, 1], prev_direct_occ_xy, prev_warp_wrong_xy,
        point_cloud_range,
        f'Direct-aligned prev vs warped prev (wrong)\n'
        f'IoU={metrics["prev_iou_wrong"]:.4f}',
        color_current=(0.2, 0.45, 1.0),
        color_other=(0.95, 0.2, 0.2))
    draw_occ_overlay(
        axes[1, 2], prev_warp_correct_xy, prev_warp_wrong_xy,
        point_cloud_range,
        'Warped prev: correct vs wrong layout\n'
        '(blue=correct, red=wrong)',
        color_current=(0.2, 0.45, 1.0),
        color_other=(0.95, 0.2, 0.2))
    draw_occ_overlay(
        axes[1, 3], oldest_direct_occ_xy, oldest_warp_correct_xy,
        point_cloud_range,
        f'Oldest frame: direct vs chained correct warp\n'
        f'IoU={metrics["oldest_iou_correct"]:.4f}',
        color_current=(0.2, 0.45, 1.0),
        color_other=(0.95, 0.2, 0.2))

    draw_occ_overlay(
        axes[2, 0], current_occ_xy, prev_direct_occ_xy, point_cloud_range,
        f'Current vs direct-aligned prev\n'
        f'IoU={metrics["current_iou_direct_prev"]:.4f}',
        color_current=(0.2, 0.9, 0.2),
        color_other=(0.2, 0.45, 1.0))
    draw_occ_overlay(
        axes[2, 1], current_occ_xy, prev_warp_correct_xy, point_cloud_range,
        f'Current vs warped prev (correct)\n'
        f'IoU={metrics["current_iou_warp_correct_prev"]:.4f}',
        color_current=(0.2, 0.9, 0.2),
        color_other=(0.95, 0.2, 0.2))
    draw_occ_overlay(
        axes[2, 2], current_occ_xy, prev_warp_wrong_xy, point_cloud_range,
        f'Current vs warped prev (wrong)\n'
        f'IoU={metrics["current_iou_warp_wrong_prev"]:.4f}',
        color_current=(0.2, 0.9, 0.2),
        color_other=(0.95, 0.2, 0.2))
    axes[2, 3].axis('off')
    axes[2, 3].text(
        0.02,
        0.98,
        'Current-vs-prev panels are useful for visual sanity checks.\n'
        'They are not the strict implementation test because moving\n'
        'objects and newly visible regions legitimately differ between\n'
        'frames.\n\n'
        'The stricter check is direct-aligned prev vs warped prev.',
        va='top',
        ha='left',
        fontsize=10)

    fig.suptitle(
        'BEVFormer temporal warp audit\n'
        f'{Path(args.config).name} | split={args.split} | index={sample_idx}',
        fontsize=14)
    fig.tight_layout()

    stem = f'{Path(args.config).stem}_{args.split}_idx{sample_idx}'
    fig.savefig(out_dir / f'{stem}_audit.png', bbox_inches='tight', dpi=180)
    plt.close(fig)

    with open(out_dir / f'{stem}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f'Saved figure to {(out_dir / f"{stem}_audit.png")}')
    print(f'Saved metrics to {(out_dir / f"{stem}_metrics.json")}')


if __name__ == '__main__':
    main()
