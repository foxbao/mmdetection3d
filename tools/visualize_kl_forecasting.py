"""Visualize KL forecasting predictions in BEV.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_kl_forecasting.py \
        --config projects/BEVFormer/configs/bevformer_lidar_kl_temporal_transfusion_forecasting_mlp.py \
        --checkpoint work_dirs/bevformer_lidar_kl_temporal_transfusion_forecasting_mlp/epoch_4.pth \
        --out-dir work_dirs/vis_bevformer_lidar_kl_temporal_transfusion_forecasting_mlp_epoch4 \
        --device cuda:0 --max-frames 8
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmengine
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


CLASS_NAMES = (
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize KL per-object forecasting trajectories.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--indices', type=int, nargs='*', default=None)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--match-dist-thr', type=float, default=2.0)
    parser.add_argument('--min-final-disp', type=float, default=0.5)
    parser.add_argument('--scan-limit', type=int, default=800)
    parser.add_argument('--point-stride', type=int, default=6)
    return parser.parse_args()


def import_cfg_modules(cfg: Config) -> None:
    register_all_modules()
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports:
        import_modules_from_strings(**custom_imports)


def get_dataset_cfg(cfg: Config, split: str):
    if split == 'val':
        return cfg.val_dataloader.dataset
    return cfg.test_dataloader.dataset


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sample_indices(dataset, max_frames: int, explicit: Sequence[int] | None,
                   min_final_disp: float, scan_limit: int) -> List[int]:
    if explicit:
        return [int(i) for i in explicit]

    scored: List[Tuple[int, int, float]] = []
    n_scan = min(len(dataset), scan_limit if scan_limit > 0 else len(dataset))
    for idx in range(n_scan):
        info = dataset.get_data_info(idx)
        moving = 0
        total_disp = 0.0
        for inst in info.get('instances', []):
            locs = np.asarray(inst.get('gt_forecasting_locs', []),
                              dtype=np.float32)
            mask = np.asarray(inst.get('gt_forecasting_mask', []),
                              dtype=bool)
            if locs.ndim != 2 or locs.shape[0] == 0 or not mask.any():
                continue
            valid_locs = locs[:len(mask)][mask]
            final_disp = float(np.linalg.norm(valid_locs[-1]))
            if final_disp >= min_final_disp:
                moving += 1
                total_disp += final_disp
        if moving > 0:
            scored.append((moving, idx, total_disp))

    scored.sort(key=lambda x: (x[0], x[2]), reverse=True)
    return [idx for _, idx, _ in scored[:max_frames]]


def extract_gt(info: Dict, min_final_disp: float) -> List[Dict]:
    gts = []
    for gt_idx, inst in enumerate(info.get('instances', [])):
        locs = np.asarray(inst.get('gt_forecasting_locs', []),
                          dtype=np.float32)
        mask = np.asarray(inst.get('gt_forecasting_mask', []), dtype=bool)
        if locs.ndim != 2 or locs.shape[-1] != 2 or not mask.any():
            continue
        steps = min(locs.shape[0], mask.shape[0])
        locs = locs[:steps]
        mask = mask[:steps]
        valid_locs = locs[mask]
        final_disp = float(np.linalg.norm(valid_locs[-1]))
        if final_disp < min_final_disp:
            continue
        box = np.asarray(inst['bbox_3d'], dtype=np.float32)
        label = int(inst.get('bbox_label_3d', inst.get('bbox_label', -1)))
        gts.append(
            dict(
                gt_idx=gt_idx,
                center=box[:2],
                box=box,
                label=label,
                traj=box[:2][None, :] + locs,
                mask=mask,
                final_disp=final_disp))
    return gts


def extract_pred(pred_instances, score_thr: float, topk: int) -> List[Dict]:
    if len(pred_instances) == 0:
        return []
    scores = to_numpy(pred_instances.scores_3d)
    keep = np.where(scores >= score_thr)[0]
    if keep.size == 0:
        return []
    keep = keep[np.argsort(-scores[keep])]
    if topk > 0:
        keep = keep[:topk]

    boxes = to_numpy(pred_instances.bboxes_3d.tensor)
    labels = to_numpy(pred_instances.labels_3d).astype(np.int64)
    trajs = to_numpy(pred_instances.forecasting_3d)

    preds = []
    for pred_idx in keep:
        center = boxes[pred_idx, :2]
        preds.append(
            dict(
                pred_idx=int(pred_idx),
                center=center,
                box=boxes[pred_idx],
                label=int(labels[pred_idx]),
                score=float(scores[pred_idx]),
                traj=center[None, :] + trajs[pred_idx]))
    return preds


def greedy_match(gts: List[Dict], preds: List[Dict],
                 dist_thr: float) -> List[Tuple[int, int, float]]:
    matches = []
    used_gt = set()
    used_pred = set()
    candidates = []
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            if gt['label'] != pred['label']:
                continue
            dist = float(np.linalg.norm(gt['center'] - pred['center']))
            if dist <= dist_thr:
                candidates.append((dist, gi, pi))
    for dist, gi, pi in sorted(candidates):
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, dist))
    return matches


def get_points(data) -> np.ndarray | None:
    points = data.get('inputs', {}).get('points', None)
    if isinstance(points, (list, tuple)):
        points = points[-1] if points else None
    if points is None:
        return None
    points = to_numpy(points)
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    return points


def draw_box(ax, box: np.ndarray, color: str, alpha: float = 0.35) -> None:
    x, y, _, length, width, _, yaw = box[:7]
    corners = np.array([
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
        [length / 2, width / 2],
    ])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]])
    pts = corners @ rot.T + np.array([x, y])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.8, alpha=alpha)


def plot_traj(ax, center: np.ndarray, traj: np.ndarray, mask: np.ndarray | None,
              color: str, label: str | None, lw: float, alpha: float,
              marker: str) -> None:
    pts = np.concatenate([center[None, :], traj], axis=0)
    if mask is not None:
        keep = np.concatenate([[True], mask.astype(bool)])
        pts = pts[keep]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha,
            marker=marker, markersize=3, label=label)
    ax.scatter([center[0]], [center[1]], color=color, s=14, alpha=alpha)


def render(points: np.ndarray | None, gts: List[Dict], preds: List[Dict],
           matches: List[Tuple[int, int, float]], pc_range: Sequence[float],
           save_path: str, title: str, point_stride: int) -> Dict:
    matched_gt = {gi for gi, _, _ in matches}
    matched_pred = {pi for _, pi, _ in matches}

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')

    if points is not None and points.size:
        pts = points[::max(1, point_stride)]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.08, c='white', alpha=0.14,
                   linewidths=0)

    gt_label_once = True
    pred_label_once = True
    miss_label_once = True
    unmatched_pred_label_once = True

    for gi, gt in enumerate(gts):
        draw_box(ax, gt['box'], '#2dd36f', alpha=0.45)
        label = 'GT future' if gt_label_once else None
        gt_label_once = False
        plot_traj(ax, gt['center'], gt['traj'], gt['mask'], '#2dd36f', label,
                  lw=1.8, alpha=0.95 if gi in matched_gt else 0.5,
                  marker='o')
        if gi not in matched_gt:
            label = 'missed GT' if miss_label_once else None
            miss_label_once = False
            ax.scatter(gt['center'][0], gt['center'][1], s=38,
                       facecolors='none', edgecolors='#ff5c5c',
                       linewidths=1.2, label=label)

    for pi, pred in enumerate(preds):
        if pi in matched_pred:
            label = 'Pred future' if pred_label_once else None
            pred_label_once = False
            alpha = 0.95
            color = '#ffb000'
            lw = 1.6
        else:
            label = 'Unmatched pred' if unmatched_pred_label_once else None
            unmatched_pred_label_once = False
            alpha = 0.28
            color = '#b0b0b0'
            lw = 0.9
        draw_box(ax, pred['box'], color, alpha=alpha * 0.5)
        plot_traj(ax, pred['center'], pred['traj'], None, color, label,
                  lw=lw, alpha=alpha, marker='x')

    for gi, pi, _ in matches:
        gt = gts[gi]
        pred = preds[pi]
        final_steps = min(gt['traj'].shape[0], pred['traj'].shape[0])
        if final_steps:
            err = float(np.linalg.norm(gt['traj'][:final_steps] -
                                       pred['traj'][:final_steps], axis=1)
                        .mean())
            ax.text(gt['center'][0], gt['center'][1], f'{err:.1f}m',
                    color='white', fontsize=6)

    ax.set_xlim(float(pc_range[0]), float(pc_range[3]))
    ax.set_ylim(float(pc_range[1]), float(pc_range[4]))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(color='#333333', linewidth=0.4)
    ax.set_xlabel('x / m', color='white')
    ax.set_ylabel('y / m', color='white')
    ax.tick_params(colors='white')
    ax.set_title(title, color='white')
    leg = ax.legend(loc='upper right', fontsize=7, facecolor='#222222',
                    edgecolor='#555555')
    for text in leg.get_texts():
        text.set_color('white')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return dict(
        out_file=save_path,
        num_gt=len(gts),
        num_pred=len(preds),
        num_matched=len(matches),
        num_missed_gt=len(gts) - len(matched_gt),
        num_unmatched_pred=len(preds) - len(matched_pred))


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_cfg_modules(cfg)

    dataset = DATASETS.build(get_dataset_cfg(cfg, args.split))
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    indices = sample_indices(dataset, args.max_frames, args.indices,
                             args.min_final_disp, args.scan_limit)
    mmengine.mkdir_or_exist(args.out_dir)

    summaries = []
    for idx in indices:
        info = dataset.get_data_info(idx)
        data = dataset[idx]
        token = info.get('token', str(idx))
        with torch.no_grad():
            pred_sample = model.test_step(pseudo_collate([data]))[0]

        points = get_points(data)
        gts = extract_gt(info, args.min_final_disp)
        preds = extract_pred(pred_sample.pred_instances_3d, args.score_thr,
                             args.topk)
        matches = greedy_match(gts, preds, args.match_dist_thr)

        save_path = osp.join(args.out_dir, f'{idx:06d}_{token}.png')
        title = (f'idx={idx} token={token[:8]}  '
                 f'matched={len(matches)}/{len(gts)} '
                 f'score_thr={args.score_thr}')
        summary = render(points, gts, preds, matches, cfg.point_cloud_range,
                         save_path, title, args.point_stride)
        summary.update(index=int(idx), token=token)
        summaries.append(summary)
        print(f'[OK] {save_path}')

    mmengine.dump(summaries, osp.join(args.out_dir, 'summary.json'))
    print(f'Summary: {osp.join(args.out_dir, "summary.json")}')


if __name__ == '__main__':
    main()
