"""Visualize KL LiDAR detections with velocity arrows in BEV.

Two modes are supported:

1. Online inference from a checkpoint (requires a CUDA-visible environment
   for voxel / spconv models such as BEVFormerLidar):

   python tools/visualize_kl_velocity.py \
       --config projects/BEVFormer/configs/bevformer_lidar_kl_temporal_centerhead.py \
       --checkpoint work_dirs/bevformer_lidar_kl_temporal_centerhead/epoch_6.pth \
       --out-dir work_dirs/vis_temporal_centerhead_epoch6 \
       --device cuda:0 \
       --max-frames 20

2. Offline visualization from a saved ``results_nusc.json`` file:

   python tools/visualize_kl_velocity.py \
       --config projects/BEVFormer/configs/bevformer_lidar_kl_temporal_centerhead.py \
       --results-json work_dirs/.../results_nusc.json \
       --out-dir work_dirs/vis_from_results \
       --max-frames 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
from typing import Dict, Iterable, List, Sequence

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
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize KL detections with velocity arrows.')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument(
        '--checkpoint',
        help='checkpoint file for online inference mode')
    parser.add_argument(
        '--results-json',
        help='results_nusc.json for offline visualization mode')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='directory to save rendered images')
    parser.add_argument(
        '--split',
        default='val',
        choices=['val', 'test'],
        help='which dataloader / ann file to use from the config')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='device for online inference, e.g. cuda:0')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.2,
        help='prediction score threshold')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=20,
        help='number of frames to render; 0 means all')
    parser.add_argument(
        '--indices',
        type=int,
        nargs='*',
        default=None,
        help='explicit dataset/info indices to render')
    parser.add_argument(
        '--tokens',
        nargs='*',
        default=None,
        help='explicit sample tokens to render')
    parser.add_argument(
        '--topk',
        type=int,
        default=80,
        help='max predictions per frame after thresholding')
    parser.add_argument(
        '--point-stride',
        type=int,
        default=4,
        help='subsample factor for point rendering')
    parser.add_argument(
        '--vel-scale',
        type=float,
        default=3.0,
        help='meters shown per m/s of vx/vy')
    parser.add_argument(
        '--min-vel-draw',
        type=float,
        default=0.2,
        help='minimum speed magnitude to draw velocity arrow')
    parser.add_argument(
        '--annotate',
        action='store_true',
        help='annotate predicted boxes with class / score / speed')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip frames whose output image already exists')
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.results_json):
        raise ValueError(
            'Specify exactly one of --checkpoint or --results-json.')
    if args.point_stride <= 0:
        raise ValueError('--point-stride must be > 0.')
    if args.topk <= 0:
        raise ValueError('--topk must be > 0.')
    return args


def import_cfg_modules(cfg: Config) -> None:
    register_all_modules(init_default_scope=True)
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)


def get_dataset_cfg(cfg: Config, split: str):
    if split == 'val':
        return cfg.val_dataloader.dataset
    return cfg.test_dataloader.dataset


def resolve_info_path(cfg: Config, dataset_cfg) -> str:
    data_root = dataset_cfg.get('data_root', cfg.get('data_root', ''))
    ann_file = dataset_cfg['ann_file']
    if osp.isabs(ann_file):
        return ann_file
    return osp.join(data_root, ann_file)


def load_info_list(info_path: str) -> List[dict]:
    loaded = mmengine.load(info_path)
    if isinstance(loaded, dict):
        if 'data_list' in loaded:
            return loaded['data_list']
        if 'infos' in loaded:
            return loaded['infos']
    if isinstance(loaded, list):
        return loaded
    raise TypeError(f'Unsupported info file structure in {info_path}.')


def resolve_lidar_path(cfg: Config, dataset_cfg, info: dict) -> str:
    data_root = dataset_cfg.get('data_root', cfg.get('data_root', ''))
    data_prefix = dataset_cfg.get('data_prefix', cfg.get('data_prefix', {}))
    pts_prefix = data_prefix.get('pts', '')
    lidar_rel = info['lidar_points']['lidar_path']
    return osp.join(data_root, pts_prefix, lidar_rel)


def load_points(cfg: Config, dataset_cfg, info: dict) -> np.ndarray:
    lidar_path = resolve_lidar_path(cfg, dataset_cfg, info)
    num_feats = int(info['lidar_points'].get('num_pts_feats', 5))
    points = np.fromfile(lidar_path, dtype=np.float32)
    return points.reshape(-1, num_feats)


def filter_gt_instances(info: dict, use_valid_flag: bool) -> List[dict]:
    instances = info.get('instances', [])
    filtered = []
    for inst in instances:
        if use_valid_flag:
            keep = bool(inst.get('bbox_3d_isvalid', False))
        else:
            keep = int(inst.get('num_lidar_pts', 0)) > 0
        if keep:
            filtered.append(inst)
    return filtered


def gt_arrays_from_info(info: dict, use_valid_flag: bool) -> Dict[str, np.ndarray]:
    instances = filter_gt_instances(info, use_valid_flag)
    if not instances:
        return dict(
            boxes=np.zeros((0, 9), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            scores=np.ones((0,), dtype=np.float32),
        )

    boxes = []
    labels = []
    for inst in instances:
        box = np.asarray(inst['bbox_3d'], dtype=np.float32)
        vel = np.asarray(inst.get('velocity', [0.0, 0.0]), dtype=np.float32)
        if box.shape[0] == 7:
            box = np.concatenate([box, vel], axis=0)
        else:
            box = box.copy()
            if box.shape[0] >= 9:
                box[7:9] = vel[:2]
        boxes.append(box[:9])
        labels.append(int(inst['bbox_label_3d']))

    return dict(
        boxes=np.stack(boxes, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.ones((len(boxes),), dtype=np.float32),
    )


def quat_to_yaw(quat: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def pred_arrays_from_results(entries: List[dict],
                             score_thr: float,
                             topk: int) -> Dict[str, np.ndarray]:
    kept = [entry for entry in entries
            if float(entry.get('detection_score', 0.0)) >= score_thr]
    kept.sort(key=lambda item: float(item.get('detection_score', 0.0)),
              reverse=True)
    kept = kept[:topk]

    if not kept:
        return dict(
            boxes=np.zeros((0, 9), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            scores=np.zeros((0,), dtype=np.float32),
        )

    boxes = []
    labels = []
    scores = []
    for entry in kept:
        x, y, z = entry['translation']
        w, l, h = entry['size']
        yaw = quat_to_yaw(entry['rotation'])
        vx, vy = entry.get('velocity', [0.0, 0.0])
        boxes.append([x, y, z, l, w, h, yaw, vx, vy])
        labels.append(CLASS_NAME_TO_ID[entry['detection_name']])
        scores.append(float(entry['detection_score']))

    return dict(
        boxes=np.asarray(boxes, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
    )


def gt_arrays_from_ann_info(ann_info: dict) -> Dict[str, np.ndarray]:
    gt_boxes = ann_info.get('gt_bboxes_3d')
    gt_labels = ann_info.get('gt_labels_3d')

    if gt_boxes is None or gt_labels is None:
        return dict(
            boxes=np.zeros((0, 9), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            scores=np.ones((0,), dtype=np.float32),
        )

    if hasattr(gt_boxes, 'tensor'):
        boxes = gt_boxes.tensor.detach().cpu().numpy().astype(np.float32)
    else:
        boxes = np.asarray(gt_boxes, dtype=np.float32)
    if boxes.shape[1] < 9:
        pad = np.zeros((boxes.shape[0], 9 - boxes.shape[1]), dtype=boxes.dtype)
        boxes = np.concatenate([boxes, pad], axis=1)

    labels = np.asarray(gt_labels, dtype=np.int64)
    return dict(
        boxes=boxes[:, :9],
        labels=labels,
        scores=np.ones((len(labels),), dtype=np.float32),
    )


def to_numpy_boxes(instances_3d) -> np.ndarray:
    if len(instances_3d) == 0:
        return np.zeros((0, 9), dtype=np.float32)
    boxes = instances_3d.bboxes_3d.tensor.detach().cpu().numpy().astype(
        np.float32)
    if boxes.shape[1] < 9:
        pad = np.zeros((boxes.shape[0], 9 - boxes.shape[1]), dtype=boxes.dtype)
        boxes = np.concatenate([boxes, pad], axis=1)
    return boxes[:, :9]


def to_numpy_labels(instances_3d) -> np.ndarray:
    if len(instances_3d) == 0:
        return np.zeros((0,), dtype=np.int64)
    return instances_3d.labels_3d.detach().cpu().numpy().astype(np.int64)


def to_numpy_scores(instances_3d) -> np.ndarray:
    if len(instances_3d) == 0:
        return np.zeros((0,), dtype=np.float32)
    return instances_3d.scores_3d.detach().cpu().numpy().astype(np.float32)


def sample_indices(total: int,
                   max_frames: int,
                   indices: Sequence[int] | None) -> List[int]:
    if indices:
        out = []
        for idx in indices:
            if idx < 0 or idx >= total:
                raise IndexError(f'Index {idx} is out of range 0..{total - 1}.')
            out.append(int(idx))
        return out
    if max_frames == 0 or max_frames >= total:
        return list(range(total))
    return list(range(max_frames))


def compute_box_corners_bev(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    corners = []
    for box in boxes:
        x, y, _, l, w, _, yaw = box[:7]
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        local = np.array([
            [l / 2, w / 2],
            [l / 2, -w / 2],
            [-l / 2, -w / 2],
            [-l / 2, w / 2],
        ], dtype=np.float32)
        corners.append(local @ rot.T + np.array([x, y], dtype=np.float32))
    return np.stack(corners, axis=0)


def lidar_xy_to_display(xy: np.ndarray) -> np.ndarray:
    """Rotate LiDAR BEV for display so ego-forward points upward.

    LiDAR convention in this repo follows the common autonomous-driving frame:
    +X forward, +Y left. For visualization we rotate by +90 deg on the page so
    forward becomes +display-Y (up) and left becomes -display-X.
    """
    disp = np.empty_like(xy, dtype=np.float32)
    disp[..., 0] = -xy[..., 1]
    disp[..., 1] = xy[..., 0]
    return disp


def draw_boxes(ax,
               boxes: np.ndarray,
               labels: np.ndarray,
               scores: np.ndarray | None,
               color: str,
               vel_scale: float,
               min_vel_draw: float,
               annotate: bool = False,
               alpha: float = 1.0) -> None:
    if boxes.size == 0:
        return

    corners = compute_box_corners_bev(boxes)
    for idx, box in enumerate(boxes):
        poly = corners[idx]
        poly_disp = lidar_xy_to_display(poly)
        closed = np.concatenate([poly, poly[:1]], axis=0)
        closed_disp = np.concatenate([poly_disp, poly_disp[:1]], axis=0)
        ax.plot(closed_disp[:, 0], closed_disp[:, 1],
                color=color, linewidth=1.4,
                alpha=alpha)

        center_xy_disp = lidar_xy_to_display(
            np.array([[box[0], box[1]]], dtype=np.float32))[0]
        center_x = float(center_xy_disp[0])
        center_y = float(center_xy_disp[1])
        # Draw a short heading tick for box yaw. This is independent of
        # velocity and only indicates the box front direction.
        yaw = float(box[6])
        heading_len = min(max(float(box[3]) * 0.22, 0.45), 1.35)
        heading_vec_lidar = np.array(
            [[math.cos(yaw), math.sin(yaw)]], dtype=np.float32)
        heading_vec_disp = lidar_xy_to_display(heading_vec_lidar)[0]
        heading_tip_x = center_x + float(heading_vec_disp[0]) * heading_len
        heading_tip_y = center_y + float(heading_vec_disp[1]) * heading_len
        ax.plot([center_x, heading_tip_x], [center_y, heading_tip_y],
                color=color, linewidth=0.9, linestyle='--', alpha=0.6)

        vx = float(box[7])
        vy = float(box[8])
        speed = math.hypot(vx, vy)
        if speed >= min_vel_draw:
            vel_disp = lidar_xy_to_display(
                np.array([[vx, vy]], dtype=np.float32))[0]
            dx = float(vel_disp[0]) * vel_scale
            dy = float(vel_disp[1]) * vel_scale
            disp_len = math.hypot(dx, dy)
            head_length = min(max(disp_len * 0.22, 0.18), 0.9)
            head_width = min(max(disp_len * 0.14, 0.12), 0.6)
            shaft_width = min(max(disp_len * 0.012, 0.015), 0.05)
            ax.arrow(
                center_x,
                center_y,
                dx,
                dy,
                color=color,
                width=shaft_width,
                head_width=head_width,
                head_length=head_length,
                length_includes_head=True,
                alpha=alpha)

        if annotate:
            label_name = CLASS_NAMES[int(labels[idx])]
            speed_text = f'{speed:.2f}m/s'
            if scores is None or len(scores) == 0:
                text = f'{label_name} {speed_text}'
            else:
                text = f'{label_name} {float(scores[idx]):.2f} {speed_text}'
            ax.text(
                center_x,
                center_y,
                text,
                color=color,
                fontsize=6,
                ha='left',
                va='bottom',
                alpha=alpha)


def render_frame(points: np.ndarray,
                 gt_data: Dict[str, np.ndarray],
                 pred_data: Dict[str, np.ndarray],
                 save_path: str,
                 token: str,
                 pc_range: Sequence[float],
                 vel_scale: float,
                 min_vel_draw: float,
                 point_stride: int,
                 annotate: bool) -> None:
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    disp_x_min = -y_max
    disp_x_max = -y_min
    disp_y_min = x_min
    disp_y_max = x_max

    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    pts = points[::point_stride, :3]
    pts_disp = lidar_xy_to_display(pts[:, :2])
    ax.scatter(
        pts_disp[:, 0],
        pts_disp[:, 1],
        s=0.12,
        c='white',
        alpha=0.35,
        linewidths=0)

    draw_boxes(
        ax,
        gt_data['boxes'],
        gt_data['labels'],
        gt_data['scores'],
        color='#00ff88',
        vel_scale=vel_scale,
        min_vel_draw=min_vel_draw,
        annotate=False,
        alpha=0.95)
    draw_boxes(
        ax,
        pred_data['boxes'],
        pred_data['labels'],
        pred_data['scores'],
        color='#ff4d4d',
        vel_scale=vel_scale,
        min_vel_draw=min_vel_draw,
        annotate=annotate,
        alpha=0.95)

    ax.plot(0.0, 0.0, marker='o', markersize=4, color='#ffd166')
    ax.arrow(
        0.0, 0.0, 0.0, 3.0,
        color='#ffd166',
        width=0.03,
        head_width=0.5,
        head_length=0.6,
        length_includes_head=True)

    ax.set_xlim(disp_x_min, disp_x_max)
    ax.set_ylim(disp_y_min, disp_y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Lateral (m)', color='white')
    ax.set_ylabel('Forward (m)', color='white')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#888888')
    ax.grid(color='#444444', linestyle='--', linewidth=0.5, alpha=0.4)

    gt_speed = np.linalg.norm(gt_data['boxes'][:, 7:9], axis=1) \
        if len(gt_data['boxes']) else np.zeros((0,), dtype=np.float32)
    pred_speed = np.linalg.norm(pred_data['boxes'][:, 7:9], axis=1) \
        if len(pred_data['boxes']) else np.zeros((0,), dtype=np.float32)
    title = (
        f'KL BEV velocity view | token={token}\n'
        f'GT={len(gt_data["boxes"])} mean_speed={gt_speed.mean():.2f} m/s | '
        f'Pred={len(pred_data["boxes"])} '
        f'mean_speed={pred_speed.mean():.2f} m/s'
    )
    ax.set_title(title, color='white', fontsize=10)

    legend_lines = [
        plt.Line2D([0], [0], color='#00ff88', lw=2),
        plt.Line2D([0], [0], color='#ff4d4d', lw=2),
        plt.Line2D([0], [0], color='#ffd166', lw=2),
    ]
    ax.legend(
        legend_lines,
        ['GT boxes + velocity', 'Pred boxes + velocity', 'Ego heading'],
        loc='upper right',
        facecolor='black',
        edgecolor='#888888',
        labelcolor='white',
        fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def select_tokens_from_infos(info_list: List[dict],
                             results_by_token: Dict[str, list],
                             args) -> List[str]:
    ordered_tokens = [info['token'] for info in info_list
                      if info['token'] in results_by_token]
    if args.tokens:
        missing = [token for token in args.tokens if token not in results_by_token]
        if missing:
            raise KeyError(f'Tokens not found in results: {missing}')
        return list(args.tokens)
    if args.indices:
        return [ordered_tokens[idx] for idx in sample_indices(
            len(ordered_tokens), args.max_frames, args.indices)]
    if args.max_frames == 0 or args.max_frames >= len(ordered_tokens):
        return ordered_tokens
    return ordered_tokens[:args.max_frames]


def run_offline(cfg: Config, args) -> None:
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    info_path = resolve_info_path(cfg, dataset_cfg)
    info_list = load_info_list(info_path)
    info_by_token = {info['token']: info for info in info_list}

    with open(args.results_json, 'r') as f:
        results_payload = json.load(f)
    results_by_token = results_payload['results']

    tokens = select_tokens_from_infos(info_list, results_by_token, args)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for token in tokens:
        info = info_by_token[token]
        save_path = osp.join(args.out_dir, f'{token}.png')
        if args.skip_existing and osp.exists(save_path):
            summary.append(dict(
                token=token,
                out_file=save_path,
                skipped=True))
            print(f'[SKIP] {save_path}')
            continue

        points = load_points(cfg, dataset_cfg, info)
        gt_data = gt_arrays_from_info(
            info, bool(dataset_cfg.get('use_valid_flag', False)))
        pred_data = pred_arrays_from_results(
            results_by_token.get(token, []), args.score_thr, args.topk)

        render_frame(
            points=points,
            gt_data=gt_data,
            pred_data=pred_data,
            save_path=save_path,
            token=token,
            pc_range=cfg.point_cloud_range,
            vel_scale=args.vel_scale,
            min_vel_draw=args.min_vel_draw,
            point_stride=args.point_stride,
            annotate=args.annotate)
        summary.append(dict(
            token=token,
            out_file=save_path,
            num_gt=int(len(gt_data['boxes'])),
            num_pred=int(len(pred_data['boxes']))))
        print(f'[OK] {save_path}')

    mmengine.dump(summary, osp.join(args.out_dir, 'summary.json'))


def run_online(cfg: Config, args) -> None:
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset = DATASETS.build(dataset_cfg)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    indices = sample_indices(len(dataset), args.max_frames, args.indices)
    if args.tokens:
        token_set = set(args.tokens)
        indices = []
        for idx in range(len(dataset)):
            data = dataset.get_data_info(idx)
            if data['token'] in token_set:
                indices.append(idx)
        missing = token_set - {dataset.get_data_info(i)['token'] for i in indices}
        if missing:
            raise KeyError(f'Tokens not found in dataset: {sorted(missing)}')

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []

    for idx in indices:
        info = dataset.get_data_info(idx)
        data = dataset[idx]
        token = info['token']
        save_path = osp.join(args.out_dir, f'{idx:06d}_{token}.png')
        if args.skip_existing and osp.exists(save_path):
            summary.append(dict(
                index=int(idx),
                token=token,
                out_file=save_path,
                skipped=True))
            print(f'[SKIP] {save_path}')
            continue

        points = data['inputs']['points']
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        with torch.no_grad():
            pred_sample = model.test_step(pseudo_collate([data]))[0]

        pred_instances = pred_sample.pred_instances_3d
        keep = pred_instances.scores_3d > args.score_thr
        pred_instances = pred_instances[keep]
        if len(pred_instances) > args.topk:
            pred_instances = pred_instances[:args.topk]

        gt_data = gt_arrays_from_ann_info(info.get('eval_ann_info', {}))
        pred_data = dict(
            boxes=to_numpy_boxes(pred_instances),
            labels=to_numpy_labels(pred_instances),
            scores=to_numpy_scores(pred_instances))

        render_frame(
            points=points,
            gt_data=gt_data,
            pred_data=pred_data,
            save_path=save_path,
            token=token,
            pc_range=cfg.point_cloud_range,
            vel_scale=args.vel_scale,
            min_vel_draw=args.min_vel_draw,
            point_stride=args.point_stride,
            annotate=args.annotate)
        summary.append(dict(
            index=int(idx),
            token=token,
            out_file=save_path,
            num_gt=int(len(gt_data['boxes'])),
            num_pred=int(len(pred_data['boxes']))))
        print(f'[OK] {save_path}')

    mmengine.dump(summary, osp.join(args.out_dir, 'summary.json'))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_cfg_modules(cfg)

    if args.results_json:
        run_offline(cfg, args)
    else:
        run_online(cfg, args)


if __name__ == '__main__':
    main()
