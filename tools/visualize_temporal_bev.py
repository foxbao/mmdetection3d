# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize temporal LiDAR detection results in BEV.

This script runs samples through the configured dataset pipeline, so temporal
fields such as ``adj_infos``, ``adj_points`` and ``adj_ego_motions`` are kept.
It is intended for quick sanity checks of temporal BEVFusion models.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


PALETTE = [
    (255, 99, 71),
    (30, 144, 255),
    (50, 205, 50),
    (255, 215, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 140, 0),
    (220, 20, 60),
    (154, 205, 50),
    (70, 130, 180),
    (255, 105, 180),
    (46, 139, 87),
    (210, 105, 30),
    (123, 104, 238),
    (0, 191, 255),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Temporal BEVFusion BEV visualization')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', default='vis_temporal_bev')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max-frames', type=int, default=20)
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--no-traj',
        action='store_true',
        help='Disable trajectory drawing even when forecasting_3d exists.')
    parser.add_argument(
        '--indices',
        type=str,
        default='',
        help='Comma-separated dataset indices. Overrides --max-frames.')
    return parser.parse_args()


def points_to_numpy(points):
    if hasattr(points, 'tensor'):
        points = points.tensor
    if torch.is_tensor(points):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def boxes_to_numpy(boxes):
    if boxes is None:
        return np.zeros((0, 7), dtype=np.float32)
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    return np.asarray(boxes)


def tensor_to_numpy(x, dtype=None):
    if x is None:
        arr = np.zeros((0,), dtype=np.float32)
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def bev_xy_to_px(x, y, xlim, ylim, width, height):
    # KL uses FLU coordinates: +X forward, +Y left.  Render BEV as the
    # driver would expect: +X goes up and +Y goes left on the image.
    px = (ylim[1] - y) / (ylim[1] - ylim[0]) * (width - 1)
    py = (xlim[1] - x) / (xlim[1] - xlim[0]) * (height - 1)
    return np.stack([px, py], axis=-1)


def box_corners_bev(box):
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array([
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2, dy / 2],
    ], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def draw_boxes(img, boxes, labels, scores, xlim, ylim, color_mode,
               score_thr=0.0, thickness=2):
    height, width = img.shape[:2]
    for i, box in enumerate(boxes):
        score = None if scores is None or len(scores) == 0 else scores[i]
        if score is not None and score < score_thr:
            continue

        if color_mode == 'gt':
            color = (80, 230, 80)
        else:
            label = int(labels[i]) if labels is not None and len(labels) else 0
            color = PALETTE[label % len(PALETTE)]

        corners = box_corners_bev(box)
        pts = bev_xy_to_px(
            corners[:, 0], corners[:, 1], xlim, ylim, width, height)
        pts = np.round(pts).astype(np.int32)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)

        front = np.mean(pts[:2], axis=0).astype(np.int32)
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.line(img, tuple(center), tuple(front), color, thickness,
                 cv2.LINE_AA)

        if score is not None:
            cv2.putText(img, f'{score:.2f}', tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                        cv2.LINE_AA)


def draw_trajectories(img, boxes, trajs, masks, xlim, ylim, color,
                      score_thr=0.0, scores=None, thickness=2):
    """Draw future center trajectories in BEV.

    Trajectories are stored as per-step [dx, dy] displacement from the current
    box center.  GT uses ``masks`` to hide invalid future steps; predictions
    normally pass an all-true mask.
    """
    if boxes is None or trajs is None:
        return
    if len(boxes) == 0 or len(trajs) == 0:
        return

    height, width = img.shape[:2]
    num_items = min(len(boxes), len(trajs))
    for i in range(num_items):
        if scores is not None and len(scores) and scores[i] < score_thr:
            continue

        traj = np.asarray(trajs[i], dtype=np.float32)
        if traj.ndim != 2 or traj.shape[-1] != 2:
            continue

        if masks is None:
            valid = np.ones((traj.shape[0], ), dtype=bool)
        else:
            valid = np.asarray(masks[i], dtype=bool).reshape(-1)
            valid = valid[:traj.shape[0]]
        if not valid.any():
            continue

        center = np.asarray(boxes[i][:2], dtype=np.float32)
        future = center[None, :] + traj[:len(valid)]
        path = np.concatenate([center[None, :], future[valid]], axis=0)
        pix = bev_xy_to_px(path[:, 0], path[:, 1], xlim, ylim, width, height)
        pix = np.round(pix).astype(np.int32)

        cv2.polylines(img, [pix], False, color, thickness, cv2.LINE_AA)
        for step, point in enumerate(pix[1:], start=1):
            radius = 2 + int(step == len(pix) - 1)
            cv2.circle(img, tuple(point), radius, color, -1, cv2.LINE_AA)


def draw_bev(points, pred_boxes, pred_labels, pred_scores, gt_boxes,
             pred_trajs, gt_trajs, gt_masks, sample_name, adj_count,
             score_thr, xlim, ylim, out_path):
    width, height = 1200, 800
    img = np.full((height, width, 3), 18, dtype=np.uint8)

    if points.size:
        mask = (
            (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
            (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]))
        pts = points[mask]
        if len(pts):
            pix = bev_xy_to_px(pts[:, 0], pts[:, 1], xlim, ylim, width,
                               height)
            pix = np.round(pix).astype(np.int32)
            valid = (
                (pix[:, 0] >= 0) & (pix[:, 0] < width) &
                (pix[:, 1] >= 0) & (pix[:, 1] < height))
            pix = pix[valid]
            img[pix[:, 1], pix[:, 0]] = (150, 150, 150)

    for x in range(int(np.ceil(xlim[0] / 20) * 20), int(xlim[1]) + 1, 20):
        p0 = bev_xy_to_px(np.array([x]), np.array([ylim[0]]), xlim, ylim,
                          width, height)[0].astype(int)
        p1 = bev_xy_to_px(np.array([x]), np.array([ylim[1]]), xlim, ylim,
                          width, height)[0].astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (45, 45, 45), 1)
    for y in range(int(np.ceil(ylim[0] / 20) * 20), int(ylim[1]) + 1, 20):
        p0 = bev_xy_to_px(np.array([xlim[0]]), np.array([y]), xlim, ylim,
                          width, height)[0].astype(int)
        p1 = bev_xy_to_px(np.array([xlim[1]]), np.array([y]), xlim, ylim,
                          width, height)[0].astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (45, 45, 45), 1)

    origin = bev_xy_to_px(np.array([0.0]), np.array([0.0]), xlim, ylim,
                          width, height)[0].astype(int)
    x_axis = bev_xy_to_px(np.array([10.0]), np.array([0.0]), xlim, ylim,
                          width, height)[0].astype(int)
    y_axis = bev_xy_to_px(np.array([0.0]), np.array([10.0]), xlim, ylim,
                          width, height)[0].astype(int)
    cv2.arrowedLine(img, tuple(origin), tuple(x_axis), (80, 80, 255), 2)
    cv2.arrowedLine(img, tuple(origin), tuple(y_axis), (80, 255, 80), 2)

    draw_boxes(img, gt_boxes, None, None, xlim, ylim, 'gt', thickness=2)
    draw_boxes(img, pred_boxes, pred_labels, pred_scores, xlim, ylim, 'pred',
               score_thr=score_thr, thickness=2)
    draw_trajectories(img, gt_boxes, gt_trajs, gt_masks, xlim, ylim,
                      (60, 255, 60), thickness=2)
    draw_trajectories(img, pred_boxes, pred_trajs, None, xlim, ylim,
                      (255, 255, 255), score_thr=score_thr,
                      scores=pred_scores, thickness=2)

    pred_count = int((pred_scores >= score_thr).sum()) \
        if len(pred_scores) else 0
    title = (f'{sample_name}  pred>={score_thr:.2f}: {pred_count}  '
             f'gt: {len(gt_boxes)}  valid_adj: {adj_count}/2')
    cv2.rectangle(img, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        'green=GT box/traj, colored=prediction box, white=pred traj, '
        'red=+X front, green=+Y left',
        (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (220, 220, 220), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)

    dataset_cfg = cfg.val_dataloader.dataset if args.split == 'val' else \
        cfg.test_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    point_cloud_range = cfg.get('point_cloud_range',
                                [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0])
    xlim = (float(point_cloud_range[0]), float(point_cloud_range[3]))
    ylim = (float(point_cloud_range[1]), float(point_cloud_range[4]))

    if args.indices:
        indices = [int(x) for x in args.indices.split(',') if x.strip()]
    else:
        indices = list(range(min(args.max_frames, len(dataset))))

    for out_idx, data_idx in enumerate(indices):
        data = dataset[data_idx]
        with torch.no_grad():
            result = model.test_step(pseudo_collate([data]))[0]

        data_sample = data['data_samples']
        points = points_to_numpy(data['inputs']['points'])
        pred = result.pred_instances_3d
        pred_boxes = boxes_to_numpy(pred.bboxes_3d)
        pred_labels = tensor_to_numpy(pred.labels_3d, np.int64)
        pred_scores = tensor_to_numpy(pred.scores_3d, np.float32)
        pred_trajs = None
        if not args.no_traj and hasattr(pred, 'forecasting_3d'):
            pred_trajs = tensor_to_numpy(pred.forecasting_3d, np.float32)

        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_trajs = None
        gt_masks = None
        if getattr(data_sample, 'eval_ann_info', None) is not None:
            eval_ann = data_sample.eval_ann_info
            gt_boxes = boxes_to_numpy(
                eval_ann.get('gt_bboxes_3d'))
            if not args.no_traj and 'gt_forecasting_locs' in eval_ann:
                gt_trajs = tensor_to_numpy(
                    eval_ann.get('gt_forecasting_locs'), np.float32)
                gt_masks = tensor_to_numpy(
                    eval_ann.get('gt_forecasting_mask'), np.bool_)

        adj_points = data['inputs'].get('adj_points', [])
        adj_count = sum(1 for item in adj_points if item is not None)
        token = data_sample.metainfo.get('token', f'{data_idx:06d}')
        sample_name = f'{data_idx:06d}_{token}'
        out_path = out_dir / f'{out_idx:03d}_{sample_name}.png'
        draw_bev(points, pred_boxes, pred_labels, pred_scores, gt_boxes,
                 pred_trajs, gt_trajs, gt_masks, sample_name, adj_count,
                 args.score_thr, xlim, ylim, out_path)
        print(f'[VIS] wrote {out_path}')

    print(f'[VIS] done: {len(indices)} frames -> {out_dir}')


if __name__ == '__main__':
    os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
    main()
