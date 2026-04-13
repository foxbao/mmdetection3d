"""Camera-only inference + visualization for DETR3D / BEVFusion-cam.

mmdet3d's built-in Det3DLocalVisualizer skips file output for `multi-view_det`
(only mono_det / multi-modality_det branches in
``local_visualizer.py:1058`` produce drawn_img_3d that gets imwrite'd).
This script projects predicted (and GT) 3D boxes onto every camera view and
saves a stitched image per validation frame, which is what we actually want
for inspecting a multi-view camera detector like DETR3D.

Usage:
    python tools/infer_camera_vis.py \
        --config projects/DETR3D/configs/detr3d_r101_gridmask_kl.py \
        --checkpoint work_dirs/detr3d_r101_gridmask_kl/epoch_10.pth \
        --out-dir vis_detr3d \
        --score-thr 0.3 \
        --max-frames 50
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

# Reuse the existing visualization helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.visualize_tools import (  # noqa: E402
    _project_boxes_to_corners_cam,
    _stitch_max_height_grid,
    draw_boxes_on_image,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out-dir', default='vis_camera')
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-frames', type=int, default=0,
                   help='0 = all val frames')
    p.add_argument('--num-cols', type=int, default=3)
    p.add_argument('--no-gt', action='store_true',
                   help='only draw predictions, skip GT overlay')
    p.add_argument('--bev-range', type=float, default=60.0,
                   help='half-side of BEV plot in meters (square ±range)')
    p.add_argument('--bev-size', type=int, default=800,
                   help='BEV canvas size in pixels')
    p.add_argument('--no-bev', action='store_true',
                   help='disable the BEV side panel')
    p.add_argument('--match-debug', action='store_true',
                   help='only draw pred-gt pairs matched by xy distance, '
                        'and annotate each box with its bottom_z')
    p.add_argument('--match-radius', type=float, default=3.0,
                   help='xy distance threshold for greedy 1-1 matching (m)')
    return p.parse_args()


def greedy_match(pred_boxes, gt_boxes, radius):
    """Greedy 1-1 match between pred and gt by xy L2 distance.

    Returns list of (pred_idx, gt_idx) pairs.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    px = pred_boxes[:, :2]
    gx = gt_boxes[:, :2]
    d = np.linalg.norm(px[:, None, :] - gx[None, :, :], axis=-1)
    pairs = []
    used_g = set()
    order = np.argsort(d.min(axis=1))
    for pi in order:
        gi = int(np.argmin(d[pi]))
        if gi in used_g:
            row = d[pi].copy()
            for u in used_g:
                row[u] = np.inf
            gi = int(np.argmin(row))
        if d[pi, gi] <= radius and gi not in used_g:
            pairs.append((int(pi), gi))
            used_g.add(gi)
    return pairs


def draw_bev(pred_boxes, gt_boxes, half_range=60.0, size=800):
    """Top-down BEV plot with y forward/up and x right.

    GT in green, pred in red. Ego at center as a yellow triangle pointing +Y.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size // 2
    scale = size / (2.0 * half_range)  # px per meter

    def world_to_px(x, y):
        # Display convention: +Y is up/forward, +X is right.
        u = int(round(cx + x * scale))
        v = int(round(cy - y * scale))
        return u, v

    def draw_axes():
        axis_len = min(20.0, half_range * 0.35)
        origin = world_to_px(0.0, 0.0)
        x_tip = world_to_px(axis_len, 0.0)
        y_tip = world_to_px(0.0, axis_len)
        neg_x = world_to_px(-axis_len, 0.0)
        neg_y = world_to_px(0.0, -axis_len)

        cv2.arrowedLine(canvas, origin, x_tip, (0, 180, 255), 3,
                        cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(canvas, origin, y_tip, (255, 180, 0), 3,
                        cv2.LINE_AA, tipLength=0.15)
        cv2.line(canvas, origin, neg_x, (0, 90, 160), 1, cv2.LINE_AA)
        cv2.line(canvas, origin, neg_y, (120, 90, 0), 1, cv2.LINE_AA)

        cv2.putText(canvas, '+X right', (x_tip[0] + 8, x_tip[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(canvas, '+Y forward', (y_tip[0] + 8, y_tip[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(canvas, '-X', (neg_x[0] + 8, neg_x[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 160), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, '-Y', (neg_y[0] + 8, neg_y[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 90, 0), 1,
                    cv2.LINE_AA)

    # Grid every 10 m
    grid_color = (50, 50, 50)
    for r in range(10, int(half_range) + 1, 10):
        rp = int(round(r * scale))
        cv2.circle(canvas, (cx, cy), rp, grid_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, 0), (cx, size), grid_color, 1)
    cv2.line(canvas, (0, cy), (size, cy), grid_color, 1)

    def draw_boxes(boxes, color):
        if boxes is None or len(boxes) == 0:
            return
        for b in boxes:
            x, y = float(b[0]), float(b[1])
            l, w = float(b[3]), float(b[4])
            yaw = float(b[6])
            c, s = np.cos(yaw), np.sin(yaw)
            local = np.array([
                [ l / 2,  w / 2],
                [ l / 2, -w / 2],
                [-l / 2, -w / 2],
                [-l / 2,  w / 2],
            ])
            R = np.array([[c, -s], [s, c]])
            world = local @ R.T + np.array([x, y])
            pts = np.array([world_to_px(px, py) for px, py in world],
                           dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color,
                          thickness=2, lineType=cv2.LINE_AA)
            # heading line: from center to mid of front edge
            front_mid = (world[0] + world[1]) / 2
            p0 = world_to_px(x, y)
            p1 = world_to_px(front_mid[0], front_mid[1])
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

    draw_boxes(gt_boxes, (0, 255, 0))
    draw_boxes(pred_boxes, (0, 0, 255))
    draw_axes()

    # Ego marker (yellow triangle pointing up = +Y forward)
    ego = np.array([
        [cx, cy - 10],
        [cx - 7, cy + 8],
        [cx + 7, cy + 8],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [ego], (0, 255, 255))

    cv2.putText(canvas, 'BEV (lidar)', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'+/- {int(half_range)}m, 10m grid',
                (15, size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def stitch_cam_and_bev(cam_grid, bev_img):
    """Place BEV panel to the right of the camera grid, matching height."""
    h_cam = cam_grid.shape[0]
    h_bev, w_bev = bev_img.shape[:2]
    new_w = int(round(w_bev * h_cam / h_bev))
    bev_resized = cv2.resize(bev_img, (new_w, h_cam),
                             interpolation=cv2.INTER_AREA)
    return np.hstack([cam_grid, bev_resized])


def build_val_dataset(cfg):
    val_cfg = cfg.val_dataloader.dataset
    val_cfg['_scope_'] = 'mmdet3d'
    return DATASETS.build(val_cfg)


def draw_one_frame(img_paths, lidar2cam, cam2img,
                   pred_boxes, gt_boxes, num_cols):
    """Render every camera view with pred (red) + GT (green) boxes."""
    images = []
    for cam_id, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARN] cannot read {img_path}')
            continue

        T = np.asarray(lidar2cam[cam_id])
        K = np.asarray(cam2img[cam_id])[:3, :3]
        R = T[:3, :3]
        t = T[:3, 3]

        if gt_boxes is not None and len(gt_boxes) > 0:
            gt_corners = _project_boxes_to_corners_cam(gt_boxes, R, t, z_bottom=True)
            img = draw_boxes_on_image(img, gt_corners, K,
                                      color=(0, 255, 0), thickness=2)
        if pred_boxes is not None and len(pred_boxes) > 0:
            pred_corners = _project_boxes_to_corners_cam(pred_boxes, R, t, z_bottom=True)
            img = draw_boxes_on_image(img, pred_corners, K,
                                      color=(0, 0, 255), thickness=2)

        cv2.putText(img, f'CAM_{cam_id}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2,
                    cv2.LINE_AA)
        images.append(img)

    if not images:
        return None
    return _stitch_max_height_grid(images, num_cols)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    dataset = build_val_dataset(cfg)
    print(f'[INFO] val dataset size = {len(dataset)}')

    n = len(dataset) if args.max_frames == 0 else min(args.max_frames,
                                                      len(dataset))
    for idx in range(n):
        data = dataset[idx]

        with torch.no_grad():
            result = model.test_step(pseudo_collate([data]))

        pred = result[0].pred_instances_3d
        keep = pred.scores_3d > args.score_thr
        pred_boxes = pred.bboxes_3d.tensor[keep].cpu().numpy()

        ds = data['data_samples']
        gt_boxes = None
        if not args.no_gt:
            gt_info = getattr(ds, 'eval_ann_info', None)
            if gt_info is not None and 'gt_bboxes_3d' in gt_info:
                gt_boxes = gt_info['gt_bboxes_3d'].tensor.cpu().numpy()

        meta = ds.metainfo
        img_paths = ds.img_path
        lidar2cam = meta['lidar2cam']
        cam2img = meta['cam2img']

        stitched = draw_one_frame(img_paths, lidar2cam, cam2img,
                                  pred_boxes, gt_boxes, args.num_cols)
        if stitched is None:
            continue

        if not args.no_bev:
            bev = draw_bev(pred_boxes, gt_boxes,
                           half_range=args.bev_range, size=args.bev_size)
            stitched = stitch_cam_and_bev(stitched, bev)

        sample_idx = meta.get('sample_idx', idx)
        out_file = os.path.join(args.out_dir, f'{idx:05d}_{sample_idx}.jpg')
        cv2.imwrite(out_file, stitched)
        if (idx + 1) % 10 == 0 or idx == n - 1:
            print(f'[{idx + 1}/{n}] saved {out_file}  '
                  f'(pred={len(pred_boxes)}, '
                  f'gt={0 if gt_boxes is None else len(gt_boxes)})')

    print(f'[DONE] wrote {n} stitched images to {args.out_dir}')


if __name__ == '__main__':
    main()
