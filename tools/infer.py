# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import cv2
import mmcv
import torch   # 一定要有

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS, DATASETS
from pathlib import Path
import numpy as np
import copy
import pyvista as pv

from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis import inference_multi_modality_detector

from utils.visualize_tools import (
    class_names,
    box_to_corners_3d,
    BOX_EDGES,
    create_pyvista_box_lines,
    stitch_images_grid,
    visualize_black_bg_vista,
    save_multi_cam_images_with_boxes,
    save_multi_cam_images_from_path,
    show_multi_cam_images_from_path,
)


def project_boxes_to_image(img, boxes_3d, lidar2cam, cam_intrinsic, color, thickness=2):
    """Project 3D boxes onto a camera image (in-place).

    Args:
        img: BGR image (will be modified in-place).
        boxes_3d: (N, >=7) 3D boxes in lidar frame.
        lidar2cam: (4, 4) lidar-to-camera transform.
        cam_intrinsic: (3, 3) or (4, 4) camera intrinsic.
        color: BGR color tuple for drawing.
        thickness: line thickness.
    """
    K = np.array(cam_intrinsic)[:3, :3]
    R = np.array(lidar2cam)[:3, :3]
    t = np.array(lidar2cam)[:3, 3]

    for box in boxes_3d:
        corners_lidar = box_to_corners_3d(box, z_bottom=True)
        corners_cam = corners_lidar @ R.T + t

        # skip if all corners behind camera
        if np.all(corners_cam[:, 2] <= 0):
            continue

        # project to image (clip z<=0 to avoid division issues)
        corners_clip = corners_cam.copy()
        corners_clip[corners_clip[:, 2] <= 0, 2] = 1e-6
        proj = (K @ corners_clip.T)
        proj = (proj[:2] / proj[2:3]).T.astype(int)  # (8, 2)

        for i, j in BOX_EDGES:
            if corners_cam[i, 2] <= 0 or corners_cam[j, 2] <= 0:
                continue
            cv2.line(img, tuple(proj[i]), tuple(proj[j]), color, thickness)


def build_camera_grid(img_paths, lidar2cam=None, cam2img=None,
                      gt_boxes=None, pred_boxes=None, num_cols=3):
    """Read multi-view images, draw projected boxes, stitch into grid."""
    imgs = []
    for cam_id, p in enumerate(img_paths):
        p = Path(p)
        img = cv2.imread(str(p))
        if img is None:
            continue

        # Project boxes onto this camera
        if lidar2cam is not None and cam2img is not None:
            l2c = lidar2cam[cam_id]
            K = cam2img[cam_id]
            if gt_boxes is not None and len(gt_boxes) > 0:
                project_boxes_to_image(img, gt_boxes, l2c, K,
                                       color=(0, 255, 0), thickness=2)
            if pred_boxes is not None and len(pred_boxes) > 0:
                project_boxes_to_image(img, pred_boxes, l2c, K,
                                       color=(0, 0, 255), thickness=2)

        cam_name = p.parent.name
        cv2.putText(img, cam_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        imgs.append(img)

    grid = stitch_images_grid(imgs, num_cols)
    if grid is None:
        return None
    return cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)


def visualize_combined(
    points, boxes=None, labels=None, scores=None,
    score_thresh=0.0, gt_boxes=None, img_paths=None,
    lidar2cam=None, cam2img=None,
):
    """Combined visualization: camera images (left) + 3D scene (right)."""
    # Filter pred boxes by score for image projection
    filtered_pred = None
    if boxes is not None:
        if scores is not None:
            mask = scores >= score_thresh
            filtered_pred = boxes[mask]
        else:
            filtered_pred = boxes

    cam_grid = build_camera_grid(
        img_paths, lidar2cam=lidar2cam, cam2img=cam2img,
        gt_boxes=gt_boxes, pred_boxes=filtered_pred,
    ) if img_paths is not None else None
    has_cam = cam_grid is not None

    if has_cam:
        plotter = pv.Plotter(shape=(1, 2), window_size=(2400, 900))

        # --- Left: camera images ---
        plotter.subplot(0, 0)
        plotter.set_background('black')
        h_img, w_img = cam_grid.shape[:2]
        aspect = w_img / h_img
        plane = pv.Plane(
            center=(0, 0, 0), direction=(0, 0, 1),
            i_size=aspect * 10, j_size=10)
        tex = pv.numpy_to_texture(cam_grid)
        plotter.add_mesh(plane, texture=tex, lighting=False)
        plotter.view_xy()
        plotter.enable_image_style()

        # --- Right: 3D scene ---
        plotter.subplot(0, 1)
    else:
        plotter = pv.Plotter(window_size=(1600, 900))

    plotter.set_background('black')

    # Point cloud
    pc = pv.PolyData(points[:, :3])
    plotter.add_points(pc, color='white', point_size=1,
                       render_points_as_spheres=True)

    # Axes
    for axis_end, color in [((5, 0, 0), 'red'), ((0, 5, 0), 'green'), ((0, 0, 5), 'blue')]:
        plotter.add_mesh(pv.Line((0, 0, 0), axis_end), color=color, line_width=3)

    # GT boxes (green)
    if gt_boxes is not None:
        for box in gt_boxes:
            plotter.add_mesh(create_pyvista_box_lines(box), color='green', line_width=2)

    # Pred boxes (red)
    if boxes is not None:
        for i, box in enumerate(boxes):
            if scores is not None and scores[i] < score_thresh:
                continue
            plotter.add_mesh(create_pyvista_box_lines(box), color='red', line_width=2)

    plotter.show()


def build_val_dataset(cfg):
    val_dataset_cfg = cfg.val_dataloader.dataset
    return DATASETS.build(val_dataset_cfg)

def build_train_dataset(cfg, no_aug=False):
    train_cfg = copy.deepcopy(cfg.train_dataloader.dataset)

    if no_aug:
        print('[VIS] Use train dataset without augmentation')

        if 'dataset' in train_cfg:
            pipeline = train_cfg['dataset']['pipeline']
        else:
            pipeline = train_cfg['pipeline']

        aug_types = {
            'ObjectSample',
            'GlobalRotScaleTrans',
            'BEVFusionRandomFlip3D',
            'PointShuffle',
        }
        new_pipeline = [s for s in pipeline if s.get('type', '') not in aug_types]

        if 'dataset' in train_cfg:
            train_cfg['dataset']['pipeline'] = new_pipeline
        else:
            train_cfg['pipeline'] = new_pipeline

        removed = [s.get('type') for s in pipeline if s not in new_pipeline]
        print('[VIS] Removed augmentation steps:')
        for t in removed:
            print('   -', t)

    return DATASETS.build(train_cfg)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pcd_dir', type=str, help='Point cloud file')
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--vis-mode',
        type=str,
        default='lidar',
        choices=['lidar', 'multi','gt'],
        help='visualization mode: lidar or multi-modality'
    )
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo1', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    parser.add_argument(
        '--no-aug',
        action='store_true',
        help='disable train data augmentation for GT visualization'
    )
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.config)

    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    # =====================================================
    # 1. LiDAR visualization
    # =====================================================
    if args.vis_mode == 'lidar':
        print('[VIS] LiDAR mode')

        if args.pcd_dir is not None:
            pcd_dir = Path(args.pcd_dir)
            for pcd in sorted(pcd_dir.glob('*.bin')):
                with torch.no_grad():
                    result, data = inference_detector(model, [str(pcd)])

                points = data[0]['inputs']['points'].cpu().numpy()
                pred = result[0].pred_instances_3d

                visualize_black_bg_vista(
                    points,
                    pred.bboxes_3d.tensor.cpu().numpy(),
                    labels=pred.labels_3d.cpu().numpy(),
                    scores=pred.scores_3d.cpu().numpy(),
                    score_thresh=args.score_thr,
                )
            return

        dataset = build_val_dataset(cfg)
        for data in dataset:
            pcd_path = data['data_samples'].lidar_path
            gt_info = data['data_samples'].eval_ann_info
            gt_boxes = gt_info['gt_bboxes_3d'].tensor.cpu().numpy()

            with torch.no_grad():
                result, infer_data = inference_detector(model, [pcd_path])

            points = infer_data[0]['inputs']['points'].cpu().numpy()
            pred = result[0].pred_instances_3d

            visualize_black_bg_vista(
                points,
                pred.bboxes_3d.tensor.cpu().numpy(),
                labels=pred.labels_3d.cpu().numpy(),
                scores=pred.scores_3d.cpu().numpy(),
                score_thresh=args.score_thr,
                gt_boxes=gt_boxes,
            )

    # =====================================================
    # 2. Multi-modality visualization
    # =====================================================
    elif args.vis_mode == 'multi':
        print('[VIS] Multi-modality mode')

        dataset = build_val_dataset(cfg)
        for idx, data in enumerate(dataset):
            with torch.no_grad():
                result = model.test_step(
                    pseudo_collate([data])
                )
            points = data['inputs']['points'].cpu().numpy()
            pred = result[0].pred_instances_3d

            # GT boxes
            gt_info = data['data_samples'].eval_ann_info
            gt_boxes = gt_info['gt_bboxes_3d'].tensor.cpu().numpy() \
                if gt_info is not None and 'gt_bboxes_3d' in gt_info else None

            # Get projection matrices from metainfo
            meta = data['data_samples'].metainfo
            lidar2cam = np.array(meta['lidar2cam'])
            cam2img = np.array(meta['cam2img'])

            visualize_combined(
                points,
                pred.bboxes_3d.tensor.cpu().numpy(),
                labels=pred.labels_3d.cpu().numpy(),
                scores=pred.scores_3d.cpu().numpy(),
                score_thresh=args.score_thr,
                gt_boxes=gt_boxes,
                img_paths=data['data_samples'].img_path,
                lidar2cam=lidar2cam,
                cam2img=cam2img,
            )

    # =====================================================
    # 3. GroundTruth only (train dataset)
    # =====================================================
    elif args.vis_mode == 'gt':
        print('[VIS] GroundTruth only mode (TRAIN dataset)')

        focus_classes = [
            # 'WheelCrane',
            # 'Forklift',
            # 'Crane',
        ]
        focus_class_ids = {
            k for k, v in class_names.items()
            if v in focus_classes
        }
        dataset = build_train_dataset(cfg, no_aug=args.no_aug)

        for data in dataset:
            points = data['inputs']['points'].cpu().numpy()
            img_paths = data['data_samples'].img_path

            gt_instances = data['data_samples'].gt_instances_3d
            gt_boxes = gt_instances.bboxes_3d.tensor.cpu().numpy()
            gt_labels = gt_instances.labels_3d.cpu().numpy()

            # 帧级过滤：只看包含 focus_classes 的帧
            if len(focus_class_ids) > 0:
                if not any(cls in focus_class_ids for cls in gt_labels):
                    continue

            save_multi_cam_images_with_boxes(
                img_paths=img_paths,
                gt_boxes=gt_boxes,
                data_sample=data['data_samples'],
                save_dir='vis_multi'
            )
            visualize_black_bg_vista(
                points=points,
                boxes=None,
                labels=None,
                scores=None,
                score_thresh=0.0,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                focus_class_ids=focus_class_ids,
            )


if __name__ == '__main__':
    args = parse_args()
    main(args)
