# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
import torch   # 一定要有

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS
from pathlib import Path
import numpy as np
import open3d as o3d
import math
import copy
import pyvista as pv

from mmengine.runner import Runner
from mmdet3d.registry import DATASETS
from mmengine.config import Config
from mmdet3d.apis import inference_multi_modality_detector,inference_multi_modality_detector_bao

from utils.visualize_tools import save_multi_cam_images_from_path,save_multi_cam_images_with_boxes,visualize_black_bg_vista

class_names = {
    0:"Pedestrian", 
    1:"Car", 
    2:"IGV-Full", 
    3:"Truck", 
    4:"Trailer-Empty", 
    5:"Trailer-Full", 
    6:"IGV-Empty", 
    7:"Crane", 
    8:"OtherVehicle", 
    9:"Cone", 
    10:"ContainerForklift", 
    11:"Forklift", 
    12:"Lorry", 
    13:"ConstructionVehicle", 
    14:"WheelCrane"
}

def create_pointcloud(points):
    """
    points: (M, 4) -> x, y, z, intensity
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # 灰色点云
    colors = np.ones((points.shape[0], 3)) * 0.6
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_3d_box(box, color):
    """
    box: [x, y, z, dx, dy, dz, yaw, vx, vy]
    yaw: rad, z-axis
    """
    x, y, z, dx, dy, dz, yaw = box[:7]

    # Open3D 的 box 是以中心为原点的
    obb = o3d.geometry.OrientedBoundingBox(
        center=[x, y, z],
        R=o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]),
        extent=[dx, dy, dz]
    )

    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lineset.paint_uniform_color(color)

    return lineset

def visualize_black_bg(
    points,
    boxes,
    labels=None,
    scores=None,
    score_thresh=0.0,
    gt_boxes=None,
    gt_labels=None
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="PointCloud + 3D Boxes",
        width=1600,
        height=900
    )

    # ---------- 渲染参数 ----------
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = 0.8
    opt.line_width = 2.0

    # ---------- 原点坐标轴 ----------
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0,
        origin=[0, 0, 0]
    )
    vis.add_geometry(axis)

    # ---------- 点云 ----------
    pcd = create_pointcloud(points)
    vis.add_geometry(pcd)

    # =========================================================
    # 1️⃣ GT（绿色）
    # =========================================================
    if gt_boxes is not None:
        for i in range(len(gt_boxes)):
            gt_color = np.array([0.0, 1.0, 0.0])  # Green
            box3d = create_3d_box(gt_boxes[i], gt_color)
            vis.add_geometry(box3d)

    # =========================================================
    # 2️⃣ Pred（红色）
    # =========================================================
    if boxes is not None:
        for i in range(len(boxes)):
            if scores is not None and scores[i] < score_thresh:
                continue

            pred_color = np.array([1.0, 0.0, 0.0])  # Red
            box3d = create_3d_box(boxes[i], pred_color)
            vis.add_geometry(box3d)

    vis.run()
    vis.destroy_window()



def build_val_dataset(cfg):
    val_dataset_cfg = cfg.val_dataloader.dataset
    val_dataset = DATASETS.build(val_dataset_cfg)
    return val_dataset

def build_train_dataset(cfg, no_aug=False):
    train_cfg = copy.deepcopy(cfg.train_dataloader.dataset)

    if no_aug:
        print('[VIS] Use train dataset without augmentation')

        # 取出原始 pipeline（考虑 wrapper）
        if 'dataset' in train_cfg:
            pipeline = train_cfg['dataset']['pipeline']
        else:
            pipeline = train_cfg['pipeline']

        # 过滤掉增强算子
        new_pipeline = []
        for step in pipeline:
            step_type = step.get('type', '')
            if step_type in (
                'ObjectSample',
                'GlobalRotScaleTrans',
                'BEVFusionRandomFlip3D',
                'PointShuffle',
            ):
                continue
            new_pipeline.append(step)

        # 回写 pipeline
        if 'dataset' in train_cfg:
            train_cfg['dataset']['pipeline'] = new_pipeline
        else:
            train_cfg['pipeline'] = new_pipeline

        print('[VIS] Removed augmentation steps:')
        for s in pipeline:
            if s.get('type') not in [x.get('type') for x in new_pipeline]:
                print('   -', s.get('type'))

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
    # 1️⃣ LiDAR visualization
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

                visualize_black_bg(
                    points,
                    pred.bboxes_3d.tensor.cpu().numpy(),
                    pred.scores_3d.cpu().numpy(),
                    args.score_thr
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

            visualize_black_bg(
                points,
                pred.bboxes_3d.tensor.cpu().numpy(),
                pred.scores_3d.cpu().numpy(),
                args.score_thr,
                gt_boxes
            )

    # =====================================================
    # 2️⃣ Multi-modality visualization
    # =====================================================
    elif args.vis_mode == 'multi':
        print('[VIS] Multi-modality mode')

        # assert args.img_dir is not None, 'multi mode requires --img-dir'

        dataset = build_val_dataset(cfg)
        for data in dataset:
            sample = data['data_samples']

            pcd_path = sample.lidar_path
            img_path = sample.img_path
            ann_path = dataset.ann_file
            gt_info = sample.eval_ann_info
            gt_boxes = gt_info['gt_bboxes_3d'].tensor.cpu().numpy()

            with torch.no_grad():
                result, infer_data = inference_multi_modality_detector_bao(
                    model,
                    pcd_path,
                    img_path,
                    # ann_path
                )

            points = infer_data['inputs']['points'].cpu().numpy()
            pred = result.pred_instances_3d

            visualize_black_bg(
                points,
                pred.bboxes_3d.tensor.cpu().numpy(),
                pred.scores_3d.cpu().numpy(),
                args.score_thr,
                gt_boxes
            )
            
    elif args.vis_mode == 'gt':
        print('[VIS] GroundTruth only mode (TRAIN dataset)')

        focus_classes = [
            # 'WheelCrane',
            # 以后想加直接加
            # 'Forklift',
            # 'Crane',
        ]
        focus_class_ids = {
            k for k, v in class_names.items()
            if v in focus_classes
        }
        dataset = build_train_dataset(cfg, no_aug=args.no_aug)

        for data in dataset:
            # -------- points --------
            points = data['inputs']['points'].cpu().numpy()
            
            # -------- images --------
            img_paths=data['data_samples'].img_path
            
            # -------- GT --------
            gt_instances = data['data_samples'].gt_instances_3d
            gt_boxes = gt_instances.bboxes_3d.tensor.cpu().numpy()
            gt_labels = gt_instances.labels_3d.cpu().numpy()
            
            # =============================
            # 帧级过滤逻辑（关键）
            # =============================
            if len(focus_class_ids) > 0:
                # 这一帧是否包含任意一个 required class
                if not any(cls in focus_class_ids for cls in gt_labels):
                    continue

            # save_multi_cam_images_from_path(img_paths)
            save_multi_cam_images_with_boxes(
                img_paths=img_paths,
                gt_boxes=gt_boxes,
                lidar2img=data['data_samples'].lidar2img
            )
            # ✅ GT-only：boxes / labels / scores 直接传 None
            # visualize_black_bg(
            visualize_black_bg_vista(
                points=points,
                boxes=None,
                labels=None,
                scores=None,
                score_thresh=0.0,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                focus_class_ids=focus_class_ids,   # ⭐ 新名字
            )



if __name__ == '__main__':
    args = parse_args()
    main(args)
