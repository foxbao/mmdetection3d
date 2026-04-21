"""Temporal BEVFusion + MotionHead on KL (no augmentation).

Extends the temporal noaug config with a MotionHead that predicts 6-step
future trajectories for each detected object.
"""

_base_ = ['./bevfusion_lidar_kl_temp2_noaug.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane'
]

model = dict(
    motion_head=dict(
        type='MotionHead',
        in_channels=128,
        forecast_steps=6,
        hidden_channels=256,
        num_layers=2,
        dropout=0.1,
        loss_weight=0.5))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=5, use_dim=4, backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='LoadTemporalData', load_dim=5, use_dim=4,
         min_time_diff=0.2, max_time_diff=1.2),
    dict(type='PointsRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
               'gt_bboxes', 'gt_labels',
               'adj_points', 'adj_ego_motions',
               'gt_forecasting_locs', 'gt_forecasting_mask'],
         meta_keys=[
             'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
             'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
             'lidar_path', 'img_path', 'transformation_3d_flow',
             'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
             'img_aug_matrix', 'lidar_aug_matrix', 'ego2global',
             'lidar_coord_frame'])
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='kl_infos_train_with_velocity.pkl',
            pipeline=train_pipeline)))

val_dataloader = dict(
    dataset=dict(ann_file='kl_infos_val_with_velocity.pkl'))
test_dataloader = val_dataloader

val_evaluator = dict(
    metric=['bbox', 'forecasting'],
    forecast_match_dist_thr=2.0,
    forecast_miss_thr=2.0)
test_evaluator = val_evaluator

work_dir = './work_dirs/bevfusion_lidar_kl_temp2_noaug_motion6'
