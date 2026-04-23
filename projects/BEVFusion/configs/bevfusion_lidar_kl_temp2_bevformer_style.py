"""KL LiDAR detector with BEVFormer-style short history recurrence.

This variant extends the `temp1` prev_bev config from a single previous frame
to a short oldest-to-newest queue:

  - dataset attaches `t-2, t-1` via token linkage
  - pipeline loads `prev_points_queue`
  - model recursively rolls the queue into one history BEV
  - final current frame still uses the same PrevBEVTemporalFuser
"""

_base_ = ['./bevfusion_lidar_kl_temp1_bevformer_style.py']

num_prev_frames = 2

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadPrevFrameQueuePoints',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range={{_base_.point_cloud_range}}),
    dict(type='ObjectRangeFilter', point_cloud_range={{_base_.point_cloud_range}}),
    dict(type='ObjectNameFilter', classes={{_base_.class_names}}),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'prev_points_queue', 'img', 'gt_bboxes_3d',
            'gt_labels_3d', 'gt_bboxes', 'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
            'lidar_aug_matrix', 'scene_token', 'ego2global',
            'prev_ego2global_queue', 'prev_bev_exists_queue',
            'lidar_coord_frame'
        ]),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadPrevFrameQueuePoints',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(type='PointsRangeFilter', point_cloud_range={{_base_.point_cloud_range}}),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'prev_points_queue', 'gt_bboxes_3d',
              'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views',
            'lidar_aug_matrix', 'scene_token', 'ego2global',
            'prev_ego2global_queue', 'prev_bev_exists_queue',
            'lidar_coord_frame'
        ]),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        load_prev_frame=False,
        load_prev_frame_queue=num_prev_frames))

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        load_prev_frame=False,
        load_prev_frame_queue=num_prev_frames))
test_dataloader = val_dataloader

work_dir = './work_dirs/bevfusion_lidar_kl_temp2_bevformer_style'
