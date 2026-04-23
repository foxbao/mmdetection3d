"""KL LiDAR detector with BEVFormer-style in-sample prev_bev history.

This config keeps the single-frame BEVFusion LiDAR detector, but changes the
temporal contract to BEVFormer-style previous-frame loading inside each sample:

  - dataset attaches the previous frame via token linkage
  - pipeline loads ``prev_points`` alongside the current frame
  - model computes ``prev_bev`` on the fly, then warps it into the current
    frame before fusion
  - no scene-ordered sampler or cache-reset hook is required
"""

_base_ = ['./bevfusion_lidar_kl_base.py']

batch_size = 4

# Training schedule override for this derived config.
max_epochs = 6
val_interval = 6
warmup_epochs = min(3, max_epochs)
cosine_epochs = max(1, max_epochs - warmup_epochs)

model = dict(
    temporal_fuser=dict(
        type='PrevBEVTemporalFuser',
        in_channels=512,
        bev_xbound=[-80.0, 80.0, 0.4],
        bev_ybound=[-48.0, 48.0, 0.4],
        num_heads=8,
        dropout=0.1,
        ffn_ratio=2.0,
        use_motion_embed=True))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadPrevFramePoints',
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
            'points', 'prev_points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_bboxes', 'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
            'lidar_aug_matrix', 'scene_token', 'ego2global',
            'prev_ego2global', 'prev_bev_exists',
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
        type='LoadPrevFramePoints',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(type='PointsRangeFilter', point_cloud_range={{_base_.point_cloud_range}}),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'prev_points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views',
            'lidar_aug_matrix', 'scene_token', 'ego2global',
            'prev_ego2global', 'prev_bev_exists',
            'lidar_coord_frame'
        ]),
]

train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='KlDataset',
        data_root='data/kl_8/',
        ann_file='kl_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo={{_base_.metainfo}},
        modality={{_base_.input_modality}},
        test_mode=False,
        data_prefix={{_base_.data_prefix}},
        use_valid_flag=True,
        load_prev_frame=True,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='KlDataset',
        data_root='data/kl_8/',
        ann_file='kl_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo={{_base_.metainfo}},
        modality={{_base_.input_modality}},
        test_mode=True,
        data_prefix={{_base_.data_prefix}},
        use_valid_flag=True,
        load_prev_frame=True,
        box_type_3d='LiDAR'))
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.2,
        begin=0,
        end=warmup_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=cosine_epochs,
        eta_min_ratio=1e-4,
        begin=warmup_epochs,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=warmup_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=warmup_epochs,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=val_interval)

custom_hooks = []

work_dir = './work_dirs/bevfusion_lidar_kl_temp1_bevformer_style'
