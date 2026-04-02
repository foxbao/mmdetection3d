"""BEVFusion camera-only config for KL dataset.

Based on the working NuScenes config (bevfusion_cam_voxel0075_..._nus-3d.py).
Only changed what is necessary for the KL dataset:
  - Dataset type, paths, class names, task heads
  - Image size: KL stores 640x512 (pre-resized from 1920x1536)
  - No multi-sweep (KL has single-frame LiDAR only)
  - No velocity annotations → vel code_weights zeroed
  - Backbone: ResNet50 (vs NuScenes Swin-T), to handle larger 512x640 images
  - Spatial range narrowed for short-range camera debug: x ±48m, y ±64m

Spatial chain:
  xbound [-48, 48, 0.4], ybound [-64, 64, 0.4] → BEV 240x320
  DepthLSSTransform downsample=2 → 120x160
  GeneralizedResNet: 120x160 → 60x80 → 30x40 → 30x40
  LSSFPN scale_factor=4 → 120x160
  CenterHead: grid [960, 1280] / out_factor 8 = 120x160 ✓
"""

_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

# ---- KL dataset ----
point_cloud_range = [-48.0, -64.0, -2.0, 48.0, 64.0, 6.0]
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane'
]
metainfo = dict(classes=class_names)
dataset_type = 'KlDataset'
data_root = 'data/kl_8/'
data_prefix = dict(
    pts='v1.0-trainval/samples',
    img='v1.0-trainval/sample',
    sweeps='v1.0-trainval/samples')
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# ---- Model (same architecture as NuScenes, adapted for KL images) ----
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    # ResNet50 instead of Swin-T: KL images are 512x640 (larger than NuScenes 256x704)
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[512, 1024, 2048],  # ResNet50 channels (vs Swin [192,384,768])
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[512, 640],    # KL stored image size
        feature_size=[64, 80],    # image_size / backbone_stride(8)
        xbound=[-48.0, 48.0, 0.4],
        ybound=[-64.0, 64.0, 0.4],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    # BEV backbone + neck: identical to NuScenes
    pts_backbone=dict(
        type='GeneralizedResNet',
        in_channels=80,
        blocks=[[2, 128, 2], [2, 256, 2], [2, 512, 1]]),
    pts_neck=dict(
        type='LSSFPN',
        in_channels=[512, 128],
        in_indices=[-1, 0],
        out_channels=256,
        scale_factor=4),
    # CenterHead: 10 task heads for 15 KL classes (grouped by shape/size)
    bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Car']),
            dict(num_class=2, class_names=['IGV-Full', 'IGV-Empty']),
            dict(num_class=2, class_names=['Truck', 'Lorry']),
            dict(num_class=2, class_names=['Trailer-Empty', 'Trailer-Full']),
            dict(num_class=1, class_names=['Crane']),
            dict(num_class=2, class_names=['OtherVehicle', 'ConstructionVehicle']),
            dict(num_class=2, class_names=['ContainerForklift', 'Forklift']),
            dict(num_class=1, class_names=['Cone']),
            dict(num_class=1, class_names=['WheelCrane']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=[-55.0, -72.0, -10.0, 55.0, 72.0, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[960, 1280, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # KL has NO velocity → zero out vel loss weights
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        test_cfg=dict(
            post_center_limit_range=[-55.0, -72.0, -10.0, 55.0, 72.0, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            # per-task min_radius: sized by object footprint
            # Ped, Car, IGV, Truck, Trailer, Crane, Other, Forklift, Cone, WheelCrane
            min_radius=[1, 4, 8, 10, 12, 10, 8, 4, 0.5, 10],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            # per-task NMS type (ref NuScenes YAML)
            nms_type=['circle', 'rotate', 'rotate', 'rotate', 'rotate',
                      'rotate', 'rotate', 'rotate', 'circle', 'rotate'],
            nms_scale=[[2.5], [1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                       [1.0], [1.0, 1.0], [1.0, 1.0], [2.5], [1.0]],
            pre_max_size=1000,
            post_max_size=200,
            nms_thr=0.2)))

# ---- ObjectSample (same structure as NuScenes) ----
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kl_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points={
            'Pedestrian': 10, 'Car': 50, 'IGV-Full': 50, 'Truck': 50,
            'Trailer-Empty': 50, 'Trailer-Full': 50, 'IGV-Empty': 50,
            'Crane': 50, 'OtherVehicle': 50, 'Cone': 10,
            'ContainerForklift': 50, 'Forklift': 50, 'Lorry': 50,
            'ConstructionVehicle': 50, 'WheelCrane': 100
        }),
    classes=class_names,
    sample_groups={
        'Pedestrian': 5, 'Car': 5, 'IGV-Full': 5, 'Truck': 5,
        'Trailer-Empty': 5, 'Trailer-Full': 5, 'IGV-Empty': 5,
        'Crane': 5, 'OtherVehicle': 0, 'Cone': 5,
        'ContainerForklift': 5, 'Forklift': 1, 'Lorry': 1,
        'ConstructionVehicle': 5, 'WheelCrane': 1
    },
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args))

# ---- Pipelines ----
# KL differences from NuScenes:
#   - No LoadPointsFromMultiSweeps (KL has single frame only)
#   - LiDAR: load_dim=5 use_dim=4 (x,y,z,intensity)
#   - ImageAug3D: final_dim=[512,640], resize_lim=[0.9,1.1]
#     (KL images already stored at 640x512, mild augmentation)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ImageAug3D',
        final_dim=[512, 640],
        resize_lim=[0.9, 1.1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.3925, 0.3925],
        translation_std=0),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[512, 640],
        resize_lim=[1.0, 1.0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views'
        ])
]

# ---- Dataloaders (same structure as NuScenes) ----
train_dataloader = dict(
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kl_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            box_type_3d='LiDAR')))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kl_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='KlMetric',
    data_root=data_root,
    ann_file=data_root + 'kl_infos_val.pkl',
    point_cloud_range=point_cloud_range,
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ---- Optimizer (from NuScenes, clip_grad lowered to 5 for stability) ----
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

# ---- LR schedule: same cyclic policy as NuScenes (40% warmup, 60% decay) ----
# NuScenes: 50 epochs, warmup 0-20, decay 20-50
# KL: 24 epochs, warmup 0-10, decay 10-24  (same 40/60 ratio)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.2,
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=14,
        eta_min_ratio=1e-4,
        begin=10,
        end=24,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=10,
        end=24,
        by_epoch=True,
        convert_to_iter_based=True)
]

# ---- Runtime ----
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=5)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=16)

log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_last=True))

# Disable ObjectSample at 75% of training (NuScenes: epoch 15/50=30%, here: 18/24=75%)
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=18)]
