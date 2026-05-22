"""LiDAR-only BEVFormer on KL_8 — single-frame CenterHead baseline.

Standalone config. The only project-local Python it depends on is
``projects.BEVFormer.bevformer``; everything else (voxel encoder,
backbone, neck, head, dataset, transforms) comes from mmdet3d's
built-in modules.

This config wires the standard mmdet3d LiDAR voxel stack to a CenterHead so
we can verify the detector trains end-to-end. Because this variant is
strictly single-frame, velocity channels are kept in the head for shape
compatibility but their supervision is disabled. Later temporal variants
replace the post-neck path with a vendored BEVFormer Temporal
Self-Attention encoder.
"""

_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.BEVFormer.bevformer', 'projects.KL8'],
    allow_failed_imports=False,
)

# ----------------------------- dataset / classes -----------------------------

dataset_type = 'KlDataset'
data_root = 'data/kl_8/'
data_prefix = dict(
    pts='v1.0-trainval/samples',
    img='v1.0-trainval/sample',
    sweeps='v1.0-trainval/samples')
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]
metainfo = dict(classes=class_names)

# ------------------------------- voxelization -------------------------------

voxel_size = [0.1, 0.1, 0.2]
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
# Sparse grid in (D, H, W) = (Z, Y, X) order — mmdet3d's SparseEncoder
# convention. Outputs (B, C, Y, X) BEV which is what CenterHead expects.
# Range -80..80 X / -48..48 Y / -2..6 Z at voxel 0.1/0.1/0.2:
#   X bins = 1600, Y bins = 960, Z bins = 40 -> sparse_shape Z=41 (one extra)
sparse_shape = [41, 960, 1600]

# ----------------------------------- model ----------------------------------

model = dict(
    type='BEVFormerLidar',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000])),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                          (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=512,
        # Single task with all KL classes — keeps the single-frame head
        # minimal. The later temporal CenterHead variant reuses the same
        # detection contract and adds a BEVFormer-style temporal encoder,
        # so per-class anchor/task tuning is deferred.
        tasks=[dict(num_class=len(class_names), class_names=class_names)],
        common_heads=dict(
            # Keep the velocity branch in the head so later temporal stages can
            # reuse the same output contract without reshaping checkpoints.
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-100.0, -60.0, -10.0, 100.0, 60.0, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1600, 960, 41],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # Single-frame LiDAR has no temporal cue for velocity, so
            # supervision is disabled here and only re-enabled once temporal
            # context exists.
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            # Front-back symmetric classes: allow theta / theta+pi target
            # flipping in CenterHead bbox loss, matching TransFusionHead.
            pi_symmetric_class_indices=[2, 6, 14])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-100.0, -60.0, -10.0, 100.0, 60.0, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

# -------------------------------- pipelines --------------------------------

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

# ------------------------------- dataloaders -------------------------------

train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=12,
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

# -------------------------------- evaluator --------------------------------

val_evaluator = dict(
    type='KlMetric',
    data_root=data_root,
    ann_file=data_root + 'kl_infos_val.pkl',
    metric='bbox',
    point_cloud_range=point_cloud_range,
    pi_symmetric_classes=['IGV-Full', 'IGV-Empty', 'WheelCrane'],
    backend_args=backend_args)
test_evaluator = val_evaluator

# ---------------------------- training schedule ----------------------------

lr = 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Short schedule for the single-frame baseline.
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=6)
val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=5, eta_min_ratio=1e-3, by_epoch=True,
         begin=1, end=6, convert_to_iter_based=True),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))

log_processor = dict(window_size=50)
auto_scale_lr = dict(enable=False, base_batch_size=32)

work_dir = './work_dirs/bevformer_lidar_kl_singleframe_centerhead'
