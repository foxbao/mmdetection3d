"""Common KL LiDAR BEVDETR experiment base.

This file intentionally avoids choosing a concrete BEV encoder. Temporal
LiDAR BEVFormer and the UniAD-style LiDAR BEV encoder both inherit the same
dataset, LiDAR backbone, BEVDETR decoder, optimizer, evaluator, and schedule
settings from here.
"""

_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'projects.BEVFormer.bevformer',
        'projects.KL8',
    ],
    allow_failed_imports=False,
)

# ----------------------------- dataset / classes -----------------------------

dataset_type = 'KlBEVFormerDataset'
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
num_classes = len(class_names)

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
voxel_size = [0.1, 0.1, 0.2]
# SparseEncoderXYZ uses sparse axes in [X, Y, Z] order.
sparse_shape = [1600, 960, 41]
queue_length = 4

# Front-back symmetric port-scene classes: yaw and yaw + pi are physically
# identical (IGV-Full=2, IGV-Empty=6, WheelCrane=14). Must stay in sync with
# class_names ordering and with the evaluator's pi_symmetric_classes list.
pi_symmetric_class_indices = [2, 6, 14]

# ----------------------------------- model ----------------------------------

model = dict(
    data_preprocessor=dict(
        type='BEVFormerDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000])),
    voxel_coord_order='xyz',
    bev_feature_layout='xy',
    point_cloud_range=point_cloud_range,
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoderXYZ',
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                          (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
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
        type='BEVDETRHead',
        in_channels=512,
        num_classes=num_classes,
        num_query=600,
        embed_dims=256,
        num_decoder_layers=6,
        num_heads=8,
        num_points=4,
        ffn_channels=1024,
        dropout=0.1,
        code_size=10,
        pc_range=point_cloud_range,
        bev_feature_layout='xy',
        with_box_refine=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 0.2, 0.2],
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=2.0),
        loss_bbox=dict(
            type='mmdet.L1Loss',
            reduction='mean',
            loss_weight=0.25)),
    train_cfg=dict(
        pts=dict(
            pi_symmetric_class_indices=pi_symmetric_class_indices,
            assigner=dict(
                type='BEVDETRHungarianAssigner3D',
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=2.0),
                reg_cost=dict(type='BEVDETRBBox3DL1Cost', weight=0.25),
                pc_range=point_cloud_range,
                pi_symmetric_class_indices=pi_symmetric_class_indices))),
    test_cfg=dict(
        pts=dict(
            max_num=500,
            score_threshold=0.05,
            post_center_range=[
                -80.0, -48.0, -10.0, 80.0, 48.0, 10.0
            ])))

# -------------------------------- pipelines --------------------------------

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
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
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

# ------------------------------- dataloaders -------------------------------

# Batch sizes are per GPU in distributed training.
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kl_infos_train.pkl',
        queue_length=queue_length,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        use_valid_flag=True,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kl_infos_val.pkl',
        queue_length=queue_length,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kl_infos_val.pkl',
        queue_length=queue_length,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

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

lr = 2e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Keep this DETR-head run longer than the TransFusion schedule. The head is
# trained from scratch and is still improving after epoch 2.
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=11,
        eta_min_ratio=1e-3,
        by_epoch=True,
        begin=1,
        end=12,
        convert_to_iter_based=True),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))

log_processor = dict(window_size=50)
auto_scale_lr = dict(enable=False, base_batch_size=32)
