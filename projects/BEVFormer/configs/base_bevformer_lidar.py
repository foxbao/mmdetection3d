"""LiDAR counterpart of UniAD's base BEVFormer detector config.

This is the current mainline detection-stage base for the LiDAR UniAD stack.
It uses detector-owned object queries and logit-space reference points,
matching UniAD's detection-query boundary more closely than the earlier
sigmoid-reference ablation.

The config intentionally adopts the standard mmdet3d / UniAD axis convention:
voxel coordinates ``[b, z, y, x]`` and BEV feature layout ``[B, C, Y, X]``.
The detector / head therefore carry no layout flags or swap steps. The older
BEVFusion-style XYZ path lives in ``bevformer_lidar_kl_bev_detr_base.py`` and
remains unchanged for the legacy ``BEVDETRHead`` configs.
"""

_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'projects.BEVFormer.bevformer',
        'projects.BEVFormer.bevformer.modules.lidar_bevformer_encoder',
        'projects.BEVFormer.bevformer.modules.lidar_perception_transformer',
        'projects.BEVFormer.bevformer.modules.lidar_spatial_cross_attention',
        'projects.BEVFormer.bevformer.dense_heads.bev_detr_lidar_uniad_head',
        'projects.BEVFormer.bevformer.detectors.bevformer_lidar_uniad',
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
# Standard mmdet3d SparseEncoder consumes coords as [b, z, y, x]; sparse_shape
# is listed in the same order: [Z, Y, X].
sparse_shape = [41, 960, 1600]
queue_length = 4

# Front-back symmetric port-scene classes: yaw and yaw + pi are physically
# identical (IGV-Full=2, IGV-Empty=6, WheelCrane=14). Must stay in sync with
# class_names ordering and with the evaluator's pi_symmetric_classes list.
pi_symmetric_class_indices = [2, 6, 14]

# ----------------------------------- model ----------------------------------

model = dict(
    type='BEVFormerLidarUniAD',
    num_query=600,
    embed_dims=256,
    video_test_mode=True,
    data_preprocessor=dict(
        type='BEVFormerDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000])),
    point_cloud_range=point_cloud_range,
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                          (128, 128)),
        # Each entry is per-conv padding within a stage (not per-axis). Only
        # Stage 2's last conv uses a 3-D padding tuple, which is in
        # (Z, Y, X) order to match the SparseEncoder coords convention.
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (0, 1, 1)), (0, 0)),
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
        out_channels=[128, 128],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='BEVFormerLiDARHead',
        # BEV memory throughout is [B, C, H=Y, W=X], matching UniAD's
        # query-first BEV transformer boundary and standard mmdet3d axes.
        lidar_in_channels=256,
        in_channels=256,
        num_classes=num_classes,
        num_query=600,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        embed_dims=256,
        code_size=10,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 0.2, 0.2],
        bev_h=120,
        bev_w=200,
        bev_embed_dims=256,
        transformer=dict(
            type='PerceptionTransformer',
            # UniAD-style temporal alignment lives in the transformer:
            # rotate prev_bev for ego yaw and pass ego translation as shifted
            # reference points to temporal self-attention.
            point_cloud_range=point_cloud_range,
            use_shift=True,
            rotate_prev_bev=True,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                num_heads=8,
                temporal_num_points=4,
                spatial_num_points=8,
                ffn_channels=512,
                dropout=0.1),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1,
                            num_points=4),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[
                -80.0, -48.0, -10.0, 80.0, 48.0, 10.0
            ],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=120,
            col_num_embed=200),
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
                # UniAD sets iou_cost weight=0.0 as a DETR-compatible fake
                # cost. This assigner directly matches with cls + normalized
                # 3D bbox L1, so the zero-weight IoU cost is intentionally
                # omitted.
                reg_cost=dict(type='BEVDETRBBox3DL1Cost', weight=0.25),
                pc_range=point_cloud_range,
                pi_symmetric_class_indices=pi_symmetric_class_indices))),
    test_cfg=dict(
        pts=dict(
            max_num=300,
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

train_dataloader = dict(
    batch_size=1,
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
    batch_size=1,
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
    batch_size=1,
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

work_dir = './work_dirs/base_bevformer_lidar'
