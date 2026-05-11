"""Standalone single-frame LiDAR BEVFormer with TransFusion on KL_8.

This branch keeps the BEVFormerLidar detector shell and KL single-frame data setup,
but switches the LiDAR voxel stack to BEVFusion's sparse-axis convention:

- voxel coors entering the sparse encoder are reordered to ``[b, x, y, z]``
- sparse_shape is interpreted as ``[x_len, y_len, z_len]``
- the resulting BEV tensor is laid out as ``[B, C, X, Y]``

TransFusionHead in ``projects/BEVFusion`` is already patched to train against
that ``[X, Y]`` heatmap order, so the whole branch stays internally
consistent. As with the single-frame CenterHead baseline, this variant is
strictly single-frame, so velocity channels stay in the head but their
supervision is disabled until the temporal TransFusion variant adds temporal
context.
"""

_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'projects.BEVFormer.bevformer',
        'projects.BEVFusion.bevfusion',
        'projects.KL8',
    ],
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
num_classes = len(class_names)

# ------------------------------- voxelization -------------------------------

voxel_size = [0.1, 0.1, 0.2]
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
# BEVFusionSparseEncoder uses sparse axes in [X, Y, Z] order.
sparse_shape = [1600, 960, 41]

# ----------------------------------- model ----------------------------------

model = dict(
    type='BEVFormerLidar',
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
        type='BEVFusionSparseEncoder',
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
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=num_classes,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        common_heads=dict(
            # Keep the velocity branch in the head so the temporal TransFusion
            # variant can reuse the same output contract once fusion is added.
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-80.0, -48.0, -10.0, 80.0, 48.0, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=10),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)),
    train_cfg=dict(
        pts=dict(
            dataset='KL',
            point_cloud_range=point_cloud_range,
            grid_size=sparse_shape,
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            # Single-frame LiDAR has no temporal cue for velocity; the
            # temporal variant re-enables these last two weights after fusion
            # is added.
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 0.0, 0.0],
            pi_symmetric_class_indices=[2, 6, 14],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(
                    type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)))),
    test_cfg=dict(
        pts=dict(
            dataset='KL',
            grid_size=sparse_shape,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type=None)))

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
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.5,
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
    batch_size=8,
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
    batch_size=4,
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
test_dataloader = dict(
    batch_size=4,
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

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=6)
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
        T_max=5,
        eta_min_ratio=1e-3,
        by_epoch=True,
        begin=1,
        end=6,
        convert_to_iter_based=True),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))

log_processor = dict(window_size=50)
auto_scale_lr = dict(enable=False, base_batch_size=32)

work_dir = './work_dirs/bevformer_lidar_kl_singleframe_transfusion'
