"""Archived pre-merge track config name for old work-dir bookkeeping.

The implementation classes now use logit-space references directly under the
main class names, so this file no longer restores a separate sigmoid-reference
code path by itself.
"""

_base_ = ['./bevformer_lidar_kl_uniad_bev_detr_sigmoid_legacy.py']

custom_imports = dict(
    imports=[
        'projects.BEVFormer.bevformer',
        'projects.BEVFormer.bevformer.modules.lidar_bevformer_encoder',
        'projects.BEVFormer.bevformer.modules.lidar_perception_transformer',
        'projects.BEVFormer.bevformer.modules.lidar_spatial_cross_attention',
        'projects.BEVFormer.bevformer.dense_heads.bev_detr_lidar_uniad_head',
        'projects.BEVFormer.bevformer.detectors.bevformer_lidar_uniad',
        'projects.BEVFormer.bevformer.detectors.bevformer_lidar_uniad_track',
        'projects.KL8',
    ],
    allow_failed_imports=False,
)

data_root = 'data/kl_8/'
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]
queue_length = 3
num_classes = len(class_names)
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
pi_symmetric_class_indices = [2, 6, 14]

model = dict(
    type='UniADTrackLiDAR',
    num_query=600,
    embed_dims=256,
    num_classes=num_classes,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    miss_tolerance=5,
    reset_track_query_each_frame=False,
    qim_args=dict(
        random_drop=0.1,
        fp_ratio=0.3,
        merger_dropout=0.0,
        update_query_pos=True),
    track_loss_cfg=dict(
        type='ClipMatcher',
        num_classes=num_classes,
        pc_range=point_cloud_range,
        pi_symmetric_class_indices=pi_symmetric_class_indices,
        weight_dict=dict(loss_cls=2.0, loss_bbox=0.25),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='BEVDETRHungarianAssigner3D',
            cls_cost=dict(
                type='mmdet.FocalLossCost', gamma=2.0, alpha=0.25, weight=2.0),
            reg_cost=dict(type='BEVDETRBBox3DL1Cost', weight=0.25),
            pc_range=point_cloud_range,
            pi_symmetric_class_indices=pi_symmetric_class_indices),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=1.0)))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=5, use_dim=4, backend_args=None),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_track_ids_3d']),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='KlBEVFormerDataset',
        queue_length=queue_length,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))

lr = 2e-4
optim_wrapper = dict(
    optimizer=dict(lr=lr),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False,
         begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=6, eta_min_ratio=1e-3,
         by_epoch=True, begin=0, end=6, convert_to_iter_based=True),
]

load_from = (
    './work_dirs/bevformer_lidar_kl_uniad_bev_detr_sigmoid_legacy/'
    'epoch_2.pth')
work_dir = './work_dirs/bevformer_lidar_kl_uniad_bev_detr_track_sigmoid_legacy'

find_unused_parameters = True

val_evaluator = [
    dict(
        type='KlMetric',
        data_root=data_root,
        ann_file=data_root + 'kl_infos_val.pkl',
        metric='bbox',
        point_cloud_range=point_cloud_range,
        pi_symmetric_classes=['IGV-Full', 'IGV-Empty', 'WheelCrane'],
        backend_args=None),
    dict(
        type='KlTrackingMetric',
        ann_file=data_root + 'kl_infos_val.pkl',
        match_threshold=2.0,
        num_thresholds=40,
        evaluate_predicted_samples_only=True),
]
test_evaluator = val_evaluator
