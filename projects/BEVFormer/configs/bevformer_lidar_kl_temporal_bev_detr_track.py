"""MOTR-style tracking fine-tune on top of the trained DETR detector.

Inherits the detection config so the head + temporal encoder + sparse stack
stay identical. The differences are:

  * detector type: ``BEVFormerLidarTrack`` (clip-level loss + online inference)
  * train loss adds a ``ClipMatcher`` driving frame-by-frame matching;
    detection-side losses on the current frame are still computed via the same
    head loss function inside the matcher (cls + L1) for a clean shared path.
  * batch_size = 1 (clip mode keeps gradients across all frames).
  * queue_length = 3 (was 4 for detection — track training is heavier).
  * train pipeline drops GlobalRotScaleTrans / RandomFlip3D — those would have
    to be sync'd across the queue and conjugated through ``ego_motion_delta``
    to be safe; first version skips them. Pack3DDetInputs adds
    ``gt_track_ids_3d`` so every frame's gt_instances_3d carries track ids.
  * load_from = epoch_12 of the detection run. Track fine-tune lr is 10x
    smaller than detection lr.
"""
_base_ = ['./bevformer_lidar_kl_temporal_bev_detr.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
queue_length = 3
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]
pi_symmetric_class_indices = [2, 6, 14]
num_classes = len(class_names)

# ---------------------------------------------------------------------- #
# Track-specific model wiring
# ---------------------------------------------------------------------- #

model = dict(
    type='BEVFormerLidarTrack',
    num_query=600,
    embed_dims=256,
    num_classes=num_classes,
    score_thresh=0.4,
    filter_score_thresh=0.3,
    miss_tolerance=3,
    qim_args=dict(
        random_drop=0.1,
        fp_ratio=0.3,
        merger_dropout=0.0,
        update_query_pos=False),
    track_loss_cfg=dict(
        type='ClipMatcher',
        num_classes=num_classes,
        pc_range=point_cloud_range,
        pi_symmetric_class_indices=pi_symmetric_class_indices,
        weight_dict=dict(loss_cls=2.0, loss_bbox=0.25, loss_iou=2.0),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='BEVDETRHungarianAssigner3D',
            cls_cost=dict(
                type='mmdet.FocalLossCost', gamma=2.0, alpha=0.25, weight=2.0),
            reg_cost=dict(type='BEVDETRBBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='BEVDETRIoU3DCost', weight=0.25),
            iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
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
            type='mmdet.L1Loss', reduction='mean', loss_weight=1.0),
        loss_iou=dict(
            type='mmdet.GIoULoss', reduction='mean', loss_weight=1.0),
        iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar')))

# ---------------------------------------------------------------------- #
# Train pipeline: keep loading + range filters; drop BEV-frame augmentations
# (need clip-level sync); add gt_track_ids_3d to Pack keys.
# ---------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------- #
# Dataloaders: batch=1, queue=3.
# ---------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------- #
# Schedule: short fine-tune from detection ckpt.
# ---------------------------------------------------------------------- #

lr = 2e-5
optim_wrapper = dict(
    optimizer=dict(lr=lr),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=2)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True,
         begin=0, end=1, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=5, eta_min_ratio=1e-3,
         by_epoch=True, begin=1, end=6, convert_to_iter_based=True),
]

load_from = './work_dirs/bevformer_lidar_kl_temporal_bev_detr/epoch_12.pth'
work_dir = './work_dirs/bevformer_lidar_kl_temporal_bev_detr_track'

# QIM branches activate only when there are active tracks (iou > 0.5). On
# frames/iterations without any carry-over, its params don't receive grad
# and DDP would abort — enable find_unused_parameters for safety.
find_unused_parameters = True

# ---------------------------------------------------------------------- #
# Evaluator: detection + tracking metrics.
# ---------------------------------------------------------------------- #

data_root = 'data/kl_8/'
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
        # The BEVFormer queue dataset drops frames that cannot form a full
        # temporal segment. Match detection eval and score tracking only on
        # frames the model actually predicts, while keeping queue_length=3.
        evaluate_predicted_samples_only=True),
]
test_evaluator = val_evaluator
