"""Track-query motion ablation on top of the UniAD-like LiDAR tracker.

This keeps det/track unchanged and adds an optional motion head that only
consumes the final ``outs_track`` state. The motion branch is detached from
the shared track graph, so it does not backprop into detection or tracking.
"""

_base_ = ['./uniad_lidar_kl_track.py']

data_root = 'data/kl_8/'
train_batch_size = 1
eval_batch_size = 1
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]

model = dict(
    freeze_lidar_backbone=True,
    freeze_lidar_neck=True,
    freeze_bev_encoder=True,
    motion_head=dict(
        type='TrackMotionHead',
        embed_dims=256,
        hidden_dims=256,
        num_steps=6,
        num_classes=15,
        dropout=0.1,
        pc_range=[-80.0, -48.0, -2.0, 80.0, 48.0, 6.0],
        use_center=True,
        use_velocity=True,
        use_class_embed=True,
        motion_weight_clamp=(0.5, 5.0),
        smooth_l1_beta=0.5,
        loss_weight=1.0),
)

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
         keys=[
             'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_track_ids_3d',
             'gt_forecasting_locs', 'gt_forecasting_mask'
         ]),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=eval_batch_size)
test_dataloader = dict(batch_size=eval_batch_size)

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

work_dir = './work_dirs/uniad_lidar_kl_track_motion_mlp'
load_from = './work_dirs/uniad_lidar_kl_track/epoch_5.pth'

val_evaluator = [
    dict(
        type='KlMetric',
        data_root=data_root,
        ann_file=data_root + 'kl_infos_val.pkl',
        metric=['bbox', 'forecasting'],
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
