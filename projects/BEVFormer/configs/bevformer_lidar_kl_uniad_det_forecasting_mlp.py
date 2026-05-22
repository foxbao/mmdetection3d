"""Phase-1 KL motion forecasting on frozen UniAD-style LiDAR BEV.

This config keeps the current query-driven detector fixed and trains only a
small BEV forecasting head against ``gt_forecasting_locs`` /
``gt_forecasting_mask``. The goal is to verify whether the frozen BEV feature
already carries enough motion signal before wiring forecasting into the track
query path.
"""

_base_ = ['./base_bevformer_lidar.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]

model = dict(
    train_bbox_head=False,
    forecasting_detach_bev=True,
    freeze_detector_for_forecasting=True,
    forecasting_head=dict(
        type='BEVForecastingHead',
        embed_dims=256,
        hidden_dims=256,
        num_steps=6,
        num_classes=len(class_names),
        dropout=0.1,
        pc_range=point_cloud_range,
        use_velocity=True,
        use_class_embed=True,
        motion_weight_clamp=(0.5, 5.0),
        smooth_l1_beta=0.5,
        loss_weight=1.0))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
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
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_forecasting_locs', 'gt_forecasting_mask'
        ]),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=6,
        eta_min_ratio=1e-2,
        by_epoch=True,
        begin=0,
        end=6,
        convert_to_iter_based=True),
]

val_evaluator = dict(metric=['bbox', 'forecasting'])
test_evaluator = val_evaluator

load_from = './work_dirs/base_bevformer_lidar/epoch_5.pth'
work_dir = './work_dirs/base_bevformer_lidar_forecasting_mlp'
