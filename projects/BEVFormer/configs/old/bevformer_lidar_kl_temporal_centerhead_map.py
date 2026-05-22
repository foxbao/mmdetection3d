"""Temporal CenterHead extension: add KL map supervision on top of BEV fusion."""

_base_ = ['./bevformer_lidar_kl_temporal_centerhead.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
batch_size = 4
map_mask_shape = (120, 200)
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]

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
    dict(
        type='GenerateKLMapMask',
        map_file='data/kl_8/map/base_map.txt',
        map_origin='data/kl_8/map/map_origin.yaml',
        point_cloud_range=point_cloud_range,
        mask_shape=map_mask_shape,
        target='drivable'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_seg_map']),
]

model = dict(
    map_head=dict(
        type='BEVMapHead',
        in_channels=512,
        hidden_channels=128,
        loss_weight=1.0))

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=batch_size)
test_dataloader = dict(batch_size=batch_size)

work_dir = './work_dirs/bevformer_lidar_kl_temporal_centerhead_map'
