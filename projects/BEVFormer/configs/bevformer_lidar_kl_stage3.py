"""Stage 3: LiDAR temporal encoder with TSA + CenterHead."""

_base_ = ['./bevformer_lidar_kl_stage2.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
batch_size = 4
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]

# Stage 3 temporal fusion consumes ``ego_motion_delta`` to warp ``prev_bev``.
# Keep the train pipeline geometry-preserving for now: queue-level shared
# rotation/flip is still not enough unless the motion delta is also conjugated
# into the augmented coordinate frame. Range/name filtering and point shuffle
# are safe because they do not redefine the BEV frame.
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
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
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

model = dict(
    data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
    point_cloud_range=point_cloud_range,
    temporal_encoder=dict(
        type='BEVTemporalEncoder',
        embed_dims=512,
        num_layers=3,
        num_heads=8,
        num_points=4,
        ffn_channels=1024,
        dropout=0.1),
)

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=batch_size)
test_dataloader = dict(batch_size=batch_size)

work_dir = './work_dirs/bevformer_lidar_kl_stage3'
