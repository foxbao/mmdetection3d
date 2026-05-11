"""Temporal LiDAR BEV encoder with TSA + CenterHead.

This config inherits the single-frame CenterHead baseline and configures
queue-aware loading locally. Unlike the single-frame variant, it has temporal
context from the BEV queue, so velocity supervision is enabled again for the
shared CenterHead velocity channels.
"""

_base_ = ['./bevformer_lidar_kl_singleframe_centerhead.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
queue_length = 4
batch_size = 4
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]

# Temporal fusion consumes ``ego_motion_delta`` to warp ``prev_bev``. Keep the
# train pipeline geometry-preserving for now: queue-level shared rotation/flip
# is still not enough unless the motion delta is also conjugated into the
# augmented coordinate frame. Range/name filtering and point shuffle are safe
# because they do not redefine the BEV frame.
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
    train_cfg=dict(
        pts=dict(
            # This temporal variant has BEV queue context, so velocity
            # supervision is enabled again.
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 0.2, 0.2])),
)

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        type='KlBEVFormerDataset',
        queue_length=queue_length,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))
test_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))

work_dir = './work_dirs/bevformer_lidar_kl_temporal_centerhead'
