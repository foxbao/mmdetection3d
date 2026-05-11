"""Temporal TSA on top of the single-frame TransFusion detector.

The sparse stack and TransFusion head stay in ``[X, Y]`` layout. The current
BEVFormer temporal modules still assume ``[Y, X]``, so the detector adapts the
layout only at the temporal boundary. Queue-aware dataloading is configured
here directly; there is no separate stage2 config layer. Because this stage
has temporal context, it also re-enables supervision for the shared
TransFusion velocity channels.
"""

_base_ = ['./bevformer_lidar_kl_singleframe_transfusion.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
queue_length = 4
# This temporal variant keeps a full BEV queue in memory and adds temporal
# attention on top. On an idle 24 GB RTX 4090, per-GPU train batch size 4 is
# stable. Evaluation is kept at 1 to leave more memory headroom for decoding
# and metrics.
train_batch_size = 4
eval_batch_size = 1
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
    point_cloud_range=point_cloud_range,
    bev_feature_layout='xy',
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
            # This temporal variant has BEV queue context, so it can supervise
            # the TransFusion velocity channels again.
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 0.2, 0.2])),
)

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        type='KlBEVFormerDataset',
        queue_length=queue_length,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=eval_batch_size,
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))
test_dataloader = dict(
    batch_size=eval_batch_size,
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))

work_dir = './work_dirs/bevformer_lidar_kl_temporal_transfusion'
