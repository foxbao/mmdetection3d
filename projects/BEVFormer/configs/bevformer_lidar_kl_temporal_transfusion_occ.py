"""Temporal TransFusion with lightweight box-derived occupancy supervision.

This is the first occupancy bootstrap config: targets are generated online
from KL 3D boxes instead of precomputed dense scene completion labels. The
canonical occupancy grid is ``[X, Y, Z]`` with label 0 as empty/background and
box class labels shifted by +1.
"""

_base_ = ['./bevformer_lidar_kl_temporal_transfusion.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
occ_size = [200, 120, 10]
occ_num_classes = 16
ego_ignore_range = [-8.0, -2.0, -2.0, 8.0, 2.0, 6.0]
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]

# Occupancy logits add a moderate memory cost. Start lower than the temporal
# detector baseline, then raise this after a smoke train confirms headroom.
train_batch_size = 2

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
    dict(
        type='GenerateKLOccFromBoxes',
        point_cloud_range=point_cloud_range,
        occ_size=occ_size,
        empty_idx=0,
        label_offset=1,
        mode='bbox_fill',
        ego_ignore_range=ego_ignore_range,
        current_frame_only=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occ']),
]

model = dict(
    occ_head=dict(
        type='BEVOccHead2D',
        in_channels=512,
        hidden_channels=128,
        num_classes=occ_num_classes,
        num_z=occ_size[2],
        empty_idx=0,
        empty_weight=0.05,
        loss_weight=1.0))

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(pipeline=train_pipeline))

work_dir = './work_dirs/bevformer_lidar_kl_temporal_transfusion_occ'
