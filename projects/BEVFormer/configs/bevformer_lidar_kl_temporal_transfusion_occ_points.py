"""Point-conditioned occupancy target for temporal TransFusion.

Compared with ``bevformer_lidar_kl_temporal_transfusion_occ.py``, this config
does not fill whole GT boxes. It labels only voxels that contain LiDAR points
inside a GT box, optionally expands them by one voxel in BEV, and marks the
remaining box interior as ``ignore_idx`` instead of empty.
"""

_base_ = ['./bevformer_lidar_kl_temporal_transfusion_occ.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
occ_size = [200, 120, 10]
ego_ignore_range = [-8.0, -2.0, -2.0, 8.0, 2.0, 6.0]
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
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='GenerateKLOccFromBoxes',
        point_cloud_range=point_cloud_range,
        occ_size=occ_size,
        empty_idx=0,
        ignore_idx=255,
        label_offset=1,
        mode='points_in_boxes',
        min_points_per_voxel=1,
        dilation_xy=1,
        mark_unobserved_box_ignore=True,
        ego_ignore_range=ego_ignore_range,
        current_frame_only=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occ']),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = './work_dirs/bevformer_lidar_kl_temporal_transfusion_occ_points'
