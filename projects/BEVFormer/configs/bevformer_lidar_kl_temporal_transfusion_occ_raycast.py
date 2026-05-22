"""Ray-cast occupancy target for temporal TransFusion.

Target labels:

- 0: observed free space
- 1..15: KL semantic object classes
- 16: ground
- 17: other unannotated obstacle
- 255: unknown / ignored

Unlike the box-fill bootstrap, this config does not supervise occluded space as
empty. Free voxels are only produced along LiDAR rays before the first observed
hit; unobserved space remains ``ignore_idx``.
"""

_base_ = ['./bevformer_lidar_kl_temporal_transfusion_occ.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
occ_size = [200, 120, 10]
occ_num_classes = 18
ground_label = 16
obstacle_label = 17
ego_ignore_range = [-8.0, -2.0, -2.0, 8.0, 2.0, 6.0]
# Per-GPU batch size. With GPUs 1-7 this gives a total batch size of 14.
train_batch_size = 4
occ_class_weight = [0.05] + [1.0] * 15 + [1.0, 0.25]
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
        mode='raycast',
        min_points_per_voxel=1,
        dilation_xy=1,
        mark_unobserved_box_ignore=True,
        ray_origin=[0.0, 0.0, 0.0],
        ground_label=ground_label,
        obstacle_label=obstacle_label,
        label_scene=True,
        ground_height_threshold=0.55,
        ground_smooth_radius=3,
        fill_ground=True,
        ground_fill_radius=2,
        ground_fill_min_neighbors=5,
        remove_ground_under_obstacle=True,
        obstacle_min_points_per_voxel=2,
        obstacle_min_component_voxels=8,
        obstacle_small_component_keep_min_points=300,
        obstacle_thin_component_min_major_span=4.0,
        obstacle_thin_component_max_minor_span=2.4,
        obstacle_thin_component_max_z_span=1.6,
        obstacle_thin_component_keep_min_points=300,
        obstacle_box_ignore_margin=0.8,
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
        class_weight=occ_class_weight,
        loss_weight=1.0))

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(pipeline=train_pipeline))

work_dir = './work_dirs/bevformer_lidar_kl_temporal_transfusion_occ_raycast'
