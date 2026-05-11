"""Multi-frame ray-cast occupancy target for temporal TransFusion.

This keeps the same OCC network as ``occ_raycast.py`` and only changes target
generation. Current-frame points still define free-space rays and object hits;
history frames only densify static scene/background hits after ego-motion
alignment.
"""

_base_ = ['./bevformer_lidar_kl_temporal_transfusion_occ_raycast.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
occ_size = [200, 120, 10]
ground_label = 16
obstacle_label = 17
ego_ignore_range = [-8.0, -2.0, -2.0, 8.0, 2.0, 6.0]
train_batch_size = 4
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
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_track_ids_3d']),
]

queue_post_pipeline = [
    dict(
        type='GenerateKLMultiFrameOccFromQueue',
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
        obstacle_min_component_voxels=4,
        obstacle_box_ignore_margin=0.8,
        ego_ignore_range=ego_ignore_range,
        aggregate_history_scene=True,
        history_scene_only=True,
        aggregate_dynamic_instances=False,
        dynamic_instance_min_points=1),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        pipeline=train_pipeline,
        post_pipeline=queue_post_pipeline))

work_dir = './work_dirs/' \
           'bevformer_lidar_kl_temporal_transfusion_occ_raycast_multiframe'
