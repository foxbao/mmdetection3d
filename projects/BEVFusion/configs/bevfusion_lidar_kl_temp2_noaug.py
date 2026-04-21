"""No-augmentation variant of temporal LiDAR BEVFusion on KL.

This config is for isolating temporal BEV fusion quality. It keeps temporal
history in the original LiDAR coordinate frame by disabling DB sampling,
random global rotation/scale/translation, and BEV flips.
"""

_base_ = ['./bevfusion_lidar_kl_temp2.py']

# Re-declare variables used in pipeline (base Python vars are not inherited)
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane'
]

# Keep all data in the original LiDAR coordinate frame so adj_ego_motions
# remain consistent with current and historical BEV features.
# Removed: ObjectSample, GlobalRotScaleTrans, BEVFusionRandomFlip3D
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=5, use_dim=4, backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='LoadTemporalData', load_dim=5, use_dim=4,
         min_time_diff=0.2, max_time_diff=1.2),
    dict(type='PointsRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
               'gt_bboxes', 'gt_labels',
               'adj_points', 'adj_ego_motions'],
         meta_keys=[
             'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
             'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
             'lidar_path', 'img_path', 'transformation_3d_flow',
             'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
             'img_aug_matrix', 'lidar_aug_matrix', 'ego2global',
             'lidar_coord_frame'])
]

train_dataloader = dict(
    dataset=dict(dataset=dict(pipeline=train_pipeline)))

work_dir = './work_dirs/bevfusion_lidar_kl_temp2_noaug'
