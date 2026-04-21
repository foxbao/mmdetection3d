"""Temporal BEVFusion on KL with global augmentation restored.

Inherits the temporal architecture from ``bevfusion_lidar_kl_temp2.py``
(same model, same dataset setup) and rebuilds the train pipeline so that:

  1. ``BEVFusionGlobalRotScaleTrans`` + ``BEVFusionRandomFlip3D`` accumulate
     the current-frame LiDAR-frame augmentation into ``lidar_aug_matrix``.
  2. ``SyncTemporalAug`` re-applies the same matrix to ``adj_points`` and
     conjugates ``adj_ego_motions`` so ``TemporalBEVFuser`` keeps aligning
     history to the augmented current frame.
  3. ``PointsRangeFilter`` prunes out-of-range points for both current and
     adjacent frames (extension added to mmdet3d).
  4. ``ObjectSample`` (DBSampler) operates on the current frame only;
     inserted objects simply lack a temporal history, which is the same
     regime as single-frame training for those targets.
"""
_base_ = ['./bevfusion_lidar_kl_temp2.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane'
]
data_root = 'data/kl_8/'
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kl_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points={
            'Pedestrian': 10, 'Car': 50, 'IGV-Full': 50, 'Truck': 50,
            'Trailer-Empty': 50, 'Trailer-Full': 50, 'IGV-Empty': 50,
            'Crane': 50, 'OtherVehicle': 50, 'Cone': 10,
            'ContainerForklift': 50, 'Forklift': 50, 'Lorry': 50,
            'ConstructionVehicle': 50, 'WheelCrane': 100}),
    classes=class_names,
    sample_groups={
        'Pedestrian': 5, 'Car': 5, 'IGV-Full': 5, 'Truck': 5,
        'Trailer-Empty': 5, 'Trailer-Full': 5, 'IGV-Empty': 5,
        'Crane': 5, 'OtherVehicle': 0, 'Cone': 5,
        'ContainerForklift': 5, 'Forklift': 1, 'Lorry': 1,
        'ConstructionVehicle': 5, 'WheelCrane': 1},
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=5, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='LoadTemporalData', load_dim=5, use_dim=4,
         min_time_diff=0.2, max_time_diff=1.2),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='BEVFusionGlobalRotScaleTrans',
         scale_ratio_range=[0.9, 1.1],
         rot_range=[-0.78539816, 0.78539816],
         translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='SyncTemporalAug'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
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
             'lidar_aug_matrix', 'ego2global', 'lidar_coord_frame'])
]

train_dataloader = dict(
    dataset=dict(dataset=dict(pipeline=train_pipeline)))

custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]

work_dir = './work_dirs/bevfusion_lidar_kl_temp2_syncaug'
