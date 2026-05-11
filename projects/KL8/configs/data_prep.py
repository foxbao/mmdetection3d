gt_annotation_filter = dict(
    enable=True,
    min_points_by_class={
        'Pedestrian': 10,
        'Car': 50,
        'IGV-Full': 50,
        'Truck': 50,
        'Trailer-Empty': 50,
        'Trailer-Full': 50,
        'IGV-Empty': 50,
        'Crane': 50,
        'OtherVehicle': 50,
        'Cone': 5,
        'ContainerForklift': 50,
        'Forklift': 50,
        'Lorry': 50,
        'ConstructionVehicle': 50,
        'WheelCrane': 200,
    })


lidar_selection = dict(
    enable=False,
    use_lidars=[
        'bp_front_left', 'bp_front_right',
        'bp_rear_left', 'bp_rear_right',
        'helios_front_left', 'helios_rear_right',
        'm1_front', 'm1_rear',
    ])


camera_selection = dict(
    enable=True,
    use_cameras=[
        'front',
        'left_front',
        'left_rear',
        'rear',
        'right_front',
        'right_rear',
    ])


# Sensor-level soft-sync settings for tools/create_data.py --cfg
# projects/KL8/configs/data_prep.py
sensor_sync_cfg = dict(
    lidar_max_diff=0.05,
    camera_max_diff=0.05,
    localization_max_diff=0.15,
    require_valid_localization=True,
    sensor_time_offsets={})


# Temporal adjacency rules only affect prev/next chain construction and
# downstream forecasting/temporal datasets. Single-frame training ignores them.
temporal_chain_cfg = dict(
    enable=True,
    min_adj_time_diff=0.35,
    max_adj_time_diff=0.75)


# Forecasting labels are optional and only make sense when temporal links are
# enabled. Keep them off by default while the KL pipeline is detection-first.
forecast_cfg = dict(
    enable=False,
    forecast_steps=6)


# Per-instance velocity from track_id centered differences (nuScenes style).
# Written in place to ``instance['velocity']`` so KlDataset(with_velocity=True)
# loads it directly into ``gt_bboxes_3d[:, 7:9]`` and KlMetric reads non-zero
# GT for vel_err. Disable only if you intentionally want zero-velocity
# placeholders for a detection-only sanity run.
velocity_cfg = dict(
    enable=True,
    min_dt=1e-3,
    max_time_diff=1.5,
    max_speed=60.0)


# Backward-compatible merged view for older tooling that still reads sync_cfg.
sync_cfg = dict(sensor_sync_cfg)
if temporal_chain_cfg.get('enable', True):
    sync_cfg.update(
        min_adj_time_diff=temporal_chain_cfg['min_adj_time_diff'],
        max_adj_time_diff=temporal_chain_cfg['max_adj_time_diff'])
