gt_annotation_filter = dict(
    enable=True,
    min_points_by_class={
        "Pedestrian": 10,
        "Car": 50,
        "IGV-Full": 50,
        "Truck": 50,
        "Trailer-Empty": 50,
        "Trailer-Full": 50,
        "IGV-Empty": 50,
        "Crane": 50,
        "OtherVehicle": 50,
        "Cone": 5,
        "ContainerForklift": 50,
        "Forklift": 50,
        "Lorry": 50,
        "ConstructionVehicle": 50,
        "WheelCrane": 200
    }
)


lidar_selection = dict(
    enable=False,
    use_lidars=[
        "bp_front_left", "bp_front_right",
        "bp_rear_left", "bp_rear_right",
        "helios_front_left", "helios_rear_right",
        "m1_front", "m1_rear"
    ]
)

camera_selection = dict(
    enable=True,
    use_cameras=[
        "front",
        "left_front",
        "left_rear",
        "rear",
        "right_front",
        "right_rear"
    ]
)

# Soft-sync settings for tools/create_data.py --cfg projects/BEVFusion/configs/kl_8.py
# A positive sensor offset means raw sensor timestamps are later than the
# label/frame timestamp by that many seconds. Defaults below preserve the
# previous nearest-neighbor behavior while making the thresholds explicit.
sync_cfg = dict(
    lidar_max_diff=0.05,
    camera_max_diff=0.05,
    localization_max_diff=0.15,
    require_valid_localization=True,
    min_adj_time_diff=0.2,
    max_adj_time_diff=1.2,
    sensor_time_offsets={}
)

# gt_box_clamp = dict(
#     enable=True,
#     by_class={
#         "WheelCrane": dict(
#             z=dict(
#                 max=4.0
#             )
#         )
#     }
# )
