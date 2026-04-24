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
    enable=True,
    use_lidars=[
        "bp_front_left", "bp_front_right",
        "bp_rear_left", "bp_rear_right",
        "helios_front_left", "helios_rear_right"
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
