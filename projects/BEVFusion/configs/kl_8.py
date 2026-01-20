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
        "WheelCrane": 100
    }
)