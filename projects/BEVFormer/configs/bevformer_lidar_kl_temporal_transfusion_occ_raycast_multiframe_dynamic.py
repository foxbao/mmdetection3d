"""Multi-frame OCC target with ID-aligned dynamic instance points.

Compared with ``occ_raycast_multiframe.py``, this keeps the same conservative
static-scene aggregation and additionally uses ``track_id`` to move historical
object-box points into the current frame's matching object box.
"""

_base_ = ['./bevformer_lidar_kl_temporal_transfusion_occ_raycast_multiframe.py']

train_dataloader = dict(
    dataset=dict(
        post_pipeline=[
            dict(
                type='GenerateKLMultiFrameOccFromQueue',
                point_cloud_range=[
                    -80.0, -48.0, -2.0, 80.0, 48.0, 6.0],
                occ_size=[200, 120, 10],
                empty_idx=0,
                ignore_idx=255,
                label_offset=1,
                mode='raycast',
                min_points_per_voxel=1,
                dilation_xy=1,
                mark_unobserved_box_ignore=True,
                ray_origin=[0.0, 0.0, 0.0],
                ground_label=16,
                obstacle_label=17,
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
                ego_ignore_range=[
                    -8.0, -2.0, -2.0, 8.0, 2.0, 6.0],
                aggregate_history_scene=True,
                history_scene_only=True,
                aggregate_dynamic_instances=True,
                dynamic_instance_min_points=1),
        ]))

work_dir = './work_dirs/' \
           'bevformer_lidar_kl_temporal_transfusion_occ_raycast_mf_dynamic'
