"""LiDAR BEVFusion with 2-frame temporal fusion, synced aug, velocities."""

_base_ = ['./bevfusion_lidar_kl_temp2_syncaug.py']

train_dataloader = dict(
    dataset=dict(
        dataset=dict(ann_file='kl_infos_train_with_velocity.pkl')))

val_dataloader = dict(
    dataset=dict(ann_file='kl_infos_val_with_velocity.pkl'))
test_dataloader = val_dataloader

work_dir = './work_dirs/bevfusion_lidar_kl_temp2_syncaug_vel'
