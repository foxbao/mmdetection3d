"""Single-frame LiDAR BEVFusion baseline on KL with real velocities."""

_base_ = ['./bevfusion_lidar_kl_base.py']

train_dataloader = dict(
    dataset=dict(
        dataset=dict(ann_file='kl_infos_train_with_velocity.pkl')))

val_dataloader = dict(
    dataset=dict(ann_file='kl_infos_val_with_velocity.pkl'))
test_dataloader = val_dataloader

work_dir = './work_dirs/bevfusion_lidar_kl_base_vel'
