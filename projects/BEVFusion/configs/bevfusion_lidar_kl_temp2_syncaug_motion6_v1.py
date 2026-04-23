"""Temporal BEVFusion + V1 motion decoder on KL with synced augmentation."""

_base_ = ['./bevfusion_lidar_kl_temp2_syncaug_motion6.py']

model = dict(
    motion_head=dict(
        _delete_=True,
        type='MotionDecoderHead',
        in_channels=128,
        hidden_dim=256,
        forecast_steps=6,
        num_modes=3,
        num_attn_layers=2,
        num_attn_heads=8,
        dropout=0.1,
        loss_weight=0.5,
        cls_weight=0.2))

work_dir = './work_dirs/bevfusion_lidar_kl_temp2_syncaug_motion6_v1'
