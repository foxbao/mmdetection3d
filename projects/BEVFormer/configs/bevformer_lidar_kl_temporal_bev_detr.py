"""Temporal LiDAR BEVFormer with a BEVDETR detection head."""

_base_ = ['./bevformer_lidar_kl_bev_detr_base.py']

model = dict(
    type='BEVFormerLidar',
    temporal_encoder=dict(
        type='BEVTemporalEncoder',
        embed_dims=512,
        num_layers=3,
        num_heads=8,
        num_points=4,
        ffn_channels=1024,
        dropout=0.1))

work_dir = './work_dirs/bevformer_lidar_kl_temporal_bev_detr'
