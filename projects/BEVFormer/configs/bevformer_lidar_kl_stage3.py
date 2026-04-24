"""Stage 3: LiDAR temporal encoder with TSA + CenterHead."""

_base_ = ['./bevformer_lidar_kl_stage2.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]

model = dict(
    data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
    point_cloud_range=point_cloud_range,
    temporal_encoder=dict(
        type='BEVTemporalEncoder',
        embed_dims=512,
        num_layers=3,
        num_heads=8,
        num_points=4,
        ffn_channels=1024,
        dropout=0.1),
)

work_dir = './work_dirs/bevformer_lidar_kl_stage3'
