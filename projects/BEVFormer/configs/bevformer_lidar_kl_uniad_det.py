"""Main UniAD-like query-driven LiDAR BEV detector.

This is the current mainline config. It uses detector-owned object queries and
logit-space reference points, matching UniAD's track-head boundary more closely
than the earlier sigmoid-reference ablation.
"""

_base_ = ['./bevformer_lidar_kl_uniad_base.py']

custom_imports = dict(
    imports=[
        'projects.BEVFormer.bevformer',
        'projects.BEVFormer.bevformer.modules.lidar_bevformer_encoder',
        'projects.BEVFormer.bevformer.modules.lidar_perception_transformer',
        'projects.BEVFormer.bevformer.modules.lidar_spatial_cross_attention',
        'projects.BEVFormer.bevformer.dense_heads.bev_detr_lidar_uniad_head',
        'projects.BEVFormer.bevformer.detectors.bevformer_lidar_uniad',
        'projects.KL8',
    ],
    allow_failed_imports=False,
)

queue_length = 3
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]

model = dict(
    type='BEVFormerLidarUniAD',
    num_query=600,
    embed_dims=256,
    video_test_mode=True,
    pts_neck=dict(out_channels=[128, 128]),
    pts_bbox_head=dict(
        type='BEVFormerDETRHead',
        # BEV memory throughout is [B, C, H=Y, W=X], matching UniAD's
        # query-first BEV transformer boundary and standard mmdet3d axes.
        lidar_in_channels=256,
        in_channels=256,
        embed_dims=256,
        ffn_channels=512,
        bev_h=120,
        bev_w=200,
        bev_embed_dims=256,
        transformer=dict(
            type='PerceptionTransformer',
            # UniAD-style temporal alignment lives in the transformer:
            # rotate prev_bev for ego yaw and pass ego translation as shifted
            # reference points to temporal self-attention.
            point_cloud_range=point_cloud_range,
            use_shift=True,
            rotate_prev_bev=True,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                num_heads=8,
                temporal_num_points=4,
                spatial_num_points=8,
                ffn_channels=512,
                dropout=0.1))),
    test_cfg=dict(pts=dict(max_num=300)))

# This UniAD-like BEV encoder is heavier than the baseline temporal encoder.
# Keep per-GPU batch size explicit here so it does not inherit the common
# detection base's train batch_size=4.
train_dataloader = dict(
    batch_size=4,
    dataset=dict(queue_length=queue_length))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(queue_length=queue_length))
test_dataloader = dict(
    batch_size=1,
    dataset=dict(queue_length=queue_length))

work_dir = './work_dirs/bevformer_lidar_kl_uniad_det'
