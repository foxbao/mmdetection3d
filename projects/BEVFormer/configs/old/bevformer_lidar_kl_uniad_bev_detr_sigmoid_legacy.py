"""Archived pre-merge config name for old work-dir bookkeeping.

The implementation classes now use logit-space references directly under the
main class names, so this file no longer restores a separate sigmoid-reference
code path by itself.
"""

_base_ = ['../bevformer_lidar_kl_bev_detr_base.py']

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

model = dict(
    type='BEVFormerLidarUniAD',
    pts_neck=dict(out_channels=[128, 128]),
    pts_bbox_head=dict(
        type='BEVFormerTrackHead',
        # Detector layout is [B, C, X, Y]; head BEV encoder receives
        # [B, C, Y, X], matching UniAD's query-first BEV transformer boundary.
        lidar_in_channels=256,
        in_channels=256,
        embed_dims=256,
        bev_h=120,
        bev_w=200,
        bev_embed_dims=256,
        transformer=dict(
            type='PerceptionTransformer',
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                num_heads=8,
                temporal_num_points=4,
                spatial_num_points=8,
                ffn_channels=1024,
                dropout=0.1))),
    test_cfg=dict(pts=dict(max_num=300)))

train_dataloader = dict(
    batch_size=3,
    dataset=dict(queue_length=queue_length))
val_dataloader = dict(
    batch_size=3,
    dataset=dict(queue_length=queue_length))
test_dataloader = dict(
    batch_size=3,
    dataset=dict(queue_length=queue_length))

work_dir = './work_dirs/bevformer_lidar_kl_uniad_bev_detr_sigmoid_legacy'
