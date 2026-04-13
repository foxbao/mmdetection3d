"""BEVFusion LiDAR-camera config for KL dataset with ResNet-50 + LSS.

This variant keeps the validated LiDAR-camera LSS setup and only swaps the
camera image backbone from Swin-T to ResNet-50.
"""

_base_ = ['./bevfusion_lidar_camera_yx_kl_lss.py']

model = dict(
    img_backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        _delete_=True,
        type='GeneralizedLSSFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)))

train_dataloader = dict(batch_size=2)
