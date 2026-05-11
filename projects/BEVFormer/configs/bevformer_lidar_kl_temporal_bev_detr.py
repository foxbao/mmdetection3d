"""Temporal LiDAR BEVFormer with a DETR-style BEV detection head.

This is the first bridge toward UniAD-style tracking: keep the existing
LiDAR BEVFormer encoder and replace the dense detection head with learned
object queries plus Hungarian set prediction.
"""

_base_ = ['./bevformer_lidar_kl_temporal_transfusion.py']

custom_imports = dict(
    imports=[
        'projects.BEVFormer.bevformer',
        'projects.BEVFusion.bevfusion',
        'projects.KL8',
    ],
    allow_failed_imports=False,
)

num_classes = 15
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
# Front-back symmetric port-scene classes: yaw and yaw + pi are physically
# identical (IGV-Full=2, IGV-Empty=6, WheelCrane=14). Must stay in sync with
# class_names ordering in the base config and with the val_evaluator's
# pi_symmetric_classes list.
pi_symmetric_class_indices = [2, 6, 14]

model = dict(
    pts_bbox_head=dict(
        _delete_=True,
        type='BEVDETRHead',
        in_channels=512,
        num_classes=num_classes,
        num_query=600,
        embed_dims=256,
        num_decoder_layers=6,
        num_heads=8,
        num_points=4,
        ffn_channels=1024,
        dropout=0.1,
        code_size=10,
        pc_range=point_cloud_range,
        bev_feature_layout='xy',
        with_box_refine=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 0.2, 0.2],
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=2.0),
        loss_bbox=dict(
            type='mmdet.L1Loss',
            reduction='mean',
            loss_weight=0.25)),
    train_cfg=dict(
        pts=dict(
            _delete_=True,
            pi_symmetric_class_indices=pi_symmetric_class_indices,
            assigner=dict(
                type='BEVDETRHungarianAssigner3D',
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=2.0),
                reg_cost=dict(type='BEVDETRBBox3DL1Cost', weight=0.25),
                pc_range=point_cloud_range,
                pi_symmetric_class_indices=pi_symmetric_class_indices))),
    test_cfg=dict(
        pts=dict(
            _delete_=True,
            max_num=600,
            score_threshold=0.05,
            post_center_range=[
                -80.0, -48.0, -10.0, 80.0, 48.0, 10.0
            ])))

# Deformable cross-attention cuts cross-attn memory ~10x vs full attention,
# so batch can stay at 3 even with 6 decoder layers and 600 queries.
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=2)
test_dataloader = dict(batch_size=2)

lr = 2e-4
optim_wrapper = dict(
    optimizer=dict(lr=lr),
    clip_grad=dict(max_norm=35, norm_type=2))

# Keep this DETR-head run longer than the inherited TransFusion schedule.
# The head is trained from scratch and is still improving after epoch 2.
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=2)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=11,
        eta_min_ratio=1e-3,
        by_epoch=True,
        begin=1,
        end=12,
        convert_to_iter_based=True),
]

work_dir = './work_dirs/bevformer_lidar_kl_temporal_bev_detr'
