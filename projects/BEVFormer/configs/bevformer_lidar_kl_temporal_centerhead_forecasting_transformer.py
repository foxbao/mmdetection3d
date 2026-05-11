"""Temporal CenterHead forecasting (transformer): head on frozen BEV.

Same training setup as
``bevformer_lidar_kl_temporal_centerhead_forecasting_mlp.py`` but swaps in
``TransformerForecastingHead`` — a 2-layer transformer decoder with
self-attention among queries (inter-object reasoning) and cross-attention to
BEV (global scene context, not just bilinear single sample). Use this config
when:

  * B's macro mADE is in the 0.40-0.70 m band — i.e. B can match
    constant-velocity extrapolation but isn't adding intelligence beyond
    it. Inter-object attention should help especially for port hard
    cases (Truck following, Crane-Trailer coordination).
  * You want the per-instance ``motion_query`` representation that a later
    planning head can consume (not exposed by the MLP variant).

Cost: ~1M extra params (vs 205K for B), training time per epoch ~10-20%
slower than B due to attention over 24000 BEV tokens.

Run the MLP and transformer variants separately on the same temporal
checkpoint to ablate "head capacity" vs "BEV signal sufficiency" — they
share the frozen backbone.
"""

_base_ = ['./bevformer_lidar_kl_temporal_centerhead.py']

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
batch_size = 4
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]

# --- model: temporal detector + transformer forecasting head --------------

model = dict(
    forecasting_head=dict(
        type='TransformerForecastingHead',
        bev_dim=512,
        embed_dims=256,
        num_layers=2,
        num_heads=8,
        ffn_dims=512,
        num_steps=6,
        num_classes=15,
        dropout=0.1,
        pc_range=point_cloud_range,
        use_velocity=True,
        use_class_embed=True,
        motion_weight_clamp=(0.5, 5.0),
        smooth_l1_beta=0.5,
        loss_weight=1.0))

# --- pipeline: same as B (Pack keys must include forecasting GT) ----------

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
              'gt_forecasting_locs', 'gt_forecasting_mask']),
]

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(pipeline=train_pipeline))

# --- freeze temporal detector modules; train only forecasting_head --------

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'pts_voxel_encoder':  dict(lr_mult=0.0),
        'pts_middle_encoder': dict(lr_mult=0.0),
        'pts_backbone':       dict(lr_mult=0.0),
        'pts_neck':           dict(lr_mult=0.0),
        'temporal_encoder':   dict(lr_mult=0.0),
        'pts_bbox_head':      dict(lr_mult=0.0),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=6, eta_min_ratio=1e-2,
         by_epoch=True, begin=0, end=6, convert_to_iter_based=True),
]

load_from = './work_dirs/bevformer_lidar_kl_temporal_centerhead/epoch_6.pth'
val_evaluator = dict(metric=['bbox', 'forecasting'])
test_evaluator = val_evaluator

work_dir = (
    './work_dirs/'
    'bevformer_lidar_kl_temporal_centerhead_forecasting_transformer')
