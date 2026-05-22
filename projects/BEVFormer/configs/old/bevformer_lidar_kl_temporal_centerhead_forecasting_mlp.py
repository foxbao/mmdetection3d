"""Temporal CenterHead forecasting (MLP): head on frozen temporal BEV.

Adds ``BEVForecastingHead`` (bilinear-sample BEV + small MLP, ~205K params)
to the temporal CenterHead detector, freezes all detector weights via
``lr_mult=0`` on every existing module, and trains only the new head against
``gt_forecasting_locs`` / ``gt_forecasting_mask`` packed by Pack3DDetInputs.

The point of this config is to answer **"does the temporal BEV feature
already carry enough motion info that a tiny head can decode multi-step
trajectories?"** Decision criteria from val baselines:

  * macro mADE < 0.40 m  → BEV signal sufficient, head adds value beyond
                          constant-velocity extrapolation. Proceed to
                          the downstream planning stage.
  * macro mADE 0.40-0.70 m → head only matches const-vel — no real
                          intelligence added. Consider the transformer
                          forecasting variant or unfreeze TSA encoder.
  * macro mADE > 1.0 m   → head close to predict-zero; diagnose loss /
                          freeze / lr / motion-weight clamp.

Constant-velocity (using GT velocity) baseline on val:
  macro mADE 0.397 m, mFDE 0.708 m, MR 7.1%
Predict-zero baseline:
  macro mADE 1.256 m, mFDE 2.041 m, MR 24.1%

Pre-requisite: ``BEVFormerLidar.__init__`` must accept ``forecasting_head``
arg + call its loss/predict in detector forward.
This config will not build until that wiring is in place.
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

# --- model: temporal detector + new forecasting head ----------------------

model = dict(
    forecasting_head=dict(
        type='BEVForecastingHead',
        embed_dims=512,
        hidden_dims=256,
        num_steps=6,
        num_classes=15,
        dropout=0.1,
        pc_range=point_cloud_range,
        use_velocity=True,         # concat current vel into MLP input
        use_class_embed=True,      # concat class one-hot into MLP input
        motion_weight_clamp=(0.5, 5.0),  # long-tail: bound static vs fast
        smooth_l1_beta=0.5,
        loss_weight=1.0))

# --- pipeline: extend Pack3DDetInputs.keys with forecasting GT ------------
# The temporal train pipeline doesn't pack forecasting fields (memory:
# project_motionhead_phase01 finding). We must add them so the
# detector's loss can pull gt_forecasting_locs / gt_forecasting_mask off
# data_samples.gt_instances_3d. Inference (test_pipeline) is unchanged
# because KlMetric reads forecasting GT from raw pkl, not pipeline output.

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
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
# ``lr_mult=0`` zeros gradient updates for any param whose name matches.
# Anything not listed (i.e. ``forecasting_head.*``) keeps the full lr.
# Caveat: BN running stats still update during forward — for 6 epochs the
# drift is small, but if eval shows detection degradation, a future
# refactor should toggle frozen modules to eval mode in detector.train().

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

# --- schedule: 6 epochs, cosine anneal, val every epoch -------------------
# Override the single-frame baseline's [LinearLR warmup + Cosine 1..6] with a
# flat cosine
# over the full 6 epochs — head is from scratch but main backbone frozen,
# so warmup isn't needed. val_interval=1 to track mADE convergence.

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=6, eta_min_ratio=1e-2,
         by_epoch=True, begin=0, end=6, convert_to_iter_based=True),
]

val_evaluator = dict(metric=['bbox', 'forecasting'])
test_evaluator = val_evaluator

# Override this from the CLI if you want a different finished temporal ckpt.
load_from = './work_dirs/bevformer_lidar_kl_temporal_centerhead/epoch_6.pth'
work_dir = (
    './work_dirs/bevformer_lidar_kl_temporal_centerhead_forecasting_mlp')
