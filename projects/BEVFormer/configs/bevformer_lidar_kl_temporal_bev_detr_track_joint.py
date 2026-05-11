"""Joint detection + tracking training from scratch (UniAD-style).

Instead of the two-stage approach (train detection -> fine-tune tracking),
this config trains both simultaneously. The BEVDETRClipMatcher provides
full detection supervision (cls + L1 + IoU on every decoder layer, every
frame) so the detection head learns alongside the tracking components.

Differences from the fine-tune config:
  * No ``load_from`` - everything trains from scratch
  * lr = 2e-4 (detection-level, 10x higher than fine-tune)
  * max_epochs = 12 (same as standalone detection training)
  * Warmup: 1 epoch linear, cosine for remaining 11
"""
_base_ = ['./bevformer_lidar_kl_temporal_bev_detr_track.py']

# ---------------------------------------------------------------------- #
# Schedule: full training from scratch.
# ---------------------------------------------------------------------- #

lr = 2e-4
optim_wrapper = dict(
    optimizer=dict(lr=lr),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=2)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True,
         begin=0, end=1, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=11, eta_min_ratio=1e-3,
         by_epoch=True, begin=1, end=12, convert_to_iter_based=True),
]

load_from = None
work_dir = './work_dirs/bevformer_lidar_kl_temporal_bev_detr_track_joint'
