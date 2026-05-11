"""One-iteration smoke test for the track-mode config."""
_base_ = ['./bevformer_lidar_kl_temporal_bev_detr_track.py']

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=1,
    val_interval=999)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False)
val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

default_hooks = dict(
    checkpoint=dict(interval=999),
    logger=dict(interval=1))

work_dir = './work_dirs/smoke_bevformer_lidar_kl_temporal_bev_detr_track'
