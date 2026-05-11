"""Short smoke train for the ray-cast occupancy branch."""

_base_ = ['./bevformer_lidar_kl_temporal_transfusion_occ_raycast.py']

load_from = './work_dirs/bevformer_lidar_kl_temporal_transfusion/epoch_6.pth'

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=10,
    val_interval=100000)
param_scheduler = []
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)

val_cfg = None
val_dataloader = None
val_evaluator = None

default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(interval=100000, by_epoch=False),
)

work_dir = './work_dirs/smoke_occ_raycast'
