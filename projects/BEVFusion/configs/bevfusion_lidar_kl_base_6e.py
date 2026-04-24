_base_ = ['./bevfusion_lidar_kl_base.py']

# 6-epoch schedule: 0-2 warmup, 2-6 cosine decay.
lr = 0.0001
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.2,
        begin=0,
        end=2,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        eta_min_ratio=1e-4,
        begin=2,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=2,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=2,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=6)
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=4)]

work_dir = './work_dirs/bevfusion_lidar_kl_base_6e'
