# KL BEVFusion Fair Comparison Matrix

Use this matrix for clean comparisons after the temporal warp, missing-history
handling, and velocity-label fixes.

## Detection / Temporal Ablation

| ID | Config | Work dir | Purpose |
| --- | --- | --- | --- |
| A | `bevfusion_lidar_kl_base_vel.py` | `work_dirs/bevfusion_lidar_kl_base_vel` | Single-frame detection baseline |
| B | `bevfusion_lidar_kl_temp2_noaug_vel.py` | `work_dirs/bevfusion_lidar_kl_temp2_noaug_vel` | 2-frame temporal fusion without train-time geometric augmentation |
| C | `bevfusion_lidar_kl_temp2_syncaug_vel.py` | `work_dirs/bevfusion_lidar_kl_temp2_syncaug_vel` | 2-frame temporal fusion with synchronized augmentation |

Compare A/B/C with the same epoch, preferably epoch 6 first.

## Motion-Head Ablation

| ID | Config | Work dir | Purpose |
| --- | --- | --- | --- |
| B | `bevfusion_lidar_kl_temp2_noaug_vel.py` | `work_dirs/bevfusion_lidar_kl_temp2_noaug_vel` | Temporal detection baseline |
| D | `bevfusion_lidar_kl_temp2_noaug_motion6_vel.py` | `work_dirs/bevfusion_lidar_kl_temp2_noaug_motion6_vel` | Adds `MotionHead` and forecasting metrics |

Compare B/D with the same epoch.  Use detection metrics for both and
forecasting metrics only for D.

## Data Contract

All configs in this matrix use:

- `data/kl_8/kl_infos_train_with_velocity.pkl`
- `data/kl_8/kl_infos_val_with_velocity.pkl`

Do not compare these directly against older checkpoints trained on
`kl_infos_train.pkl`, because those labels contained zero velocities.

## Recommended Commands

```bash
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_kl_base_vel.py
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_kl_temp2_noaug_vel.py
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_kl_temp2_syncaug_vel.py
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_kl_temp2_noaug_motion6_vel.py
```

Evaluate with the matching config and checkpoint:

```bash
python tools/test.py <config.py> <work_dir>/epoch_6.pth --work-dir work_dirs/eval_<name>_epoch6
```

## Primary Metrics

For A/B/C:

- `NDS`
- `mAP`
- `mATE`
- `mASE`
- `mAOE`
- `mAVE`
- per-class `AP_dist_0.5`

For D additionally:

- `forecast/mADE`
- `forecast/mFDE`
- `forecast/MR_2m`
- `forecast/recall`
