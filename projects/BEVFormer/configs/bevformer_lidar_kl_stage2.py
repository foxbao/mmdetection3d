"""Stage 2: BEVFormer-style temporal queue loading on KL.

Inherits Stage 1 and only swaps the dataset type to KlBEVFormerDataset with
queue_length=4. The model stays single-frame (Stage 1 skeleton) — this stage
only validates that the data layer produces the Q-frame queue, per-frame
metas, ``prev_bev_exists`` and ``ego_motion_delta`` correctly. Stage 3 will
introduce the TSA encoder that actually consumes ``history_points`` + queue
metas.
"""

_base_ = ['./bevformer_lidar_kl_base.py']

queue_length = 4

train_dataloader = dict(
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))

# Eval also goes through the queue loader so the temporal_encoder sees
# prev_bev at val/test time, matching the training distribution.
val_dataloader = dict(
    dataset=dict(type='KlBEVFormerDataset', queue_length=queue_length))
test_dataloader = val_dataloader

work_dir = './work_dirs/bevformer_lidar_kl_stage2'
