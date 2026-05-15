"""MemoryBank ablation for the UniAD-like LiDAR BEV tracker."""

_base_ = ['./bevformer_lidar_kl_uniad_track.py']

model = dict(
    type='LidarUniADTrackMemory',
    mem_args=dict(
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
        memory_bank_save_period=3))

work_dir = './work_dirs/bevformer_lidar_kl_uniad_track_memory'
