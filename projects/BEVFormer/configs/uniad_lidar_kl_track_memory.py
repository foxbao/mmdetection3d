"""MemoryBank ablation for the UniAD-like LiDAR BEV tracker."""

_base_ = ['./base_track_map_lidar.py']

model = dict(
    type='UniADTrackLiDARMemory',
    mem_args=dict(
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
        memory_bank_save_period=3))

work_dir = './work_dirs/uniad_lidar_kl_track_memory'
