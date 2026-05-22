"""Compatibility alias for the main UniAD-like LiDAR track config."""

_base_ = ['../base_track_map_lidar.py']

load_from = './work_dirs/bevformer_lidar_kl_uniad_logit_bev_detr/epoch_2.pth'
work_dir = './work_dirs/bevformer_lidar_kl_uniad_logit_bev_detr_track'
