"""Compatibility alias for the main UniAD-like LiDAR track config."""

_base_ = ['../uniad_lidar_kl_track.py']

load_from = './work_dirs/bevformer_lidar_kl_uniad_logit_bev_detr/epoch_2.pth'
work_dir = './work_dirs/bevformer_lidar_kl_uniad_logit_bev_detr_track'
