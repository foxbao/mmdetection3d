_base_ = './kl8_lidar_bevfusion.py'

# LiDAR-camera BEVFusion configs load processed multi-view images from the KL
# infos, so generate camera_undist images and camera calibration entries.
camera_processing_cfg = dict(
    enable=True,
    img_scale=1.0 / 3.0)
