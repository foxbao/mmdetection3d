_base_ = './kl8_lidar_bevformer.py'

# BEVFusion LiDAR configs use ObjectSample/db_sampler, so generate the
# ground-truth database in addition to the KL infos and merged LiDAR samples.
gt_database_cfg = dict(
    enable=True)
