"""Lower inference thresholds for tracking eval (no retrain needed).

For testing only: temporarily lower score_thresh / filter_score_thresh /
miss_tolerance to see how many true detections were being filtered out of
the tracking metric. Use with:

    python tools/test.py <this_config> <checkpoint>

Training fields are inherited from the base config and can still be reused.
"""
_base_ = ['./bevformer_lidar_kl_temporal_bev_detr_track.py']

model = dict(
    score_thresh=0.2,
    filter_score_thresh=0.1,
    miss_tolerance=5,
)
