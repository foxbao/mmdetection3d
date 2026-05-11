"""Runtime tracker that assigns and retires obj_ids during inference.

During training, ClipMatcher writes obj_idxes directly from GT track ids; at
inference we have no GT, so a simple online policy promotes high-score new
tracks to fresh obj_ids and retires inactive ones after ``miss_tolerance``
frames. This is the same behaviour as UniAD's RuntimeTrackerBase, but the
optional 3D-IoU dedup (which UniAD calls with denormalize_bbox) is removed —
the detector can apply pi-symmetric IoU dedup outside if needed.
"""
from .track_instance import Instances


class RuntimeTrackerBase(object):
    """Online obj_id allocator + miss-tolerance retirement."""

    def __init__(self,
                 score_thresh: float = 0.4,
                 filter_score_thresh: float = 0.3,
                 miss_tolerance: int = 3) -> None:
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self) -> None:
        self.max_obj_id = 0

    def update(self, track_instances: Instances) -> None:
        track_instances.disappear_time[
            track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if (track_instances.obj_idxes[i] == -1 and
                    track_instances.scores[i] >= self.score_thresh):
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif (track_instances.obj_idxes[i] >= 0 and
                  track_instances.scores[i] < self.filter_score_thresh):
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    track_instances.obj_idxes[i] = -1
