"""LiDAR-only BEVFormer detector — Stage 1 skeleton.

Inherits ``MVXTwoStageDetector`` to mirror BEVFormer's original
detector structure (the upstream ``BEVFormer`` class also subclasses
``MVXTwoStageDetector``). For Stage 1 this class is a thin alias: it
runs the standard mmdet3d voxel → sparse-encoder → backbone → neck →
head pipeline in a single forward pass, with no temporal logic.

Stage 3 will override ``extract_feat`` (or add a sibling
``obtain_history_bev``) to consume the temporal queue and chain
``prev_bev`` through a TSA-based BEV encoder, matching BEVFormer.
"""

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVFormerLidar(MVXTwoStageDetector):
    pass
