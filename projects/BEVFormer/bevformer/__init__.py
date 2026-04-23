"""LiDAR-only BEVFormer for the KL_8 dataset.

A clean re-implementation of BEVFormer's temporal architecture for LiDAR
input. Built only on mmdet3d built-in modules; the BEVFusion project is
not a dependency.

Stage 1 (current): skeleton — single-frame LiDAR detector wired up end
to end with the standard mmdet3d voxel stack + CenterHead. No temporal
yet. Subsequent stages will progressively add:

  * Stage 2: queue-based temporal data loading.
  * Stage 3: vendored BEVFormer Temporal Self-Attention + multi-layer
    BEV encoder, replacing the post-neck path.
  * Stage 4: full 24-epoch training schedule (BEVFormer baseline).
"""
from .detectors import BEVFormerLidar

__all__ = ['BEVFormerLidar']
