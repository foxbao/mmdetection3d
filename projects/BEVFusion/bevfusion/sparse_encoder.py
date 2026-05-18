# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models.middle_encoders import SparseEncoderXYZ
from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVFusionSparseEncoder(SparseEncoderXYZ):
    r"""Backward-compatible name for the xyz-order sparse encoder.

    New cross-project configs should prefer ``SparseEncoderXYZ``. This alias is
    kept so existing BEVFusion configs and checkpoints can keep using
    ``type='BEVFusionSparseEncoder'`` without changing model structure.
    """

    pass
