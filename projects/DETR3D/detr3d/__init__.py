from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmdet3d.registry import MODELS

# In mmdet >= 3.0.0 the DetrTransformerDecoderLayer API was completely
# refactored and the class is no longer registered. DETR3D configs use
# the old mmcv BaseTransformerLayer-style API (attn_cfgs / operation_order),
# so we register BaseTransformerLayer under the legacy name.
if 'DetrTransformerDecoderLayer' not in MODELS._module_dict:
    MODELS.register_module(
        name='DetrTransformerDecoderLayer', module=BaseTransformerLayer)

from .detr3d import DETR3D
from .detr3d_head import DETR3DHead
from .detr3d_transformer import (Detr3DCrossAtten, Detr3DTransformer,
                                 Detr3DTransformerDecoder)
from .hungarian_assigner_3d import HungarianAssigner3D
from .match_cost import BBox3DL1Cost
from .nms_free_coder import NMSFreeCoder
from .vovnet import VoVNet

__all__ = [
    'VoVNet', 'DETR3D', 'DETR3DHead', 'Detr3DTransformer',
    'Detr3DTransformerDecoder', 'Detr3DCrossAtten', 'HungarianAssigner3D',
    'BBox3DL1Cost', 'NMSFreeCoder'
]
