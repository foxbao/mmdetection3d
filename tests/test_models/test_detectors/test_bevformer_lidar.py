import numpy as np
import torch
from mmengine import DefaultScope
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from projects.BEVFormer.bevformer.detectors.bevformer_lidar import (
    BEVFormerLidar)
from projects.BEVFormer.bevformer.modules import (BEVTemporalEncoder,
                                                  warp_prev_bev)


@MODELS.register_module()
class _DummyTemporalEncoder(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, current_bev, prev_bev=None):
        if prev_bev is None:
            return current_bev + 10
        return current_bev + prev_bev


@MODELS.register_module()
class _DummyBBoxHead(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.last_pts_feats = None

    def loss(self, pts_feats, batch_data_samples, **kwargs):
        self.last_pts_feats = pts_feats
        return dict(dummy_loss=pts_feats[0].sum())

    def predict(self, pts_feats, batch_data_samples, **kwargs):
        self.last_pts_feats = pts_feats
        return [InstanceData() for _ in batch_data_samples]


def _make_queue_metas():
    eye = np.eye(4, dtype=np.float32)
    return {
        0: dict(prev_bev_exists=False, ego_motion_delta=eye),
        1: dict(prev_bev_exists=True, ego_motion_delta=eye),
        2: dict(prev_bev_exists=True, ego_motion_delta=eye),
    }


def test_warp_prev_bev_identity():
    prev_bev = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    delta = np.eye(4, dtype=np.float32)

    warped = warp_prev_bev(prev_bev, [delta], [0, 0, -1, 4, 4, 1])
    assert torch.allclose(warped, prev_bev, atol=1e-5)


def test_warp_prev_bev_translation():
    prev_bev = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    prev_bev[0, 0, 1, 1] = 1.0
    delta = np.eye(4, dtype=np.float32)
    delta[0, 3] = 1.0

    warped = warp_prev_bev(prev_bev, [delta], [0, 0, -1, 4, 4, 1])
    assert warped[0, 0, 1, 2] > 0.99


def test_bev_temporal_encoder_shape():
    encoder = BEVTemporalEncoder(
        embed_dims=8,
        num_layers=2,
        num_heads=2,
        num_points=2,
        ffn_channels=16,
        dropout=0.0)
    current_bev = torch.randn((2, 8, 3, 4))
    prev_bev = torch.randn((2, 8, 3, 4))

    out = encoder(current_bev, prev_bev)
    out_no_prev = encoder(current_bev)

    assert out.shape == current_bev.shape
    assert out_no_prev.shape == current_bev.shape


def test_bevformer_lidar_loss_uses_history_chain():
    DefaultScope.get_instance('test_bevformer_lidar', scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            temporal_encoder=dict(type='_DummyTemporalEncoder'),
            point_cloud_range=[0, 0, -1, 4, 4, 1],
        ))

    def _fake_extract(points, batch_data_samples):
        bev = []
        for pts in points:
            value = pts[0, 0]
            bev.append(torch.full((4, 2, 2), value, dtype=pts.dtype))
        return [torch.stack(bev, dim=0)]

    detector.extract_pts_bev_from_points = _fake_extract

    sample = Det3DDataSample()
    sample.set_metainfo(dict(queue_metas=_make_queue_metas()))
    batch_data_samples = [sample]
    batch_inputs = dict(
        points=[torch.tensor([[2.0, 0.0, 0.0, 0.0]])],
        history_points=[[torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]])]])

    losses = detector.loss(batch_inputs, batch_data_samples)

    assert 'dummy_loss' in losses
    fused_bev = detector.pts_bbox_head.last_pts_feats[0]
    assert fused_bev.shape == (1, 4, 2, 2)
    assert torch.allclose(fused_bev,
                          torch.full_like(fused_bev, 13.0),
                          atol=1e-5)
