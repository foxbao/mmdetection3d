import numpy as np
import torch
from mmengine import DefaultScope
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, PointData
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
class _RecordingTemporalEncoder(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.last_current_bev = None
        self.last_prev_bev = None

    def forward(self, current_bev, prev_bev=None):
        self.last_current_bev = current_bev.clone()
        self.last_prev_bev = (None if prev_bev is None else prev_bev.clone())
        if prev_bev is None:
            return current_bev
        return current_bev + 2 * prev_bev


@MODELS.register_module()
class _RecordingForecastingHead(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.last_bev_feat = None

    def loss(self, bev_feat, centers_list, velocities_list, labels_list,
             gt_locs_list, gt_mask_list):
        self.last_bev_feat = bev_feat.clone()
        return dict(loss_traj=bev_feat.sum() * 0)

    def forward(self, bev_feat, centers_list, velocities_list=None,
                labels_list=None):
        self.last_bev_feat = bev_feat.clone()
        outputs = []
        for centers in centers_list:
            outputs.append(bev_feat.new_zeros((centers.shape[0], 6, 2)))
        return outputs


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


@MODELS.register_module()
class _DummyBBoxHeadExpectMetas(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.last_batch_input_metas = None

    def predict(self, pts_feats, batch_input_metas, **kwargs):
        self.last_batch_input_metas = batch_input_metas
        return [InstanceData() for _ in batch_input_metas]


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


def test_bevformer_lidar_reorders_voxel_coors_for_bevfusion_encoder():
    DefaultScope.get_instance('test_bevformer_lidar_voxel_order',
                              scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            voxel_coord_order='xyz',
        ))

    captured = {}

    def _fake_voxelize(points, batch_data_samples):
        return dict(
            voxels=torch.zeros((1, 2, 4), dtype=torch.float32),
            coors=torch.tensor([[0, 4, 5, 6]], dtype=torch.int32),
            num_points=torch.tensor([2], dtype=torch.int32))

    def _fake_extract(voxel_dict, points=None, batch_input_metas=None):
        captured['coors'] = voxel_dict['coors'].clone()
        return [torch.zeros((1, 4, 2, 2), dtype=torch.float32)]

    detector.data_preprocessor.voxelize = _fake_voxelize
    detector.extract_pts_feat = _fake_extract

    sample = Det3DDataSample()
    detector.extract_pts_bev_from_points([torch.zeros((2, 4))], [sample])

    assert torch.equal(captured['coors'],
                       torch.tensor([[0, 6, 5, 4]], dtype=torch.int32))


def test_bevformer_lidar_temporal_encoder_adapts_xy_layout():
    DefaultScope.get_instance('test_bevformer_lidar_temporal_xy',
                              scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            temporal_encoder=dict(type='_RecordingTemporalEncoder'),
            bev_feature_layout='xy',
        ))

    current = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)
    prev = current + 10

    fused = detector._fuse_bev(current, prev)

    assert torch.equal(detector.temporal_encoder.last_current_bev,
                       current.transpose(-1, -2))
    assert torch.equal(detector.temporal_encoder.last_prev_bev,
                       prev.transpose(-1, -2))
    assert fused.shape == current.shape
    assert torch.equal(fused, current + 2 * prev)


def test_bevformer_lidar_warp_prev_bev_adapts_xy_layout():
    DefaultScope.get_instance('test_bevformer_lidar_warp_xy',
                              scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            bev_feature_layout='xy',
            point_cloud_range=[0, 0, -1, 3, 4, 1],
        ))

    prev_bev = torch.zeros((1, 1, 3, 4), dtype=torch.float32)
    prev_bev[0, 0, 1, 1] = 1.0  # [x=1, y=1] in detector layout [X, Y]
    delta = np.eye(4, dtype=np.float32)
    delta[0, 3] = 1.0

    warped = detector._warp_prev_bev_if_needed(
        prev_bev, [dict(prev_bev_exists=True, ego_motion_delta=delta)])

    assert warped[0, 0, 2, 1] > 0.99
    assert warped[0, 0, 1, 2] < 1e-4


def test_bevformer_lidar_loss_includes_map_head():
    DefaultScope.get_instance('test_bevformer_lidar_map', scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            temporal_encoder=dict(type='_DummyTemporalEncoder'),
            map_head=dict(type='BEVMapHead', in_channels=4, hidden_channels=4),
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
    sample.gt_pts_seg = PointData(seg_map=torch.ones((1, 2, 2)))
    batch_data_samples = [sample]
    batch_inputs = dict(
        points=[torch.tensor([[2.0, 0.0, 0.0, 0.0]])],
        history_points=[[torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]])]])

    losses = detector.loss(batch_inputs, batch_data_samples)

    assert 'dummy_loss' in losses
    assert 'loss_map' in losses
    assert losses['loss_map'].item() > 0


def test_bevformer_lidar_loss_includes_occ_head():
    DefaultScope.get_instance('test_bevformer_lidar_occ', scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            temporal_encoder=dict(type='_DummyTemporalEncoder'),
            occ_head=dict(
                type='BEVOccHead2D',
                in_channels=4,
                hidden_channels=4,
                num_classes=3,
                num_z=2),
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
    occ = torch.zeros((2, 2, 2), dtype=torch.uint8)
    occ[0, 0, 0] = 1
    sample.gt_pts_seg = PointData(occ=occ)
    batch_inputs = dict(
        points=[torch.tensor([[2.0, 0.0, 0.0, 0.0]])],
        history_points=[[torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]])]])

    losses = detector.loss(batch_inputs, [sample])

    assert 'dummy_loss' in losses
    assert 'loss_occ_ce' in losses
    assert losses['loss_occ_ce'].item() > 0


def test_bevformer_lidar_predict_attaches_occ_head():
    DefaultScope.get_instance('test_bevformer_lidar_occ_pred',
                              scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            occ_head=dict(
                type='BEVOccHead2D',
                in_channels=4,
                hidden_channels=4,
                num_classes=3,
                num_z=2),
            bev_feature_layout='xy',
        ))

    def _fake_extract(points, batch_data_samples):
        return [torch.zeros((len(points), 4, 2, 3), dtype=torch.float32)]

    detector.extract_pts_bev_from_points = _fake_extract

    sample = Det3DDataSample()
    detector.eval()
    with torch.no_grad():
        out = detector.predict(
            dict(points=[torch.zeros((2, 4), dtype=torch.float32)]),
            [sample])

    assert out[0].pred_pts_seg.occ.shape == (2, 3, 2)


def _make_gt_instances(n: int):
    """Build a small InstanceData with 9-dim LiDAR boxes + forecasting GT."""
    boxes_t = torch.zeros(n, 9)
    boxes_t[:, :2] = torch.randn(n, 2) * 1.5     # cx, cy in pc range
    boxes_t[:, 3:6] = torch.tensor([1.0, 1.0, 1.0])
    boxes_t[:, 7:9] = torch.randn(n, 2) * 0.5
    gi = InstanceData()
    gi.bboxes_3d = LiDARInstance3DBoxes(boxes_t, box_dim=9)
    gi.labels_3d = torch.zeros(n, dtype=torch.long)
    gi.forecasting_locs = torch.randn(n, 6, 2)
    gi.forecasting_mask = torch.ones(n, 6, dtype=torch.bool)
    return gi


def test_bevformer_lidar_loss_includes_forecasting_head():
    DefaultScope.get_instance('test_bevformer_lidar_fcst', scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            temporal_encoder=dict(type='_DummyTemporalEncoder'),
            forecasting_head=dict(
                type='BEVForecastingHead',
                embed_dims=4, hidden_dims=8, num_steps=6, num_classes=3,
                pc_range=[0, 0, -1, 4, 4, 1],
                use_velocity=True, use_class_embed=True,
                dropout=0.0),
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
    sample.gt_instances_3d = _make_gt_instances(n=3)
    batch_data_samples = [sample]
    batch_inputs = dict(
        points=[torch.tensor([[2.0, 0.0, 0.0, 0.0]])],
        history_points=[[torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]])]])

    losses = detector.loss(batch_inputs, batch_data_samples)

    assert 'dummy_loss' in losses
    assert 'loss_traj' in losses
    assert losses['loss_traj'].item() > 0


def test_bevformer_lidar_loss_skips_forecasting_when_gt_missing():
    """No-op fallback: if the pipeline didn't pack forecasting GT, the head
    must silently return no loss — not crash on missing attributes."""
    DefaultScope.get_instance('test_bevformer_lidar_fcst_noop',
                              scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            temporal_encoder=dict(type='_DummyTemporalEncoder'),
            forecasting_head=dict(
                type='BEVForecastingHead',
                embed_dims=4, hidden_dims=8, num_steps=6, num_classes=3,
                pc_range=[0, 0, -1, 4, 4, 1],
                dropout=0.0),
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
    # gt_instances_3d without forecasting_locs/mask (e.g. stage3 pipeline)
    gi = InstanceData()
    gi.bboxes_3d = LiDARInstance3DBoxes(torch.zeros(2, 9), box_dim=9)
    gi.labels_3d = torch.zeros(2, dtype=torch.long)
    sample.gt_instances_3d = gi
    batch_data_samples = [sample]
    batch_inputs = dict(
        points=[torch.tensor([[2.0, 0.0, 0.0, 0.0]])],
        history_points=[[torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]])]])

    losses = detector.loss(batch_inputs, batch_data_samples)

    assert 'dummy_loss' in losses
    assert 'loss_traj' not in losses  # silently skipped


def test_bevformer_lidar_predict_attaches_forecasting_3d():
    """predict() should attach forecasting_3d to each pred_instances_3d when
    forecasting_head is wired in."""
    DefaultScope.get_instance('test_bevformer_lidar_fcst_pred',
                              scope_name='mmdet3d')

    @MODELS.register_module(force=True)
    class _DummyBBoxHeadWithBoxes(BaseModule):
        def __init__(self, **kwargs):
            super().__init__()

        def predict(self, pts_feats, batch_data_samples, **kwargs):
            results = []
            for _ in batch_data_samples:
                pi = InstanceData()
                pi.bboxes_3d = LiDARInstance3DBoxes(
                    torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
                                   0.5, -0.2],
                                  [2.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0,
                                   -0.3, 0.4]]),
                    box_dim=9)
                pi.labels_3d = torch.tensor([0, 1])
                pi.scores_3d = torch.tensor([0.9, 0.8])
                results.append(pi)
            return results

    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHeadWithBoxes'),
            temporal_encoder=dict(type='_DummyTemporalEncoder'),
            forecasting_head=dict(
                type='BEVForecastingHead',
                embed_dims=4, hidden_dims=8, num_steps=6, num_classes=3,
                pc_range=[0, 0, -1, 4, 4, 1],
                dropout=0.0),
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
        points=[torch.tensor([[2.0, 0.0, 0.0, 0.0]])])

    detector.eval()
    with torch.no_grad():
        out = detector.predict(batch_inputs, batch_data_samples)

    assert hasattr(out[0].pred_instances_3d, 'forecasting_3d')
    assert out[0].pred_instances_3d.forecasting_3d.shape == (2, 6, 2)


def test_bevformer_lidar_predict_attaches_language_selected_score():
    DefaultScope.get_instance('test_bevformer_lidar_lang_pred',
                              scope_name='mmdet3d')

    @MODELS.register_module(force=True)
    class _DummyBBoxHeadWithBoxesForLanguage(BaseModule):
        def __init__(self, **kwargs):
            super().__init__()

        def predict(self, pts_feats, batch_data_samples, **kwargs):
            results = []
            for _ in batch_data_samples:
                pi = InstanceData()
                pi.bboxes_3d = LiDARInstance3DBoxes(
                    torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
                                   0.5, -0.2],
                                  [2.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0,
                                   -0.3, 0.4]]),
                    box_dim=9)
                pi.labels_3d = torch.tensor([0, 1])
                pi.scores_3d = torch.tensor([0.9, 0.8])
                results.append(pi)
            return results

    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHeadWithBoxesForLanguage'),
            temporal_encoder=dict(type='_DummyTemporalEncoder'),
            forecasting_head=dict(
                type='LanguageConditionedForecastingHead',
                bev_dim=4, hidden_dims=8, text_embed_dims=4, vocab_size=34,
                num_steps=6, num_classes=3, pc_range=[0, 0, -1, 4, 4, 1],
                dropout=0.0),
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
    sample.set_metainfo(dict(
        queue_metas=_make_queue_metas(),
        language_tokens=torch.tensor([2, 4, 5, 0]),
        language_token_mask=torch.tensor([True, True, True, False])))
    batch_data_samples = [sample]
    batch_inputs = dict(points=[torch.tensor([[2.0, 0.0, 0.0, 0.0]])])

    detector.eval()
    with torch.no_grad():
        out = detector.predict(batch_inputs, batch_data_samples)

    pred = out[0].pred_instances_3d
    assert hasattr(pred, 'forecasting_3d')
    assert hasattr(pred, 'language_selected_score')
    assert pred.language_selected_score.shape == (2, )


def test_bevformer_lidar_forecasting_adapts_xy_layout():
    DefaultScope.get_instance('test_bevformer_lidar_fcst_xy',
                              scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHead'),
            forecasting_head=dict(type='_RecordingForecastingHead'),
            bev_feature_layout='xy',
            point_cloud_range=[0, 0, -1, 4, 6, 1],
        ))

    bev = torch.arange(24, dtype=torch.float32).view(1, 4, 2, 3)

    def _fake_extract(points, batch_data_samples):
        return [bev.clone()]

    detector.extract_pts_bev_from_points = _fake_extract

    sample = Det3DDataSample()
    sample.set_metainfo(dict(queue_metas=_make_queue_metas()))
    sample.gt_instances_3d = _make_gt_instances(n=2)

    detector.loss(
        dict(points=[torch.tensor([[1.0, 0.0, 0.0, 0.0]])]),
        [sample])

    expected = bev.transpose(-1, -2).contiguous()
    assert torch.equal(detector.forecasting_head.last_bev_feat, expected)


def test_bevformer_lidar_predict_adapts_meta_based_bbox_head():
    DefaultScope.get_instance('test_bevformer_lidar_predict_meta_head',
                              scope_name='mmdet3d')
    detector = MODELS.build(
        dict(
            type='BEVFormerLidar',
            data_preprocessor=dict(type='BEVFormerDataPreprocessor'),
            pts_bbox_head=dict(type='_DummyBBoxHeadExpectMetas'),
        ))

    def _fake_extract(points, batch_data_samples):
        return [torch.zeros((len(points), 4, 2, 2), dtype=torch.float32)]

    detector.extract_pts_bev_from_points = _fake_extract

    sample = Det3DDataSample()
    sample.set_metainfo(dict(scene_token='scene-1'))
    detector.eval()
    with torch.no_grad():
        detector.predict(dict(points=[torch.zeros((2, 4))]), [sample])

    assert detector.pts_bbox_head.last_batch_input_metas == [
        dict(scene_token='scene-1')
    ]
