import torch

from mmdet3d.structures import Det3DDataSample, PointData
from projects.BEVFormer.bevformer.dense_heads import BEVOccHead2D


def _make_occ_samples(batch_size, shape):
    samples = []
    for idx in range(batch_size):
        occ = torch.zeros(shape, dtype=torch.uint8)
        occ[idx % shape[0], idx % shape[1], idx % shape[2]] = 1
        sample = Det3DDataSample()
        sample.gt_pts_seg = PointData(occ=occ)
        samples.append(sample)
    return samples


def test_bev_occ_head_xy_loss_and_predict():
    head = BEVOccHead2D(
        in_channels=8,
        hidden_channels=4,
        num_classes=3,
        num_z=2,
        bev_feature_layout='xy',
        empty_weight=0.1)
    bev_feat = torch.randn((2, 8, 3, 5))
    batch_data_samples = _make_occ_samples(2, (3, 5, 2))

    logits = head(bev_feat)
    losses = head.loss(bev_feat, batch_data_samples)
    pred_samples = head.predict(bev_feat, batch_data_samples)

    assert logits.shape == (2, 3, 3, 5, 2)
    assert 'loss_occ_ce' in losses
    assert losses['loss_occ_ce'].item() > 0
    assert len(pred_samples) == 2
    assert pred_samples[0].pred_pts_seg.occ.shape == (3, 5, 2)


def test_bev_occ_head_yx_returns_canonical_xyz_grid():
    head = BEVOccHead2D(
        in_channels=8,
        hidden_channels=4,
        num_classes=3,
        num_z=2,
        bev_feature_layout='yx')
    bev_feat = torch.randn((1, 8, 5, 3))
    batch_data_samples = _make_occ_samples(1, (3, 5, 2))

    logits = head(bev_feat)
    losses = head.loss(bev_feat, batch_data_samples)

    assert logits.shape == (1, 3, 3, 5, 2)
    assert losses['loss_occ_ce'].item() > 0
