import torch

from mmdet3d.structures import Det3DDataSample, PointData
from projects.BEVFormer.bevformer.dense_heads import BEVMapHead


def test_bev_map_head_loss_and_predict():
    head = BEVMapHead(in_channels=8, hidden_channels=4, loss_weight=1.0)
    bev_feat = torch.randn((2, 8, 3, 4))

    batch_data_samples = []
    for _ in range(2):
        sample = Det3DDataSample()
        sample.gt_pts_seg = PointData(seg_map=torch.ones((1, 3, 4)))
        batch_data_samples.append(sample)

    losses = head.loss(bev_feat, batch_data_samples)
    assert 'loss_map' in losses
    assert losses['loss_map'].item() > 0

    pred_samples = head.predict(bev_feat, batch_data_samples)
    assert len(pred_samples) == 2
    assert pred_samples[0].pred_pts_seg.seg_map.shape == (1, 3, 4)
    assert pred_samples[0].pred_pts_seg.seg_map_mask.shape == (1, 3, 4)
