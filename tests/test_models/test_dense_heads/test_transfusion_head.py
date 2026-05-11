import torch
from mmdet.models.task_modules import AssignResult, PseudoSampler
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet3d.structures import LiDARInstance3DBoxes
from projects.BEVFusion.bevfusion.transfusion_head import TransFusionHead


class _FakeBBoxCoder:

    code_size = 10

    def decode(self, score, rot, dim, center, height, vel=None):
        num_boxes = center.shape[-1]
        boxes = center.new_zeros((num_boxes, self.code_size))
        return [dict(bboxes=boxes)]

    def encode(self, boxes):
        return boxes.new_zeros((boxes.shape[0], self.code_size))


class _FakeAssigner:

    def assign(self, bboxes, gt_bboxes, gt_labels, score, train_cfg):
        gt_inds = bboxes.new_tensor([1, 0], dtype=torch.long)
        max_overlaps = bboxes.new_tensor([1.0, 0.0])
        labels = bboxes.new_tensor([0, -1], dtype=torch.long)
        return AssignResult(
            num_gts=1,
            gt_inds=gt_inds,
            max_overlaps=max_overlaps,
            labels=labels)


def _make_head_for_target_test():
    head = TransFusionHead.__new__(TransFusionHead)
    torch.nn.Module.__init__(head)
    head.num_classes = 1
    head.num_proposals = 2
    head.auxiliary = False
    head.num_decoder_layers = 1
    head.bbox_coder = _FakeBBoxCoder()
    head.bbox_assigner = _FakeAssigner()
    head.bbox_sampler = PseudoSampler()
    head.train_cfg = ConfigDict(
        assigner=ConfigDict(type='HungarianAssigner3D'),
        grid_size=[16, 8, 1],
        point_cloud_range=[0.0, 0.0, -1.0, 16.0, 8.0, 1.0],
        voxel_size=[1.0, 1.0, 1.0],
        out_size_factor=1,
        gaussian_overlap=0.1,
        min_radius=0,
        pos_weight=-1)
    return head


def _make_preds():
    return dict(
        heatmap=torch.zeros((1, 1, 2)),
        center=torch.zeros((1, 2, 2)),
        height=torch.zeros((1, 1, 2)),
        dim=torch.ones((1, 3, 2)),
        rot=torch.zeros((1, 2, 2)),
        vel=torch.zeros((1, 2, 2)))


def test_transfusion_heatmap_target_uses_xy_feature_order():
    head = _make_head_for_target_test()
    gt_instances = InstanceData()
    gt_instances.bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor([[3.2, 5.4, 0.0, 0.2, 0.2, 1.0, 0.0, 0.0, 0.0]]),
        box_dim=9)
    gt_instances.labels_3d = torch.tensor([0], dtype=torch.long)

    targets = head.get_targets_single(gt_instances, _make_preds(), 0)
    heatmap = targets[7]

    assert heatmap.shape == (1, 1, 16, 8)
    assert heatmap[0, 0, 3, 5] == 1
    assert heatmap[0, 0, 5, 3] == 0
