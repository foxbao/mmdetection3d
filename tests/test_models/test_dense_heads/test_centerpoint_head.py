import pytest
import torch
from mmdet3d.structures import LiDARInstance3DBoxes

from mmdet3d.models.dense_heads.centerpoint_head import (_expand_nms_scales,
                                                         _expand_nms_types,
                                                         _limit_max_per_img)


def test_expand_nms_types():
    assert _expand_nms_types('rotate', 2) == ['rotate', 'rotate']
    assert _expand_nms_types(['circle', 'rotate'],
                             2) == ['circle', 'rotate']

    with pytest.raises(AssertionError):
        _expand_nms_types(['circle'], 2)


def test_expand_nms_scales():
    assert _expand_nms_scales(None, [1, 2]) == [[1.0], [1.0, 1.0]]
    assert _expand_nms_scales(2.5, [1, 2]) == [[2.5], [2.5, 2.5]]
    assert _expand_nms_scales([1.0, 2.0], [1, 2]) == [[1.0], [2.0, 2.0]]
    assert _expand_nms_scales([1.0, 2.0], [2]) == [[1.0, 2.0]]
    assert _expand_nms_scales([[1.0], [1.0, 2.0]],
                              [1, 2]) == [[1.0], [1.0, 2.0]]

    with pytest.raises(AssertionError):
        _expand_nms_scales([[1.0], [1.0]], [1, 2])


def test_limit_max_per_img():
    bboxes = LiDARInstance3DBoxes(torch.arange(45).reshape(5, 9).float(), 9)
    scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
    labels = torch.tensor([0, 1, 2, 3, 4])

    kept_bboxes, kept_scores, kept_labels = _limit_max_per_img(
        bboxes, scores, labels, 3)

    assert len(kept_scores) == 3
    assert torch.allclose(kept_scores, torch.tensor([0.9, 0.8, 0.3]))
    assert torch.equal(kept_labels, torch.tensor([1, 3, 2]))
    assert torch.equal(kept_bboxes.tensor, bboxes.tensor[[1, 3, 2]])
