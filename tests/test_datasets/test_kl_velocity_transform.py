"""Unit tests for ``ComputeVelocityFromForecasting``.

The transform pulls step-0 displacement from ``gt_forecasting_locs`` and
writes it into ``gt_bboxes_3d[:, 7:9]`` (or expands a 7-dim box to 9-dim
when ``with_velocity=False`` somewhere upstream). Mask gating ensures rows
with no valid first-step are left at zero velocity rather than poisoning
the loss with garbage from a masked-out future.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mmdet3d.structures import LiDARInstance3DBoxes
from projects.KL8.transforms import ComputeVelocityFromForecasting


FORECAST_STEPS = 6


def _make_boxes(coords, box_dim=9):
    """Build a (N, box_dim) LiDARInstance3DBoxes with zero velocity slots."""
    n = len(coords)
    boxes = np.zeros((n, box_dim), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        boxes[i, 0] = x
        boxes[i, 1] = y
        boxes[i, 3:6] = [2.0, 2.0, 2.0]  # lwh
    return LiDARInstance3DBoxes(boxes, box_dim=box_dim)


def test_overwrites_velocity_in_9dim_box():
    """KlDataset default with_velocity=True path: 9-dim in, 9-dim out, vel set."""
    boxes = _make_boxes([(0.0, 0.0), (10.0, 0.0)], box_dim=9)
    # step 0 displacement: (1.0, 0.5) for row 0, (-2.0, 0.0) for row 1.
    locs = np.zeros((2, FORECAST_STEPS, 2), dtype=np.float32)
    locs[0, 0] = (1.0, 0.5)
    locs[1, 0] = (-2.0, 0.0)
    mask = np.ones((2, FORECAST_STEPS), dtype=np.bool_)

    out = ComputeVelocityFromForecasting(dt=0.5)(dict(
        gt_bboxes_3d=boxes,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    new_boxes = out['gt_bboxes_3d']
    assert new_boxes.box_dim == 9
    # vel = displacement / dt = displacement / 0.5 = 2 * displacement
    assert torch.allclose(new_boxes.tensor[:, 7:9],
                          torch.tensor([[2.0, 1.0], [-4.0, 0.0]]))
    # untouched dims preserved
    assert torch.allclose(new_boxes.tensor[:, :7], boxes.tensor[:, :7])


def test_expands_7dim_box_to_9dim():
    """7-dim input (with_velocity=False upstream) gets expanded to 9-dim."""
    boxes = _make_boxes([(0.0, 0.0)], box_dim=7)
    locs = np.zeros((1, FORECAST_STEPS, 2), dtype=np.float32)
    locs[0, 0] = (3.0, 4.0)
    mask = np.ones((1, FORECAST_STEPS), dtype=np.bool_)

    out = ComputeVelocityFromForecasting(dt=0.5)(dict(
        gt_bboxes_3d=boxes,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    new_boxes = out['gt_bboxes_3d']
    assert new_boxes.box_dim == 9
    assert new_boxes.tensor.shape == (1, 9)
    assert torch.allclose(new_boxes.tensor[0, 7:9], torch.tensor([6.0, 8.0]))
    # geometry preserved
    assert torch.allclose(new_boxes.tensor[0, :7], boxes.tensor[0, :7])


def test_mask_gating_zeros_invalid_rows():
    """Rows whose step-0 mask is False must end up with velocity = 0."""
    boxes = _make_boxes([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], box_dim=9)
    locs = np.full((3, FORECAST_STEPS, 2), fill_value=99.0, dtype=np.float32)
    # row 1's first step is masked-out — its vel should NOT be 99/dt.
    mask = np.ones((3, FORECAST_STEPS), dtype=np.bool_)
    mask[1, 0] = False

    out = ComputeVelocityFromForecasting(dt=0.5)(dict(
        gt_bboxes_3d=boxes,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    vel = out['gt_bboxes_3d'].tensor[:, 7:9]
    assert torch.allclose(vel[0], torch.tensor([198.0, 198.0]))
    assert torch.allclose(vel[1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(vel[2], torch.tensor([198.0, 198.0]))


def test_empty_boxes_9dim_passthrough():
    """N=0 case: no crash, type preserved, no expansion needed if already 9-dim."""
    boxes = _make_boxes([], box_dim=9)
    locs = np.zeros((0, FORECAST_STEPS, 2), dtype=np.float32)
    mask = np.zeros((0, FORECAST_STEPS), dtype=np.bool_)

    out = ComputeVelocityFromForecasting(dt=0.5)(dict(
        gt_bboxes_3d=boxes,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    assert out['gt_bboxes_3d'].tensor.shape == (0, 9)
    assert out['gt_bboxes_3d'].box_dim == 9


def test_empty_boxes_7dim_expanded():
    """N=0 + 7-dim still must surface as 9-dim so the head sees consistent code_size."""
    boxes = _make_boxes([], box_dim=7)
    locs = np.zeros((0, FORECAST_STEPS, 2), dtype=np.float32)
    mask = np.zeros((0, FORECAST_STEPS), dtype=np.bool_)

    out = ComputeVelocityFromForecasting(dt=0.5)(dict(
        gt_bboxes_3d=boxes,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    assert out['gt_bboxes_3d'].tensor.shape == (0, 9)
    assert out['gt_bboxes_3d'].box_dim == 9


def test_row_mismatch_raises():
    """Lost row alignment is a fatal pipeline bug — fail loudly."""
    boxes = _make_boxes([(0.0, 0.0), (1.0, 0.0)], box_dim=9)
    locs = np.zeros((3, FORECAST_STEPS, 2), dtype=np.float32)  # wrong N
    mask = np.ones((3, FORECAST_STEPS), dtype=np.bool_)

    with pytest.raises(ValueError, match='row alignment'):
        ComputeVelocityFromForecasting(dt=0.5)(dict(
            gt_bboxes_3d=boxes,
            gt_forecasting_locs=locs,
            gt_forecasting_mask=mask))


def test_no_op_when_keys_missing():
    """Test/inference pipelines without forecasting must pass through cleanly."""
    boxes = _make_boxes([(0.0, 0.0)], box_dim=9)
    results = dict(gt_bboxes_3d=boxes)  # no locs / mask
    out = ComputeVelocityFromForecasting(dt=0.5)(results)
    assert out['gt_bboxes_3d'] is boxes


def test_dt_validation():
    """dt must be strictly positive."""
    with pytest.raises(AssertionError):
        ComputeVelocityFromForecasting(dt=0.0)
    with pytest.raises(AssertionError):
        ComputeVelocityFromForecasting(dt=-0.5)


def test_dtype_round_trip_from_float64_locs():
    """KL pkl stores float64; transform must coerce to box tensor dtype."""
    boxes = _make_boxes([(0.0, 0.0)], box_dim=9)
    locs = np.zeros((1, FORECAST_STEPS, 2), dtype=np.float64)
    locs[0, 0] = (1.0, 2.0)
    mask = np.ones((1, FORECAST_STEPS), dtype=np.bool_)

    out = ComputeVelocityFromForecasting(dt=0.5)(dict(
        gt_bboxes_3d=boxes,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    # box tensor dtype must remain float32 — no silent upcast from float64 locs.
    assert out['gt_bboxes_3d'].tensor.dtype == boxes.tensor.dtype
    assert torch.allclose(out['gt_bboxes_3d'].tensor[0, 7:9],
                          torch.tensor([2.0, 4.0]))
