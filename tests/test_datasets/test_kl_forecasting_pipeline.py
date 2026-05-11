"""Phase 1 verification: forecasting GT flows through the Stage 3 data pipeline.

These tests anchor the invariant that ``gt_forecasting_locs`` /
``gt_forecasting_mask`` stay row-aligned with ``gt_bboxes_3d`` through every
transform, and land in ``data_samples.gt_instances_3d`` when
``Pack3DDetInputs`` is given the forecasting keys. The current Stage 3 config
omits them from ``Pack3DDetInputs.keys`` — that's fine for detection-only
training, but MotionHead config must add them (``test_pack_*`` pins both
behaviours).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from mmengine import dump

from mmdet3d.datasets.transforms import (LoadAnnotations3D, ObjectNameFilter,
                                         ObjectRangeFilter, Pack3DDetInputs)
from mmdet3d.structures import LiDARInstance3DBoxes
from projects.BEVFormer.bevformer.datasets.kl_bevformer_dataset import \
    KlBEVFormerDataset
from projects.KL8.transforms import GenerateKLLanguageQuery


FORECAST_STEPS = 6


def _boxes_and_forecasts(coords, labels):
    """Build a (gt_bboxes_3d, gt_labels_3d, locs, mask) quadruple from XY list.

    Each coord is used both as the box center (z=0, lwh=(2,2,2), yaw=0, vel=0)
    and as a distinctive fill value inside the forecasting arrays so row
    alignment is easy to check downstream.
    """
    n = len(coords)
    boxes = np.zeros((n, 7), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        boxes[i] = [x, y, 0.0, 2.0, 2.0, 2.0, 0.0]
    gt_boxes = LiDARInstance3DBoxes(boxes, box_dim=7)
    gt_labels = np.asarray(labels, dtype=np.int64)
    locs = np.stack([
        np.full((FORECAST_STEPS, 2), fill_value=float(i), dtype=np.float32)
        for i in range(n)
    ])
    mask = np.ones((n, FORECAST_STEPS), dtype=np.bool_)
    mask[:, -1] = False  # distinguish a non-trivial pattern
    return gt_boxes, gt_labels, locs, mask


def test_object_range_filter_syncs_forecasting():
    gt_boxes, gt_labels, locs, mask = _boxes_and_forecasts(
        coords=[(0.0, 0.0), (40.0, 0.0), (-3.0, 0.0)],
        labels=[0, 0, 1])  # middle one is out of [-5, 5] x range

    transform = ObjectRangeFilter(point_cloud_range=[-5, -5, -5, 5, 5, 5])
    out = transform(dict(
        gt_bboxes_3d=copy.deepcopy(gt_boxes),
        gt_labels_3d=gt_labels.copy(),
        gt_forecasting_locs=locs.copy(),
        gt_forecasting_mask=mask.copy()))

    kept = out['gt_bboxes_3d'].tensor.shape[0]
    assert kept == 2
    assert out['gt_forecasting_locs'].shape == (kept, FORECAST_STEPS, 2)
    assert out['gt_forecasting_mask'].shape == (kept, FORECAST_STEPS)
    # the filler value uniquely identifies the source row — verify
    # surviving rows are the original {0, 2}, in that order.
    surviving = out['gt_forecasting_locs'][:, 0, 0].tolist()
    assert surviving == [0.0, 2.0]


def test_object_name_filter_syncs_forecasting():
    gt_boxes, gt_labels, locs, mask = _boxes_and_forecasts(
        coords=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        labels=[0, 1, 2])

    transform = ObjectNameFilter(classes=['a', 'c'])
    # ObjectNameFilter reads self.labels based on the classes list order.
    out = transform(dict(
        gt_bboxes_3d=copy.deepcopy(gt_boxes),
        gt_labels_3d=gt_labels.copy(),
        gt_forecasting_locs=locs.copy(),
        gt_forecasting_mask=mask.copy()))

    # classes=['a','c'] -> self.labels = [0, 1] => keep gt_labels in {0,1}
    kept = out['gt_bboxes_3d'].tensor.shape[0]
    assert kept == 2
    assert out['gt_forecasting_locs'].shape == (kept, FORECAST_STEPS, 2)
    assert out['gt_forecasting_mask'].shape == (kept, FORECAST_STEPS)
    surviving = out['gt_forecasting_locs'][:, 0, 0].tolist()
    assert surviving == [0.0, 1.0]


def test_pack_inputs_delivers_forecasting_to_gt_instances_3d():
    gt_boxes, gt_labels, locs, mask = _boxes_and_forecasts(
        coords=[(0.0, 0.0), (1.0, 0.0)],
        labels=[0, 1])

    pack = Pack3DDetInputs(keys=[
        'points', 'gt_bboxes_3d', 'gt_labels_3d',
        'gt_forecasting_locs', 'gt_forecasting_mask'])
    out = pack(dict(
        points=torch.zeros(10, 4),
        gt_bboxes_3d=gt_boxes,
        gt_labels_3d=gt_labels,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    gi3d = out['data_samples'].gt_instances_3d
    assert hasattr(gi3d, 'forecasting_locs')
    assert hasattr(gi3d, 'forecasting_mask')
    assert gi3d.forecasting_locs.shape == (2, FORECAST_STEPS, 2)
    assert gi3d.forecasting_mask.shape == (2, FORECAST_STEPS)
    # value preservation (row 0 is all 0.0, row 1 is all 1.0 by construction)
    assert torch.allclose(gi3d.forecasting_locs[0],
                          torch.zeros(FORECAST_STEPS, 2))
    assert torch.allclose(gi3d.forecasting_locs[1],
                          torch.ones(FORECAST_STEPS, 2))


def test_language_query_pack_inputs_to_meta_and_gt_instances():
    gt_boxes, gt_labels, locs, mask = _boxes_and_forecasts(
        coords=[(10.0, 0.0), (50.0, 0.0), (-5.0, 0.0)],
        labels=[0, 0, 0])
    transform = GenerateKLLanguageQuery(
        query_types=('front', ), distance=40.0, max_tokens=8,
        fallback_to_all=False)
    results = transform(dict(
        points=torch.zeros(4, 4),
        gt_bboxes_3d=gt_boxes,
        gt_labels_3d=gt_labels,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    pack = Pack3DDetInputs(
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_forecasting_locs', 'gt_forecasting_mask',
            'gt_language_target_mask',
        ],
        meta_keys=[
            'language_prompt', 'language_query_type',
            'language_tokens', 'language_token_mask',
        ])
    out = pack(results)

    gi3d = out['data_samples'].gt_instances_3d
    assert hasattr(gi3d, 'language_target_mask')
    assert gi3d.language_target_mask.tolist() == [True, False, False]
    assert out['data_samples'].metainfo['language_query_type'] == 'front'
    assert out['data_samples'].metainfo['language_token_mask'].sum() > 0


def test_pack_inputs_drops_forecasting_when_not_in_keys():
    """Stage 3's current keys list drops forecasting — pin that behaviour."""
    gt_boxes, gt_labels, locs, mask = _boxes_and_forecasts(
        coords=[(0.0, 0.0)], labels=[0])

    pack = Pack3DDetInputs(
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])  # no forecasting
    out = pack(dict(
        points=torch.zeros(4, 4),
        gt_bboxes_3d=gt_boxes,
        gt_labels_3d=gt_labels,
        gt_forecasting_locs=locs,
        gt_forecasting_mask=mask))

    gi3d = out['data_samples'].gt_instances_3d
    assert not hasattr(gi3d, 'forecasting_locs')
    assert not hasattr(gi3d, 'forecasting_mask')


# ------------------- end-to-end via KlBEVFormerDataset -------------------

def _make_pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def _make_info_with_forecast(sample_idx: int, scene_token: str,
                             timestamp: float, x: float, prev: str = '',
                             nxt: str = '') -> dict:
    locs = np.full((FORECAST_STEPS, 2), 0.3, dtype=np.float32).tolist()
    mask = [True] * FORECAST_STEPS
    return dict(
        sample_idx=sample_idx,
        token=f'token-{sample_idx}',
        prev=prev,
        next=nxt,
        scene_token=scene_token,
        timestamp=float(timestamp),
        ego2global=_make_pose(x),
        lidar_points=dict(lidar_path=f'{sample_idx:03d}.bin', num_pts_feats=4),
        instances=[
            dict(
                bbox_3d=[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                bbox_label_3d=0,
                bbox_3d_isvalid=True,
                num_lidar_pts=5,
                gt_forecasting_locs=locs,
                gt_forecasting_mask=mask,
            ),
            dict(
                bbox_3d=[5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                bbox_label_3d=0,
                bbox_3d_isvalid=True,
                num_lidar_pts=5,
                gt_forecasting_locs=locs,
                gt_forecasting_mask=mask,
            ),
        ])


class _FakePointLoader:
    """Skip real point loading — the queue loader runs the pipeline on each
    frame, so a disk-free stub keeps the test hermetic."""

    def __call__(self, info):
        info['points'] = torch.zeros(1, 4)
        return info


def test_kl_bevformer_dataset_exposes_forecasting_end_to_end(tmp_path):
    data_list = [
        _make_info_with_forecast(0, 'scene-a', 100.0, 0.0, nxt='token-1'),
        _make_info_with_forecast(1, 'scene-a', 100.5, 1.0, prev='token-0',
                                  nxt='token-2'),
        _make_info_with_forecast(2, 'scene-a', 101.0, 2.0, prev='token-1'),
    ]
    ann_file = tmp_path / 'kl_infos.pkl'
    dump(dict(metainfo=dict(dataset='kl'), data_list=data_list), ann_file)

    pipeline = [
        _FakePointLoader(),
        LoadAnnotations3D(
            with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
        ObjectRangeFilter(point_cloud_range=[-10, -10, -5, 10, 10, 5]),
        ObjectNameFilter(classes=['Pedestrian']),
        Pack3DDetInputs(keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_forecasting_locs', 'gt_forecasting_mask']),
    ]

    dataset = KlBEVFormerDataset(
        data_root=str(tmp_path),
        ann_file='kl_infos.pkl',
        data_prefix=dict(pts='points'),
        pipeline=pipeline,
        queue_length=2,
        filter_empty_gt=False,
        with_velocity=False,
        use_valid_flag=True,
        metainfo=dict(classes=['Pedestrian']),
        modality=dict(use_lidar=True, use_camera=False))

    sample = dataset.prepare_data(2)
    assert sample is not None
    gi3d = sample['data_samples'].gt_instances_3d
    # both instances are inside the filter range and have label 'Pedestrian'
    assert gi3d.bboxes_3d.tensor.shape[0] == 2
    assert gi3d.forecasting_locs.shape == (2, FORECAST_STEPS, 2)
    assert gi3d.forecasting_mask.shape == (2, FORECAST_STEPS)
    expected = torch.full((2, FORECAST_STEPS, 2), 0.3,
                          dtype=gi3d.forecasting_locs.dtype)
    assert torch.allclose(gi3d.forecasting_locs, expected)
    assert bool(gi3d.forecasting_mask.all())
