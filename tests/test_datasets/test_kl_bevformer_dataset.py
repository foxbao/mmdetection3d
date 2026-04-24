# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine import dump

from mmdet3d.structures import Det3DDataSample
from projects.BEVFormer.bevformer.datasets.kl_bevformer_dataset import \
    KlBEVFormerDataset


class PackKlBEVFormerInputs:

    def __call__(self, info):
        sample_idx = int(info['sample_idx'])
        points = torch.full((sample_idx + 1, 4), float(sample_idx))
        return dict(
            inputs=dict(points=points), data_samples=Det3DDataSample())


def _make_pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def _make_info(sample_idx: int,
               scene_token: str,
               timestamp: float,
               x: float,
               prev: str = '') -> dict:
    return dict(
        sample_idx=sample_idx,
        token=f'token-{sample_idx}',
        prev=prev,
        scene_token=scene_token,
        timestamp=float(timestamp),
        ego2global=_make_pose(x),
        lidar_points=dict(
            lidar_path=f'{sample_idx:03d}.bin', num_pts_feats=4),
        instances=[
            dict(
                bbox_3d=[0.0] * 7,
                bbox_label_3d=0,
                num_lidar_pts=1,
            )
        ])


def _build_dataset(tmp_path, data_list, queue_length=3):
    ann_file = tmp_path / 'kl_infos.pkl'
    dump(dict(metainfo=dict(dataset='kl'), data_list=data_list), ann_file)
    return KlBEVFormerDataset(
        data_root=str(tmp_path),
        ann_file='kl_infos.pkl',
        data_prefix=dict(pts='points'),
        pipeline=[PackKlBEVFormerInputs()],
        queue_length=queue_length,
        filter_empty_gt=False,
        with_velocity=False,
        modality=dict(use_lidar=True, use_camera=False))


def test_prepare_data_follows_prev_chain(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        [
            _make_info(0, 'scene-a', 100.0, 0.0),
            _make_info(1, 'scene-a', 100.5, 1.0, prev='token-0'),
            _make_info(2, 'scene-a', 101.0, 3.0, prev='token-1'),
        ])

    sample = dataset.prepare_data(2)

    assert sample is not None
    assert sample['inputs']['points'].shape == (3, 4)
    assert len(sample['inputs']['history_points']) == 2
    assert sample['inputs']['history_points'][0].shape == (1, 4)
    assert sample['inputs']['history_points'][1].shape == (2, 4)
    assert torch.all(sample['inputs']['points'] == 2)
    assert torch.all(sample['inputs']['history_points'][0] == 0)
    assert torch.all(sample['inputs']['history_points'][1] == 1)

    queue_metas = sample['data_samples'].metainfo['queue_metas']
    assert [queue_metas[i]['token'] for i in range(3)] == [
        'token-0', 'token-1', 'token-2'
    ]
    assert [queue_metas[i]['prev_bev_exists'] for i in range(3)] == [
        False, True, True
    ]
    np.testing.assert_allclose(
        queue_metas[0]['ego_motion_delta'], np.eye(4, dtype=np.float64))
    np.testing.assert_allclose(
        queue_metas[1]['ego_motion_delta'],
        np.linalg.inv(_make_pose(1.0)) @ _make_pose(0.0))
    np.testing.assert_allclose(
        queue_metas[2]['ego_motion_delta'],
        np.linalg.inv(_make_pose(3.0)) @ _make_pose(1.0))
    assert queue_metas[0]['time_delta'] == 0.0
    assert queue_metas[1]['time_delta'] == 0.5
    assert queue_metas[2]['time_delta'] == 0.5


def test_prepare_data_returns_none_when_prev_chain_too_short(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        [
            _make_info(0, 'scene-a', 100.0, 0.0),
            _make_info(1, 'scene-a', 100.5, 1.0, prev='token-0'),
        ])

    assert dataset.prepare_data(1) is None
    assert dataset.prepare_data(0) is None


def test_prepare_data_returns_none_when_prev_chain_breaks(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        [
            _make_info(0, 'scene-a', 100.0, 0.0),
            _make_info(1, 'scene-a', 100.5, 1.0, prev='token-0'),
            _make_info(2, 'scene-a', 101.0, 2.0),
            _make_info(3, 'scene-a', 101.5, 3.0, prev='token-2'),
            _make_info(4, 'scene-b', 200.0, 10.0, prev='token-3'),
            _make_info(5, 'scene-b', 200.5, 11.0, prev='missing-token'),
        ])

    assert dataset.prepare_data(3) is None
    assert dataset.prepare_data(4) is None
    assert dataset.prepare_data(5) is None
