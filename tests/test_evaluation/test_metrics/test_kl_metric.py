import pytest

from mmdet3d.evaluation.metrics.kl_metric import KlDetectionBox, load_gt


class DummyKlDevKit:

    def __init__(self, data_list):
        self.data = dict(data_list=data_list)


def _make_sample(token, label=0):
    return dict(
        token=token,
        instances=[
            dict(
                bbox_3d=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1],
                bbox_label=label,
                velocity=[0.0, 0.0],
                num_lidar_pts=1,
                num_radar_pts=0)
        ])


def test_load_gt_filters_by_sample_tokens():
    kl_devkit = DummyKlDevKit([
        _make_sample('token-a'),
        _make_sample('token-b'),
        _make_sample('token-c'),
    ])

    gt_boxes = load_gt(
        kl_devkit,
        'val',
        KlDetectionBox,
        sample_tokens=['token-b', 'token-c'])

    assert set(gt_boxes.sample_tokens) == {'token-b', 'token-c'}
    assert gt_boxes['token-b'][0].sample_token == 'token-b'
    assert gt_boxes['token-c'][0].sample_token == 'token-c'


def test_load_gt_rejects_invalid_split():
    kl_devkit = DummyKlDevKit([_make_sample('token-a')])

    with pytest.raises(ValueError, match='Invalid eval_split'):
        load_gt(kl_devkit, 'test', KlDetectionBox)
