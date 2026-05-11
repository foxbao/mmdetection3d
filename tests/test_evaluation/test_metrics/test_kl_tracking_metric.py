"""Unit tests for KlTrackingMetric (AMOTA/IDS)."""
import tempfile
import pickle
import numpy as np
import pytest

from mmdet3d.evaluation.metrics.kl_tracking_metric import (
    KlTrackingMetric, _greedy_match)


def _make_pkl(tmp_path, data_list):
    """Write a minimal pkl file and return its path."""
    path = str(tmp_path / 'test_infos.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'data_list': data_list}, f)
    return path


class TestGreedyMatch:
    def test_empty_preds(self):
        matches, dists = _greedy_match(
            np.zeros((0, 3)), np.array([[1, 2, 3]]), 2.0)
        assert matches == [] and dists == []

    def test_empty_gt(self):
        matches, dists = _greedy_match(
            np.array([[1, 2, 3]]), np.zeros((0, 3)), 2.0)
        assert matches == [] and dists == []

    def test_perfect_match(self):
        pts = np.array([[0, 0, 0], [5, 5, 0], [10, 10, 0]], dtype=np.float32)
        matches, dists = _greedy_match(pts, pts, 2.0)
        assert len(matches) == 3
        assert all(d == 0.0 for d in dists)

    def test_threshold_rejection(self):
        pred = np.array([[0, 0, 0]], dtype=np.float32)
        gt = np.array([[3, 0, 0]], dtype=np.float32)
        matches, dists = _greedy_match(pred, gt, 2.0)
        assert matches == []

    def test_greedy_order(self):
        pred = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float32)
        gt = np.array([[1, 0, 0]], dtype=np.float32)
        matches, dists = _greedy_match(pred, gt, 2.0)
        assert len(matches) == 1
        assert matches[0] == (1, 0)  # pred[1] is closer to gt[0]


class TestKlTrackingMetric:
    @pytest.fixture
    def two_scene_data(self, tmp_path):
        """Two scenes, 3 frames each. Known track IDs for IDS/FRAG testing."""
        data_list = []
        # Scene A: 3 frames, 2 objects (track_id=1, track_id=2)
        for i in range(3):
            data_list.append(dict(
                sample_idx=i,
                scene_token='scene_A',
                timestamp=1000.0 + i * 0.5,
                instances=[
                    dict(bbox_3d=[0, 0, 0, 2, 2, 2, 0],
                         bbox_label_3d=1, track_id=1),
                    dict(bbox_3d=[10, 10, 0, 2, 2, 2, 0],
                         bbox_label_3d=1, track_id=2),
                ]))
        # Scene B: 3 frames, 1 object (track_id=3)
        for i in range(3):
            data_list.append(dict(
                sample_idx=3 + i,
                scene_token='scene_B',
                timestamp=2000.0 + i * 0.5,
                instances=[
                    dict(bbox_3d=[5, 5, 0, 2, 2, 2, 0],
                         bbox_label_3d=1, track_id=3),
                ]))
        ann_file = _make_pkl(tmp_path, data_list)
        return ann_file, data_list

    def test_perfect_tracking(self, two_scene_data):
        """Perfect predictions with consistent track IDs -> IDS=0, high AMOTA."""
        ann_file, data_list = two_scene_data
        metric = KlTrackingMetric(
            ann_file=ann_file, match_threshold=2.0, num_thresholds=10)

        # Simulate perfect predictions.
        results = []
        for info in data_list:
            instances = info['instances']
            centers = np.array([inst['bbox_3d'][:3] for inst in instances],
                              dtype=np.float32)
            labels = np.array([inst['bbox_label_3d'] for inst in instances],
                             dtype=np.int64)
            track_ids = np.array([inst['track_id'] for inst in instances],
                                dtype=np.int64)
            results.append(dict(
                sample_idx=info['sample_idx'],
                pred_centers=centers,
                pred_scores=np.ones(len(instances), dtype=np.float32),
                pred_labels=labels,
                pred_track_ids=track_ids,
            ))

        out = metric.compute_metrics(results)
        assert 'AMOTA' in out
        assert 'IDS' in out
        assert out['IDS'] == 0
        assert out['FRAG'] == 0
        assert out['AMOTA'] > 0.5

    def test_identity_switch(self, two_scene_data):
        """Swap track IDs between frames -> IDS > 0."""
        ann_file, data_list = two_scene_data
        metric = KlTrackingMetric(
            ann_file=ann_file, match_threshold=2.0, num_thresholds=10)

        results = []
        for info in data_list:
            instances = info['instances']
            centers = np.array([inst['bbox_3d'][:3] for inst in instances],
                              dtype=np.float32)
            labels = np.array([inst['bbox_label_3d'] for inst in instances],
                             dtype=np.int64)
            track_ids = np.array([inst['track_id'] for inst in instances],
                                dtype=np.int64)
            # In scene_A frame 1, swap the two track IDs.
            if info['sample_idx'] == 1:
                track_ids = track_ids[::-1].copy()
            results.append(dict(
                sample_idx=info['sample_idx'],
                pred_centers=centers,
                pred_scores=np.ones(len(instances), dtype=np.float32),
                pred_labels=labels,
                pred_track_ids=track_ids,
            ))

        out = metric.compute_metrics(results)
        assert out['IDS'] > 0

    def test_empty_predictions(self, two_scene_data):
        """No predictions -> all FN, AMOTA should be 0 or negative."""
        ann_file, data_list = two_scene_data
        metric = KlTrackingMetric(
            ann_file=ann_file, match_threshold=2.0, num_thresholds=10)

        results = []
        for info in data_list:
            results.append(dict(
                sample_idx=info['sample_idx'],
                pred_centers=np.zeros((0, 3), dtype=np.float32),
                pred_scores=np.zeros(0, dtype=np.float32),
                pred_labels=np.zeros(0, dtype=np.int64),
                pred_track_ids=np.zeros(0, dtype=np.int64),
            ))

        out = metric.compute_metrics(results)
        assert out['AMOTA'] == 0.0
        assert out['IDS'] == 0

    def test_fragmentation(self, two_scene_data):
        """Object disappears then reappears -> FRAG > 0."""
        ann_file, data_list = two_scene_data
        metric = KlTrackingMetric(
            ann_file=ann_file, match_threshold=2.0, num_thresholds=10)

        results = []
        for info in data_list:
            instances = info['instances']
            centers = np.array([inst['bbox_3d'][:3] for inst in instances],
                              dtype=np.float32)
            labels = np.array([inst['bbox_label_3d'] for inst in instances],
                             dtype=np.int64)
            track_ids = np.array([inst['track_id'] for inst in instances],
                                dtype=np.int64)
            # In scene_B frame 1 (sample_idx=4), drop the prediction.
            if info['sample_idx'] == 4:
                centers = np.zeros((0, 3), dtype=np.float32)
                labels = np.zeros(0, dtype=np.int64)
                track_ids = np.zeros(0, dtype=np.int64)
            results.append(dict(
                sample_idx=info['sample_idx'],
                pred_centers=centers,
                pred_scores=np.ones(len(centers), dtype=np.float32),
                pred_labels=labels,
                pred_track_ids=track_ids,
            ))

        out = metric.compute_metrics(results)
        assert out['FRAG'] >= 1
