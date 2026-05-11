"""Regression tests for ``tools/dataset_converters/add_velocity.py``.

Covers three corrections:
  A. ``scene_token`` missing/empty must raise — the (scene, track_id) key
     would otherwise collapse and cross-scene track_id reuse (rampant in
     KL pkl) would silently produce wrong velocities.
  B. When centered diff is rejected by the time guard but a single-sided
     neighbour is in range, the fallback must still stamp a velocity.
  D. Per-class rejection counter must populate so silent zero-fallback
     drops are visible in the conversion log.

Plus one basic sanity test that the standard centered-diff path produces
the expected velocity magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest
from mmengine import dump

from tools.dataset_converters.add_velocity import (
    _build_track_records, _estimate_track_velocities, _lidar2ego_from_frame,
    _segment_records, add_velocity_to_pkl)


def _make_info(frame_idx, scene_token, timestamp, instances,
               ego2global=None) -> dict:
    """Build a minimal KL info dict that ``_build_track_records`` accepts."""
    return dict(
        sample_idx=frame_idx,
        scene_token=scene_token,
        timestamp=float(timestamp),
        ego2global=(np.eye(4, dtype=np.float64).tolist()
                    if ego2global is None else ego2global),
        instances=instances)


def _inst(track_id, x, y=0.0, z=0.0, label=0) -> dict:
    return dict(
        track_id=int(track_id),
        bbox_3d=[x, y, z, 2.0, 2.0, 2.0, 0.0],
        bbox_label_3d=int(label),
        bbox_3d_isvalid=True,
        num_lidar_pts=10,
        num_radar_pts=0)


# ----------------------------- A: scene_token --------------------------------

def test_missing_scene_token_raises():
    infos = [
        _make_info(0, 'scene-a', 0.0, [_inst(1, 0.0)]),
        _make_info(1, None,      0.5, [_inst(1, 1.0)]),  # missing
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    with pytest.raises(ValueError, match='scene_token'):
        _build_track_records(infos, lidar2ego)


def test_empty_scene_token_raises():
    infos = [
        _make_info(0, 'scene-a', 0.0, [_inst(1, 0.0)]),
        _make_info(1, '',        0.5, [_inst(1, 1.0)]),  # empty string
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    with pytest.raises(ValueError, match='scene_token'):
        _build_track_records(infos, lidar2ego)


# ----------------------------- B: fallback diff ------------------------------

def test_centered_rejection_falls_back_to_one_sided():
    """Middle record of a track t=[0, 0.5, 5.0] — centered fails (5s>3s),
    forward fails (4.5s>1.5s), backward succeeds (0.5s<=1.5s)."""
    infos = [
        _make_info(0, 's', 0.0, [_inst(1, 0.0)]),
        _make_info(1, 's', 0.5, [_inst(1, 1.0)]),  # 1m forward in 0.5s
        _make_info(2, 's', 5.0, [_inst(1, 10.0)]),  # huge gap
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    tracks = _build_track_records(infos, lidar2ego)

    velocities, valid, rej_t, rej_s, singletons, by_label = (
        _estimate_track_velocities(
            tracks, min_dt=1e-3, max_time_diff=1.5, max_speed=60.0))

    # Middle record (frame 1) should now have a velocity from the backward
    # diff. Without the fallback it would be in rejected_time.
    assert (1, 0) in velocities, \
        'middle record lost velocity — fallback to one-sided diff missing'
    v = velocities[(1, 0)]
    # backward diff: (1.0 - 0.0) / 0.5 = 2.0 m/s in global x
    assert np.isclose(v[0], 2.0)
    assert np.isclose(v[1], 0.0)


def test_centered_diff_used_when_balanced():
    """Sanity: with three close neighbours, centered diff (over t_prev..t_next)
    is preferred and gives the expected velocity."""
    infos = [
        _make_info(0, 's', 0.0, [_inst(1, 0.0)]),
        _make_info(1, 's', 0.5, [_inst(1, 1.0)]),
        _make_info(2, 's', 1.0, [_inst(1, 2.0)]),
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    tracks = _build_track_records(infos, lidar2ego)
    velocities, *_ = _estimate_track_velocities(
        tracks, min_dt=1e-3, max_time_diff=1.5, max_speed=60.0)

    # Middle record: centered diff = (2.0 - 0.0) / 1.0 = 2.0 m/s
    assert np.isclose(velocities[(1, 0)][0], 2.0)


# ------------------------- D: per-class rejection ----------------------------

def test_singleton_bumps_rejected_by_label():
    """Singletons (track length < 2) feed into rejected_by_label so logs
    can show which classes lose velocity coverage."""
    infos = [
        _make_info(0, 's', 0.0, [
            _inst(1, 0.0, label=3),  # only frame for track 1 -> singleton
            _inst(2, 5.0, label=7),
        ]),
        _make_info(1, 's', 0.5, [
            _inst(2, 6.0, label=7),  # track 2 has 2 records, gets a velocity
        ]),
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    tracks = _build_track_records(infos, lidar2ego)

    velocities, valid, rej_t, rej_s, singletons, by_label = (
        _estimate_track_velocities(
            tracks, min_dt=1e-3, max_time_diff=1.5, max_speed=60.0))

    assert singletons == 1  # track 1
    assert by_label[3] == 1  # singleton was label 3
    assert by_label.get(7, 0) == 0  # track 2 succeeded — not counted
    assert valid >= 2  # both records of track 2 stamped


def test_speed_cap_triggers_speed_rejection_path():
    """A 100 m jump in 0.5s should hit the speed guard, not the time guard."""
    infos = [
        _make_info(0, 's', 0.0, [_inst(1, 0.0, label=2)]),
        _make_info(1, 's', 0.5, [_inst(1, 100.0, label=2)]),  # 200 m/s
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    tracks = _build_track_records(infos, lidar2ego)
    velocities, valid, rej_t, rej_s, singletons, by_label = (
        _estimate_track_velocities(
            tracks, min_dt=1e-3, max_time_diff=1.5, max_speed=60.0))

    assert valid == 0
    assert rej_s == 2  # both records hit the speed guard via their only strategy
    assert rej_t == 0
    assert by_label[2] == 2


# --------------- segment splitting (variable FPS + scene gaps) ---------------

def _rec(t, x):
    return dict(timestamp=float(t), center_global=np.array([x, 0., 0.]),
                frame_idx=0, inst_idx=0, bbox_label=0)


def test_segment_records_no_gap():
    """All adjacent dts within max_time_diff → single segment."""
    recs = [_rec(t, t) for t in [0.0, 0.5, 1.0, 1.5]]
    segs = _segment_records(recs, max_time_diff=1.5)
    assert len(segs) == 1
    assert len(segs[0]) == 4


def test_segment_records_splits_at_internal_gap():
    """A 4.5s gap > max_time_diff splits the track into two contiguous pieces."""
    recs = [_rec(t, t) for t in [0.0, 0.5, 5.0, 5.5, 6.0]]
    segs = _segment_records(recs, max_time_diff=1.5)
    assert len(segs) == 2
    assert [r['timestamp'] for r in segs[0]] == [0.0, 0.5]
    assert [r['timestamp'] for r in segs[1]] == [5.0, 5.5, 6.0]


def test_segment_records_handles_variable_fps_within_segment():
    """1Hz → 2Hz → 5Hz mix all stay in one segment if every dt ≤ max_time_diff."""
    recs = [_rec(t, t) for t in [0.0, 1.0, 1.5, 1.7, 1.9]]
    segs = _segment_records(recs, max_time_diff=1.5)
    assert len(segs) == 1


def test_lopsided_centered_diff_no_longer_crosses_gap():
    """Pre-fix: a record with prev=close (0.5s) and next=far (2.0s, just past
    max_time_diff) would silently get a centered diff over the 2.5s span,
    averaging through the gap. Post-fix: the gap splits the track, so this
    record is the segment's last record and falls back to backward diff."""
    infos = [
        _make_info(0, 's', 0.0, [_inst(1, 0.0)]),
        _make_info(1, 's', 0.5, [_inst(1, 1.0)]),  # 1m forward in 0.5s
        _make_info(2, 's', 2.5, [_inst(1, 100.0)]),  # huge jump after gap
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    tracks = _build_track_records(infos, lidar2ego)
    velocities, *_ = _estimate_track_velocities(
        tracks, min_dt=1e-3, max_time_diff=1.5, max_speed=60.0)

    # Frame 1 should NOT use the (frame 0, frame 2) centered diff —
    # the (1→2) side has a gap. Backward diff = (1.0 - 0.0)/0.5 = 2 m/s.
    v = velocities.get((1, 0))
    assert v is not None
    assert np.isclose(v[0], 2.0), \
        f'frame 1 vel={v} — expected backward diff ≈ 2 m/s, got centered'

    # Frame 2 is alone in its post-gap segment → singleton, no velocity.
    assert (2, 0) not in velocities


def test_track_id_reuse_across_gap_does_not_mix_segments():
    """If track_id is reused across a long un-annotated gap (KL allows this
    because the user splits scenes by motion), pre-split would diff across
    unrelated trajectories. Post-split each segment is independent."""
    infos = [
        _make_info(0, 's', 0.0,  [_inst(1, 0.0)]),
        _make_info(1, 's', 0.5,  [_inst(1, 1.0)]),
        _make_info(2, 's', 60.0, [_inst(1, 500.0)]),  # different physical obj
        _make_info(3, 's', 60.5, [_inst(1, 501.0)]),
    ]
    lidar2ego = _lidar2ego_from_frame('FLU')
    tracks = _build_track_records(infos, lidar2ego)
    velocities, *_ = _estimate_track_velocities(
        tracks, min_dt=1e-3, max_time_diff=1.5, max_speed=60.0)

    # Each segment should produce reasonable in-segment velocities (~2 m/s).
    for key in [(0, 0), (1, 0), (2, 0), (3, 0)]:
        v = velocities.get(key)
        assert v is not None
        assert abs(v[0] - 2.0) < 1e-6, \
            f'{key} vel={v} — must be in-segment, not cross-gap'


# ------------------------ end-to-end via add_velocity_to_pkl -----------------

def test_add_velocity_to_pkl_writes_in_place(tmp_path):
    """Smoke test: the top-level entrypoint runs end-to-end and writes a
    non-zero ``instance['velocity']`` for a moving track."""
    infos = [
        _make_info(0, 's', 0.0, [_inst(1, 0.0)]),
        _make_info(1, 's', 0.5, [_inst(1, 1.0)]),
    ]
    pkl = tmp_path / 'tiny.pkl'
    dump(dict(metainfo=dict(lidar_coord_frame='FLU',
                            classes=['Car']),
              data_list=infos), pkl)

    add_velocity_to_pkl(str(pkl), in_place=True)

    import mmengine
    out = mmengine.load(pkl)
    velocities = [inst['velocity']
                  for info in out['data_list']
                  for inst in info['instances']]
    # FLU: ego_x = lidar_x, so global +x motion stays as lidar +x.
    # Forward diff for both (since each record only has one neighbour).
    speeds = [float(np.hypot(v[0], v[1])) for v in velocities]
    assert all(s > 1.5 for s in speeds), \
        f'expected ~2 m/s for a 1m / 0.5s motion, got {speeds}'
