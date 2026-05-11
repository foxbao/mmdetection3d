#!/usr/bin/env python
"""Add per-object velocity annotations to KL dataset v2 pkl files.

Velocity is estimated from object tracks.  For each instance with a valid
``track_id``, the script transforms the 3D box center from the frame's LiDAR
coordinates into global coordinates, differentiates positions over time within
the same scene and track, then rotates the velocity vector back into the
current LiDAR frame.  The resulting ``[vx, vy]`` is written to each instance's
``velocity`` field, matching the nuScenes-style annotation consumed by
``KlDataset(with_velocity=True)``.

By default this script writes a sibling file with ``_with_velocity`` appended
to the input stem.  Use ``--in-place`` only when you intentionally want to
overwrite the existing pkl.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np


def _lidar2ego_from_frame(frame: str) -> np.ndarray:
    """Return LiDAR-to-ego transform for KL pkl coordinate metadata."""
    mat = np.eye(4, dtype=np.float64)
    if frame == 'RFU':
        mat[:2, :2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    elif frame != 'FLU':
        raise ValueError(f'unknown lidar_coord_frame: {frame!r}')
    return mat


def _center_global(info: dict, inst: dict, lidar2ego: np.ndarray) -> np.ndarray:
    """Transform one box center from frame LiDAR coordinates to global."""
    box = inst['bbox_3d']
    center_lidar = np.array([box[0], box[1], box[2], 1.0], dtype=np.float64)
    ego2global = np.array(info['ego2global'], dtype=np.float64)
    return (ego2global @ lidar2ego @ center_lidar)[:3]


def _global_vel_to_lidar_xy(info: dict, vel_global: np.ndarray,
                            lidar2ego: np.ndarray) -> np.ndarray:
    """Rotate a global velocity vector into the current LiDAR frame."""
    ego2global = np.array(info['ego2global'], dtype=np.float64)
    lidar2global = ego2global @ lidar2ego
    global2lidar_rot = np.linalg.inv(lidar2global[:3, :3])
    vel_lidar = global2lidar_rot @ vel_global
    return vel_lidar[:2].astype(np.float32)


def _build_track_records(infos: list, lidar2ego: np.ndarray) -> dict:
    """Collect sortable records grouped by (scene_token, track_id).

    ``scene_token`` MUST be present and non-empty on every frame. Without
    it, the (scene, track_id) compound key collapses and same-track_id
    boxes from different scenes merge — KL pkl reuses ~93% of track_ids
    across scenes, so a silent fallback would produce garbage velocities
    spanning unrelated trajectories.
    """
    tracks = defaultdict(list)
    for frame_idx, info in enumerate(infos):
        scene = info.get('scene_token')
        if not scene:
            raise ValueError(
                f'frame {frame_idx} (sample_idx={info.get("sample_idx")}) '
                f'has missing/empty scene_token; cannot safely group tracks.')
        timestamp = float(info.get('timestamp', 0.0))
        for inst_idx, inst in enumerate(info.get('instances', [])):
            track_id = inst.get('track_id', -1)
            if track_id is None or int(track_id) < 0:
                continue
            center_global = _center_global(info, inst, lidar2ego)
            tracks[(scene, int(track_id))].append({
                'frame_idx': frame_idx,
                'inst_idx': inst_idx,
                'timestamp': timestamp,
                'center_global': center_global,
                'bbox_label': int(inst.get('bbox_label_3d', -1)),
            })

    for records in tracks.values():
        records.sort(key=lambda x: x['timestamp'])
    return tracks


def _segment_records(records: list, max_time_diff: float) -> list:
    """Split a time-sorted track into sub-segments at any internal gap
    larger than ``max_time_diff``.

    Two scene-level data quirks make this necessary:
      * KL annotates at variable FPS (mostly 2 Hz, sometimes 1/5 Hz) and
        skips intervals where nothing moves — so two adjacent records in
        the same ``(scene, track_id)`` may legitimately be 5+ seconds
        apart, and differencing across that gap is meaningless.
      * If track_ids are reused across un-annotated stationary intervals,
        sorting+diffing across the gap mixes two unrelated trajectories.

    Splitting at ``> max_time_diff`` matches the same trust-window the
    one-sided guards use, so within a sub-segment every adjacent dt is
    safe and centered diff has both sides bounded by construction.
    """
    if not records:
        return []
    segments = [[records[0]]]
    for r in records[1:]:
        if r['timestamp'] - segments[-1][-1]['timestamp'] > max_time_diff:
            segments.append([r])
        else:
            segments[-1].append(r)
    return segments


def _estimate_track_velocities(tracks: dict, min_dt: float,
                               max_time_diff: float,
                               max_speed: float) -> tuple:
    """Estimate global velocity for each frame/instance occurrence.

    Tries strategies in order — centered diff (when both neighbours exist),
    forward, then backward — and takes the first one whose ``dt`` and
    speed pass the guards. Earlier this function used a hard if/elif: a
    centered attempt rejected by ``max_time_diff`` would skip the record
    entirely, even when one of the two neighbours was perfectly close.

    Each track is first cut at internal gaps via ``_segment_records`` so
    sub-segments span only contiguous-FPS observation windows.
    """
    velocities = {}
    valid_count = 0
    rejected_time = 0
    rejected_speed = 0
    singletons = 0
    rejected_by_label: Counter = Counter()

    for track_records in tracks.values():
        for records in _segment_records(track_records, max_time_diff):
            if len(records) < 2:
                singletons += len(records)
                for rec in records:
                    rejected_by_label[rec['bbox_label']] += 1
                continue
            for i, rec in enumerate(records):
                prev_rec = records[i - 1] if i > 0 else None
                next_rec = records[i + 1] if i + 1 < len(records) else None

                strategies = []
                if prev_rec is not None and next_rec is not None:
                    strategies.append(
                        (prev_rec, next_rec, 2.0 * max_time_diff))
                if next_rec is not None:
                    strategies.append((rec, next_rec, max_time_diff))
                if prev_rec is not None:
                    strategies.append((prev_rec, rec, max_time_diff))

                vel_global = None
                saw_dt_ok = False
                for r0, r1, allowed_dt in strategies:
                    dt = float(r1['timestamp'] - r0['timestamp'])
                    if dt <= min_dt or dt > allowed_dt:
                        continue
                    saw_dt_ok = True
                    v = (r1['center_global'] - r0['center_global']) / dt
                    if float(np.linalg.norm(v[:2])) > max_speed:
                        continue
                    vel_global = v
                    break

                if vel_global is None:
                    if saw_dt_ok:
                        rejected_speed += 1
                    else:
                        rejected_time += 1
                    rejected_by_label[rec['bbox_label']] += 1
                    continue

                key = (rec['frame_idx'], rec['inst_idx'])
                velocities[key] = vel_global
                valid_count += 1

    return (velocities, valid_count, rejected_time, rejected_speed,
            singletons, rejected_by_label)


def add_velocity_to_pkl(pkl_path: str,
                        out_path: str = None,
                        in_place: bool = False,
                        min_dt: float = 1e-3,
                        max_time_diff: float = 1.5,
                        max_speed: float = 60.0) -> None:
    """Process one KL info pkl and write velocity annotations."""
    pkl_path = Path(pkl_path)
    if in_place:
        out_path = pkl_path
    elif out_path is None:
        out_path = pkl_path.with_name(f'{pkl_path.stem}_with_velocity.pkl')
    else:
        out_path = Path(out_path)

    print(f'\n{"=" * 60}')
    print(f'Processing: {pkl_path}')
    print(f'Output:     {out_path}')

    data = mmengine.load(pkl_path)
    infos = data['data_list']
    frame = data.get('metainfo', {}).get('lidar_coord_frame', 'FLU')
    lidar2ego = _lidar2ego_from_frame(frame)

    print(f'Total frames: {len(infos)}  lidar_coord_frame: {frame}')
    tracks = _build_track_records(infos, lidar2ego)
    print(f'Tracks with valid track_id: {len(tracks)}')

    (velocities, valid_count, rejected_time, rejected_speed, singletons,
     rejected_by_label) = _estimate_track_velocities(
        tracks, min_dt=min_dt, max_time_diff=max_time_diff,
        max_speed=max_speed)
    metainfo = data.get('metainfo', {})
    # KL pkl stores ``categories: {name: idx}``; older formats use ``classes: [names]``.
    # Build a ``label_to_name`` dict that handles both.
    label_to_name = {}
    if isinstance(metainfo.get('categories'), dict):
        label_to_name = {int(idx): name
                         for name, idx in metainfo['categories'].items()}
    elif isinstance(metainfo.get('classes'), (list, tuple)):
        label_to_name = {i: name for i, name in enumerate(metainfo['classes'])}

    total_instances = 0
    zero_count = 0
    speeds = []
    for frame_idx, info in enumerate(infos):
        for inst_idx, inst in enumerate(info.get('instances', [])):
            total_instances += 1
            vel_global = velocities.get((frame_idx, inst_idx))
            if vel_global is None:
                inst['velocity'] = [0.0, 0.0]
                zero_count += 1
                continue

            vel_xy = _global_vel_to_lidar_xy(info, vel_global, lidar2ego)
            inst['velocity'] = vel_xy.tolist()
            speeds.append(float(np.linalg.norm(vel_xy)))

    nonzero_count = total_instances - zero_count
    print(f'Instances: {total_instances}')
    print(f'Estimated velocities: {valid_count}')
    print(f'Written non-zero-capable velocities: {nonzero_count}')
    print(f'Zero velocity fallback: {zero_count}')
    print(f'Single-frame track instances: {singletons}')
    print(f'Rejected by time guard (>{max_time_diff:g}s one-sided, '
          f'>{2 * max_time_diff:g}s centered): {rejected_time}')
    print(f'Rejected by speed guard (>{max_speed:g} m/s): {rejected_speed}')
    if rejected_by_label:
        print('Zero-fallback breakdown by class label:')
        for label in sorted(rejected_by_label):
            name = label_to_name.get(label, f'<label {label}>')
            print(f'  {name:25s} {rejected_by_label[label]}')
    if speeds:
        arr = np.asarray(speeds, dtype=np.float64)
        print('Speed stats m/s: '
              f'mean={arr.mean():.3f}, p50={np.percentile(arr, 50):.3f}, '
              f'p95={np.percentile(arr, 95):.3f}, max={arr.max():.3f}')

    mmengine.mkdir_or_exist(out_path.parent)
    mmengine.dump(data, out_path)
    print(f'Saved to: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Add track-difference velocity annotations to KL pkl files.')
    parser.add_argument(
        '--pkl-path', nargs='+', required=True,
        help='Path(s) to KL v2 pkl files.')
    parser.add_argument(
        '--out-path', default=None,
        help='Output path. Only valid when one --pkl-path is provided.')
    parser.add_argument(
        '--in-place', action='store_true',
        help='Overwrite input pkl files instead of writing *_with_velocity.pkl.')
    parser.add_argument(
        '--min-dt', type=float, default=1e-3,
        help='Minimum time delta in seconds for differencing.')
    parser.add_argument(
        '--max-time-diff', type=float, default=1.5,
        help='Maximum allowed adjacent-frame time difference in seconds. '
        'Centered differences allow twice this value, matching nuScenes.')
    parser.add_argument(
        '--max-speed', type=float, default=60.0,
        help='Reject estimated speeds above this threshold in m/s.')
    args = parser.parse_args()

    if args.out_path is not None and len(args.pkl_path) != 1:
        raise ValueError('--out-path can only be used with one --pkl-path.')

    for p in args.pkl_path:
        add_velocity_to_pkl(
            p,
            out_path=args.out_path,
            in_place=args.in_place,
            min_dt=args.min_dt,
            max_time_diff=args.max_time_diff,
            max_speed=args.max_speed)

    print('\nDone.')


if __name__ == '__main__':
    main()
