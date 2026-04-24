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
from collections import defaultdict
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
    """Collect sortable records grouped by (scene_token, track_id)."""
    tracks = defaultdict(list)
    for frame_idx, info in enumerate(infos):
        scene = info.get('scene_token', '')
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
            })

    for records in tracks.values():
        records.sort(key=lambda x: x['timestamp'])
    return tracks


def _estimate_track_velocities(tracks: dict, min_dt: float,
                               max_time_diff: float,
                               max_speed: float) -> tuple:
    """Estimate global velocity for each frame/instance occurrence."""
    velocities = {}
    valid_count = 0
    rejected_time = 0
    rejected_speed = 0
    singletons = 0

    for records in tracks.values():
        if len(records) < 2:
            singletons += len(records)
            continue

        for i, rec in enumerate(records):
            prev_rec = records[i - 1] if i > 0 else None
            next_rec = records[i + 1] if i + 1 < len(records) else None

            if prev_rec is not None and next_rec is not None:
                p0 = prev_rec['center_global']
                p1 = next_rec['center_global']
                t0 = prev_rec['timestamp']
                t1 = next_rec['timestamp']
                allowed_dt = 2.0 * max_time_diff
            elif next_rec is not None:
                p0 = rec['center_global']
                p1 = next_rec['center_global']
                t0 = rec['timestamp']
                t1 = next_rec['timestamp']
                allowed_dt = max_time_diff
            else:
                p0 = prev_rec['center_global']
                p1 = rec['center_global']
                t0 = prev_rec['timestamp']
                t1 = rec['timestamp']
                allowed_dt = max_time_diff

            dt = float(t1 - t0)
            if dt <= min_dt:
                rejected_time += 1
                continue
            if dt > allowed_dt:
                rejected_time += 1
                continue

            vel_global = (p1 - p0) / dt
            speed = float(np.linalg.norm(vel_global[:2]))
            if speed > max_speed:
                rejected_speed += 1
                continue

            key = (rec['frame_idx'], rec['inst_idx'])
            velocities[key] = vel_global
            valid_count += 1

    return velocities, valid_count, rejected_time, rejected_speed, singletons


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

    velocities, valid_count, rejected_time, rejected_speed, singletons = (
        _estimate_track_velocities(tracks, min_dt=min_dt,
                                   max_time_diff=max_time_diff,
                                   max_speed=max_speed))

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
