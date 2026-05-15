#!/usr/bin/env python
"""Check KL ego-motion delta and velocity-frame conventions.

This script compares several constant-velocity propagation hypotheses against
GT boxes of the same ``track_id`` in consecutive frames.  The best hypothesis
should match how tracking code updates reference points between frames.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mmengine
import numpy as np


def lidar2ego_from_frame(frame: str) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if frame == 'RFU':
        mat[:2, :2] = np.array([[0.0, 1.0], [-1.0, 0.0]],
                               dtype=np.float64)
    elif frame != 'FLU':
        raise ValueError(f'unknown lidar_coord_frame: {frame!r}')
    return mat


def transform_points(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return xyz @ mat[:3, :3].T + mat[:3, 3]


def xy_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred[:2] - target[:2]))


def safe_track_id(inst: dict) -> int:
    track_id = inst.get('track_id', -1)
    if track_id is None:
        return -1
    return int(track_id)


def load_infos(path: str) -> Tuple[List[dict], dict]:
    data = mmengine.load(path)
    if isinstance(data, dict):
        return data['data_list'], data.get('metainfo', {})
    return data, {}


def iter_pairs(infos: List[dict], max_time_gap: float):
    token_to_info = {
        info.get('token'): info
        for info in infos
        if info.get('token') is not None
    }
    for curr in infos:
        prev_token = curr.get('prev', '')
        if not prev_token:
            continue
        prev = token_to_info.get(prev_token)
        if prev is None:
            continue
        if curr.get('scene_token') != prev.get('scene_token'):
            continue

        dt = float(curr.get('timestamp', 0.0)) - float(
            prev.get('timestamp', 0.0))
        if dt <= 0 or dt > max_time_gap:
            continue

        prev_by_track: Dict[int, dict] = {}
        for inst in prev.get('instances', []):
            track_id = safe_track_id(inst)
            if track_id >= 0:
                prev_by_track[track_id] = inst

        for curr_inst in curr.get('instances', []):
            track_id = safe_track_id(curr_inst)
            if track_id < 0 or track_id not in prev_by_track:
                continue
            prev_inst = prev_by_track[track_id]
            yield prev, curr, prev_inst, curr_inst, dt


def summarize(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p90=np.nan,
                    p95=np.nan, max=np.nan)
    return dict(
        n=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        max=float(arr.max()))


def print_table(rows: List[Tuple[str, dict]]) -> None:
    headers = ('candidate', 'n', 'mean', 'median', 'p90', 'p95', 'max')
    widths = [34, 8, 10, 10, 10, 10, 10]
    fmt = ''.join(f'{{:<{w}}}' for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*['-' * (w - 1) for w in widths]))
    for name, stat in rows:
        print(fmt.format(
            name,
            stat['n'],
            f'{stat["mean"]:.4f}',
            f'{stat["median"]:.4f}',
            f'{stat["p90"]:.4f}',
            f'{stat["p95"]:.4f}',
            f'{stat["max"]:.4f}',
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_file', help='KL info pkl, e.g. kl_infos_val.pkl')
    parser.add_argument('--max-time-gap', type=float, default=1.5)
    parser.add_argument('--max-pairs', type=int, default=0,
                        help='0 means use all matched consecutive GT pairs')
    parser.add_argument('--examples', type=int, default=5)
    args = parser.parse_args()

    ann_path = Path(args.ann_file)
    infos, metainfo = load_infos(str(ann_path))
    frame = metainfo.get('lidar_coord_frame', 'FLU')
    lidar2ego = lidar2ego_from_frame(frame)
    ego2lidar = np.linalg.inv(lidar2ego)

    errors = defaultdict(list)
    examples = []
    num_pairs = 0
    num_label_mismatch = 0
    dt_values = []

    for prev, curr, prev_inst, curr_inst, dt in iter_pairs(
            infos, args.max_time_gap):
        if args.max_pairs and num_pairs >= args.max_pairs:
            break

        prev_center = np.asarray(prev_inst['bbox_3d'][:3], dtype=np.float64)
        curr_center = np.asarray(curr_inst['bbox_3d'][:3], dtype=np.float64)
        prev_vel = np.asarray(prev_inst.get('velocity', [0.0, 0.0]),
                              dtype=np.float64)
        curr_vel = np.asarray(curr_inst.get('velocity', [0.0, 0.0]),
                              dtype=np.float64)

        prev_ego2global = np.asarray(prev['ego2global'], dtype=np.float64)
        curr_ego2global = np.asarray(curr['ego2global'], dtype=np.float64)
        prev_to_curr_ego = np.linalg.inv(curr_ego2global) @ prev_ego2global
        prev_to_curr_lidar = ego2lidar @ prev_to_curr_ego @ lidar2ego
        curr_to_prev_lidar = np.linalg.inv(prev_to_curr_lidar)

        prev_vel_xyz = np.array([prev_vel[0], prev_vel[1], 0.0],
                                dtype=np.float64)
        curr_vel_xyz = np.array([curr_vel[0], curr_vel[1], 0.0],
                                dtype=np.float64)

        candidates = {
            'ego_only_prev_to_curr':
            transform_points(prev_to_curr_lidar, prev_center),
            'prev_vel_then_ego':
            transform_points(prev_to_curr_lidar,
                             prev_center + prev_vel_xyz * dt),
            'ego_then_prev_vel':
            transform_points(prev_to_curr_lidar, prev_center) +
            prev_vel_xyz * dt,
            'ego_then_curr_vel':
            transform_points(prev_to_curr_lidar, prev_center) +
            curr_vel_xyz * dt,
            'prev_vel_no_ego':
            prev_center + prev_vel_xyz * dt,
            'minus_prev_vel_then_ego':
            transform_points(prev_to_curr_lidar,
                             prev_center - prev_vel_xyz * dt),
            'ego_only_inverse_delta':
            transform_points(curr_to_prev_lidar, prev_center),
        }

        pair_errors = {
            name: xy_error(pred, curr_center)
            for name, pred in candidates.items()
        }
        for name, err in pair_errors.items():
            errors[name].append(err)

        if int(prev_inst.get('bbox_label_3d', -1)) != int(
                curr_inst.get('bbox_label_3d', -2)):
            num_label_mismatch += 1

        dt_values.append(dt)
        if len(examples) < args.examples:
            best = min(pair_errors.items(), key=lambda x: x[1])
            examples.append(dict(
                scene=curr.get('scene_token'),
                prev_token=prev.get('token'),
                curr_token=curr.get('token'),
                track_id=safe_track_id(curr_inst),
                dt=dt,
                prev_center=prev_center,
                curr_center=curr_center,
                prev_vel=prev_vel,
                best=best,
                errors=pair_errors,
            ))
        num_pairs += 1

    rows = sorted(
        ((name, summarize(vals)) for name, vals in errors.items()),
        key=lambda x: x[1]['mean'])

    print(f'file: {ann_path}')
    print(f'frames: {len(infos)}')
    print(f'lidar_coord_frame: {frame}')
    print(f'matched consecutive track pairs: {num_pairs}')
    print(f'label mismatches: {num_label_mismatch}')
    if dt_values:
        dt_arr = np.asarray(dt_values, dtype=np.float64)
        print('dt seconds: '
              f'mean={dt_arr.mean():.4f} '
              f'median={np.median(dt_arr):.4f} '
              f'min={dt_arr.min():.4f} max={dt_arr.max():.4f}')
    print()
    print_table(rows)

    if examples:
        print('\nexamples:')
        for ex in examples:
            print(
                f"track={ex['track_id']} dt={ex['dt']:.3f} "
                f"best={ex['best'][0]} err={ex['best'][1]:.4f}")
            print(f"  prev_center={np.round(ex['prev_center'], 3).tolist()} "
                  f"prev_vel={np.round(ex['prev_vel'], 3).tolist()}")
            print(f"  curr_center={np.round(ex['curr_center'], 3).tolist()}")


if __name__ == '__main__':
    main()
