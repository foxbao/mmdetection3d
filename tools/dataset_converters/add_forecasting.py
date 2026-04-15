#!/usr/bin/env python
"""Add GT forecasting trajectories to KL dataset v2 pkl files.

For each object with a track_id, follows the `next` frame chain and finds
the same track_id in future frames.  Future positions are transformed into
the current frame's ego coordinate system and stored as per-instance fields:

  - gt_forecasting_locs : list[list[float]]  (forecast_steps, 2)  — [dx, dy]
        relative displacement from the object's current (x, y) position,
        expressed in the current ego frame.
  - gt_forecasting_mask : list[bool]  (forecast_steps,)
        True where the track_id was found in the future frame.

Usage:
    python tools/dataset_converters/add_forecasting.py \
        --pkl-path data/kl_8/kl_infos_train.pkl \
        --forecast-steps 6

    # Multiple files:
    python tools/dataset_converters/add_forecasting.py \
        --pkl-path data/kl_8/kl_infos_train.pkl data/kl_8/kl_infos_val.pkl
"""

import argparse
import numpy as np
import mmengine
from pathlib import Path


MAX_DISPLACEMENT = 100.0  # metres; reject per-step displacements beyond this

# KL dataset: LiDAR frame is RFU (col-0=Right, col-1=Forward, col-2=Up);
# ego frame (used by ego2global) is FLU (X=Forward, Y=Left, Z=Up).
# GT boxes (bbox_3d) are stored in the LiDAR RFU frame.
# lidar2ego = 90° rotation about Z:
LIDAR2EGO = np.eye(4, dtype=np.float64)
LIDAR2EGO[:2, :2] = np.array([[0.0, 1.0],
                              [-1.0, 0.0]])


def build_indices(infos):
    """Build lookup dicts from a data_list.

    Returns:
        token_to_idx: {token_str: index_in_infos}
        token_tid_to_box: {(token_str, track_id): [x, y, z, ...]}
    """
    token_to_idx = {}
    token_tid_to_box = {}

    for idx, info in enumerate(infos):
        tok = info['token']
        token_to_idx[tok] = idx
        for inst in info.get('instances', []):
            tid = inst.get('track_id', -1)
            if tid < 0:
                continue
            token_tid_to_box[(tok, tid)] = inst['bbox_3d']

    return token_to_idx, token_tid_to_box


def compute_forecasting(infos, token_to_idx, token_tid_to_box,
                         forecast_steps=6):
    """Add gt_forecasting_locs and gt_forecasting_mask to each instance."""
    num_found = 0
    num_total = 0
    num_rejected_scene = 0
    num_rejected_range = 0

    for info in mmengine.track_iter_progress(infos):
        ego2global_curr = np.array(info['ego2global'], dtype=np.float64)
        lidar2global_curr = ego2global_curr @ LIDAR2EGO
        global2lidar_curr = np.linalg.inv(lidar2global_curr)
        curr_scene = info.get('scene_token', '')

        for inst in info.get('instances', []):
            tid = inst.get('track_id', -1)
            curr_xy = inst['bbox_3d'][:2]  # (x, y) in current LiDAR RFU frame

            locs = []
            mask = []

            future_token = info.get('next', '')
            for _ in range(forecast_steps):
                if not future_token or future_token not in token_to_idx:
                    # Scene boundary or chain ended
                    locs.append([0.0, 0.0])
                    mask.append(False)
                    future_token = ''
                    continue

                future_idx = token_to_idx[future_token]
                future_info = infos[future_idx]

                # Scene guard: track_id is only unique within a scene,
                # so break the chain as soon as scene_token changes.
                if curr_scene and future_info.get('scene_token', '') != curr_scene:
                    locs.append([0.0, 0.0])
                    mask.append(False)
                    future_token = ''
                    num_rejected_scene += 1
                    continue

                key = (future_token, tid)

                if tid >= 0 and key in token_tid_to_box:
                    # Future position in future LiDAR RFU frame
                    fut_box = token_tid_to_box[key]
                    pos_fut_lidar = np.array(
                        [fut_box[0], fut_box[1], fut_box[2], 1.0],
                        dtype=np.float64)

                    # Future LiDAR → global → current LiDAR
                    ego2global_fut = np.array(
                        future_info['ego2global'], dtype=np.float64)
                    lidar2global_fut = ego2global_fut @ LIDAR2EGO
                    pos_global = lidar2global_fut @ pos_fut_lidar
                    pos_curr = global2lidar_curr @ pos_global

                    # Store as displacement from current position (LiDAR frame)
                    dx = float(pos_curr[0] - curr_xy[0])
                    dy = float(pos_curr[1] - curr_xy[1])
                    if abs(dx) > MAX_DISPLACEMENT or abs(dy) > MAX_DISPLACEMENT:
                        # Implausible jump — likely track_id reuse we missed.
                        locs.append([0.0, 0.0])
                        mask.append(False)
                        num_rejected_range += 1
                    else:
                        locs.append([dx, dy])
                        mask.append(True)
                        num_found += 1
                else:
                    # Track lost in this future frame
                    locs.append([0.0, 0.0])
                    mask.append(False)

                num_total += 1
                future_token = future_info.get('next', '')

            inst['gt_forecasting_locs'] = locs
            inst['gt_forecasting_mask'] = mask

    return num_found, num_total, num_rejected_scene, num_rejected_range


def add_forecasting_to_pkl(pkl_path, forecast_steps=6):
    """Process a single pkl file."""
    pkl_path = str(pkl_path)
    print(f'\n{"="*60}')
    print(f'Processing: {pkl_path}')
    print(f'Forecast steps: {forecast_steps}')

    data = mmengine.load(pkl_path)
    infos = data['data_list']
    print(f'Total frames: {len(infos)}')

    # Build indices
    token_to_idx, token_tid_to_box = build_indices(infos)
    print(f'Unique (token, track_id) pairs: {len(token_tid_to_box)}')

    # Compute forecasting
    num_found, num_total, rej_scene, rej_range = compute_forecasting(
        infos, token_to_idx, token_tid_to_box, forecast_steps)

    hit_rate = num_found / max(num_total, 1) * 100
    print(f'Future track matches: {num_found}/{num_total} ({hit_rate:.1f}%)')
    print(f'Rejected by scene guard: {rej_scene}')
    print(f'Rejected by range guard (>{int(MAX_DISPLACEMENT)}m): {rej_range}')

    # Save back
    mmengine.dump(data, pkl_path)
    print(f'Saved to: {pkl_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Add GT forecasting trajectories to KL pkl files.')
    parser.add_argument(
        '--pkl-path', nargs='+', required=True,
        help='Path(s) to v2 pkl files.')
    parser.add_argument(
        '--forecast-steps', type=int, default=6,
        help='Number of future frames to predict (default: 6).')
    args = parser.parse_args()

    for p in args.pkl_path:
        add_forecasting_to_pkl(p, args.forecast_steps)

    print('\nDone.')


if __name__ == '__main__':
    main()
