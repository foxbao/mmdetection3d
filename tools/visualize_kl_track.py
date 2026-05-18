#!/usr/bin/env python
"""Visualize KL online tracking results in BEV.

Example:
    CUDA_VISIBLE_DEVICES=6 python tools/visualize_kl_track.py \
        --config projects/BEVFormer/configs/uniad_lidar_kl_track.py \
        --checkpoint work_dirs/uniad_lidar_kl_track/epoch_4.pth \
        --out-dir work_dirs/vis_track_epoch4_scene0 \
        --start-index 0 --max-frames 24 \
        --score-thr 0.0 --annotate
"""

from __future__ import annotations

import argparse
import math
import os
import os.path as osp
import subprocess
from collections import defaultdict
from typing import Dict, List, Sequence

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmengine
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules

from visualize_kl_velocity import (CLASS_NAMES, compute_box_corners_bev,
                                   lidar_xy_to_display, sample_indices,
                                   to_numpy_boxes, to_numpy_labels,
                                   to_numpy_scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize KL online tracking ids in BEV.')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument(
        '--split',
        default='val',
        choices=['val', 'test'],
        help='which dataloader config to use')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument(
        '--token',
        default=None,
        help='start from this sample token instead of --start-index')
    parser.add_argument(
        '--scene-token',
        default=None,
        help='start from the first sample of this scene token')
    parser.add_argument('--max-frames', type=int, default=24)
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='extra visualization threshold after tracker emission')
    parser.add_argument('--topk', type=int, default=120)
    parser.add_argument('--point-stride', type=int, default=4)
    parser.add_argument('--vel-scale', type=float, default=3.0)
    parser.add_argument('--min-vel-draw', type=float, default=0.2)
    parser.add_argument('--trail-length', type=int, default=12)
    parser.add_argument(
        '--draw-trails',
        action='store_true',
        help='draw historical center trails for each emitted track id')
    parser.add_argument(
        '--view-mode',
        default='overlay',
        choices=['overlay', 'split'],
        help='overlay draws GT and tracks on one BEV; split draws GT on the '
        'left and tracker output on the right.')
    parser.add_argument('--annotate', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument(
        '--webm-fps',
        type=float,
        default=3.0,
        help='FPS for the generated track_vis.webm. Set <= 0 to skip WebM.')
    parser.add_argument(
        '--webm-crf',
        type=int,
        default=34,
        help='VP9 CRF for generated WebM; lower is higher quality.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, e.g. model.score_thresh=0.7')
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise ValueError('--max-frames must be positive.')
    if args.point_stride <= 0:
        raise ValueError('--point-stride must be positive.')
    if args.topk <= 0:
        raise ValueError('--topk must be positive.')
    return args


def import_cfg_modules(cfg: Config) -> None:
    register_all_modules(init_default_scope=True)
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)


def get_dataset_cfg(cfg: Config, split: str):
    return cfg.val_dataloader.dataset if split == 'val' else cfg.test_dataloader.dataset


def resolve_start_index(dataset, args: argparse.Namespace) -> int:
    if args.token is None and args.scene_token is None:
        if args.start_index < 0 or args.start_index >= len(dataset):
            raise IndexError(f'--start-index {args.start_index} is out of range')
        return args.start_index

    for idx in range(len(dataset)):
        info = dataset.get_data_info(idx)
        if args.token is not None and info.get('token') == args.token:
            return idx
        if args.scene_token is not None and (
                info.get('scene_token') == args.scene_token):
            return idx
    key = args.token if args.token is not None else args.scene_token
    raise KeyError(f'Cannot find requested token/scene: {key}')


def consecutive_scene_indices(dataset, start_index: int,
                              max_frames: int) -> List[int]:
    start_info = dataset.get_data_info(start_index)
    scene_token = start_info.get('scene_token')
    indices = []
    for idx in range(start_index, len(dataset)):
        info = dataset.get_data_info(idx)
        if info.get('scene_token') != scene_token:
            break
        indices.append(idx)
        if len(indices) >= max_frames:
            break
    return indices


def reset_track_state(model) -> None:
    for name in ('_test_track_instances', '_test_prev_bev',
                 '_test_scene_token'):
        if hasattr(model, name):
            setattr(model, name, None)
    if hasattr(model, '_debug_prev_centers'):
        model._debug_prev_centers = {}
    if hasattr(model, 'track_base') and hasattr(model.track_base, 'clear'):
        model.track_base.clear()


def ids_to_numpy(instances_3d) -> np.ndarray:
    if len(instances_3d) == 0 or not hasattr(instances_3d, 'instance_id'):
        return np.zeros((0,), dtype=np.int64)
    return instances_3d.instance_id.detach().cpu().numpy().astype(np.int64)


def gt_arrays_from_info(info: dict, use_valid_flag: bool) -> Dict[str, np.ndarray]:
    instances = info.get('instances', [])
    boxes = []
    labels = []
    track_ids = []
    for inst in instances:
        if use_valid_flag:
            keep = bool(inst.get('bbox_3d_isvalid', False))
        else:
            keep = int(inst.get('num_lidar_pts', 0)) > 0
        if not keep:
            continue

        box = np.asarray(inst['bbox_3d'], dtype=np.float32)
        vel = np.asarray(inst.get('velocity', [0.0, 0.0]),
                         dtype=np.float32)
        if box.shape[0] == 7:
            box = np.concatenate([box, vel], axis=0)
        else:
            box = box.copy()
            if box.shape[0] >= 9:
                box[7:9] = vel[:2]
        boxes.append(box[:9])
        labels.append(int(inst['bbox_label_3d']))
        track_ids.append(int(inst.get('track_id', -1)))

    if not boxes:
        return dict(
            boxes=np.zeros((0, 9), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            scores=np.ones((0,), dtype=np.float32),
            track_ids=np.zeros((0,), dtype=np.int64))

    return dict(
        boxes=np.stack(boxes, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.ones((len(boxes), ), dtype=np.float32),
        track_ids=np.asarray(track_ids, dtype=np.int64))


def color_from_track_id(track_id: int) -> tuple:
    if track_id < 0:
        return (1.0, 0.3, 0.3)
    hue = ((track_id * 0.61803398875) % 1.0)
    # HSV to RGB, fixed saturation/value.
    import colorsys
    return colorsys.hsv_to_rgb(hue, 0.82, 1.0)


def draw_gt_boxes(ax,
                  boxes: np.ndarray,
                  labels: np.ndarray,
                  track_ids: np.ndarray,
                  vel_scale: float,
                  min_vel_draw: float,
                  annotate: bool) -> None:
    if boxes.size == 0:
        return
    corners = compute_box_corners_bev(boxes)
    for idx, box in enumerate(boxes):
        poly = corners[idx]
        poly_disp = lidar_xy_to_display(poly)
        closed = np.concatenate([poly_disp, poly_disp[:1]], axis=0)
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color='#9cffbf',
            linewidth=0.8,
            alpha=0.45)
        center_disp = lidar_xy_to_display(
            np.array([[box[0], box[1]]], dtype=np.float32))[0]
        vx = float(box[7])
        vy = float(box[8])
        speed = math.hypot(vx, vy)
        if speed >= min_vel_draw:
            vel_disp = lidar_xy_to_display(
                np.array([[vx, vy]], dtype=np.float32))[0]
            ax.arrow(
                float(center_disp[0]),
                float(center_disp[1]),
                float(vel_disp[0]) * vel_scale,
                float(vel_disp[1]) * vel_scale,
                color='#9cffbf',
                width=0.02,
                head_width=0.38,
                head_length=0.46,
                length_includes_head=True,
                alpha=0.75)
        if annotate:
            label_name = CLASS_NAMES[int(labels[idx])]
            track_id = int(track_ids[idx]) if idx < len(track_ids) else -1
            text = f'gt_id={track_id} {label_name} {speed:.1f}m/s'
            ax.text(
                float(center_disp[0]),
                float(center_disp[1]),
                text,
                color='#9cffbf',
                fontsize=6,
                ha='left',
                va='bottom',
                alpha=0.95)


def draw_track_boxes(ax,
                     boxes: np.ndarray,
                     labels: np.ndarray,
                     scores: np.ndarray,
                     track_ids: np.ndarray,
                     trails: Dict[int, List[np.ndarray]],
                     vel_scale: float,
                     min_vel_draw: float,
                     annotate: bool,
                     draw_trails: bool) -> None:
    if boxes.size == 0:
        return
    corners = compute_box_corners_bev(boxes)
    for idx, box in enumerate(boxes):
        track_id = int(track_ids[idx]) if idx < len(track_ids) else -1
        color = color_from_track_id(track_id)
        poly_disp = lidar_xy_to_display(corners[idx])
        closed = np.concatenate([poly_disp, poly_disp[:1]], axis=0)
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.6)

        center_xy = np.array([[box[0], box[1]]], dtype=np.float32)
        center_disp = lidar_xy_to_display(center_xy)[0]
        if draw_trails and track_id >= 0:
            trail = trails.get(track_id, [])
            if len(trail) >= 2:
                trail_arr = np.asarray(trail, dtype=np.float32)
                trail_disp = lidar_xy_to_display(trail_arr)
                ax.plot(
                    trail_disp[:, 0],
                    trail_disp[:, 1],
                    color=color,
                    linewidth=1.0,
                    alpha=0.55)

        yaw = float(box[6])
        heading_len = min(max(float(box[3]) * 0.22, 0.45), 1.35)
        heading_vec = lidar_xy_to_display(
            np.array([[math.cos(yaw), math.sin(yaw)]],
                     dtype=np.float32))[0]
        ax.plot(
            [center_disp[0], center_disp[0] + heading_vec[0] * heading_len],
            [center_disp[1], center_disp[1] + heading_vec[1] * heading_len],
            color=color,
            linewidth=0.9,
            linestyle='--',
            alpha=0.65)

        vx = float(box[7])
        vy = float(box[8])
        speed = math.hypot(vx, vy)
        if speed >= min_vel_draw:
            vel_disp = lidar_xy_to_display(
                np.array([[vx, vy]], dtype=np.float32))[0]
            ax.arrow(
                float(center_disp[0]),
                float(center_disp[1]),
                float(vel_disp[0]) * vel_scale,
                float(vel_disp[1]) * vel_scale,
                color=color,
                width=0.025,
                head_width=0.45,
                head_length=0.55,
                length_includes_head=True,
                alpha=0.85)

        if annotate:
            label_name = CLASS_NAMES[int(labels[idx])]
            text = (
                f'id={track_id} {label_name} '
                f'{float(scores[idx]):.2f} {speed:.1f}m/s')
            ax.text(
                float(center_disp[0]),
                float(center_disp[1]),
                text,
                color=color,
                fontsize=6,
                ha='left',
                va='bottom')


def setup_bev_axis(ax, pc_range: Sequence[float], subtitle: str) -> None:
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    ax.set_facecolor('black')
    ax.set_xlim(-y_max, -y_min)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Lateral (m)', color='white')
    ax.set_ylabel('Forward (m)', color='white')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#888888')
    ax.grid(color='#444444', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.set_title(subtitle, color='white', fontsize=10)


def draw_points_and_ego(ax, points: np.ndarray, point_stride: int) -> None:
    pts = points[::point_stride, :3]
    pts_disp = lidar_xy_to_display(pts[:, :2])
    ax.scatter(
        pts_disp[:, 0],
        pts_disp[:, 1],
        s=0.12,
        c='white',
        alpha=0.32,
        linewidths=0)

    ax.plot(0.0, 0.0, marker='o', markersize=4, color='#ffd166')
    ax.arrow(
        0.0,
        0.0,
        0.0,
        3.0,
        color='#ffd166',
        width=0.03,
        head_width=0.5,
        head_length=0.6,
        length_includes_head=True)


def render_frame(points: np.ndarray,
                 gt_data: Dict[str, np.ndarray],
                 pred_data: Dict[str, np.ndarray],
                 trails: Dict[int, List[np.ndarray]],
                 save_path: str,
                 title: str,
                 pc_range: Sequence[float],
                 point_stride: int,
                 vel_scale: float,
                 min_vel_draw: float,
                 annotate: bool,
                 view_mode: str,
                 draw_trails: bool) -> None:
    if view_mode == 'split':
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 8.8), dpi=160)
        fig.patch.set_facecolor('black')
        left_ax, right_ax = axes

        setup_bev_axis(left_ax, pc_range,
                       f'GT boxes ({len(gt_data["boxes"])})')
        setup_bev_axis(
            right_ax, pc_range,
            f'Tracker output ({len(pred_data["boxes"])}) | arrow=velocity')
        draw_points_and_ego(left_ax, points, point_stride)
        draw_points_and_ego(right_ax, points, point_stride)
        draw_gt_boxes(
            left_ax,
            gt_data['boxes'],
            gt_data['labels'],
            gt_data['track_ids'],
            vel_scale=vel_scale,
            min_vel_draw=min_vel_draw,
            annotate=annotate)
        draw_track_boxes(
            right_ax,
            pred_data['boxes'],
            pred_data['labels'],
            pred_data['scores'],
            pred_data['track_ids'],
            trails,
            vel_scale=vel_scale,
            min_vel_draw=min_vel_draw,
            annotate=annotate,
            draw_trails=draw_trails)
        fig.suptitle(title, color='white', fontsize=11)
        fig.subplots_adjust(
            left=0.055, right=0.985, bottom=0.07, top=0.91, wspace=0.06)
        fig.savefig(save_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.2, 9.2), dpi=160)
    fig.patch.set_facecolor('black')
    setup_bev_axis(ax, pc_range, title)
    draw_points_and_ego(ax, points, point_stride)

    draw_gt_boxes(
        ax,
        gt_data['boxes'],
        gt_data['labels'],
        gt_data['track_ids'],
        vel_scale=vel_scale,
        min_vel_draw=min_vel_draw,
        annotate=False)
    draw_track_boxes(
        ax,
        pred_data['boxes'],
        pred_data['labels'],
        pred_data['scores'],
        pred_data['track_ids'],
        trails,
        vel_scale=vel_scale,
        min_vel_draw=min_vel_draw,
        annotate=annotate,
        draw_trails=draw_trails)
    legend_lines = [
        plt.Line2D([0], [0], color='#9cffbf', lw=2),
        plt.Line2D([0], [0], color='#ffffff', lw=2),
        plt.Line2D([0], [0], color='#ffd166', lw=2),
    ]
    ax.legend(
        legend_lines,
        ['GT boxes', 'Track boxes colored by id', 'Ego heading'],
        loc='upper right',
        facecolor='black',
        edgecolor='#888888',
        labelcolor='white',
        fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.91)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def update_trails(trails: Dict[int, List[np.ndarray]],
                  boxes: np.ndarray,
                  track_ids: np.ndarray,
                  trail_length: int) -> None:
    active_ids = set()
    for box, track_id in zip(boxes, track_ids):
        track_id = int(track_id)
        if track_id < 0:
            continue
        active_ids.add(track_id)
        trails[track_id].append(np.asarray(box[:2], dtype=np.float32))
        if len(trails[track_id]) > trail_length:
            trails[track_id] = trails[track_id][-trail_length:]
    # Keep short history for recently missing tracks, but avoid unbounded growth.
    for track_id in list(trails.keys()):
        if track_id not in active_ids and len(trails[track_id]) == 0:
            del trails[track_id]


def write_html_player(out_dir: str, frame_files: List[str]) -> None:
    frames = [osp.basename(path) for path in frame_files]
    if not frames:
        return
    frame_items = ',\n      '.join(f'"{name}"' for name in frames)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KL Track Sequence</title>
  <style>
    body {{
      margin: 0;
      background: #101114;
      color: #f2f5fa;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 18px 0 28px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 12px;
    }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    #frameText {{ color: #9aa4b2; font-size: 13px; white-space: nowrap; }}
    .stage {{
      display: grid;
      place-items: center;
      min-height: 360px;
      background: #050608;
      border: 1px solid #343844;
      border-radius: 6px;
      overflow: hidden;
    }}
    #frameImage {{
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 210px);
      width: auto;
      height: auto;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
      padding: 12px;
      background: #181a20;
      border: 1px solid #343844;
      border-radius: 6px;
    }}
    .row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    button {{
      height: 34px;
      min-width: 42px;
      padding: 0 12px;
      border: 1px solid #343844;
      border-radius: 5px;
      background: #222631;
      color: #f2f5fa;
      font: inherit;
      cursor: pointer;
    }}
    button.primary {{ background: #113548; border-color: #26637d; }}
    input[type="range"] {{
      flex: 1 1 300px;
      min-width: 160px;
      accent-color: #64d2ff;
    }}
    input[type="number"] {{
      width: 72px;
      height: 32px;
      border: 1px solid #343844;
      border-radius: 5px;
      background: #11141a;
      color: #f2f5fa;
      padding: 0 8px;
      font: inherit;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #9aa4b2;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>KL Track Sequence</h1>
      <div id="frameText"></div>
    </header>
    <section class="stage">
      <img id="frameImage" alt="tracking frame">
    </section>
    <section class="controls">
      <div class="row">
        <button id="prevBtn">Prev</button>
        <button id="playBtn" class="primary">Play</button>
        <button id="nextBtn">Next</button>
        <label>FPS <input id="fpsInput" type="number" value="3" min="0.5" max="30" step="0.5"></label>
        <label><input id="loopInput" type="checkbox" checked> Loop</label>
      </div>
      <div class="row">
        <input id="frameSlider" type="range" min="0" max="{len(frames) - 1}" value="0" step="1">
      </div>
    </section>
  </main>
  <script>
    const frames = [
      {frame_items}
    ];
    const image = document.getElementById("frameImage");
    const text = document.getElementById("frameText");
    const slider = document.getElementById("frameSlider");
    const playBtn = document.getElementById("playBtn");
    const fpsInput = document.getElementById("fpsInput");
    const loopInput = document.getElementById("loopInput");
    let index = 0;
    let timer = null;
    function setFrame(nextIndex) {{
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      image.src = frames[index];
      slider.value = index;
      text.textContent = `Frame ${{index + 1}} / ${{frames.length}} | ${{frames[index]}}`;
    }}
    function stop() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
      }}
      playBtn.textContent = "Play";
    }}
    function play() {{
      stop();
      playBtn.textContent = "Pause";
      const fps = Math.max(0.5, Number(fpsInput.value) || 3);
      timer = setInterval(() => {{
        if (index >= frames.length - 1) {{
          if (!loopInput.checked) {{
            stop();
            return;
          }}
          setFrame(0);
          return;
        }}
        setFrame(index + 1);
      }}, 1000 / fps);
    }}
    function togglePlay() {{ timer === null ? play() : stop(); }}
    document.getElementById("prevBtn").addEventListener("click", () => {{ stop(); setFrame(index - 1); }});
    document.getElementById("nextBtn").addEventListener("click", () => {{ stop(); setFrame(index + 1); }});
    playBtn.addEventListener("click", togglePlay);
    fpsInput.addEventListener("change", () => {{ if (timer !== null) play(); }});
    slider.addEventListener("input", () => {{ stop(); setFrame(Number(slider.value)); }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === " ") {{ event.preventDefault(); togglePlay(); }}
      else if (event.key === "ArrowLeft") {{ stop(); setFrame(index - 1); }}
      else if (event.key === "ArrowRight") {{ stop(); setFrame(index + 1); }}
    }});
    setFrame(0);
  </script>
</body>
</html>
"""
    with open(osp.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


def write_webm_video(out_dir: str,
                     fps: float = 3.0,
                     crf: int = 34) -> None:
    if fps <= 0:
        return
    output_path = osp.join(out_dir, 'track_vis.webm')
    input_glob = osp.join(out_dir, '*.png')
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate',
        str(float(fps)),
        '-pattern_type',
        'glob',
        '-i',
        input_glob,
        '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v',
        'libvpx-vp9',
        '-b:v',
        '0',
        '-crf',
        str(int(crf)),
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('[WARN] ffmpeg not found; skipped WebM generation.')
    except subprocess.CalledProcessError as exc:
        print(f'[WARN] ffmpeg failed with exit code {exc.returncode}; '
              'skipped WebM generation.')


def run_online(cfg: Config, args: argparse.Namespace) -> None:
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset = DATASETS.build(dataset_cfg)
    start_index = resolve_start_index(dataset, args)
    indices = consecutive_scene_indices(dataset, start_index, args.max_frames)
    if not indices:
        raise RuntimeError('No frames selected for visualization.')

    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()
    reset_track_state(model)

    os.makedirs(args.out_dir, exist_ok=True)
    trails = defaultdict(list)
    summary = []
    frame_files = []
    for frame_id, idx in enumerate(indices):
        info = dataset.get_data_info(idx)
        data = dataset[idx]
        token = info['token']
        save_path = osp.join(args.out_dir, f'{frame_id:03d}_{idx:06d}_{token}.png')
        if args.skip_existing and osp.exists(save_path):
            print(f'[SKIP] {save_path}')
            continue

        points = data['inputs']['points']
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        with torch.no_grad():
            pred_sample = model.test_step(pseudo_collate([data]))[0]

        pred = getattr(pred_sample, 'pred_track_instances_3d',
                       pred_sample.pred_instances_3d)
        scores = pred.scores_3d
        order = torch.argsort(scores, descending=True)
        keep = order[scores[order] >= args.score_thr][:args.topk]
        pred = pred[keep]

        boxes = to_numpy_boxes(pred)
        labels = to_numpy_labels(pred)
        scores_np = to_numpy_scores(pred)
        track_ids = ids_to_numpy(pred)
        update_trails(trails, boxes, track_ids, args.trail_length)

        gt_data = gt_arrays_from_info(
            info, bool(dataset_cfg.get('use_valid_flag', False)))
        pred_data = dict(
            boxes=boxes,
            labels=labels,
            scores=scores_np,
            track_ids=track_ids)
        scene = str(info.get('scene_token', ''))
        title = (
            f'KL online track | frame={frame_id} index={idx} '
            f'token={token[:8]} scene={scene[-8:]}\n'
            f'GT={len(gt_data["boxes"])} Track={len(boxes)} '
            f'Unique IDs={len(set(track_ids.tolist())) if len(track_ids) else 0}')
        render_frame(
            points=points,
            gt_data=gt_data,
            pred_data=pred_data,
            trails=trails,
            save_path=save_path,
            title=title,
            pc_range=cfg.point_cloud_range,
            point_stride=args.point_stride,
            vel_scale=args.vel_scale,
            min_vel_draw=args.min_vel_draw,
            annotate=args.annotate,
            view_mode=args.view_mode,
            draw_trails=args.draw_trails)
        frame_files.append(save_path)
        summary.append(dict(
            frame=int(frame_id),
            index=int(idx),
            token=token,
            scene_token=info.get('scene_token'),
            out_file=save_path,
            num_gt=int(len(gt_data['boxes'])),
            num_track=int(len(boxes)),
            track_ids=[int(x) for x in track_ids.tolist()]))
        print(f'[OK] {save_path}')

    mmengine.dump(summary, osp.join(args.out_dir, 'summary.json'))
    write_html_player(args.out_dir, frame_files)
    write_webm_video(args.out_dir, fps=args.webm_fps, crf=args.webm_crf)


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_cfg_modules(cfg)
    run_online(cfg, args)


if __name__ == '__main__':
    main()
