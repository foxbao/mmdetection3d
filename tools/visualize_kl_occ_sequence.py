"""Export a lightweight animated KL occupancy sequence page.

The page stores occupied voxel indices for each frame and renders them as
instanced Three.js cubes in the browser. Free voxels and ignore voxels are
hidden by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


LABEL_COLORS = {
    1: '#ef4444',
    2: '#2563eb',
    3: '#7c3aed',
    4: '#f97316',
    5: '#ec4899',
    6: '#dc2626',
    7: '#38bdf8',
    8: '#a855f7',
    9: '#64748b',
    10: '#facc15',
    11: '#06b6d4',
    12: '#14b8a6',
    13: '#fb7185',
    14: '#c084fc',
    15: '#8b5cf6',
    16: '#2fb344',
    17: '#6b7280',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize consecutive KL occupancy frames.')
    parser.add_argument(
        '--config',
        default=(
            'projects/BEVFormer/configs/'
            'bevformer_lidar_kl_temporal_transfusion_occ_raycast.py'),
        help='Occupancy config to build the dataset from.')
    parser.add_argument(
        '--compare-config',
        default=None,
        help='Build a second dataset from this config and render side-by-side '
        'GT OCC comparison. The left side uses --config.')
    parser.add_argument(
        '--compare-left-title',
        default='Single-frame GT',
        help='Left panel title for --compare-config.')
    parser.add_argument(
        '--compare-right-title',
        default='Multi-frame GT',
        help='Right panel title for --compare-config.')
    parser.add_argument(
        '--out',
        default=(
            'work_dirs/occ_debug/'
            'kl_occ_3d_sequence_raycast_15frames.html'),
        help='Output HTML path.')
    parser.add_argument(
        '--num-frames', type=int, default=15, help='Number of frames.')
    parser.add_argument(
        '--start-index',
        type=int,
        default=None,
        help='Raw dataset index to start from. If omitted, find one.')
    parser.add_argument(
        '--cube-scale',
        type=float,
        default=0.84,
        help='Cube size relative to voxel size; <1 leaves visible gaps.')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='If set, also run model prediction and visualize pred OCC.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Inference device used with --checkpoint.')
    parser.add_argument(
        '--pred-only',
        action='store_true',
        help='Only export predicted OCC frames when --checkpoint is set.')
    parser.add_argument(
        '--mask-pred-to-observed',
        action='store_true',
        help='Set pred voxels with GT ignore label 255 back to 255 before '
        'visualization. This is useful for supervised-region comparison.')
    parser.add_argument(
        '--max-voxels-per-label',
        type=int,
        default=60000,
        help='Cap rendered voxels per label per frame. Stats remain full.')
    parser.add_argument(
        '--gt-points-boxes',
        action='store_true',
        help='Render GT OCC on the left and current points + GT 3D boxes on '
        'the right. This mode does not need --checkpoint.')
    parser.add_argument(
        '--gt-pred-points-boxes',
        action='store_true',
        help='Render GT OCC, predicted OCC, and current points + GT 3D boxes '
        'in a synchronized three-panel view. Requires --checkpoint.')
    parser.add_argument(
        '--pred-points-boxes',
        action='store_true',
        help='Render predicted OCC on the left and current points + GT 3D '
        'boxes on the right. Requires --checkpoint.')
    parser.add_argument(
        '--max-points-per-frame',
        type=int,
        default=50000,
        help='Cap rendered points per frame for --gt-points-boxes.')
    return parser.parse_args()


def get_occ_array(sample: dict) -> np.ndarray:
    occ = sample['data_samples'].gt_pts_seg.occ
    if hasattr(occ, 'detach'):
        occ = occ.detach().cpu().numpy()
    return np.asarray(occ)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'detach'):
        return json_safe(value.detach().cpu().numpy())
    return value


def get_occ_meta(sample: dict) -> dict:
    meta = sample['data_samples'].metainfo.get('gt_occ_meta', {})
    return json_safe(meta)


def dataset_raw_len(dataset) -> int:
    if hasattr(dataset, 'raw_data_len') and dataset.raw_data_len:
        return int(dataset.raw_data_len)
    return len(dataset)


def collect_sequence(dataset, start_index: int, num_frames: int) -> List[int]:
    if hasattr(dataset, 'full_init'):
        dataset.full_init()
    token2index = getattr(dataset, 'token2index', {})
    raw_indices = []
    cur_index = start_index
    scene_token = None
    for _ in range(num_frames):
        info = dataset._get_raw_data_info(cur_index)
        if scene_token is None:
            scene_token = info.get('scene_token')
        if info.get('scene_token') != scene_token:
            raise RuntimeError('Sequence crosses scene boundary.')
        if (hasattr(dataset, '_collect_queue_indices') and
                dataset._collect_queue_indices(cur_index) is None):
            raise RuntimeError(
                f'Index {cur_index} does not have a valid temporal queue.')
        raw_indices.append(cur_index)
        if len(raw_indices) == num_frames:
            break
        next_token = info.get('next', '')
        if not next_token or next_token not in token2index:
            raise RuntimeError(
                f'Index {cur_index} has no next frame in this split.')
        cur_index = token2index[next_token]
    return raw_indices


def find_sequence(dataset, num_frames: int) -> List[int]:
    if hasattr(dataset, 'full_init'):
        dataset.full_init()
    for idx in range(dataset_raw_len(dataset)):
        try:
            return collect_sequence(dataset, idx, num_frames)
        except RuntimeError:
            continue
    raise RuntimeError(f'Cannot find {num_frames} consecutive valid frames.')


def occ_to_frame(occ: np.ndarray,
                 info: dict,
                 raw_index: int,
                 source: str,
                 max_voxels_per_label: int = 0,
                 meta: Optional[dict] = None) -> dict:
    labels = {}
    unique, counts = np.unique(occ, return_counts=True)
    stats = {
        str(int(label)): int(count)
        for label, count in zip(unique, counts)
    }
    for label in unique:
        label_int = int(label)
        if label_int in (0, 255):
            continue
        flat = np.flatnonzero(occ.reshape(-1) == label_int)
        if flat.size:
            if max_voxels_per_label > 0 and flat.size > max_voxels_per_label:
                keep = np.linspace(
                    0, flat.size - 1, max_voxels_per_label, dtype=np.int64)
                flat = flat[keep]
            labels[str(label_int)] = flat.astype(np.int32).tolist()

    return dict(
        raw_index=int(raw_index),
        source=source,
        token=info.get('token', ''),
        timestamp=float(info.get('timestamp', 0.0)),
        scene_token=info.get('scene_token', ''),
        stats=stats,
        meta=json_safe(meta or {}),
        labels=labels,
        title=f'{source.upper()} raw {raw_index}',
    )


def sample_to_frame(dataset, raw_index: int, max_voxels_per_label: int) -> dict:
    sample = dataset.prepare_data(raw_index)
    if sample is None:
        raise RuntimeError(f'prepare_data({raw_index}) returned None.')
    info = dataset._get_raw_data_info(raw_index)
    occ = get_occ_array(sample)
    return occ_to_frame(
        occ, info, raw_index, 'gt', max_voxels_per_label,
        meta=get_occ_meta(sample))


def sample_to_pred_frames(dataset, model, raw_index: int,
                          include_gt: bool,
                          mask_pred_to_observed: bool,
                          max_voxels_per_label: int) -> List[dict]:
    sample = dataset.prepare_data(raw_index)
    if sample is None:
        raise RuntimeError(f'prepare_data({raw_index}) returned None.')
    info = dataset._get_raw_data_info(raw_index)
    frames = []
    gt_occ = get_occ_array(sample)
    if include_gt:
        frames.append(
            occ_to_frame(
                gt_occ, info, raw_index, 'gt',
                max_voxels_per_label, meta=get_occ_meta(sample)))

    with torch.no_grad():
        pred_sample = model.test_step(pseudo_collate([sample]))[0]
    pred_occ = pred_sample.pred_pts_seg.occ
    if hasattr(pred_occ, 'detach'):
        pred_occ = pred_occ.detach().cpu().numpy()
    pred_occ = np.asarray(pred_occ)
    source = 'pred'
    if mask_pred_to_observed:
        pred_occ = pred_occ.copy()
        pred_occ[gt_occ == 255] = 255
        source = 'pred_observed'
    frames.append(
        occ_to_frame(
            pred_occ, info, raw_index, source,
            max_voxels_per_label))
    return frames


def get_points_array(sample: dict) -> np.ndarray:
    points = sample['inputs']['points']
    if hasattr(points, 'tensor'):
        points = points.tensor
    if hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()
    return np.asarray(points)[:, :3].astype(np.float32, copy=False)


def box_corners(box: np.ndarray) -> List[List[float]]:
    x, y, z, length, width, height, yaw = box[:7]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_xy = np.array([
        [-0.5 * length, -0.5 * width],
        [-0.5 * length, 0.5 * width],
        [0.5 * length, 0.5 * width],
        [0.5 * length, -0.5 * width],
    ], dtype=np.float32)
    corners = []
    for z_val in (z, z + height):
        for lx, ly in local_xy:
            wx = x + lx * cos_yaw - ly * sin_yaw
            wy = y + lx * sin_yaw + ly * cos_yaw
            corners.append([float(wx), float(wy), float(z_val)])
    return corners


def sample_to_points_boxes_frame(sample: dict,
                                 info: dict,
                                 raw_index: int,
                                 class_names: List[str],
                                 max_points_per_frame: int) -> dict:
    points = get_points_array(sample)
    total_points = int(points.shape[0])
    if max_points_per_frame > 0 and total_points > max_points_per_frame:
        keep = np.linspace(
            0, total_points - 1, max_points_per_frame, dtype=np.int64)
        points = points[keep]
    points = np.round(points.astype(np.float32), 3).reshape(-1).tolist()

    instances = sample['data_samples'].gt_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
    labels = instances.labels_3d.detach().cpu().numpy()
    box_items = []
    for box, label in zip(boxes, labels):
        label_int = int(label)
        occ_label = label_int + 1
        box_items.append(dict(
            label=occ_label,
            name=(class_names[label_int]
                  if 0 <= label_int < len(class_names) else str(label_int)),
            corners=box_corners(box)))

    return dict(
        raw_index=int(raw_index),
        token=info.get('token', ''),
        timestamp=float(info.get('timestamp', 0.0)),
        scene_token=info.get('scene_token', ''),
        points=points,
        numPoints=int(len(points) // 3),
        totalPoints=total_points,
        boxes=box_items,
        numBoxes=len(box_items),
        title=f'POINTS+BOXES raw {raw_index}',
    )


def sample_to_gt_points_box_pair(dataset, raw_index: int,
                                 class_names: List[str],
                                 max_voxels_per_label: int,
                                 max_points_per_frame: int) -> dict:
    sample = dataset.prepare_data(raw_index)
    if sample is None:
        raise RuntimeError(f'prepare_data({raw_index}) returned None.')
    info = dataset._get_raw_data_info(raw_index)
    gt = occ_to_frame(
        get_occ_array(sample), info, raw_index, 'gt',
        max_voxels_per_label, meta=get_occ_meta(sample))
    points_boxes = sample_to_points_boxes_frame(
        sample, info, raw_index, class_names, max_points_per_frame)
    return dict(raw_index=int(raw_index), gt=gt, pointsBoxes=points_boxes)


def sample_to_gt_pred_points_box_triple(dataset,
                                        model,
                                        raw_index: int,
                                        class_names: List[str],
                                        mask_pred_to_observed: bool,
                                        max_voxels_per_label: int,
                                        max_points_per_frame: int) -> dict:
    sample = dataset.prepare_data(raw_index)
    if sample is None:
        raise RuntimeError(f'prepare_data({raw_index}) returned None.')
    info = dataset._get_raw_data_info(raw_index)
    gt_occ = get_occ_array(sample)
    gt = occ_to_frame(
        gt_occ,
        info,
        raw_index,
        'gt',
        max_voxels_per_label,
        meta=get_occ_meta(sample))

    with torch.no_grad():
        pred_sample = model.test_step(pseudo_collate([sample]))[0]
    pred_occ = pred_sample.pred_pts_seg.occ
    if hasattr(pred_occ, 'detach'):
        pred_occ = pred_occ.detach().cpu().numpy()
    pred_occ = np.asarray(pred_occ)
    if mask_pred_to_observed:
        pred_occ = pred_occ.copy()
        pred_occ[gt_occ == 255] = 255
    pred = occ_to_frame(
        pred_occ, info, raw_index, 'pred', max_voxels_per_label)

    points_boxes = sample_to_points_boxes_frame(
        sample, info, raw_index, class_names, max_points_per_frame)
    return dict(
        raw_index=int(raw_index),
        gt=gt,
        pred=pred,
        pointsBoxes=points_boxes)


def sample_to_pred_points_box_pair(dataset,
                                   model,
                                   raw_index: int,
                                   class_names: List[str],
                                   mask_pred_to_observed: bool,
                                   max_voxels_per_label: int,
                                   max_points_per_frame: int) -> dict:
    triple = sample_to_gt_pred_points_box_triple(
        dataset,
        model,
        raw_index,
        class_names,
        mask_pred_to_observed,
        max_voxels_per_label,
        max_points_per_frame)
    return dict(
        raw_index=int(raw_index),
        pred=triple['pred'],
        pointsBoxes=triple['pointsBoxes'])


def sample_to_gt_compare_pair(left_dataset, right_dataset, raw_index: int,
                              max_voxels_per_label: int) -> dict:
    left_sample = left_dataset.prepare_data(raw_index)
    if left_sample is None:
        raise RuntimeError(
            f'left prepare_data({raw_index}) returned None.')
    right_sample = right_dataset.prepare_data(raw_index)
    if right_sample is None:
        raise RuntimeError(
            f'right prepare_data({raw_index}) returned None.')

    info = right_dataset._get_raw_data_info(raw_index)
    left_frame = occ_to_frame(
        get_occ_array(left_sample), info, raw_index, 'single_gt',
        max_voxels_per_label, meta=get_occ_meta(left_sample))
    right_frame = occ_to_frame(
        get_occ_array(right_sample), info, raw_index, 'multi_gt',
        max_voxels_per_label, meta=get_occ_meta(right_sample))
    return dict(raw_index=int(raw_index), gt=left_frame, pred=right_frame)


def build_compare_payload(left_cfg: Config, right_cfg: Config, left_dataset,
                          right_dataset, raw_indices: List[int],
                          cube_scale: float,
                          max_voxels_per_label: int,
                          left_title: str,
                          right_title: str) -> dict:
    point_cloud_range = np.asarray(
        right_cfg.point_cloud_range, dtype=np.float32)
    occ_size = np.asarray(right_cfg.occ_size, dtype=np.int64)
    voxel_size = (
        (point_cloud_range[3:] - point_cloud_range[:3]) /
        occ_size.astype(np.float32))

    class_names = list(getattr(right_cfg, 'class_names', []))
    label_names: Dict[str, str] = {
        str(i + 1): name
        for i, name in enumerate(class_names)
    }
    label_names.setdefault('16', 'Ground')
    label_names.setdefault('17', 'Other obstacle')

    frame_pairs = [
        sample_to_gt_compare_pair(
            left_dataset, right_dataset, raw_index, max_voxels_per_label)
        for raw_index in raw_indices
    ]
    frames = []
    for pair in frame_pairs:
        frames.extend([pair['gt'], pair['pred']])

    return dict(
        pointCloudRange=point_cloud_range.tolist(),
        occSize=occ_size.astype(int).tolist(),
        voxelSize=voxel_size.tolist(),
        cubeScale=float(cube_scale),
        labelNames=label_names,
        labelColors={str(k): v for k, v in LABEL_COLORS.items()},
        hiddenLabels=[0, 255],
        egoIgnoreRange=list(getattr(right_cfg, 'ego_ignore_range', [])),
        leftTitle=left_title,
        rightTitle=right_title,
        frames=frames,
        framePairs=frame_pairs)


def build_payload(cfg: Config, dataset, raw_indices: List[int],
                  cube_scale: float,
                  model: Optional[object] = None,
                  pred_only: bool = False,
                  mask_pred_to_observed: bool = False,
                  max_voxels_per_label: int = 0,
                  gt_points_boxes: bool = False,
                  gt_pred_points_boxes: bool = False,
                  pred_points_boxes: bool = False,
                  max_points_per_frame: int = 0) -> dict:
    point_cloud_range = np.asarray(cfg.point_cloud_range, dtype=np.float32)
    occ_size = np.asarray(cfg.occ_size, dtype=np.int64)
    voxel_size = (
        (point_cloud_range[3:] - point_cloud_range[:3]) /
        occ_size.astype(np.float32))

    class_names = list(getattr(cfg, 'class_names', []))
    label_names: Dict[str, str] = {
        str(i + 1): name
        for i, name in enumerate(class_names)
    }
    label_names.setdefault('16', 'Ground')
    label_names.setdefault('17', 'Other obstacle')

    frames = []
    frame_pairs = []
    gt_points_box_pairs = []
    gt_pred_points_box_triples = []
    pred_points_box_pairs = []
    for raw_index in raw_indices:
        if gt_pred_points_boxes:
            pair = sample_to_gt_pred_points_box_triple(
                dataset, model, raw_index, class_names,
                mask_pred_to_observed, max_voxels_per_label,
                max_points_per_frame)
            frames.extend([pair['gt'], pair['pred']])
            gt_pred_points_box_triples.append(pair)
        elif pred_points_boxes:
            pair = sample_to_pred_points_box_pair(
                dataset, model, raw_index, class_names,
                mask_pred_to_observed, max_voxels_per_label,
                max_points_per_frame)
            frames.append(pair['pred'])
            pred_points_box_pairs.append(pair)
        elif gt_points_boxes:
            pair = sample_to_gt_points_box_pair(
                dataset, raw_index, class_names, max_voxels_per_label,
                max_points_per_frame)
            frames.append(pair['gt'])
            gt_points_box_pairs.append(pair)
        elif model is None:
            frames.append(
                sample_to_frame(dataset, raw_index, max_voxels_per_label))
        else:
            pred_frames = sample_to_pred_frames(
                dataset, model, raw_index, include_gt=not pred_only,
                mask_pred_to_observed=mask_pred_to_observed,
                max_voxels_per_label=max_voxels_per_label)
            frames.extend(pred_frames)
            if not pred_only and len(pred_frames) == 2:
                frame_pairs.append(
                    dict(
                        raw_index=int(raw_index),
                        gt=pred_frames[0],
                        pred=pred_frames[1]))

    payload = dict(
        pointCloudRange=point_cloud_range.tolist(),
        occSize=occ_size.astype(int).tolist(),
        voxelSize=voxel_size.tolist(),
        cubeScale=float(cube_scale),
        labelNames=label_names,
        labelColors={str(k): v for k, v in LABEL_COLORS.items()},
        hiddenLabels=[0, 255],
        egoIgnoreRange=list(getattr(cfg, 'ego_ignore_range', [])),
        frames=frames,
    )
    if frame_pairs:
        payload['framePairs'] = frame_pairs
    if gt_points_box_pairs:
        payload['gtPointsBoxPairs'] = gt_points_box_pairs
    if gt_pred_points_box_triples:
        payload['gtPredPointsBoxTriples'] = gt_pred_points_box_triples
    if pred_points_box_pairs:
        payload['predPointsBoxPairs'] = pred_points_box_pairs
    return payload


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(',', ':'))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KL occupancy sequence</title>
<style>
html, body {{
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #f8fafc;
  font-family: Arial, Helvetica, sans-serif;
}}
#viewport {{
  position: fixed;
  inset: 0;
}}
.panel {{
  position: fixed;
  left: 14px;
  top: 12px;
  width: min(420px, calc(100vw - 28px));
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
  color: #0f172a;
  padding: 10px 12px;
  box-sizing: border-box;
}}
.row {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
button {{
  width: 36px;
  height: 30px;
  border: 1px solid rgba(15, 23, 42, 0.22);
  border-radius: 6px;
  background: #ffffff;
  color: #0f172a;
  cursor: pointer;
  font-size: 14px;
}}
input[type="range"] {{
  flex: 1;
  min-width: 0;
}}
.title {{
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 8px;
}}
.meta, .stats {{
  font-size: 12px;
  line-height: 1.45;
  color: #334155;
  margin-top: 8px;
  word-break: break-word;
}}
.legend {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px 10px;
  margin-top: 8px;
  font-size: 12px;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}}
.swatch {{
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 auto;
}}
.legend-text {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.error {{
  position: fixed;
  inset: 20px;
  display: none;
  align-items: center;
  justify-content: center;
  color: #991b1b;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(153, 27, 27, 0.25);
  border-radius: 8px;
  font-size: 14px;
  padding: 20px;
  box-sizing: border-box;
}}
</style>
</head>
<body>
<div id="viewport"></div>
<div class="panel">
  <div class="title">KL raycast occupancy sequence</div>
  <div class="row">
    <button id="play" title="Play or pause">Play</button>
    <input id="frame" type="range" min="0" value="0">
    <span id="frameText"></span>
  </div>
  <div id="meta" class="meta"></div>
  <div id="stats" class="stats"></div>
  <div id="legend" class="legend"></div>
</div>
<div id="error" class="error"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js">
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js">
</script>
<script>
const DATA = {data_json};
const viewport = document.getElementById('viewport');
const errorBox = document.getElementById('error');
const frameSlider = document.getElementById('frame');
const playButton = document.getElementById('play');
const frameText = document.getElementById('frameText');
const metaBox = document.getElementById('meta');
const statsBox = document.getElementById('stats');
const legendBox = document.getElementById('legend');

if (!window.THREE || !THREE.OrbitControls) {{
  errorBox.style.display = 'flex';
  errorBox.textContent = 'Three.js failed to load. Check network access.';
  throw new Error('Three.js failed to load.');
}}

const range = DATA.pointCloudRange;
const occSize = DATA.occSize;
const voxelSize = DATA.voxelSize;
const scale = DATA.cubeScale;
const frameCount = DATA.frames.length;
frameSlider.max = Math.max(0, frameCount - 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8fafc);

const camera = new THREE.PerspectiveCamera(
  50, window.innerWidth / window.innerHeight, 0.1, 1200);
camera.up.set(0, 0, 1);
const center = new THREE.Vector3(
  (range[0] + range[3]) * 0.5,
  (range[1] + range[4]) * 0.5,
  (range[2] + range[5]) * 0.5);
camera.position.set(center.x + 80, center.y - 125, center.z + 58);

let renderer;
try {{
  renderer = new THREE.WebGLRenderer({{ antialias: true }});
}} catch (err) {{
  errorBox.style.display = 'flex';
  errorBox.textContent =
    'WebGL failed to start. Please open this page in a browser with WebGL.';
  throw err;
}}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
viewport.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c2cf, 0.92));
const sun = new THREE.DirectionalLight(0xffffff, 0.72);
sun.position.set(center.x - 30, center.y - 40, center.z + 120);
scene.add(sun);

const gridSize = Math.max(range[3] - range[0], range[4] - range[1]);
const grid = new THREE.GridHelper(gridSize, 32, 0x94a3b8, 0xdbe3ee);
grid.rotation.x = Math.PI / 2;
grid.position.set(center.x, center.y, range[2]);
scene.add(grid);

function addAxis(start, end, color) {{
  const material = new THREE.LineBasicMaterial({{ color }});
  const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
  scene.add(new THREE.Line(geometry, material));
}}
addAxis(
  new THREE.Vector3(range[0], 0, range[2]),
  new THREE.Vector3(range[3], 0, range[2]),
  0xef4444);
addAxis(
  new THREE.Vector3(0, range[1], range[2]),
  new THREE.Vector3(0, range[4], range[2]),
  0x22c55e);
addAxis(
  new THREE.Vector3(0, 0, range[2]),
  new THREE.Vector3(0, 0, range[5]),
  0x2563eb);

const cubeGeometry = new THREE.BoxGeometry(
  voxelSize[0] * scale, voxelSize[1] * scale, voxelSize[2] * scale);
const dummy = new THREE.Object3D();
let activeMeshes = [];
let activeFrame = 0;
let timer = null;

function labelColor(label) {{
  return DATA.labelColors[String(label)] || '#94a3b8';
}}

function labelName(label) {{
  return DATA.labelNames[String(label)] || `class ${{label}}`;
}}

function disposeMeshes() {{
  for (const mesh of activeMeshes) {{
    scene.remove(mesh);
    mesh.material.dispose();
  }}
  activeMeshes = [];
}}

function decodeFlat(flat) {{
  const yz = occSize[1] * occSize[2];
  const x = Math.floor(flat / yz);
  const rem = flat - x * yz;
  const y = Math.floor(rem / occSize[2]);
  const z = rem - y * occSize[2];
  return [x, y, z];
}}

function voxelCenter(index) {{
  return [
    range[0] + (index[0] + 0.5) * voxelSize[0],
    range[1] + (index[1] + 0.5) * voxelSize[1],
    range[2] + (index[2] + 0.5) * voxelSize[2],
  ];
}}

function addLabelMesh(label, flats) {{
  if (!flats || flats.length === 0) {{
    return;
  }}
  const material = new THREE.MeshStandardMaterial({{
    color: new THREE.Color(labelColor(label)),
    roughness: 0.82,
    metalness: 0.02
  }});
  const mesh = new THREE.InstancedMesh(cubeGeometry, material, flats.length);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < flats.length; i++) {{
    const center = voxelCenter(decodeFlat(flats[i]));
    dummy.position.set(center[0], center[1], center[2]);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }}
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  activeMeshes.push(mesh);
}}

function formatStats(stats) {{
  const free = stats['0'] || 0;
  const ignore = stats['255'] || 0;
  let occupied = 0;
  for (const [key, value] of Object.entries(stats)) {{
    if (key !== '0' && key !== '255') {{
      occupied += value;
    }}
  }}
  return `occupied drawn: ${{occupied}} | free hidden: ${{free}} | ` +
         `ignore hidden: ${{ignore}}`;
}}

function renderLegend(frame) {{
  const items = [];
  const labels = Object.keys(frame.labels)
    .map((label) => Number(label))
    .sort((a, b) => a - b);
  for (const label of labels) {{
    const count = frame.labels[String(label)].length;
    const fullCount = frame.stats[String(label)] || count;
    const countText = fullCount === count ? `${{count}}` : `${{count}}/${{fullCount}}`;
    items.push(
      `<div class="legend-item" title="${{labelName(label)}}">` +
      `<span class="swatch" style="background:${{labelColor(label)}}"></span>` +
      `<span class="legend-text">${{label}}: ${{labelName(label)}} ` +
      `(${{countText}})</span></div>`);
  }}
  legendBox.innerHTML = items.join('');
}}

function renderFrame(index) {{
  activeFrame = Math.max(0, Math.min(frameCount - 1, index));
  const frame = DATA.frames[activeFrame];
  disposeMeshes();
  const labels = Object.keys(frame.labels)
    .map((label) => Number(label))
    .sort((a, b) => a - b);
  for (const label of labels) {{
    addLabelMesh(label, frame.labels[String(label)]);
  }}
  frameSlider.value = String(activeFrame);
  frameText.textContent = `${{activeFrame + 1}} / ${{frameCount}}`;
  metaBox.textContent =
    `${{frame.title}} | timestamp=${{frame.timestamp}} | ` +
    `scene=${{frame.scene_token}}`;
  statsBox.textContent = formatStats(frame.stats) +
    ` | ego_ignore=${{JSON.stringify(DATA.egoIgnoreRange)}}`;
  renderLegend(frame);
}}

frameSlider.addEventListener('input', () => {{
  renderFrame(Number(frameSlider.value));
}});

playButton.addEventListener('click', () => {{
  if (timer) {{
    clearInterval(timer);
    timer = null;
    playButton.textContent = 'Play';
    return;
  }}
  playButton.textContent = 'Pause';
  timer = setInterval(() => {{
    renderFrame((activeFrame + 1) % frameCount);
  }}, 650);
}});

window.addEventListener('keydown', (event) => {{
  if (event.key === 'ArrowRight') {{
    renderFrame((activeFrame + 1) % frameCount);
  }}
  if (event.key === 'ArrowLeft') {{
    renderFrame((activeFrame - 1 + frameCount) % frameCount);
  }}
}});

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

renderFrame(0);
animate();
</script>
</body>
</html>
"""


def build_split_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(',', ':'))
    html = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KL occupancy GT / Pred comparison</title>
<style>
html, body {
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #f8fafc;
  font-family: Arial, Helvetica, sans-serif;
}
#viewport {
  position: fixed;
  inset: 0;
}
.divider {
  position: fixed;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: rgba(15, 23, 42, 0.22);
  pointer-events: none;
}
.side-label {
  position: fixed;
  top: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
  pointer-events: none;
}
.side-label.gt {
  left: 14px;
}
.side-label.pred {
  left: calc(50% + 14px);
}
.panel {
  position: fixed;
  left: 14px;
  right: 14px;
  bottom: 12px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: end;
  pointer-events: none;
}
.box {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
  color: #0f172a;
  padding: 8px 10px;
  min-width: 0;
  box-sizing: border-box;
  pointer-events: auto;
}
.controls {
  width: min(460px, 42vw);
}
.title {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
}
.row {
  display: flex;
  align-items: center;
  gap: 8px;
}
button {
  width: 36px;
  height: 30px;
  border: 1px solid rgba(15, 23, 42, 0.22);
  border-radius: 6px;
  background: #ffffff;
  color: #0f172a;
  cursor: pointer;
  font-size: 14px;
}
input[type="range"] {
  flex: 1;
  min-width: 0;
}
.stats {
  font-size: 12px;
  line-height: 1.45;
  color: #334155;
  word-break: break-word;
}
.legend {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 4px 10px;
  font-size: 12px;
  margin-top: 6px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.legend-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.error {
  position: fixed;
  inset: 20px;
  display: none;
  align-items: center;
  justify-content: center;
  color: #991b1b;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(153, 27, 27, 0.25);
  border-radius: 8px;
  font-size: 14px;
  padding: 20px;
  box-sizing: border-box;
}
@media (max-width: 900px) {
  .panel {
    grid-template-columns: minmax(0, 1fr);
  }
  .controls {
    width: auto;
  }
  .legend {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
</head>
<body>
<div id="viewport"></div>
<div class="divider"></div>
<div id="leftLabel" class="side-label gt">GT</div>
<div id="rightLabel" class="side-label pred">Pred</div>
<div class="panel">
  <div id="gtStats" class="box stats"></div>
  <div class="box controls">
    <div id="title" class="title"></div>
    <div class="row">
      <button id="play" title="Play or pause">Play</button>
      <input id="frame" type="range" min="0" value="0">
      <span id="frameText"></span>
    </div>
    <div id="legend" class="legend"></div>
  </div>
  <div id="predStats" class="box stats"></div>
</div>
<div id="error" class="error"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js">
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js">
</script>
<script>
const DATA = __DATA_JSON__;
const viewport = document.getElementById('viewport');
const errorBox = document.getElementById('error');
const frameSlider = document.getElementById('frame');
const playButton = document.getElementById('play');
const frameText = document.getElementById('frameText');
const titleBox = document.getElementById('title');
const gtStatsBox = document.getElementById('gtStats');
const predStatsBox = document.getElementById('predStats');
const leftLabel = document.getElementById('leftLabel');
const rightLabel = document.getElementById('rightLabel');
const legendBox = document.getElementById('legend');

if (!window.THREE || !THREE.OrbitControls) {
  errorBox.style.display = 'flex';
  errorBox.textContent = 'Three.js failed to load. Check network access.';
  throw new Error('Three.js failed to load.');
}

const pairs = DATA.framePairs || [];
const range = DATA.pointCloudRange;
const occSize = DATA.occSize;
const voxelSize = DATA.voxelSize;
const scale = DATA.cubeScale;
const frameCount = pairs.length;
frameSlider.max = Math.max(0, frameCount - 1);
const predLeft = (
  !DATA.leftTitle &&
  pairs.length > 0 &&
  String(pairs[0].pred.source || '').startsWith('pred') &&
  String(pairs[0].gt.source || '').includes('gt')
);
const leftTitle = predLeft ? 'Pred OCC' : (DATA.leftTitle || 'GT');
const rightTitle = predLeft ? 'GT OCC' : (DATA.rightTitle || 'Pred');
leftLabel.textContent = leftTitle;
rightLabel.textContent = rightTitle;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setScissorTest(true);
viewport.appendChild(renderer.domElement);

const center = new THREE.Vector3(
  (range[0] + range[3]) * 0.5,
  (range[1] + range[4]) * 0.5,
  (range[2] + range[5]) * 0.5);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1200);
camera.up.set(0, 0, 1);
camera.position.set(center.x + 80, center.y - 125, center.z + 58);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

function makeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c2cf, 0.92));
  const sun = new THREE.DirectionalLight(0xffffff, 0.72);
  sun.position.set(center.x - 30, center.y - 40, center.z + 120);
  scene.add(sun);

  const gridSize = Math.max(range[3] - range[0], range[4] - range[1]);
  const grid = new THREE.GridHelper(gridSize, 32, 0x94a3b8, 0xdbe3ee);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, range[2]);
  scene.add(grid);

  function addAxis(start, end, color) {
    const material = new THREE.LineBasicMaterial({ color });
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    scene.add(new THREE.Line(geometry, material));
  }
  addAxis(
    new THREE.Vector3(range[0], 0, range[2]),
    new THREE.Vector3(range[3], 0, range[2]),
    0xef4444);
  addAxis(
    new THREE.Vector3(0, range[1], range[2]),
    new THREE.Vector3(0, range[4], range[2]),
    0x22c55e);
  addAxis(
    new THREE.Vector3(0, 0, range[2]),
    new THREE.Vector3(0, 0, range[5]),
    0x2563eb);
  return scene;
}

const gtScene = makeScene();
const predScene = makeScene();
const cubeGeometry = new THREE.BoxGeometry(
  voxelSize[0] * scale, voxelSize[1] * scale, voxelSize[2] * scale);
const dummy = new THREE.Object3D();
let gtMeshes = [];
let predMeshes = [];
let activeFrame = 0;
let timer = null;

function labelColor(label) {
  return DATA.labelColors[String(label)] || '#94a3b8';
}

function labelName(label) {
  return DATA.labelNames[String(label)] || `class ${label}`;
}

function decodeFlat(flat) {
  const yz = occSize[1] * occSize[2];
  const x = Math.floor(flat / yz);
  const rem = flat - x * yz;
  const y = Math.floor(rem / occSize[2]);
  const z = rem - y * occSize[2];
  return [x, y, z];
}

function voxelCenter(index) {
  return [
    range[0] + (index[0] + 0.5) * voxelSize[0],
    range[1] + (index[1] + 0.5) * voxelSize[1],
    range[2] + (index[2] + 0.5) * voxelSize[2],
  ];
}

function disposeMeshes(meshes, scene) {
  for (const mesh of meshes) {
    scene.remove(mesh);
    mesh.material.dispose();
  }
  meshes.length = 0;
}

function addLabelMesh(scene, meshes, label, flats) {
  if (!flats || flats.length === 0) {
    return;
  }
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(labelColor(label)),
    roughness: 0.82,
    metalness: 0.02
  });
  const mesh = new THREE.InstancedMesh(cubeGeometry, material, flats.length);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < flats.length; i++) {
    const center = voxelCenter(decodeFlat(flats[i]));
    dummy.position.set(center[0], center[1], center[2]);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  meshes.push(mesh);
}

function formatStats(prefix, frame) {
  const stats = frame.stats || {};
  const free = stats['0'] || 0;
  const ignore = stats['255'] || 0;
  let occupied = 0;
  for (const [key, value] of Object.entries(stats)) {
    if (key !== '0' && key !== '255') {
      occupied += value;
    }
  }
  const meta = frame.meta || {};
  const metaItems = [
    ['hit', 'num_hit_voxels'],
    ['sem', 'num_semantic_voxels'],
    ['ground', 'num_ground_voxels'],
    ['raw_obs', 'num_raw_obstacle_voxels'],
    ['obs', 'num_obstacle_voxels'],
    ['box_margin', 'num_box_margin_obstacle_voxels'],
    ['filtered_obs', 'num_filtered_obstacle_voxels']
  ].filter((item) => meta[item[1]] !== undefined)
   .map((item) => `${item[0]}=${meta[item[1]]}`);
  let text = `${prefix}: raw_index=${frame.raw_index}<br>` +
    `occupied=${occupied} | free=${free} | ignore=${ignore}`;
  if (metaItems.length > 0) {
    text += `<br>${metaItems.join(' | ')}`;
  }
  return text;
}

function renderLegend(pair) {
  const labels = new Set();
  for (const frame of [pair.gt, pair.pred]) {
    for (const label of Object.keys(frame.labels || {})) {
      labels.add(Number(label));
    }
  }
  const items = Array.from(labels).sort((a, b) => a - b).map((label) =>
    `<div class="legend-item" title="${labelName(label)}">` +
    `<span class="swatch" style="background:${labelColor(label)}"></span>` +
    `<span class="legend-text">${label}: ${labelName(label)}</span></div>`);
  legendBox.innerHTML = items.join('');
}

function addFrameToScene(scene, meshes, frame) {
  const labels = Object.keys(frame.labels || {})
    .map((label) => Number(label))
    .sort((a, b) => a - b);
  for (const label of labels) {
    addLabelMesh(scene, meshes, label, frame.labels[String(label)]);
  }
}

function renderFrame(index) {
  if (frameCount === 0) {
    titleBox.textContent = 'No frame pairs found';
    return;
  }
  activeFrame = Math.max(0, Math.min(frameCount - 1, index));
  const pair = pairs[activeFrame];
  disposeMeshes(gtMeshes, gtScene);
  disposeMeshes(predMeshes, predScene);
  const leftFrame = predLeft ? pair.pred : pair.gt;
  const rightFrame = predLeft ? pair.gt : pair.pred;
  addFrameToScene(gtScene, gtMeshes, leftFrame);
  addFrameToScene(predScene, predMeshes, rightFrame);

  frameSlider.value = String(activeFrame);
  frameText.textContent = `${activeFrame + 1} / ${frameCount}`;
  titleBox.textContent =
    `raw_index=${pair.raw_index} | token=${pair.gt.token || ''}`;
  gtStatsBox.innerHTML = formatStats(leftTitle, leftFrame);
  predStatsBox.innerHTML = formatStats(rightTitle, rightFrame);
  renderLegend(pair);
}

function draw() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const half = Math.floor(width / 2);
  renderer.clear();

  camera.aspect = half / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(0, 0, half, height);
  renderer.setScissor(0, 0, half, height);
  renderer.render(gtScene, camera);

  const rightWidth = width - half;
  camera.aspect = rightWidth / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(half, 0, rightWidth, height);
  renderer.setScissor(half, 0, rightWidth, height);
  renderer.render(predScene, camera);
}

frameSlider.addEventListener('input', () => {
  renderFrame(Number(frameSlider.value));
});

playButton.addEventListener('click', () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
    playButton.textContent = 'Play';
    return;
  }
  playButton.textContent = 'Pause';
  timer = setInterval(() => {
    renderFrame((activeFrame + 1) % frameCount);
  }, 650);
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight') {
    renderFrame((activeFrame + 1) % frameCount);
  }
  if (event.key === 'ArrowLeft') {
    renderFrame((activeFrame - 1 + frameCount) % frameCount);
  }
});

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  draw();
}

renderFrame(0);
animate();
</script>
</body>
</html>
"""
    return html.replace('__DATA_JSON__', data_json)


def build_gt_points_boxes_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(',', ':'))
    html = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KL occupancy GT / points boxes check</title>
<style>
html, body {
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #f8fafc;
  font-family: Arial, Helvetica, sans-serif;
}
#viewport {
  position: fixed;
  inset: 0;
}
.divider {
  position: fixed;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: rgba(15, 23, 42, 0.22);
  pointer-events: none;
}
.side-label {
  position: fixed;
  top: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
  pointer-events: none;
}
.side-label.gt {
  left: 14px;
}
.side-label.points {
  left: calc(50% + 14px);
}
.panel {
  position: fixed;
  left: 14px;
  right: 14px;
  bottom: 12px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: end;
  pointer-events: none;
}
.box {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
  color: #0f172a;
  padding: 8px 10px;
  min-width: 0;
  box-sizing: border-box;
  pointer-events: auto;
}
.controls {
  width: min(500px, 42vw);
}
.title {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
}
.row {
  display: flex;
  align-items: center;
  gap: 8px;
}
button {
  width: 36px;
  height: 30px;
  border: 1px solid rgba(15, 23, 42, 0.22);
  border-radius: 6px;
  background: #ffffff;
  color: #0f172a;
  cursor: pointer;
  font-size: 14px;
}
input[type="range"] {
  flex: 1;
  min-width: 0;
}
.stats {
  font-size: 12px;
  line-height: 1.45;
  color: #334155;
  word-break: break-word;
}
.legend {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 4px 10px;
  font-size: 12px;
  margin-top: 6px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.legend-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.error {
  position: fixed;
  inset: 20px;
  display: none;
  align-items: center;
  justify-content: center;
  color: #991b1b;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(153, 27, 27, 0.25);
  border-radius: 8px;
  font-size: 14px;
  padding: 20px;
  box-sizing: border-box;
}
@media (max-width: 900px) {
  .panel {
    grid-template-columns: minmax(0, 1fr);
  }
  .controls {
    width: auto;
  }
  .legend {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
</head>
<body>
<div id="viewport"></div>
<div class="divider"></div>
<div class="side-label gt">GT OCC</div>
<div class="side-label points">Points + GT Boxes</div>
<div class="panel">
  <div id="gtStats" class="box stats"></div>
  <div class="box controls">
    <div id="title" class="title"></div>
    <div class="row">
      <button id="play" title="Play or pause">Play</button>
      <input id="frame" type="range" min="0" value="0">
      <span id="frameText"></span>
    </div>
    <div id="legend" class="legend"></div>
  </div>
  <div id="pointsStats" class="box stats"></div>
</div>
<div id="error" class="error"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js">
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js">
</script>
<script>
const DATA = __DATA_JSON__;
const viewport = document.getElementById('viewport');
const errorBox = document.getElementById('error');
const frameSlider = document.getElementById('frame');
const playButton = document.getElementById('play');
const frameText = document.getElementById('frameText');
const titleBox = document.getElementById('title');
const gtStatsBox = document.getElementById('gtStats');
const pointsStatsBox = document.getElementById('pointsStats');
const legendBox = document.getElementById('legend');

if (!window.THREE || !THREE.OrbitControls) {
  errorBox.style.display = 'flex';
  errorBox.textContent = 'Three.js failed to load. Check network access.';
  throw new Error('Three.js failed to load.');
}

const pairs = DATA.gtPointsBoxPairs || [];
const range = DATA.pointCloudRange;
const occSize = DATA.occSize;
const voxelSize = DATA.voxelSize;
const scale = DATA.cubeScale;
const frameCount = pairs.length;
frameSlider.max = Math.max(0, frameCount - 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setScissorTest(true);
viewport.appendChild(renderer.domElement);

const center = new THREE.Vector3(
  (range[0] + range[3]) * 0.5,
  (range[1] + range[4]) * 0.5,
  (range[2] + range[5]) * 0.5);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1200);
camera.up.set(0, 0, 1);
camera.position.set(center.x + 80, center.y - 125, center.z + 58);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

function makeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c2cf, 0.92));
  const sun = new THREE.DirectionalLight(0xffffff, 0.72);
  sun.position.set(center.x - 30, center.y - 40, center.z + 120);
  scene.add(sun);

  const gridSize = Math.max(range[3] - range[0], range[4] - range[1]);
  const grid = new THREE.GridHelper(gridSize, 32, 0x94a3b8, 0xdbe3ee);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, range[2]);
  scene.add(grid);

  function addAxis(start, end, color) {
    const material = new THREE.LineBasicMaterial({ color });
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    scene.add(new THREE.Line(geometry, material));
  }
  addAxis(
    new THREE.Vector3(range[0], 0, range[2]),
    new THREE.Vector3(range[3], 0, range[2]),
    0xef4444);
  addAxis(
    new THREE.Vector3(0, range[1], range[2]),
    new THREE.Vector3(0, range[4], range[2]),
    0x22c55e);
  addAxis(
    new THREE.Vector3(0, 0, range[2]),
    new THREE.Vector3(0, 0, range[5]),
    0x2563eb);
  return scene;
}

const gtScene = makeScene();
const pointsScene = makeScene();
const cubeGeometry = new THREE.BoxGeometry(
  voxelSize[0] * scale, voxelSize[1] * scale, voxelSize[2] * scale);
const dummy = new THREE.Object3D();
let gtMeshes = [];
let rightObjects = [];
let activeFrame = 0;
let timer = null;

function labelColor(label) {
  return DATA.labelColors[String(label)] || '#94a3b8';
}

function labelName(label) {
  return DATA.labelNames[String(label)] || `class ${label}`;
}

function decodeFlat(flat) {
  const yz = occSize[1] * occSize[2];
  const x = Math.floor(flat / yz);
  const rem = flat - x * yz;
  const y = Math.floor(rem / occSize[2]);
  const z = rem - y * occSize[2];
  return [x, y, z];
}

function voxelCenter(index) {
  return [
    range[0] + (index[0] + 0.5) * voxelSize[0],
    range[1] + (index[1] + 0.5) * voxelSize[1],
    range[2] + (index[2] + 0.5) * voxelSize[2],
  ];
}

function disposeMeshes(meshes, scene, disposeGeometry) {
  for (const mesh of meshes) {
    scene.remove(mesh);
    if (disposeGeometry && mesh.geometry) {
      mesh.geometry.dispose();
    }
    if (mesh.material) {
      mesh.material.dispose();
    }
  }
  meshes.length = 0;
}

function addLabelMesh(scene, meshes, label, flats) {
  if (!flats || flats.length === 0) {
    return;
  }
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(labelColor(label)),
    roughness: 0.82,
    metalness: 0.02
  });
  const mesh = new THREE.InstancedMesh(cubeGeometry, material, flats.length);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < flats.length; i++) {
    const center = voxelCenter(decodeFlat(flats[i]));
    dummy.position.set(center[0], center[1], center[2]);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  meshes.push(mesh);
}

function addFrameToOccScene(scene, meshes, frame) {
  const labels = Object.keys(frame.labels || {})
    .map((label) => Number(label))
    .sort((a, b) => a - b);
  for (const label of labels) {
    addLabelMesh(scene, meshes, label, frame.labels[String(label)]);
  }
}

function addPoints(pointsFrame) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(pointsFrame.points), 3));
  const material = new THREE.PointsMaterial({
    color: 0x111827,
    size: 0.09,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.78
  });
  const cloud = new THREE.Points(geometry, material);
  pointsScene.add(cloud);
  rightObjects.push(cloud);
}

function addBox(box) {
  const c = box.corners;
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];
  const positions = [];
  for (const edge of edges) {
    positions.push(...c[edge[0]], ...c[edge[1]]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(positions), 3));
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(labelColor(box.label)),
    linewidth: 1
  });
  const line = new THREE.LineSegments(geometry, material);
  pointsScene.add(line);
  rightObjects.push(line);
}

function addPointsBoxes(pointsFrame) {
  addPoints(pointsFrame);
  for (const box of pointsFrame.boxes || []) {
    addBox(box);
  }
}

function formatGtStats(frame) {
  const stats = frame.stats || {};
  const free = stats['0'] || 0;
  const ignore = stats['255'] || 0;
  let occupied = 0;
  for (const [key, value] of Object.entries(stats)) {
    if (key !== '0' && key !== '255') {
      occupied += value;
    }
  }
  const meta = frame.meta || {};
  const metaItems = [
    ['hit', 'num_hit_voxels'],
    ['free_rays', 'num_free_voxels'],
    ['sem', 'num_semantic_voxels'],
    ['ground', 'num_ground_voxels'],
    ['raw_obs', 'num_raw_obstacle_voxels'],
    ['obs', 'num_obstacle_voxels'],
    ['box_margin', 'num_box_margin_obstacle_voxels'],
    ['filtered_obs', 'num_filtered_obstacle_voxels'],
    ['ego_ignore', 'num_ego_ignore_voxels']
  ].filter((item) => meta[item[1]] !== undefined)
   .map((item) => `${item[0]}=${meta[item[1]]}`);
  const ruleItems = [
    ['obs_min_pts', 'obstacle_min_points_per_voxel'],
    ['obs_min_comp', 'obstacle_min_component_voxels'],
    ['small_keep_pts', 'obstacle_small_component_keep_min_points'],
    ['thin_major_m', 'obstacle_thin_component_min_major_span'],
    ['thin_minor_m', 'obstacle_thin_component_max_minor_span'],
    ['thin_z_m', 'obstacle_thin_component_max_z_span'],
    ['thin_keep_pts', 'obstacle_thin_component_keep_min_points'],
    ['box_margin_m', 'obstacle_box_ignore_margin']
  ].filter((item) => meta[item[1]] !== undefined)
   .map((item) => `${item[0]}=${meta[item[1]]}`);
  let text = `GT OCC: raw_index=${frame.raw_index}<br>` +
    `occupied=${occupied} | free=${free} | ignore=${ignore}`;
  if (metaItems.length > 0) {
    text += `<br>${metaItems.join(' | ')}`;
  }
  if (ruleItems.length > 0) {
    text += `<br>${ruleItems.join(' | ')}`;
  }
  return text;
}

function formatPointsStats(frame) {
  return `Points + boxes: raw_index=${frame.raw_index}<br>` +
    `points=${frame.numPoints}/${frame.totalPoints} | boxes=${frame.numBoxes}`;
}

function renderLegend(pair) {
  const labels = new Set();
  for (const label of Object.keys(pair.gt.labels || {})) {
    labels.add(Number(label));
  }
  for (const box of pair.pointsBoxes.boxes || []) {
    labels.add(Number(box.label));
  }
  const items = Array.from(labels).sort((a, b) => a - b).map((label) =>
    `<div class="legend-item" title="${labelName(label)}">` +
    `<span class="swatch" style="background:${labelColor(label)}"></span>` +
    `<span class="legend-text">${label}: ${labelName(label)}</span></div>`);
  legendBox.innerHTML = items.join('');
}

function renderFrame(index) {
  if (frameCount === 0) {
    titleBox.textContent = 'No frames found';
    return;
  }
  activeFrame = Math.max(0, Math.min(frameCount - 1, index));
  const pair = pairs[activeFrame];
  disposeMeshes(gtMeshes, gtScene, false);
  disposeMeshes(rightObjects, pointsScene, true);
  addFrameToOccScene(gtScene, gtMeshes, pair.gt);
  addPointsBoxes(pair.pointsBoxes);

  frameSlider.value = String(activeFrame);
  frameText.textContent = `${activeFrame + 1} / ${frameCount}`;
  titleBox.textContent =
    `raw_index=${pair.raw_index} | token=${pair.gt.token || ''}`;
  gtStatsBox.innerHTML = formatGtStats(pair.gt);
  pointsStatsBox.innerHTML = formatPointsStats(pair.pointsBoxes);
  renderLegend(pair);
}

function draw() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const half = Math.floor(width / 2);
  renderer.clear();

  camera.aspect = half / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(0, 0, half, height);
  renderer.setScissor(0, 0, half, height);
  renderer.render(gtScene, camera);

  const rightWidth = width - half;
  camera.aspect = rightWidth / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(half, 0, rightWidth, height);
  renderer.setScissor(half, 0, rightWidth, height);
  renderer.render(pointsScene, camera);
}

frameSlider.addEventListener('input', () => {
  renderFrame(Number(frameSlider.value));
});

playButton.addEventListener('click', () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
    playButton.textContent = 'Play';
    return;
  }
  playButton.textContent = 'Pause';
  timer = setInterval(() => {
    renderFrame((activeFrame + 1) % frameCount);
  }, 650);
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight') {
    renderFrame((activeFrame + 1) % frameCount);
  }
  if (event.key === 'ArrowLeft') {
    renderFrame((activeFrame - 1 + frameCount) % frameCount);
  }
});

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  draw();
}

renderFrame(0);
animate();
</script>
</body>
</html>
"""
    return html.replace('__DATA_JSON__', data_json)


def build_gt_pred_points_boxes_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(',', ':'))
    html = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KL occupancy GT / Pred / points boxes check</title>
<style>
html, body {
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #f8fafc;
  font-family: Arial, Helvetica, sans-serif;
}
#viewport {
  position: fixed;
  inset: 0;
}
.divider {
  position: fixed;
  top: 0;
  bottom: 0;
  width: 1px;
  background: rgba(15, 23, 42, 0.22);
  pointer-events: none;
}
.divider.left {
  left: 33.3333%;
}
.divider.right {
  left: 66.6667%;
}
.side-label {
  position: fixed;
  top: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
  pointer-events: none;
}
.side-label.gt {
  left: 14px;
}
.side-label.pred {
  left: calc(33.3333% + 14px);
}
.side-label.points {
  left: calc(66.6667% + 14px);
}
.panel {
  position: fixed;
  left: 14px;
  right: 14px;
  bottom: 12px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
  align-items: end;
  pointer-events: none;
}
.box {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
  color: #0f172a;
  padding: 8px 10px;
  min-width: 0;
  box-sizing: border-box;
  pointer-events: auto;
}
.controls {
  width: min(520px, 34vw);
}
.title {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
}
.row {
  display: flex;
  align-items: center;
  gap: 8px;
}
button {
  width: 36px;
  height: 30px;
  border: 1px solid rgba(15, 23, 42, 0.22);
  border-radius: 6px;
  background: #ffffff;
  color: #0f172a;
  cursor: pointer;
  font-size: 14px;
}
input[type="range"] {
  flex: 1;
  min-width: 0;
}
.stats {
  font-size: 12px;
  line-height: 1.45;
  color: #334155;
  word-break: break-word;
}
.legend {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 4px 10px;
  font-size: 12px;
  margin-top: 6px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.legend-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.error {
  position: fixed;
  inset: 20px;
  display: none;
  align-items: center;
  justify-content: center;
  color: #991b1b;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(153, 27, 27, 0.25);
  border-radius: 8px;
  font-size: 14px;
  padding: 20px;
  box-sizing: border-box;
}
@media (max-width: 1200px) {
  .panel {
    grid-template-columns: minmax(0, 1fr);
  }
  .controls {
    width: auto;
  }
  .legend {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
</head>
<body>
<div id="viewport"></div>
<div class="divider left"></div>
<div class="divider right"></div>
<div class="side-label gt">GT OCC</div>
<div class="side-label pred">Pred OCC</div>
<div class="side-label points">Points + GT Boxes</div>
<div class="panel">
  <div id="gtStats" class="box stats"></div>
  <div class="box controls">
    <div id="title" class="title"></div>
    <div class="row">
      <button id="play" title="Play or pause">Play</button>
      <input id="frame" type="range" min="0" value="0">
      <span id="frameText"></span>
    </div>
    <div id="legend" class="legend"></div>
  </div>
  <div id="predStats" class="box stats"></div>
  <div id="pointsStats" class="box stats"></div>
</div>
<div id="error" class="error"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA_JSON__;
const viewport = document.getElementById('viewport');
const errorBox = document.getElementById('error');
const frameSlider = document.getElementById('frame');
const playButton = document.getElementById('play');
const frameText = document.getElementById('frameText');
const titleBox = document.getElementById('title');
const gtStatsBox = document.getElementById('gtStats');
const predStatsBox = document.getElementById('predStats');
const pointsStatsBox = document.getElementById('pointsStats');
const legendBox = document.getElementById('legend');

if (!window.THREE || !THREE.OrbitControls) {
  errorBox.style.display = 'flex';
  errorBox.textContent = 'Three.js failed to load. Check network access.';
  throw new Error('Three.js failed to load.');
}

const triples = DATA.gtPredPointsBoxTriples || [];
const range = DATA.pointCloudRange;
const occSize = DATA.occSize;
const voxelSize = DATA.voxelSize;
const scale = DATA.cubeScale;
const frameCount = triples.length;
frameSlider.max = Math.max(0, frameCount - 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setScissorTest(true);
viewport.appendChild(renderer.domElement);

const center = new THREE.Vector3(
  (range[0] + range[3]) * 0.5,
  (range[1] + range[4]) * 0.5,
  (range[2] + range[5]) * 0.5);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1200);
camera.up.set(0, 0, 1);
camera.position.set(center.x + 80, center.y - 125, center.z + 58);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

function makeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c2cf, 0.92));
  const sun = new THREE.DirectionalLight(0xffffff, 0.72);
  sun.position.set(center.x - 30, center.y - 40, center.z + 120);
  scene.add(sun);
  const gridSize = Math.max(range[3] - range[0], range[4] - range[1]);
  const grid = new THREE.GridHelper(gridSize, 32, 0x94a3b8, 0xdbe3ee);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, range[2]);
  scene.add(grid);
  function addAxis(start, end, color) {
    const material = new THREE.LineBasicMaterial({ color });
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    scene.add(new THREE.Line(geometry, material));
  }
  addAxis(
    new THREE.Vector3(range[0], 0, range[2]),
    new THREE.Vector3(range[3], 0, range[2]),
    0xef4444);
  addAxis(
    new THREE.Vector3(0, range[1], range[2]),
    new THREE.Vector3(0, range[4], range[2]),
    0x22c55e);
  addAxis(
    new THREE.Vector3(0, 0, range[2]),
    new THREE.Vector3(0, 0, range[5]),
    0x2563eb);
  return scene;
}

const gtScene = makeScene();
const predScene = makeScene();
const pointsScene = makeScene();
const cubeGeometry = new THREE.BoxGeometry(
  voxelSize[0] * scale, voxelSize[1] * scale, voxelSize[2] * scale);
const dummy = new THREE.Object3D();
let gtMeshes = [];
let predMeshes = [];
let rightObjects = [];
let activeFrame = 0;
let timer = null;

function labelColor(label) {
  return DATA.labelColors[String(label)] || '#94a3b8';
}

function labelName(label) {
  return DATA.labelNames[String(label)] || `class ${label}`;
}

function decodeFlat(flat) {
  const yz = occSize[1] * occSize[2];
  const x = Math.floor(flat / yz);
  const rem = flat - x * yz;
  const y = Math.floor(rem / occSize[2]);
  const z = rem - y * occSize[2];
  return [x, y, z];
}

function voxelCenter(index) {
  return [
    range[0] + (index[0] + 0.5) * voxelSize[0],
    range[1] + (index[1] + 0.5) * voxelSize[1],
    range[2] + (index[2] + 0.5) * voxelSize[2],
  ];
}

function disposeMeshes(meshes, scene, disposeGeometry) {
  for (const mesh of meshes) {
    scene.remove(mesh);
    if (disposeGeometry && mesh.geometry) {
      mesh.geometry.dispose();
    }
    if (mesh.material) {
      mesh.material.dispose();
    }
  }
  meshes.length = 0;
}

function addLabelMesh(scene, meshes, label, flats) {
  if (!flats || flats.length === 0) {
    return;
  }
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(labelColor(label)),
    roughness: 0.82,
    metalness: 0.02
  });
  const mesh = new THREE.InstancedMesh(cubeGeometry, material, flats.length);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < flats.length; i++) {
    const center = voxelCenter(decodeFlat(flats[i]));
    dummy.position.set(center[0], center[1], center[2]);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  meshes.push(mesh);
}

function addFrameToOccScene(scene, meshes, frame) {
  const labels = Object.keys(frame.labels || {})
    .map((label) => Number(label))
    .sort((a, b) => a - b);
  for (const label of labels) {
    addLabelMesh(scene, meshes, label, frame.labels[String(label)]);
  }
}

function addPoints(pointsFrame) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(pointsFrame.points), 3));
  const material = new THREE.PointsMaterial({
    color: 0x111827,
    size: 0.09,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.78
  });
  const cloud = new THREE.Points(geometry, material);
  pointsScene.add(cloud);
  rightObjects.push(cloud);
}

function addBox(box) {
  const c = box.corners;
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];
  const positions = [];
  for (const edge of edges) {
    positions.push(...c[edge[0]], ...c[edge[1]]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(positions), 3));
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(labelColor(box.label)),
    linewidth: 1
  });
  const line = new THREE.LineSegments(geometry, material);
  pointsScene.add(line);
  rightObjects.push(line);
}

function addPointsBoxes(pointsFrame) {
  addPoints(pointsFrame);
  for (const box of pointsFrame.boxes || []) {
    addBox(box);
  }
}

function formatOccStats(prefix, frame) {
  const stats = frame.stats || {};
  const free = stats['0'] || 0;
  const ignore = stats['255'] || 0;
  let occupied = 0;
  for (const [key, value] of Object.entries(stats)) {
    if (key !== '0' && key !== '255') {
      occupied += value;
    }
  }
  const meta = frame.meta || {};
  const metaItems = [
    ['hit', 'num_hit_voxels'],
    ['free_rays', 'num_free_voxels'],
    ['sem', 'num_semantic_voxels'],
    ['ground', 'num_ground_voxels'],
    ['raw_obs', 'num_raw_obstacle_voxels'],
    ['obs', 'num_obstacle_voxels'],
    ['box_margin', 'num_box_margin_obstacle_voxels'],
    ['filtered_obs', 'num_filtered_obstacle_voxels'],
    ['ego_ignore', 'num_ego_ignore_voxels']
  ].filter((item) => meta[item[1]] !== undefined)
   .map((item) => `${item[0]}=${meta[item[1]]}`);
  let text = `${prefix}: raw_index=${frame.raw_index}<br>` +
    `occupied=${occupied} | free=${free} | ignore=${ignore}`;
  if (metaItems.length > 0) {
    text += `<br>${metaItems.join(' | ')}`;
  }
  return text;
}

function formatPointsStats(frame) {
  return `Points + boxes: raw_index=${frame.raw_index}<br>` +
    `points=${frame.numPoints}/${frame.totalPoints} | boxes=${frame.numBoxes}`;
}

function renderLegend(triple) {
  const labels = new Set();
  for (const frame of [triple.gt, triple.pred]) {
    for (const label of Object.keys(frame.labels || {})) {
      labels.add(Number(label));
    }
  }
  for (const box of triple.pointsBoxes.boxes || []) {
    labels.add(Number(box.label));
  }
  const items = Array.from(labels).sort((a, b) => a - b).map((label) =>
    `<div class="legend-item" title="${labelName(label)}">` +
    `<span class="swatch" style="background:${labelColor(label)}"></span>` +
    `<span class="legend-text">${label}: ${labelName(label)}</span></div>`);
  legendBox.innerHTML = items.join('');
}

function renderFrame(index) {
  if (frameCount === 0) {
    titleBox.textContent = 'No frames found';
    return;
  }
  activeFrame = Math.max(0, Math.min(frameCount - 1, index));
  const triple = triples[activeFrame];
  disposeMeshes(gtMeshes, gtScene, false);
  disposeMeshes(predMeshes, predScene, false);
  disposeMeshes(rightObjects, pointsScene, true);
  addFrameToOccScene(gtScene, gtMeshes, triple.gt);
  addFrameToOccScene(predScene, predMeshes, triple.pred);
  addPointsBoxes(triple.pointsBoxes);
  frameSlider.value = String(activeFrame);
  frameText.textContent = `${activeFrame + 1} / ${frameCount}`;
  titleBox.textContent =
    `raw_index=${triple.raw_index} | token=${triple.gt.token || ''}`;
  gtStatsBox.innerHTML = formatOccStats('GT OCC', triple.gt);
  predStatsBox.innerHTML = formatOccStats('Pred OCC', triple.pred);
  pointsStatsBox.innerHTML = formatPointsStats(triple.pointsBoxes);
  renderLegend(triple);
}

function draw() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const leftWidth = Math.floor(width / 3);
  const middleWidth = Math.floor(width / 3);
  const rightWidth = width - leftWidth - middleWidth;
  renderer.clear();

  camera.aspect = leftWidth / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(0, 0, leftWidth, height);
  renderer.setScissor(0, 0, leftWidth, height);
  renderer.render(gtScene, camera);

  camera.aspect = middleWidth / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(leftWidth, 0, middleWidth, height);
  renderer.setScissor(leftWidth, 0, middleWidth, height);
  renderer.render(predScene, camera);

  camera.aspect = rightWidth / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(leftWidth + middleWidth, 0, rightWidth, height);
  renderer.setScissor(leftWidth + middleWidth, 0, rightWidth, height);
  renderer.render(pointsScene, camera);
}

frameSlider.addEventListener('input', () => {
  renderFrame(Number(frameSlider.value));
});

playButton.addEventListener('click', () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
    playButton.textContent = 'Play';
    return;
  }
  playButton.textContent = 'Pause';
  timer = setInterval(() => {
    renderFrame((activeFrame + 1) % frameCount);
  }, 650);
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight') {
    renderFrame((activeFrame + 1) % frameCount);
  }
  if (event.key === 'ArrowLeft') {
    renderFrame((activeFrame - 1 + frameCount) % frameCount);
  }
});

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  draw();
}

renderFrame(0);
animate();
</script>
</body>
</html>
"""
    return html.replace('__DATA_JSON__', data_json)


def build_pred_points_boxes_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(',', ':'))
    html = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KL occupancy Pred / points boxes check</title>
<style>
html, body {
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #f8fafc;
  font-family: Arial, Helvetica, sans-serif;
}
#viewport {
  position: fixed;
  inset: 0;
}
.divider {
  position: fixed;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: rgba(15, 23, 42, 0.22);
  pointer-events: none;
}
.side-label {
  position: fixed;
  top: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
  pointer-events: none;
}
.side-label.pred {
  left: 14px;
}
.side-label.points {
  left: calc(50% + 14px);
}
.panel {
  position: fixed;
  left: 14px;
  right: 14px;
  bottom: 12px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: end;
  pointer-events: none;
}
.box {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
  color: #0f172a;
  padding: 8px 10px;
  min-width: 0;
  box-sizing: border-box;
  pointer-events: auto;
}
.controls {
  width: min(500px, 42vw);
}
.title {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
}
.row {
  display: flex;
  align-items: center;
  gap: 8px;
}
button {
  width: 36px;
  height: 30px;
  border: 1px solid rgba(15, 23, 42, 0.22);
  border-radius: 6px;
  background: #ffffff;
  color: #0f172a;
  cursor: pointer;
  font-size: 14px;
}
input[type="range"] {
  flex: 1;
  min-width: 0;
}
.stats {
  font-size: 12px;
  line-height: 1.45;
  color: #334155;
  word-break: break-word;
}
.legend {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 4px 10px;
  font-size: 12px;
  margin-top: 6px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.legend-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.error {
  position: fixed;
  inset: 20px;
  display: none;
  align-items: center;
  justify-content: center;
  color: #991b1b;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(153, 27, 27, 0.25);
  border-radius: 8px;
  font-size: 14px;
  padding: 20px;
  box-sizing: border-box;
}
@media (max-width: 900px) {
  .panel {
    grid-template-columns: minmax(0, 1fr);
  }
  .controls {
    width: auto;
  }
  .legend {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
</head>
<body>
<div id="viewport"></div>
<div class="divider"></div>
<div class="side-label pred">Pred OCC</div>
<div class="side-label points">Points + GT Boxes</div>
<div class="panel">
  <div id="predStats" class="box stats"></div>
  <div class="box controls">
    <div id="title" class="title"></div>
    <div class="row">
      <button id="play" title="Play or pause">Play</button>
      <input id="frame" type="range" min="0" value="0">
      <span id="frameText"></span>
    </div>
    <div id="legend" class="legend"></div>
  </div>
  <div id="pointsStats" class="box stats"></div>
</div>
<div id="error" class="error"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA_JSON__;
const viewport = document.getElementById('viewport');
const errorBox = document.getElementById('error');
const frameSlider = document.getElementById('frame');
const playButton = document.getElementById('play');
const frameText = document.getElementById('frameText');
const titleBox = document.getElementById('title');
const predStatsBox = document.getElementById('predStats');
const pointsStatsBox = document.getElementById('pointsStats');
const legendBox = document.getElementById('legend');

if (!window.THREE || !THREE.OrbitControls) {
  errorBox.style.display = 'flex';
  errorBox.textContent = 'Three.js failed to load. Check network access.';
  throw new Error('Three.js failed to load.');
}

const pairs = DATA.predPointsBoxPairs || [];
const range = DATA.pointCloudRange;
const occSize = DATA.occSize;
const voxelSize = DATA.voxelSize;
const scale = DATA.cubeScale;
const frameCount = pairs.length;
frameSlider.max = Math.max(0, frameCount - 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setScissorTest(true);
viewport.appendChild(renderer.domElement);

const center = new THREE.Vector3(
  (range[0] + range[3]) * 0.5,
  (range[1] + range[4]) * 0.5,
  (range[2] + range[5]) * 0.5);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1200);
camera.up.set(0, 0, 1);
camera.position.set(center.x + 80, center.y - 125, center.z + 58);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

function makeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c2cf, 0.92));
  const sun = new THREE.DirectionalLight(0xffffff, 0.72);
  sun.position.set(center.x - 30, center.y - 40, center.z + 120);
  scene.add(sun);
  const gridSize = Math.max(range[3] - range[0], range[4] - range[1]);
  const grid = new THREE.GridHelper(gridSize, 32, 0x94a3b8, 0xdbe3ee);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, range[2]);
  scene.add(grid);
  function addAxis(start, end, color) {
    const material = new THREE.LineBasicMaterial({ color });
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    scene.add(new THREE.Line(geometry, material));
  }
  addAxis(
    new THREE.Vector3(range[0], 0, range[2]),
    new THREE.Vector3(range[3], 0, range[2]),
    0xef4444);
  addAxis(
    new THREE.Vector3(0, range[1], range[2]),
    new THREE.Vector3(0, range[4], range[2]),
    0x22c55e);
  addAxis(
    new THREE.Vector3(0, 0, range[2]),
    new THREE.Vector3(0, 0, range[5]),
    0x2563eb);
  return scene;
}

const predScene = makeScene();
const pointsScene = makeScene();
const cubeGeometry = new THREE.BoxGeometry(
  voxelSize[0] * scale, voxelSize[1] * scale, voxelSize[2] * scale);
const dummy = new THREE.Object3D();
let predMeshes = [];
let rightObjects = [];
let activeFrame = 0;
let timer = null;

function labelColor(label) {
  return DATA.labelColors[String(label)] || '#94a3b8';
}

function labelName(label) {
  return DATA.labelNames[String(label)] || `class ${label}`;
}

function decodeFlat(flat) {
  const yz = occSize[1] * occSize[2];
  const x = Math.floor(flat / yz);
  const rem = flat - x * yz;
  const y = Math.floor(rem / occSize[2]);
  const z = rem - y * occSize[2];
  return [x, y, z];
}

function voxelCenter(index) {
  return [
    range[0] + (index[0] + 0.5) * voxelSize[0],
    range[1] + (index[1] + 0.5) * voxelSize[1],
    range[2] + (index[2] + 0.5) * voxelSize[2],
  ];
}

function disposeMeshes(meshes, scene, disposeGeometry) {
  for (const mesh of meshes) {
    scene.remove(mesh);
    if (disposeGeometry && mesh.geometry) {
      mesh.geometry.dispose();
    }
    if (mesh.material) {
      mesh.material.dispose();
    }
  }
  meshes.length = 0;
}

function addLabelMesh(scene, meshes, label, flats) {
  if (!flats || flats.length === 0) {
    return;
  }
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(labelColor(label)),
    roughness: 0.82,
    metalness: 0.02
  });
  const mesh = new THREE.InstancedMesh(cubeGeometry, material, flats.length);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < flats.length; i++) {
    const center = voxelCenter(decodeFlat(flats[i]));
    dummy.position.set(center[0], center[1], center[2]);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  meshes.push(mesh);
}

function addFrameToOccScene(scene, meshes, frame) {
  const labels = Object.keys(frame.labels || {})
    .map((label) => Number(label))
    .sort((a, b) => a - b);
  for (const label of labels) {
    addLabelMesh(scene, meshes, label, frame.labels[String(label)]);
  }
}

function addPoints(pointsFrame) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(pointsFrame.points), 3));
  const material = new THREE.PointsMaterial({
    color: 0x111827,
    size: 0.09,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.78
  });
  const cloud = new THREE.Points(geometry, material);
  pointsScene.add(cloud);
  rightObjects.push(cloud);
}

function addBox(box) {
  const c = box.corners;
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];
  const positions = [];
  for (const edge of edges) {
    positions.push(...c[edge[0]], ...c[edge[1]]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(positions), 3));
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(labelColor(box.label)),
    linewidth: 1
  });
  const line = new THREE.LineSegments(geometry, material);
  pointsScene.add(line);
  rightObjects.push(line);
}

function addPointsBoxes(pointsFrame) {
  addPoints(pointsFrame);
  for (const box of pointsFrame.boxes || []) {
    addBox(box);
  }
}

function formatPredStats(frame) {
  const stats = frame.stats || {};
  const free = stats['0'] || 0;
  const ignore = stats['255'] || 0;
  let occupied = 0;
  for (const [key, value] of Object.entries(stats)) {
    if (key !== '0' && key !== '255') {
      occupied += value;
    }
  }
  return `Pred OCC: raw_index=${frame.raw_index}<br>` +
    `occupied=${occupied} | free=${free} | ignore=${ignore}`;
}

function formatPointsStats(frame) {
  return `Points + boxes: raw_index=${frame.raw_index}<br>` +
    `points=${frame.numPoints}/${frame.totalPoints} | boxes=${frame.numBoxes}`;
}

function renderLegend(pair) {
  const labels = new Set();
  for (const label of Object.keys(pair.pred.labels || {})) {
    labels.add(Number(label));
  }
  for (const box of pair.pointsBoxes.boxes || []) {
    labels.add(Number(box.label));
  }
  const items = Array.from(labels).sort((a, b) => a - b).map((label) =>
    `<div class="legend-item" title="${labelName(label)}">` +
    `<span class="swatch" style="background:${labelColor(label)}"></span>` +
    `<span class="legend-text">${label}: ${labelName(label)}</span></div>`);
  legendBox.innerHTML = items.join('');
}

function renderFrame(index) {
  if (frameCount === 0) {
    titleBox.textContent = 'No frames found';
    return;
  }
  activeFrame = Math.max(0, Math.min(frameCount - 1, index));
  const pair = pairs[activeFrame];
  disposeMeshes(predMeshes, predScene, false);
  disposeMeshes(rightObjects, pointsScene, true);
  addFrameToOccScene(predScene, predMeshes, pair.pred);
  addPointsBoxes(pair.pointsBoxes);
  frameSlider.value = String(activeFrame);
  frameText.textContent = `${activeFrame + 1} / ${frameCount}`;
  titleBox.textContent =
    `raw_index=${pair.raw_index} | token=${pair.pred.token || ''}`;
  predStatsBox.innerHTML = formatPredStats(pair.pred);
  pointsStatsBox.innerHTML = formatPointsStats(pair.pointsBoxes);
  renderLegend(pair);
}

function draw() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const half = Math.floor(width / 2);
  renderer.clear();

  camera.aspect = half / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(0, 0, half, height);
  renderer.setScissor(0, 0, half, height);
  renderer.render(predScene, camera);

  const rightWidth = width - half;
  camera.aspect = rightWidth / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(half, 0, rightWidth, height);
  renderer.setScissor(half, 0, rightWidth, height);
  renderer.render(pointsScene, camera);
}

frameSlider.addEventListener('input', () => {
  renderFrame(Number(frameSlider.value));
});

playButton.addEventListener('click', () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
    playButton.textContent = 'Play';
    return;
  }
  playButton.textContent = 'Pause';
  timer = setInterval(() => {
    renderFrame((activeFrame + 1) % frameCount);
  }, 650);
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight') {
    renderFrame((activeFrame + 1) % frameCount);
  }
  if (event.key === 'ArrowLeft') {
    renderFrame((activeFrame - 1 + frameCount) % frameCount);
  }
});

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  draw();
}

renderFrame(0);
animate();
</script>
</body>
</html>
"""
    return html.replace('__DATA_JSON__', data_json)


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0:
        raise ValueError('--num-frames must be positive.')
    if not (0.0 < args.cube_scale <= 1.0):
        raise ValueError('--cube-scale must be in (0, 1].')
    if args.gt_pred_points_boxes and args.checkpoint is None:
        raise ValueError('--gt-pred-points-boxes requires --checkpoint.')
    if args.pred_points_boxes and args.checkpoint is None:
        raise ValueError('--pred-points-boxes requires --checkpoint.')

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    if args.compare_config is not None:
        compare_cfg = Config.fromfile(args.compare_config)
        compare_dataset = DATASETS.build(compare_cfg.train_dataloader.dataset)
        if args.start_index is None:
            raw_indices = find_sequence(compare_dataset, args.num_frames)
        else:
            raw_indices = collect_sequence(
                compare_dataset, args.start_index, args.num_frames)
        payload = build_compare_payload(
            cfg,
            compare_cfg,
            dataset,
            compare_dataset,
            raw_indices,
            args.cube_scale,
            args.max_voxels_per_label,
            args.compare_left_title,
            args.compare_right_title)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_split_html(payload), encoding='utf-8')
        print(out.resolve())
        print('raw_indices:', raw_indices)
        for pair in payload['framePairs']:
            print(pair['raw_index'], pair['gt']['stats'],
                  pair['pred']['stats'])
        return

    model = None
    if args.checkpoint is not None and not args.gt_points_boxes:
        model = init_model(args.config, args.checkpoint, device=args.device)
        model.eval()
    if args.start_index is None:
        raw_indices = find_sequence(dataset, args.num_frames)
    else:
        raw_indices = collect_sequence(
            dataset, args.start_index, args.num_frames)

    payload = build_payload(
        cfg,
        dataset,
        raw_indices,
        args.cube_scale,
        model=model,
        pred_only=args.pred_only,
        mask_pred_to_observed=args.mask_pred_to_observed,
        max_voxels_per_label=args.max_voxels_per_label,
        gt_points_boxes=args.gt_points_boxes,
        gt_pred_points_boxes=args.gt_pred_points_boxes,
        pred_points_boxes=args.pred_points_boxes,
        max_points_per_frame=args.max_points_per_frame)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if payload.get('gtPredPointsBoxTriples'):
        html = build_gt_pred_points_boxes_html(payload)
    elif payload.get('predPointsBoxPairs'):
        html = build_pred_points_boxes_html(payload)
    elif payload.get('gtPointsBoxPairs'):
        html = build_gt_points_boxes_html(payload)
    elif payload.get('framePairs'):
        html = build_split_html(payload)
    else:
        html = build_html(payload)
    out.write_text(html, encoding='utf-8')

    print(out.resolve())
    print('raw_indices:', raw_indices)
    for frame in payload['frames']:
        print(frame['raw_index'], frame['stats'])


if __name__ == '__main__':
    main()
