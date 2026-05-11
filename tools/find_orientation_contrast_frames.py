"""Find and render KL orientation contrast frames for two checkpoints.

The script scans the validation set, matches predictions from a baseline model
and an improved model to the same front-back symmetric GT boxes, ranks frames
where baseline yaw error is larger than ours, and saves side-by-side BEV PNGs.
"""

from __future__ import annotations

import argparse
import math
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mmengine
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


CJK_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if osp.exists(CJK_FONT_PATH):
    font_manager.fontManager.addfont(CJK_FONT_PATH)
CJK_FONT_PROP = font_manager.FontProperties(fname=CJK_FONT_PATH)
TITLE_FONT_PROP = font_manager.FontProperties(fname=CJK_FONT_PATH, size=27)
plt.rcParams['font.family'] = CJK_FONT_PROP.get_name()
plt.rcParams['axes.unicode_minus'] = False


CLASS_NAMES = (
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane')
SYM_CLASS_IDS = {2, 6, 14}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find BEV frames where base yaw is worse than ours.')
    parser.add_argument('--base-config', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--ours-config', required=True)
    parser.add_argument('--ours-checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--scan-limit', type=int, default=500)
    parser.add_argument('--scan-indices', type=int, nargs='*')
    parser.add_argument('--topk-render', type=int, default=12)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--match-dist-thr', type=float, default=2.0)
    parser.add_argument('--point-stride', type=int, default=1)
    parser.add_argument('--point-size', type=float, default=0.35)
    parser.add_argument('--point-alpha', type=float, default=0.70)
    parser.add_argument('--point-color', default='#d8f5ff')
    parser.add_argument('--crop-radius', type=float, default=18.0)
    parser.add_argument('--base-display-name', default='Conventional')
    parser.add_argument('--ours-display-name', default='Proposed')
    parser.add_argument('--figure-prefix', default='Case')
    parser.add_argument(
        '--show-heading-arrows',
        action='store_true',
        help='Draw heading arrows on boxes. Disabled by default for patent '
        'figures because pi-symmetric boxes may have equivalent opposite '
        'headings.')
    parser.add_argument(
        '--show-box-labels',
        action='store_true',
        help='Draw text labels at box centers. Disabled by default for cleaner '
        'patent comparison figures.')
    parser.add_argument(
        '--score-mode',
        default='pi',
        choices=['pi', 'raw'],
        help='Use pi-periodic yaw error or raw yaw error to rank frames.')
    parser.add_argument('--min-base-pi-err', type=float, default=0.35)
    parser.add_argument('--max-ours-pi-err', type=float, default=0.15)
    parser.add_argument('--min-base-raw-err', type=float, default=0.8)
    parser.add_argument('--max-ours-raw-err', type=float, default=0.35)
    parser.add_argument(
        '--render-fallback',
        action='store_true',
        help='Render top scored frames even if strict thresholds find none.')
    return parser.parse_args()


def import_cfg_modules(cfg: Config) -> None:
    register_all_modules(init_default_scope=True)
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)


def get_dataset_cfg(cfg: Config, split: str):
    if split == 'val':
        return cfg.val_dataloader.dataset
    return cfg.test_dataloader.dataset


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_diff_raw(a: float, b: float) -> float:
    return abs(wrap_to_pi(a - b))


def yaw_diff_pi(a: float, b: float) -> float:
    raw = yaw_diff_raw(a, b)
    return min(raw, abs(math.pi - raw))


def get_gt_from_info(info: dict) -> Dict[str, np.ndarray]:
    ann = info.get('eval_ann_info', {})
    if ann.get('gt_bboxes_3d') is not None:
        boxes = ann['gt_bboxes_3d']
        labels = ann['gt_labels_3d']
        boxes = to_numpy(boxes.tensor if hasattr(boxes, 'tensor') else boxes)
        labels = to_numpy(labels).astype(np.int64)
        return dict(boxes=boxes[:, :7], labels=labels)

    boxes = []
    labels = []
    for inst in info.get('instances', []):
        if not inst.get('bbox_3d_isvalid', True):
            continue
        boxes.append(np.asarray(inst['bbox_3d'], dtype=np.float32)[:7])
        labels.append(int(inst['bbox_label_3d']))
    if not boxes:
        return dict(
            boxes=np.zeros((0, 7), dtype=np.float32),
            labels=np.zeros((0, ), dtype=np.int64))
    return dict(
        boxes=np.stack(boxes).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64))


def pred_to_arrays(pred_instances, score_thr: float) -> Dict[str, np.ndarray]:
    if len(pred_instances) == 0:
        return dict(
            boxes=np.zeros((0, 7), dtype=np.float32),
            labels=np.zeros((0, ), dtype=np.int64),
            scores=np.zeros((0, ), dtype=np.float32))
    scores = to_numpy(pred_instances.scores_3d).astype(np.float32)
    keep = scores >= score_thr
    boxes = to_numpy(pred_instances.bboxes_3d.tensor).astype(np.float32)
    labels = to_numpy(pred_instances.labels_3d).astype(np.int64)
    return dict(boxes=boxes[keep, :7], labels=labels[keep], scores=scores[keep])


def match_pred(gt_box: np.ndarray, gt_label: int, pred: Dict[str, np.ndarray],
               dist_thr: float):
    if len(pred['boxes']) == 0:
        return None
    cls_mask = pred['labels'] == gt_label
    if not cls_mask.any():
        return None
    inds = np.where(cls_mask)[0]
    dists = np.linalg.norm(pred['boxes'][inds, :2] - gt_box[:2], axis=1)
    best_local = int(np.argmin(dists))
    if float(dists[best_local]) > dist_thr:
        return None
    idx = int(inds[best_local])
    return dict(
        index=idx,
        box=pred['boxes'][idx],
        score=float(pred['scores'][idx]),
        center_dist=float(dists[best_local]))


def lidar_xy_to_display(xy: np.ndarray) -> np.ndarray:
    disp = np.empty_like(xy, dtype=np.float32)
    disp[..., 0] = -xy[..., 1]
    disp[..., 1] = xy[..., 0]
    return disp


def box_corners_bev(box: np.ndarray) -> np.ndarray:
    x, y, _, l, w, _, yaw = [float(v) for v in box[:7]]
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = np.array([
        [l / 2, w / 2],
        [l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2, w / 2],
    ], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def draw_box(ax,
             box: np.ndarray,
             color: str,
             linewidth: float,
             label: Optional[str] = None,
             alpha: float = 1.0,
             draw_heading: bool = False,
             linestyle='-',
             label_corner: int = 0,
             label_offset=(0.0, 0.0),
             label_anchor: Optional[str] = None):
    corners = box_corners_bev(box)
    closed = np.concatenate([corners, corners[:1]], axis=0)
    closed_disp = lidar_xy_to_display(closed)
    ax.plot(
        closed_disp[:, 0],
        closed_disp[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        solid_capstyle='butt',
        dash_capstyle='butt')

    center = lidar_xy_to_display(box[None, :2])[0]
    if draw_heading:
        yaw = float(box[6])
        heading_len = min(max(float(box[3]) * 0.38, 1.2), 3.0)
        heading = lidar_xy_to_display(
            np.array([[math.cos(yaw), math.sin(yaw)]], dtype=np.float32))[0]
        ax.arrow(
            center[0],
            center[1],
            heading[0] * heading_len,
            heading[1] * heading_len,
            color=color,
            width=0.10,
            head_width=0.75,
            head_length=0.85,
            length_includes_head=True,
            alpha=alpha)
    if label:
        if label_anchor:
            xs = closed_disp[:4, 0]
            ys = closed_disp[:4, 1]
            box_anchor_xy = {
                'upper_left': (xs.min(), ys.max()),
                'upper_right': (xs.max(), ys.max()),
                'lower_left': (xs.min(), ys.min()),
                'lower_right': (xs.max(), ys.min()),
            }
            if label_anchor.startswith('corner_'):
                # Use the actual rotated-box corner nearest to the requested
                # outer corner. This keeps labels visually attached to the box.
                target_name = label_anchor[len('corner_'):]
                target_xy = np.array(box_anchor_xy[target_name],
                                     dtype=np.float32)
                corner_idx = int(
                    np.argmin(
                        np.linalg.norm(
                            closed_disp[:4] - target_xy[None, :], axis=1)))
                label_xy = closed_disp[corner_idx]
            else:
                label_xy = np.array(box_anchor_xy[label_anchor],
                                    dtype=np.float32)
        else:
            label_xy = closed_disp[int(label_corner) % 4]
        ax.text(
            label_xy[0] + float(label_offset[0]),
            label_xy[1] + float(label_offset[1]),
            label,
            color=color,
            fontsize=27,
            fontweight='bold',
            ha='left',
            va='bottom',
            bbox=dict(
                facecolor='black',
                edgecolor='none',
                alpha=0.75,
                pad=1.0))


def setup_axis(ax,
               cfg: Config,
               title: str,
               focus_box: Optional[np.ndarray] = None,
               crop_radius: float = 18.0):
    ax.set_facecolor('black')
    if focus_box is None:
        x_min, y_min, _, x_max, y_max, _ = [
            float(v) for v in cfg.point_cloud_range
        ]
        ax.set_xlim(-y_max, -y_min)
        ax.set_ylim(x_min, x_max)
    else:
        center = lidar_xy_to_display(focus_box[None, :2])[0]
        ax.set_xlim(center[0] - crop_radius, center[0] + crop_radius)
        ax.set_ylim(center[1] - crop_radius, center[1] + crop_radius)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, color='white', fontproperties=TITLE_FONT_PROP)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#888888')
    ax.grid(color='#444444', linestyle='--', linewidth=0.4, alpha=0.35)


def render_candidate(cfg: Config, data: dict, cand: dict, out_file: str):
    points = data['inputs']['points']
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    pts = points[:, :3]
    pts_disp = lidar_xy_to_display(pts[:, :2])
    crop_radius = float(cand.get('crop_radius', 18.0))
    crop_center = lidar_xy_to_display(cand['gt_box'][None, :2])[0]
    crop_pad = 1.0
    crop_mask = (
        (pts_disp[:, 0] >= crop_center[0] - crop_radius - crop_pad)
        & (pts_disp[:, 0] <= crop_center[0] + crop_radius + crop_pad)
        & (pts_disp[:, 1] >= crop_center[1] - crop_radius - crop_pad)
        & (pts_disp[:, 1] <= crop_center[1] + crop_radius + crop_pad))
    pts_disp = pts_disp[crop_mask][::cand['point_stride']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=180)
    fig.patch.set_facecolor('black')
    base_name = cand.get('base_display_name', 'Baseline')
    ours_name = cand.get('ours_display_name', 'Ours')
    base_title = (
        f"{base_name} | {cand['class_name']} | "
        f"π周期误差 {cand['base_pi_err']:.2f} rad")
    ours_title = (
        f"{ours_name} | π周期误差 {cand['ours_pi_err']:.2f} rad")
    setup_axis(axes[0], cfg, base_title, cand['gt_box'], crop_radius)
    setup_axis(axes[1], cfg, ours_title, cand['gt_box'], crop_radius)
    for ax in axes:
        ax.scatter(
            pts_disp[:, 0],
            pts_disp[:, 1],
            s=float(cand.get('point_size', 0.35)),
            c=str(cand.get('point_color', '#d8f5ff')),
            alpha=float(cand.get('point_alpha', 0.70)),
            linewidths=0)
        draw_box(
            ax,
            cand['gt_box'],
            'white',
            2.8,
            'GT',
            1.0,
            bool(cand.get('draw_heading', False)),
            linestyle='-',
            label_corner=0,
            label_offset=(0.06, 0.06),
            label_anchor='corner_upper_right')

    draw_box(
        axes[0],
        cand['base_box'],
        'white',
        3.6,
        'A',
        1.0,
        bool(cand.get('draw_heading', False)),
        linestyle=(0, (9, 5)),
        label_corner=2,
        label_offset=(0.15, 0.15))
    draw_box(
        axes[1],
        cand['ours_box'],
        'white',
        3.6,
        'B',
        1.0,
        bool(cand.get('draw_heading', False)),
        linestyle=(0, (8, 3, 2, 3)),
        label_corner=2,
        label_offset=(0.15, 0.15),
        label_anchor='lower_left')
    fig.tight_layout()
    fig.savefig(out_file, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def predict_one(model, data: dict):
    with torch.no_grad():
        return model.test_step(pseudo_collate([data]))[0].pred_instances_3d


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_cfg = Config.fromfile(args.base_config)
    ours_cfg = Config.fromfile(args.ours_config)
    import_cfg_modules(ours_cfg)

    dataset = DATASETS.build(get_dataset_cfg(ours_cfg, args.split))
    base_model = init_model(base_cfg, args.base_checkpoint, device=args.device)
    ours_model = init_model(ours_cfg, args.ours_checkpoint, device=args.device)

    if args.scan_indices:
        scan_indices = [
            idx for idx in args.scan_indices if 0 <= idx < len(dataset)
        ]
    else:
        n_scan = min(
            len(dataset), args.scan_limit if args.scan_limit > 0 else len(dataset))
        scan_indices = list(range(n_scan))
    candidates: List[dict] = []
    for idx in mmengine.track_iter_progress(scan_indices):
        info = dataset.get_data_info(idx)
        gt = get_gt_from_info(info)
        sym_gt_inds = [
            i for i, label in enumerate(gt['labels'])
            if int(label) in SYM_CLASS_IDS
        ]
        if not sym_gt_inds:
            continue

        data = dataset[idx]
        base_pred = pred_to_arrays(predict_one(base_model, data), args.score_thr)
        ours_pred = pred_to_arrays(predict_one(ours_model, data), args.score_thr)

        for gi in sym_gt_inds:
            gt_box = gt['boxes'][gi]
            gt_label = int(gt['labels'][gi])
            base_match = match_pred(gt_box, gt_label, base_pred,
                                    args.match_dist_thr)
            ours_match = match_pred(gt_box, gt_label, ours_pred,
                                    args.match_dist_thr)
            if base_match is None or ours_match is None:
                continue

            base_raw = yaw_diff_raw(base_match['box'][6], gt_box[6])
            ours_raw = yaw_diff_raw(ours_match['box'][6], gt_box[6])
            base_pi = yaw_diff_pi(base_match['box'][6], gt_box[6])
            ours_pi = yaw_diff_pi(ours_match['box'][6], gt_box[6])
            if args.score_mode == 'pi':
                score = base_pi - ours_pi
            else:
                score = base_raw - ours_raw
            candidates.append(
                dict(
                    index=int(idx),
                    token=info['token'],
                    class_id=gt_label,
                    class_name=CLASS_NAMES[gt_label],
                    gt_box=gt_box,
                    base_box=base_match['box'],
                    ours_box=ours_match['box'],
                    base_score=base_match['score'],
                    ours_score=ours_match['score'],
                    base_raw_err=float(base_raw),
                    ours_raw_err=float(ours_raw),
                    base_pi_err=float(base_pi),
                    ours_pi_err=float(ours_pi),
                    score=float(score),
                    score_mode=args.score_mode,
                    point_stride=int(args.point_stride),
                    point_size=float(args.point_size),
                    point_alpha=float(args.point_alpha),
                    point_color=str(args.point_color),
                    crop_radius=float(args.crop_radius),
                    base_display_name=str(args.base_display_name),
                    ours_display_name=str(args.ours_display_name),
                    figure_prefix=str(args.figure_prefix),
                    draw_heading=bool(args.show_heading_arrows),
                    draw_labels=bool(args.show_box_labels)))

    candidates.sort(key=lambda item: item['score'], reverse=True)
    if args.score_mode == 'pi':
        strict = [
            cand for cand in candidates
            if cand['base_pi_err'] >= args.min_base_pi_err
            and cand['ours_pi_err'] <= args.max_ours_pi_err
        ]
    else:
        strict = [
            cand for cand in candidates
            if cand['base_raw_err'] >= args.min_base_raw_err
            and cand['ours_raw_err'] <= args.max_ours_raw_err
        ]
    selected = strict[:args.topk_render]
    if not selected and args.render_fallback:
        selected = candidates[:args.topk_render]

    summary = []
    for rank, cand in enumerate(selected, start=1):
        data = dataset[cand['index']]
        out_file = osp.join(
            args.out_dir,
            f"rank{rank:02d}_{cand['class_name']}.png")
        render_candidate(ours_cfg, data, cand, out_file)
        row = {
            k: v for k, v in cand.items()
            if k not in {'gt_box', 'base_box', 'ours_box'}
        }
        row['out_file'] = out_file
        summary.append(row)
        print(f"[OK] {out_file}")

    mmengine.dump(
        dict(num_candidates=len(candidates), selected=summary),
        osp.join(args.out_dir, 'summary.json'),
        indent=2)
    print(f'Found {len(candidates)} matched symmetric candidates; '
          f'rendered {len(selected)} images.')


if __name__ == '__main__':
    main()
