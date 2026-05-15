"""Multi-object tracking evaluation metric for the KL dataset.

Computes AMOTA, AMOTP, MOTA, MOTP, IDS, FRAG, MT, ML using greedy
center-distance matching (same convention as nuScenes tracking eval).
No external MOT library required — only numpy.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import load
from mmengine.logging import print_log

from mmdet3d.registry import METRICS

TRACKING_CLASSES = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
]


@METRICS.register_module()
class KlTrackingMetric(BaseMetric):
    """AMOTA/IDS tracking metric for the KL dataset."""

    default_prefix = 'KL_track'

    def __init__(self,
                 ann_file: str,
                 class_names: Optional[List[str]] = None,
                 match_threshold: float = 2.0,
                 num_thresholds: int = 40,
                 evaluate_predicted_samples_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.class_names = class_names or TRACKING_CLASSES
        self.match_threshold = match_threshold
        self.num_thresholds = num_thresholds
        self.evaluate_predicted_samples_only = evaluate_predicted_samples_only

    # -------------------------------------------------------------- #
    # process: collect per-frame predictions
    # -------------------------------------------------------------- #

    def process(self, data_batch: dict,
                data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample.get('pred_track_instances_3d')
            if pred is None:
                pred = data_sample['pred_instances_3d']
            bboxes_3d = pred['bboxes_3d']
            if hasattr(bboxes_3d, 'gravity_center'):
                centers = bboxes_3d.gravity_center.cpu().numpy()
            else:
                centers = bboxes_3d.tensor.cpu().numpy()[:, :3]
            scores = pred['scores_3d'].cpu().numpy()
            labels = pred['labels_3d'].cpu().numpy()
            track_ids = pred.get('instance_id', None)
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy()
            else:
                track_ids = np.full(len(scores), -1, dtype=np.int64)
            self.results.append(dict(
                sample_idx=data_sample['sample_idx'],
                pred_centers=centers,
                pred_scores=scores,
                pred_labels=labels,
                pred_track_ids=track_ids,
            ))

    # -------------------------------------------------------------- #
    # compute_metrics: aggregate and evaluate
    # -------------------------------------------------------------- #

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        data_infos = load(self.ann_file)['data_list']
        pred_by_idx = {r['sample_idx']: r for r in results}
        if self.evaluate_predicted_samples_only:
            pred_indices = set(pred_by_idx)
            raw_count = len(data_infos)
            data_infos = [
                info for info in data_infos
                if info.get('sample_idx') in pred_indices
            ]
            print_log(
                'KL tracking eval is restricted to predicted samples: '
                f'{len(data_infos)}/{raw_count} frames.',
                logger='current')

        gt_by_idx = self._build_gt_lookup(data_infos)
        scenes = self._group_by_scene(data_infos)

        num_classes = len(self.class_names)
        thresholds = np.linspace(
            1.0 / self.num_thresholds, 1.0, self.num_thresholds)

        # Per-class, per-threshold accumulators.
        # Shape: [num_thresholds, num_classes] for tp/fp/fn/ids/motp_sum
        tp_all = np.zeros((self.num_thresholds, num_classes))
        fp_all = np.zeros((self.num_thresholds, num_classes))
        fn_all = np.zeros((self.num_thresholds, num_classes))
        ids_all = np.zeros((self.num_thresholds, num_classes))
        motp_sum = np.zeros((self.num_thresholds, num_classes))

        # For FRAG/MT/ML: track per-GT-track matched/total frame counts.
        # Key: (scene_token, cls_idx, gt_track_id) to avoid cross-scene collision.
        gt_track_stats: Dict[int, Dict[tuple, List[bool]]] = defaultdict(
            lambda: defaultdict(list))  # threshold_idx -> (scene, cls, tid) -> [matched?]

        for thr_idx, score_thr in enumerate(thresholds):
            for scene_token, frame_indices in scenes.items():
                prev_matches: Dict[int, Dict[int, int]] = {}  # class -> {gt_tid: pred_tid}
                for sample_idx in frame_indices:
                    gt = gt_by_idx.get(sample_idx)
                    pred = pred_by_idx.get(sample_idx)
                    if gt is None:
                        continue
                    if pred is None:
                        # All GT are FN.
                        for cls_idx in range(num_classes):
                            cls_mask = gt['labels'] == cls_idx
                            fn_all[thr_idx, cls_idx] += cls_mask.sum()
                        continue

                    score_mask = pred['pred_scores'] >= score_thr
                    p_centers = pred['pred_centers'][score_mask]
                    p_labels = pred['pred_labels'][score_mask]
                    p_tids = pred['pred_track_ids'][score_mask]

                    for cls_idx in range(num_classes):
                        p_cls = p_labels == cls_idx
                        g_cls = gt['labels'] == cls_idx
                        pc = p_centers[p_cls]
                        pt = p_tids[p_cls]
                        gc = gt['centers'][g_cls]
                        gt_t = gt['track_ids'][g_cls]

                        matches, dists = _greedy_match(
                            pc, gc, self.match_threshold)

                        n_match = len(matches)
                        tp_all[thr_idx, cls_idx] += n_match
                        fp_all[thr_idx, cls_idx] += len(pc) - n_match
                        fn_all[thr_idx, cls_idx] += len(gc) - n_match

                        if dists:
                            motp_sum[thr_idx, cls_idx] += sum(dists)

                        # IDS: check identity consistency.
                        cur_map = {}  # gt_tid -> pred_tid
                        prev_cls = prev_matches.get(cls_idx, {})
                        for (pi, gi), _ in zip(matches, dists):
                            g_tid = int(gt_t[gi])
                            p_tid = int(pt[pi])
                            cur_map[g_tid] = p_tid
                            if g_tid in prev_cls and prev_cls[g_tid] != p_tid:
                                ids_all[thr_idx, cls_idx] += 1

                        if cls_idx not in prev_matches:
                            prev_matches[cls_idx] = {}
                        prev_matches[cls_idx] = cur_map

                        # Track per-GT-track matched state for FRAG/MT/ML.
                        matched_gt_set = {int(gt_t[gi]) for _, gi in matches}
                        for gi_local in range(len(gc)):
                            g_tid = int(gt_t[gi_local])
                            key = (scene_token, cls_idx, g_tid)
                            gt_track_stats[thr_idx][key].append(
                                g_tid in matched_gt_set)

        # Compute per-class metrics.
        gt_counts = np.zeros(num_classes)
        for info in data_infos:
            for inst in info.get('instances', []):
                lbl = inst.get('bbox_label_3d', -1)
                if 0 <= lbl < num_classes:
                    gt_counts[lbl] += 1

        metrics = self._compute_amota(
            tp_all, fp_all, fn_all, ids_all, motp_sum, gt_counts, thresholds)

        # FRAG / MT / ML at the threshold with best overall MOTA.
        best_thr_idx = self._best_threshold_idx(
            tp_all, fp_all, fn_all, ids_all, gt_counts)
        frag, mt, ml = self._compute_frag_mt_ml(
            gt_track_stats[best_thr_idx])
        metrics['FRAG'] = frag
        metrics['MT'] = mt
        metrics['ML'] = ml
        metrics['IDS'] = int(ids_all[best_thr_idx].sum())

        self._log_results(metrics)
        return metrics

    # -------------------------------------------------------------- #
    # Internal helpers
    # -------------------------------------------------------------- #

    def _build_gt_lookup(self, data_infos):
        """Build sample_idx -> {centers, labels, track_ids} from pkl."""
        gt_by_idx = {}
        for info in data_infos:
            idx = info['sample_idx']
            instances = info.get('instances', [])
            if not instances:
                gt_by_idx[idx] = dict(
                    centers=np.zeros((0, 3), dtype=np.float32),
                    labels=np.zeros(0, dtype=np.int64),
                    track_ids=np.zeros(0, dtype=np.int64))
                continue
            centers = np.array(
                [inst['bbox_3d'][:3] for inst in instances], dtype=np.float32)
            labels = np.array(
                [inst['bbox_label_3d'] for inst in instances], dtype=np.int64)
            track_ids = np.array(
                [inst.get('track_id', -1) for inst in instances],
                dtype=np.int64)
            gt_by_idx[idx] = dict(
                centers=centers, labels=labels, track_ids=track_ids)
        return gt_by_idx

    def _group_by_scene(self, data_infos):
        """Group sample indices by scene_token, sorted by timestamp."""
        scene_frames = defaultdict(list)
        for info in data_infos:
            scene = info.get('scene_token', 'unknown')
            scene_frames[scene].append(
                (info['timestamp'], info['sample_idx']))
        scenes = {}
        for scene, frames in scene_frames.items():
            frames.sort(key=lambda x: x[0])
            scenes[scene] = [idx for _, idx in frames]
        return scenes

    def _compute_amota(self, tp_all, fp_all, fn_all, ids_all, motp_sum,
                       gt_counts, thresholds):
        """Compute AMOTA/AMOTP per class and overall."""
        num_classes = len(self.class_names)
        recall_levels = np.linspace(
            1.0 / self.num_thresholds, 1.0, self.num_thresholds)

        amota_per_class = np.full(num_classes, np.nan)
        amotp_per_class = np.full(num_classes, np.nan)

        for cls_idx in range(num_classes):
            if gt_counts[cls_idx] == 0:
                continue
            gt_c = gt_counts[cls_idx]
            tp = tp_all[:, cls_idx]
            fp = fp_all[:, cls_idx]
            fn = fn_all[:, cls_idx]
            ids = ids_all[:, cls_idx]
            motp_s = motp_sum[:, cls_idx]

            recall = tp / gt_c
            mota = 1.0 - (fp + fn + ids) / np.maximum(gt_c, 1)
            motp = np.where(
                tp > 0, motp_s / np.maximum(tp, 1), self.match_threshold)

            # Interpolate MOTA at recall levels.
            # Sort by recall ascending for interpolation.
            sort_idx = np.argsort(recall)
            recall_sorted = recall[sort_idx]
            mota_sorted = mota[sort_idx]
            motp_sorted = motp[sort_idx]

            mota_interp = np.interp(
                recall_levels, recall_sorted, mota_sorted,
                left=mota_sorted[0] if len(mota_sorted) > 0 else 0,
                right=mota_sorted[-1] if len(mota_sorted) > 0 else 0)
            motp_interp = np.interp(
                recall_levels, recall_sorted, motp_sorted,
                left=self.match_threshold, right=self.match_threshold)

            amota_per_class[cls_idx] = np.mean(np.maximum(mota_interp, 0))
            valid_motp = motp_interp[mota_interp > 0]
            amotp_per_class[cls_idx] = (
                np.mean(valid_motp) if len(valid_motp) > 0
                else self.match_threshold)

        metrics = {}
        valid_mask = ~np.isnan(amota_per_class)
        metrics['AMOTA'] = float(np.mean(amota_per_class[valid_mask])) \
            if valid_mask.any() else 0.0
        metrics['AMOTP'] = float(np.mean(amotp_per_class[valid_mask])) \
            if valid_mask.any() else self.match_threshold

        for cls_idx in range(num_classes):
            name = self.class_names[cls_idx]
            if not np.isnan(amota_per_class[cls_idx]):
                metrics[f'{name}_AMOTA'] = float(amota_per_class[cls_idx])
                metrics[f'{name}_AMOTP'] = float(amotp_per_class[cls_idx])
        return metrics

    def _best_threshold_idx(self, tp_all, fp_all, fn_all, ids_all, gt_counts):
        """Find threshold index with best overall MOTA."""
        total_gt = gt_counts.sum()
        if total_gt == 0:
            return 0
        tp = tp_all.sum(axis=1)
        fp = fp_all.sum(axis=1)
        fn = fn_all.sum(axis=1)
        ids = ids_all.sum(axis=1)
        mota = 1.0 - (fp + fn + ids) / total_gt
        return int(np.argmax(mota))

    def _compute_frag_mt_ml(self, track_stats):
        """Compute FRAG, MT, ML from per-track matched-frame lists."""
        if not track_stats:
            return 0, 0.0, 0.0
        frag = 0
        mt_count = 0
        ml_count = 0
        total_tracks = 0
        for gt_tid, matched_list in track_stats.items():
            if not matched_list:
                continue
            total_tracks += 1
            ratio = sum(matched_list) / len(matched_list)
            if ratio >= 0.8:
                mt_count += 1
            elif ratio <= 0.2:
                ml_count += 1
            # FRAG: count TRACKED->LOST->TRACKED transitions.
            was_tracked = False
            was_lost = False
            for m in matched_list:
                if m:
                    if was_lost and was_tracked:
                        frag += 1
                    was_tracked = True
                    was_lost = False
                else:
                    if was_tracked:
                        was_lost = True
        mt = mt_count / max(total_tracks, 1)
        ml = ml_count / max(total_tracks, 1)
        return frag, mt, ml

    def _log_results(self, metrics):
        """Print a summary table."""
        header = f'\n{"="*60}\nKL Tracking Evaluation Results\n{"="*60}'
        lines = [header]
        for key in ['AMOTA', 'AMOTP', 'IDS', 'FRAG', 'MT', 'ML']:
            if key in metrics:
                lines.append(f'  {key:12s}: {metrics[key]:.4f}')
        lines.append(f'{"─"*60}')
        lines.append(f'  {"Class":<20s} {"AMOTA":>8s} {"AMOTP":>8s}')
        for name in self.class_names:
            amota = metrics.get(f'{name}_AMOTA', float('nan'))
            amotp = metrics.get(f'{name}_AMOTP', float('nan'))
            if not np.isnan(amota):
                lines.append(f'  {name:<20s} {amota:>8.4f} {amotp:>8.4f}')
        lines.append(f'{"="*60}\n')
        print_log('\n'.join(lines), logger='current')


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #

def _greedy_match(pred_centers: np.ndarray, gt_centers: np.ndarray,
                  threshold: float):
    """Greedy matching by 3D center distance (nuScenes convention).

    Returns:
        matches: list of (pred_idx, gt_idx) tuples
        dists: list of matched distances
    """
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return [], []
    diff = pred_centers[:, None, :] - gt_centers[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Flatten and sort by distance.
    M, N = dist_matrix.shape
    flat_indices = np.argsort(dist_matrix, axis=None)
    matched_preds = set()
    matched_gts = set()
    matches = []
    dists = []
    for flat_idx in flat_indices:
        pi = int(flat_idx // N)
        gi = int(flat_idx % N)
        d = dist_matrix[pi, gi]
        if d > threshold:
            break
        if pi in matched_preds or gi in matched_gts:
            continue
        matches.append((pi, gi))
        dists.append(float(d))
        matched_preds.add(pi)
        matched_gts.add(gi)
        if len(matched_preds) == M or len(matched_gts) == N:
            break
    return matches, dists
