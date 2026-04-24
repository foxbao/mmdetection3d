"""Evaluate AP by distance range for each class."""
import argparse
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules
from mmdet3d.structures import LiDARInstance3DBoxes


def compute_ap(recalls, precisions):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def compute_ap_for_class(gt_list, pred_list, pcr, iou_thresh=0.5, dist_range=None):
    """Compute AP for one class.

    gt_list: list of dicts with 'boxes' (N,7) and 'dists' (N,)
    pred_list: list of dicts with 'boxes' (M,7) and 'scores' (M,)
    pcr: [xmin, ymin, xmax, ymax] rectangular range filter (matches
        ObjectRangeFilter in training pipeline)
    """
    all_scores = []
    all_tp = []
    n_gt_total = 0

    for i in range(len(gt_list)):
        gt_boxes = gt_list[i]['boxes']
        gt_dists = gt_list[i]['dists']
        pred_boxes = pred_list[i]['boxes']
        pred_scores = pred_list[i]['scores']

        # Filter GT by point_cloud_range (mimic ObjectRangeFilter)
        if len(gt_boxes) > 0:
            m = ((gt_boxes[:, 0] >= pcr[0]) & (gt_boxes[:, 0] <= pcr[2]) &
                 (gt_boxes[:, 1] >= pcr[1]) & (gt_boxes[:, 1] <= pcr[3]))
            gt_boxes = gt_boxes[m]
            gt_dists = gt_dists[m]

        # Filter GT by distance
        if dist_range is not None and len(gt_boxes) > 0:
            mask = (gt_dists >= dist_range[0]) & (gt_dists < dist_range[1])
            gt_boxes = gt_boxes[mask]

        # Filter pred by distance
        if dist_range is not None and len(pred_boxes) > 0:
            pred_dists = np.sqrt(pred_boxes[:, 0]**2 + pred_boxes[:, 1]**2)
            mask = (pred_dists >= dist_range[0]) & (pred_dists < dist_range[1])
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]

        n_gt = len(gt_boxes)
        n_gt_total += n_gt

        if len(pred_scores) == 0:
            continue

        if n_gt == 0:
            for s in pred_scores:
                all_scores.append(s)
                all_tp.append(0)
            continue

        # Compute 3D IoU
        # GT raw boxes use gravity center (origin=(0.5,0.5,0.5))
        # Pred boxes use bottom center (origin=(0.5,0.5,0))
        gt_b = LiDARInstance3DBoxes(gt_boxes, origin=(0.5, 0.5, 0.5))
        pred_b = LiDARInstance3DBoxes(pred_boxes, origin=(0.5, 0.5, 0))
        iou_matrix = gt_b.overlaps(gt_b, pred_b).numpy()  # (n_gt, n_pred)

        # Sort by score
        sort_idx = np.argsort(-pred_scores)
        pred_scores = pred_scores[sort_idx]
        iou_matrix = iou_matrix[:, sort_idx]

        matched_gt = set()
        for j in range(len(pred_scores)):
            all_scores.append(pred_scores[j])
            best_gt = np.argmax(iou_matrix[:, j])
            if iou_matrix[best_gt, j] >= iou_thresh and best_gt not in matched_gt:
                all_tp.append(1)
                matched_gt.add(best_gt)
            else:
                all_tp.append(0)

    if n_gt_total == 0:
        return float('nan'), 0

    if len(all_scores) == 0:
        return 0.0, n_gt_total

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)

    sort_idx = np.argsort(-all_scores)
    all_tp = all_tp[sort_idx]

    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)
    recalls = cum_tp / n_gt_total
    precisions = cum_tp / (cum_tp + cum_fp)

    return compute_ap(recalls, precisions), n_gt_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    args = parser.parse_args()

    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = './work_dirs/eval_by_distance_tmp'
    cfg.load_from = args.checkpoint

    # Read XY range from config's point_cloud_range so this script stays
    # in sync with the training/eval range without manual edits.
    full_pcr = cfg.point_cloud_range  # [xmin, ymin, zmin, xmax, ymax, zmax]
    pcr = [full_pcr[0], full_pcr[1], full_pcr[3], full_pcr[4]]
    print(f'Using point_cloud_range XY: {pcr}')

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    runner.model.eval()

    class_names = cfg.class_names
    n_classes = len(class_names)

    # Collect per-frame, per-class GT and predictions
    gt_all = {c: [] for c in range(n_classes)}
    pred_all = {c: [] for c in range(n_classes)}

    print(f'Running inference...')
    n_batches = len(runner.val_dataloader)
    with torch.no_grad():
        for i, batch in enumerate(runner.val_dataloader):
            results = runner.model.test_step(batch)

            for r in results:
                # --- Extract GT from eval_ann_info ---
                ea = r.eval_ann_info
                gt_labels = np.array(ea['gt_labels_3d'])
                gt_instances = ea['instances']

                gt_boxes_per_class = {}
                for c in range(n_classes):
                    gt_boxes_per_class[c] = {'boxes': np.zeros((0, 7)), 'dists': np.array([])}

                for idx, inst in enumerate(gt_instances):
                    label = gt_labels[idx]
                    bbox = np.array(inst['bbox_3d'][:7], dtype=np.float32)
                    dist = np.sqrt(bbox[0]**2 + bbox[1]**2)
                    if gt_boxes_per_class[label]['boxes'].shape[0] == 0:
                        gt_boxes_per_class[label]['boxes'] = bbox.reshape(1, 7)
                        gt_boxes_per_class[label]['dists'] = np.array([dist])
                    else:
                        gt_boxes_per_class[label]['boxes'] = np.vstack([
                            gt_boxes_per_class[label]['boxes'], bbox.reshape(1, 7)])
                        gt_boxes_per_class[label]['dists'] = np.append(
                            gt_boxes_per_class[label]['dists'], dist)

                for c in range(n_classes):
                    gt_all[c].append(gt_boxes_per_class[c])

                # --- Extract predictions ---
                p = r.pred_instances_3d
                p_boxes = p.bboxes_3d.tensor.cpu().numpy()[:, :7]
                p_scores = p.scores_3d.cpu().numpy()
                p_labels = p.labels_3d.cpu().numpy()

                for c in range(n_classes):
                    mask = p_labels == c
                    pred_all[c].append({
                        'boxes': p_boxes[mask],
                        'scores': p_scores[mask],
                    })

            if (i + 1) % 100 == 0:
                print(f'  [{i+1}/{n_batches}]')

    print(f'Inference done.\n')

    # Compute AP by distance
    dist_ranges = [(0, 30), (30, 50), (50, 80), (80, 200), None]
    dist_names = ['0-30m', '30-50m', '50-80m', '80m+', 'ALL']

    header = f'{"Class":<20}'
    for dn in dist_names:
        header += f'{dn:>14}'
    print(header)
    print('-' * len(header))

    for c in range(n_classes):
        row = f'{class_names[c]:<20}'
        for dr in dist_ranges:
            ap, n_gt = compute_ap_for_class(
                gt_all[c], pred_all[c], pcr,
                iou_thresh=args.iou_thresh, dist_range=dr)
            if np.isnan(ap):
                row += f'{"N/A":>14}'
            else:
                row += f'{ap:>8.3f}({n_gt:>4})'
        print(row)


if __name__ == '__main__':
    main()
