# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
import torch
from mmengine import Config, load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)


from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionMetricDataList
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.common.data_classes import EvalBox
import time

import json
import os
import pickle
import tqdm
from mmdet3d.datasets.kl_dataset import KlDataset
from easydict import EasyDict as edict
from nuscenes.eval.common.utils import center_distance, yaw_diff, \
    velocity_l2, scale_iou, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData
from collections import defaultdict

def accumulate_pi_symmetric(gt_boxes, pred_boxes, class_name, dist_fcn,
                            dist_th, pi_symmetric_classes=None, verbose=False):
    """Same as nuscenes accumulate(), but uses period=π for orient_err
    on classes with front-back symmetry.

    Args:
        pi_symmetric_classes: iterable of class names whose orientation
            should be evaluated modulo π (front-back symmetric). 'barrier'
            is always treated as π-symmetric for nuScenes compatibility.
            If None, no class is treated as π-symmetric (except 'barrier').
    """
    if pi_symmetric_classes is None:
        pi_symmetric_classes = set()
    else:
        pi_symmetric_classes = set(pi_symmetric_classes)

    npos = len([1 for gt_box in gt_boxes.all
                if gt_box.detection_name == class_name])
    if npos == 0:
        return DetectionMetricData.no_predictions()

    pred_boxes_list = [box for box in pred_boxes.all
                       if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]
    sortind = [i for (v, i) in sorted(
        (v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp = []
    fp = []
    conf = []
    match_data = {'trans_err': [], 'vel_err': [], 'scale_err': [],
                  'orient_err': [], 'attr_err': [], 'conf': []}

    # Use period=π for barrier and pi-symmetric classes
    if class_name in pi_symmetric_classes or class_name == 'barrier':
        orient_period = np.pi
    else:
        orient_period = 2 * np.pi

    taken = set()
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            if (gt_box.detection_name == class_name
                    and (pred_box.sample_token, gt_idx) not in taken):
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        is_match = min_dist < dist_th
        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            match_data['trans_err'].append(
                center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(
                velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(
                1 - scale_iou(gt_box_match, pred_box))
            match_data['orient_err'].append(
                yaw_diff(gt_box_match, pred_box, period=orient_period))
            match_data['attr_err'].append(
                1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)
        else:
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    for key in match_data.keys():
        if key == 'conf':
            continue
        tmp = cummean(np.array(match_data[key]))
        match_data[key] = np.interp(
            conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    return DetectionMetricData(
        recall=rec, precision=prec, confidence=conf,
        trans_err=match_data['trans_err'],
        vel_err=match_data['vel_err'],
        scale_err=match_data['scale_err'],
        orient_err=match_data['orient_err'],
        attr_err=match_data['attr_err'])

class KlDevConfig:
    CLASS_MAPPING = KlDataset.METAINFO['classes']
    DETECT_RANGE = 80

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int,
                 point_cloud_range: Optional[List[float]] = None,
                 pi_symmetric_classes: Optional[List[str]] = None):
        # 找出在 class_range 中但不在 KlDataset 中的类
        extra = set(class_range.keys()) - set(KlDataset.METAINFO['classes'])
        # 找出在 KlDataset 中但不在 class_range 中的类
        missing = set(KlDataset.METAINFO['classes']) - set(class_range.keys())

        assert set(class_range.keys()) == set(KlDataset.METAINFO['classes']), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.point_cloud_range = point_cloud_range
        self.pi_symmetric_classes = list(pi_symmetric_classes) \
            if pi_symmetric_classes is not None else []

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight,
            'point_cloud_range': self.point_cloud_range,
            'pi_symmetric_classes': self.pi_symmetric_classes
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'],
                   content.get('point_cloud_range'),
                   content.get('pi_symmetric_classes'))

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)



class KlDetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self, cfg: KlDevConfig):

        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuScenes detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """
        # Summarize.
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            'nd_score': self.nd_score,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = KlDevConfig.deserialize(content['cfg'])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        return metrics

    def __eq__(self, other):
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq




class KlDevKit:
    _class_range = {name: KlDevConfig.DETECT_RANGE for name in KlDevConfig.CLASS_MAPPING}
    KLDEVKIT_CONFIG = edict({
        'class_names': KlDevConfig.CLASS_MAPPING,
        'class_range': _class_range,
        'dist_fcn': 'center_distance',
        'dist_th_tp': 2.0,
        'verbose': True,
        'dist_ths': [0.5, 1.0, 2.0, 4.0],
        'max_boxes_per_sample': 500,
        'mean_ap_weight': 5,
        'min_precision': 0.1,
        'min_recall': 0.1,
        'point_cloud_range': None,
        'pi_symmetric_classes': [],
    })

    def __init__(self, dataroot: str, mode: str = 'train'):
        self.dataroot = dataroot
        if mode == 'train':
            self.pkl_file = osp.join(dataroot, 'kl_infos_train.pkl')
        else:
            self.pkl_file = osp.join(dataroot, 'kl_infos_val.pkl')
        self.data = self.load_data()

    def load_data(self):
        if not osp.exists(self.pkl_file):
            raise FileNotFoundError(f"File not found: {self.pkl_file}")
        with open(self.pkl_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def filter_eval_boxes(self, 
                          eval_boxes: EvalBoxes,
                          max_dist: Dict[str, float],
                          point_cloud_range: Optional[List[float]] = None,
                          verbose: bool = False) -> EvalBoxes:
        class_field = 'detection_name'

        def in_point_cloud_range(box: EvalBox) -> bool:
            if point_cloud_range is None:
                return True
            x, y, z = box.translation
            return (point_cloud_range[0] <= x <= point_cloud_range[3]
                    and point_cloud_range[1] <= y <= point_cloud_range[4]
                    and point_cloud_range[2] <= z <= point_cloud_range[5])

        # Accumulators for number of filtered boxes.
        total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
        for ind, sample_token in enumerate(eval_boxes.sample_tokens):

            # Filter on distance first.
            total += len(eval_boxes[sample_token])
            if point_cloud_range is None:
                eval_boxes.boxes[sample_token] = [
                    box for box in eval_boxes[sample_token]
                    if box.ego_dist < max_dist[box.__getattribute__(class_field)]
                ]
            else:
                eval_boxes.boxes[sample_token] = [
                    box for box in eval_boxes[sample_token]
                    if in_point_cloud_range(box)
                ]
            dist_filter += len(eval_boxes[sample_token])

            # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
            eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
            point_filter += len(eval_boxes[sample_token])

        if verbose:
            print("=> Original number of boxes: %d" % total)
            print("=> After distance based filtering: %d" % dist_filter)
            print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
            print("=> After bike rack filtering: %d" % bike_rack_filter)

        return eval_boxes

def load_gt(klDevKit: KlDevKit, eval_split: str, box_cls):
    if eval_split not in ['train', 'val']:
        raise ValueError(f"Invalid eval_split: {eval_split}")
    
    data_list = klDevKit.data['data_list']
    all_annotations = EvalBoxes()
    for sample in tqdm.tqdm(data_list):
        sample_token = sample['token']
        sample_boxes = []
        for box in sample['instances']:
            box['sample_token'] = sample_token
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box['bbox_3d'][6])
            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=box['bbox_3d'][:3],
                    size=np.array(box['bbox_3d'][3:6])[[1, 0, 2]].tolist(),
                    rotation=quat.elements.tolist(),
                    velocity=box['velocity'],
                    num_pts=box['num_lidar_pts'] + box['num_radar_pts'],
                    detection_name=KlDevConfig.CLASS_MAPPING[box['bbox_label']],
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=KlDevConfig.CLASS_MAPPING[box['bbox_label']]
                )
            )
        all_annotations.add_boxes(sample_token, sample_boxes)
    return all_annotations



class KlDetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in KlDevConfig.CLASS_MAPPING, 'Error: Unknown detection_name %s' % detection_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])


class KlDetectionEval:
    def __init__(self,
                klDevKit: KlDevKit,
                config: KlDevConfig,
                result_path: str,
                eval_set: str,
                output_dir: str = None,
                verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.klDevKit = klDevKit
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, KlDetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.klDevKit, self.eval_set, KlDetectionBox)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."


        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = self.klDevKit.filter_eval_boxes(
            self.pred_boxes,
            self.cfg.class_range,
            point_cloud_range=self.cfg.point_cloud_range,
            verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = self.klDevKit.filter_eval_boxes(
            self.gt_boxes,
            self.cfg.class_range,
            point_cloud_range=self.cfg.point_cloud_range,
            verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens
    
    
    def evaluate(self) -> Tuple[Dict[str, float], List[DetectionMetricDataList]]:
        start_time = time.time()
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                callable = center_distance if self.cfg.dist_fcn == 'center_distance' else self.cfg.dist_fcn_callable
                md = accumulate_pi_symmetric(
                    self.gt_boxes, self.pred_boxes, class_name, callable,
                    dist_th, pi_symmetric_classes=self.cfg.pi_symmetric_classes)
                metric_data_list.set(class_name, dist_th, md)
        metrics = KlDetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list



    def main(self):
        metrics, metric_data_list = self.evaluate()
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err']))

        return metrics_summary
    
@METRICS.register_module()
class KlMetric(BaseMetric):
    """Nuscenes evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        eval_version (str): Configuration version of evaluation.
            Defaults to 'detection_cvpr_2019'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 detect_range: Optional[float] = None,
                 class_range: Optional[Dict[str, float]] = None,
                 point_cloud_range: Optional[List[float]] = None,
                 pi_symmetric_classes: Optional[List[str]] = None,
                 forecast_match_dist_thr: float = 2.0,
                 forecast_miss_thr: float = 2.0,
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 eval_version: str = 'detection_cvpr_2019',
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'NuScenes metric'
        super(KlMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only
        if self.format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]
        self.forecast_match_dist_thr = forecast_match_dist_thr
        self.forecast_miss_thr = forecast_miss_thr

        self.eval_version = eval_version
        # self.eval_detection_configs = config_factory(self.eval_version)
        kldevconfig_info = KlDevKit.KLDEVKIT_CONFIG
        if class_range is None:
            if detect_range is None:
                class_range = kldevconfig_info.class_range
            else:
                class_range = {
                    class_name: detect_range
                    for class_name in KlDevConfig.CLASS_MAPPING
                }
        self.eval_detection_configs = KlDevConfig(
            class_range=class_range,
            dist_fcn=kldevconfig_info.dist_fcn,
            dist_ths=kldevconfig_info.dist_ths,
            dist_th_tp=kldevconfig_info.dist_th_tp,
            min_recall=kldevconfig_info.min_recall,
            min_precision=kldevconfig_info.min_precision,
            max_boxes_per_sample=kldevconfig_info.max_boxes_per_sample,
            mean_ap_weight=kldevconfig_info.mean_ap_weight,
            point_cloud_range=point_cloud_range,
            pi_symmetric_classes=pi_symmetric_classes
        )

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['data_list']
        result_dict, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        for metric in self.metrics:
            if metric == 'bbox':
                ap_dict = self.kl_evaluate(
                    result_dict, classes=classes, metric=metric,
                    logger=logger)
                for result in ap_dict:
                    metric_dict[result] = ap_dict[result]
            elif metric == 'forecasting':
                forecast_dict = self.evaluate_forecasting(
                    results, classes=classes, logger=logger)
                metric_dict.update(forecast_dict)
            else:
                raise KeyError(f'Unsupported KL metric: {metric}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    @staticmethod
    def _to_numpy(data):
        """Convert tensor-like data to numpy without changing scalars."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _get_instance_label(instance: dict) -> int:
        if 'bbox_label_3d' in instance:
            return int(instance['bbox_label_3d'])
        if 'bbox_label' in instance:
            return int(instance['bbox_label'])
        return -1

    @staticmethod
    def _normalize_sample_idx(sample_idx):
        if isinstance(sample_idx, torch.Tensor):
            sample_idx = sample_idx.item()
        if isinstance(sample_idx, np.ndarray):
            sample_idx = sample_idx.item()
        return sample_idx

    def evaluate_forecasting(
            self,
            results: List[dict],
            classes: Optional[List[str]] = None,
            logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluate per-object 6-step trajectory forecasts.

        Predictions are matched to ground truth by current-frame center
        distance and class label. The reported ADE/FDE are computed on
        absolute future centers, i.e. current box center plus predicted/GT
        displacement, so current detection localization error is included.
        """
        sample_to_info = {}
        for idx, info in enumerate(self.data_infos):
            sample_idx = info.get('sample_idx', idx)
            sample_to_info[self._normalize_sample_idx(sample_idx)] = info

        ades = []
        fdes = []
        misses = []
        valid_steps = 0
        matched_count = 0
        gt_forecast_count = 0
        horizon_errors = defaultdict(list)
        class_errors = defaultdict(list)
        has_forecasting_pred = False

        for result in results:
            sample_idx = self._normalize_sample_idx(result['sample_idx'])
            info = sample_to_info.get(sample_idx, None)
            if info is None:
                continue

            gt_instances = info.get('instances', [])
            if len(gt_instances) == 0:
                continue

            gt_boxes = np.asarray([
                inst.get('bbox_3d', np.zeros(7, dtype=np.float32))
                for inst in gt_instances
            ], dtype=np.float32)
            gt_labels = np.asarray([
                self._get_instance_label(inst) for inst in gt_instances
            ], dtype=np.int64)

            gt_traj_list = []
            gt_mask_list = []
            max_steps = 0
            for inst in gt_instances:
                locs = np.asarray(
                    inst.get('gt_forecasting_locs',
                             np.zeros((0, 2), dtype=np.float32)),
                    dtype=np.float32)
                mask = np.asarray(
                    inst.get('gt_forecasting_mask',
                             np.zeros((0, ), dtype=np.bool_)),
                    dtype=np.bool_).reshape(-1)
                if locs.ndim != 2 or locs.shape[-1] != 2:
                    locs = np.zeros((0, 2), dtype=np.float32)
                steps = min(locs.shape[0], mask.shape[0])
                locs = locs[:steps]
                mask = mask[:steps]
                gt_traj_list.append(locs)
                gt_mask_list.append(mask)
                max_steps = max(max_steps, steps)

            if max_steps == 0:
                continue
            gt_trajs = np.zeros(
                (len(gt_instances), max_steps, 2), dtype=np.float32)
            gt_masks = np.zeros(
                (len(gt_instances), max_steps), dtype=np.bool_)
            for gt_idx, (locs, mask) in enumerate(
                    zip(gt_traj_list, gt_mask_list)):
                steps = locs.shape[0]
                gt_trajs[gt_idx, :steps] = locs
                gt_masks[gt_idx, :steps] = mask

            gt_has_forecast = gt_masks.any(axis=1)
            gt_forecast_count += int(gt_has_forecast.sum())

            pred = result['pred_instances_3d']
            if 'forecasting_3d' not in pred:
                continue
            has_forecasting_pred = True

            pred_boxes = self._to_numpy(pred['bboxes_3d'].tensor)
            pred_scores = self._to_numpy(pred['scores_3d'])
            pred_labels = self._to_numpy(pred['labels_3d']).astype(np.int64)
            pred_trajs = self._to_numpy(pred['forecasting_3d'])

            if pred_boxes.shape[0] == 0 or pred_trajs.shape[0] == 0:
                continue

            matched_gt = set()
            for pred_idx in np.argsort(-pred_scores):
                same_class_gt = np.where(
                    gt_labels == pred_labels[pred_idx])[0]
                candidates = [
                    gt_idx for gt_idx in same_class_gt
                    if gt_idx not in matched_gt and gt_has_forecast[gt_idx]
                ]
                if len(candidates) == 0:
                    continue

                dists = np.linalg.norm(
                    gt_boxes[candidates, :2] - pred_boxes[pred_idx, :2],
                    axis=1)
                best_pos = int(np.argmin(dists))
                gt_idx = candidates[best_pos]
                if dists[best_pos] > self.forecast_match_dist_thr:
                    continue

                mask = gt_masks[gt_idx].astype(bool)
                steps = min(pred_trajs.shape[1], gt_trajs.shape[1],
                            mask.shape[0])
                if steps == 0:
                    continue
                mask = mask[:steps]
                if not mask.any():
                    continue

                pred_abs = (pred_boxes[pred_idx, None, :2] +
                            pred_trajs[pred_idx, :steps, :2])
                gt_abs = (gt_boxes[gt_idx, None, :2] +
                          gt_trajs[gt_idx, :steps, :2])
                err = np.linalg.norm(pred_abs - gt_abs, axis=-1)
                valid_err = err[mask]

                ade = float(valid_err.mean())
                fde = float(valid_err[-1])
                ades.append(ade)
                fdes.append(fde)
                misses.append(float(fde > self.forecast_miss_thr))
                valid_steps += int(mask.sum())
                matched_count += 1
                matched_gt.add(gt_idx)

                label = int(gt_labels[gt_idx])
                if classes is not None and 0 <= label < len(classes):
                    class_errors[classes[label]].append(ade)
                for step_idx, step_err in enumerate(err[:steps]):
                    if mask[step_idx]:
                        horizon_errors[step_idx + 1].append(float(step_err))

        if logger is not None and not has_forecasting_pred:
            logger.warning('No forecasting_3d field found in predictions. '
                           'Forecasting metrics will be zero.')

        metric_prefix = 'pred_instances_3d_kl/forecast'
        detail = {
            f'{metric_prefix}/mADE':
            float(np.mean(ades)) if ades else 0.0,
            f'{metric_prefix}/mFDE':
            float(np.mean(fdes)) if fdes else 0.0,
            f'{metric_prefix}/MR_{self.forecast_miss_thr:g}m':
            float(np.mean(misses)) if misses else 0.0,
            f'{metric_prefix}/recall':
            float(matched_count / max(gt_forecast_count, 1)),
            f'{metric_prefix}/matched_count':
            float(matched_count),
            f'{metric_prefix}/gt_count':
            float(gt_forecast_count),
            f'{metric_prefix}/valid_steps':
            float(valid_steps),
        }

        for step_idx in sorted(horizon_errors):
            values = horizon_errors[step_idx]
            detail[f'{metric_prefix}/ADE_t{step_idx}'] = (
                float(np.mean(values)) if values else 0.0)

        for class_name in classes or []:
            values = class_errors.get(class_name, [])
            detail[f'{metric_prefix}/{class_name}_ADE'] = (
                float(np.mean(values)) if values else 0.0)

        return detail

    def kl_evaluate(self,
                     result_dict: dict,
                     metric: str = 'bbox',
                     classes: Optional[List[str]] = None,
                     logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in Nuscenes protocol.

        Args:
            result_dict (dict): Formatted results of the dataset.
            metric (str): Metrics to be evaluated. Defaults to 'bbox'.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        metric_dict = dict()
        for name in result_dict:
            print(f'Evaluating bboxes of {name}')
            ret_dict = self._evaluate_single(
                result_dict[name], classes=classes, result_name=name)
            metric_dict.update(ret_dict)
        return metric_dict

    def _evaluate_single(
            self,
            result_path: str,
            classes: Optional[List[str]] = None,
            result_name: str = 'pred_instances_3d') -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """


        output_dir = osp.join(*osp.split(result_path)[:-1])
        kl_devit = KlDevKit(self.data_root, mode='val')
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        kl_eval = KlDetectionEval(
            kl_devit,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        kl_eval.main()

        # record metrics
        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_kl'
        for name in classes:
            for k, v in metrics['label_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail

    def format_results(
        self,
        results: List[dict],
        classes: Optional[List[str]] = None,
        jsonfile_prefix: Optional[str] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the mmdet3d results to standard NuScenes json file.

        Args:
            results (List[dict]): Testing results of the dataset.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
            dict containing the json filepaths, ``tmp_dir`` is the temporal
            directory created for saving json files when ``jsonfile_prefix`` is
            not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]

        for name in results[0]:
            if 'pred' in name and '3d' in name and name[0] != '_':
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                box_type_3d = type(results_[0]['bboxes_3d'])
                if box_type_3d == LiDARInstance3DBoxes:
                    result_dict[name] = self._format_lidar_bbox(
                        results_, sample_idx_list, classes, tmp_file_)

        return result_dict, tmp_dir

    def get_attr_name(self, attr_idx: int, label_name: str) -> str:
        """Get attribute from predicted index.

        This is a workaround to predict attribute when the predicted velocity
        is not reliable. We map the predicted attribute index to the one in the
        attribute set. If it is consistent with the category, we will keep it.
        Otherwise, we will use the default attribute.

        Args:
            attr_idx (int): Attribute index.
            label_name (str): Predicted category name.

        Returns:
            str: Predicted attribute name.
        """
        # TODO: Simplify the variable name
        AttrMapping_rev2 = [
            'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
            'pedestrian.standing', 'pedestrian.sitting_lying_down',
            'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
        ]
        if label_name == 'car' or label_name == 'bus' \
            or label_name == 'truck' or label_name == 'trailer' \
                or label_name == 'construction_vehicle':
            if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
                AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
                    AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        elif label_name == 'pedestrian':
            if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
                AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
                    AttrMapping_rev2[attr_idx] == \
                    'pedestrian.sitting_lying_down':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        elif label_name == 'bicycle' or label_name == 'motorcycle':
            if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
                    AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        else:
            return self.DefaultAttribute[label_name]

    def _format_camera_bbox(self,
                            results: List[dict],
                            sample_idx_list: List[int],
                            classes: Optional[List[str]] = None,
                            jsonfile_prefix: Optional[str] = None) -> str:
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')

        # Camera types in Nuscenes datasets
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]

        CAM_NUM = 6

        for i, det in enumerate(mmengine.track_iter_progress(results)):

            sample_idx = sample_idx_list[i]

            frame_sample_idx = sample_idx // CAM_NUM
            camera_type_id = sample_idx % CAM_NUM

            if camera_type_id == 0:
                boxes_per_frame = []
                attrs_per_frame = []

            # need to merge results from images of the same sample
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_token = self.data_infos[frame_sample_idx]['token']
            camera_type = camera_types[camera_type_id]
            boxes, attrs = cam_nusc_box_to_global(
                self.data_infos[frame_sample_idx], boxes, attrs, classes,
                self.eval_detection_configs, camera_type)
            boxes_per_frame.extend(boxes)
            attrs_per_frame.extend(attrs)
            # Remove redundant predictions caused by overlap of images
            if (sample_idx + 1) % CAM_NUM != 0:
                continue
            boxes = global_nusc_box_to_cam(self.data_infos[frame_sample_idx],
                                           boxes_per_frame, classes,
                                           self.eval_detection_configs)
            cam_boxes3d, scores, labels = nusc_box_to_cam_box3d(boxes)
            # box nms 3d over 6 images in a frame
            # TODO: move this global setting into config
            nms_cfg = dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_pre=4096,
                nms_thr=0.05,
                score_thr=0.01,
                min_bbox_size=0,
                max_per_frame=500)
            nms_cfg = Config(nms_cfg)
            cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
            boxes3d = cam_boxes3d.tensor
            # generate attr scores from attr labels
            attrs = labels.new_tensor([attr for attr in attrs_per_frame])
            boxes3d, scores, labels, attrs = box3d_multiclass_nms(
                boxes3d,
                cam_boxes3d_for_nms,
                scores,
                nms_cfg.score_thr,
                nms_cfg.max_per_frame,
                nms_cfg,
                mlvl_attr_scores=attrs)
            cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
            det = bbox3d2result(cam_boxes3d, scores, labels, attrs)
            boxes, attrs = output_to_nusc_box(det)
            boxes, attrs = cam_nusc_box_to_global(
                self.data_infos[frame_sample_idx], boxes, attrs, classes,
                self.eval_detection_configs)

            for i, box in enumerate(boxes):
                name = classes[box.label]
                attr = self.get_attr_name(attrs[i], name)
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path

    def _format_lidar_bbox(self,
                           results: List[dict],
                           sample_idx_list: List[int],
                           classes: Optional[List[str]] = None,
                           jsonfile_prefix: Optional[str] = None) -> str:
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]['token']
            
            for i, box in enumerate(boxes):
                attr = classes[box.label]
                name = classes[box.label]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        # res_path = 'results_nusc.json'
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path


def output_to_nusc_box(
        detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
        NuScenesBoxes and attribute labels.
    """
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attr_labels' in detection:
        attrs = detection['attr_labels'].numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    elif isinstance(bbox3d, CameraInstance3DBoxes):
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        for i in range(len(bbox3d)):
            q1 = pyquaternion.Quaternion(
                axis=[0, 0, 1], radians=nus_box_yaw[i])
            q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes '
            'to standard NuScenesBoxes.')

    return box_list, attrs





def cam_nusc_box_to_global(
    info: dict,
    boxes: List[NuScenesBox],
    attrs: np.ndarray,
    classes: List[str],
    eval_configs: KlDevConfig,
    camera_type: str = 'CAM_FRONT',
) -> Tuple[List[NuScenesBox], List[int]]:
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        attrs (np.ndarray): Predicted attributes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.
        camera_type (str): Type of camera. Defaults to 'CAM_FRONT'.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], List[int]]: List of standard
        NuScenesBoxes in the global coordinate and attribute label.
    """
    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        # Move box to ego vehicle coord system
        cam2ego = np.array(info['images'][camera_type]['cam2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05, atol=1e-07))
        box.translate(cam2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
        attr_list.append(attr)
    return box_list, attr_list


def global_nusc_box_to_cam(info: dict, boxes: List[NuScenesBox],
                           classes: List[str],
                           eval_configs: KlDevConfig) -> List[NuScenesBox]:
    """Convert the box from global to camera coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`NuScenesBox`]: List of standard NuScenesBoxes in camera
        coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        ego2global = np.array(info['ego2global'])
        box.translate(-ego2global[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05,
                                    atol=1e-07).inverse)
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to camera coord system
        cam2ego = np.array(info['images']['CAM_FRONT']['cam2ego'])
        box.translate(-cam2ego[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05,
                                    atol=1e-07).inverse)
        box_list.append(box)
    return box_list


def nusc_box_to_cam_box3d(
    boxes: List[NuScenesBox]
) -> Tuple[CameraInstance3DBoxes, torch.Tensor, torch.Tensor]:
    """Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.

    Args:
        boxes (:obj:`List[NuScenesBox]`): List of predicted NuScenesBoxes.

    Returns:
        Tuple[:obj:`CameraInstance3DBoxes`, torch.Tensor, torch.Tensor]:
        Converted 3D bounding boxes, scores and labels.
    """
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[0::2] for b in boxes]).view(-1, 2)

    # convert nusbox to cambox convention
    dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
    rots = -rots

    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
    cam_boxes3d = CameraInstance3DBoxes(
        boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes]).cuda()
    labels = torch.LongTensor([b.label for b in boxes]).cuda()
    nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
    indices = labels.new_tensor(list(range(scores.shape[0])))
    nms_scores[indices, labels] = scores
    return cam_boxes3d, nms_scores, labels

if __name__ == "__main__":
    klDevit = KlDevKit('/media/cx/bak/model/mmdetection3d/data/kl', 'test')
    data = klDevit.data  # noqa: F841
    print(data['data_list'][0])
 
