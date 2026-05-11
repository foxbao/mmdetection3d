"""BEVFormer-style temporal queue loader for KL.

Builds a fixed-length queue by walking the explicit ``prev`` chain stored in
KL info files. A sample is only valid when all historical frames exist and
form a clean temporal segment, so broken links, missing tokens or cross-scene
history cause ``prepare_data`` to return ``None`` and let the dataloader
resample. The historical frames are passed through the same pipeline as the
current frame, then merged into a single sample that carries:

  * ``inputs['points']``         — current-frame point cloud (unchanged).
  * ``inputs['history_points']`` — list of length ``Q-1`` with older frames,
                                    oldest first.
  * ``data_samples.metainfo['queue_metas']`` — ``{0: meta_oldest, ...,
                                    Q-1: meta_current}`` with per-frame
                                    ``scene_token``, ``timestamp``,
                                    ``ego2global``, ``prev_bev_exists``
                                    and ``ego_motion_delta`` (4x4, prev→curr
                                    in ego frame). Boundary frame has
                                    ``prev_bev_exists=False`` and identity
                                    delta.

Boundary policy (Stage 2): strict temporal segments. The queue must be fully
recoverable from ``prev`` links inside one scene; otherwise the sample is
rejected before reaching the model.

Augmentation policy: matches BEVFormer v1 / UniAD — no BEV-frame aug
(``GlobalRotScaleTrans`` / ``RandomFlip3D``) in train_pipeline. If you ever
re-introduce BEV aug, you also need to (a) sync aug params across the queue
and (b) conjugate ``ego_motion_delta`` by the aug matrix. Otherwise the
temporal warp is silently misaligned by exactly the per-frame aug rotation.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from mmengine.dataset import Compose, force_full_init

from mmdet3d.datasets.kl_dataset import KlDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class KlBEVFormerDataset(KlDataset):

    def __init__(self,
                 *args,
                 queue_length: int = 4,
                 max_time_gap: float = 1.0,
                 post_pipeline: Optional[List[Union[dict, Callable]]] = None,
                 **kwargs) -> None:
        assert queue_length >= 1, f'queue_length must be >=1, got {queue_length}'
        assert max_time_gap > 0, f'max_time_gap must be >0, got {max_time_gap}'
        self.queue_length = queue_length
        self.max_time_gap = float(max_time_gap)
        self.post_pipeline = (
            Compose(post_pipeline) if post_pipeline is not None else None)
        self.token2index: Dict[str, int] = {}
        self.valid_data_indices: Optional[List[int]] = None
        self.raw_data_len = 0
        super().__init__(*args, **kwargs)

    def full_init(self) -> None:
        """Load dataset annotations and build a token lookup table."""
        if self._fully_initialized:
            return
        super().full_init()
        self.raw_data_len = len(self.data_address) if self.serialize_data else \
            len(self.data_list)
        self.token2index = {}
        for idx in range(self.raw_data_len):
            info = self._get_raw_data_info(idx)
            token = info.get('token')
            if token:
                self.token2index[token] = idx
        if self.test_mode:
            self.valid_data_indices = [
                idx for idx in range(self.raw_data_len)
                if self._collect_queue_indices(idx) is not None
            ]
        else:
            self.valid_data_indices = None

    @force_full_init
    def __len__(self) -> int:
        if self.valid_data_indices is not None:
            return len(self.valid_data_indices)
        return self.raw_data_len

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        return self._get_raw_data_info(self._to_raw_index(idx))

    def _to_raw_index(self, idx: int) -> int:
        if self.valid_data_indices is None:
            return idx
        if idx < 0:
            idx += len(self.valid_data_indices)
        return self.valid_data_indices[idx]

    def _get_raw_data_info(self, idx: int) -> dict:
        return super().get_data_info(idx)

    def prepare_data(self, index: int) -> Optional[dict]:
        """Assemble a queue of ``queue_length`` frames ending at ``index``."""
        indices = self._collect_queue_indices(self._to_raw_index(index))
        if indices is None:
            return None

        queue: List[dict] = []
        raw_meta: List[dict] = []
        for queue_idx, i in enumerate(indices):
            raw = self._get_raw_data_info(i)
            if raw is None:
                return None
            raw_meta.append(self._extract_raw_meta(raw))
            example = self._single_prepare(
                raw, is_current_frame=(queue_idx == len(indices) - 1))
            if example is None:
                return None
            queue.append(example)
        return self._union2one(queue, raw_meta)

    def _collect_queue_indices(self, index: int) -> Optional[List[int]]:
        """Collect a full queue by following explicit ``prev`` links."""
        current = self._get_raw_data_info(index)
        if current is None:
            return None

        indices = [index]
        scene_token = current.get('scene_token')
        cursor_info = current
        prev_token = current.get('prev', '')
        for _ in range(self.queue_length - 1):
            if not prev_token:
                return None
            prev_index = self.token2index.get(prev_token)
            if prev_index is None:
                return None

            prev_info = self._get_raw_data_info(prev_index)
            if prev_info is None:
                return None
            if prev_info.get('scene_token') != scene_token:
                return None
            dt = abs(float(cursor_info.get('timestamp', 0.0)) -
                     float(prev_info.get('timestamp', 0.0)))
            if dt > self.max_time_gap:
                return None

            indices.append(prev_index)
            cursor_info = prev_info
            prev_token = prev_info.get('prev', '')

        indices.reverse()
        return indices

    @staticmethod
    def _extract_raw_meta(raw: dict) -> dict:
        ego = raw.get('ego2global', np.eye(4))
        return {
            'scene_token': raw.get('scene_token'),
            'ego2global': np.asarray(ego, dtype=np.float64),
            'timestamp': float(raw.get('timestamp', 0.0)),
            'token': raw.get('token'),
        }

    def _single_prepare(self,
                        ori_input_dict: dict,
                        is_current_frame: bool = True) -> Optional[dict]:
        """Per-frame pipeline run, mirroring ``Det3DDataset.prepare_data``."""
        input_dict = copy.deepcopy(ori_input_dict)
        input_dict['box_type_3d'] = self.box_type_3d
        input_dict['box_mode_3d'] = self.box_mode_3d
        input_dict['_kl_is_current_frame'] = is_current_frame

        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)

        if not self.test_mode and self.filter_empty_gt:
            if example is None:
                return None
            labels = example['data_samples'].gt_instances_3d.labels_3d
            if len(labels) == 0:
                return None
        return example

    def _union2one(self, queue: List[dict],
                   raw_meta: List[dict]) -> dict:
        """Merge the per-frame outputs into a single sample."""
        assert len(queue) == len(raw_meta) == self.queue_length

        points_list = [frame['inputs']['points'] for frame in queue]
        history_points = points_list[:-1]
        current_points = points_list[-1]

        # Collect per-frame gt_instances_3d (used by track-mode detectors that
        # supervise every frame in the queue, not just the current frame).
        # Detection-mode detectors ignore this and read only the current
        # frame's gt_instances_3d as before. List is ordered oldest→newest;
        # the last entry is the current frame.
        queue_gt_instances_3d = [
            frame['data_samples'].gt_instances_3d
            if hasattr(frame['data_samples'], 'gt_instances_3d') else None
            for frame in queue
        ]

        queue_metas: Dict[int, dict] = {}
        prev_ego2global = None
        prev_timestamp = None
        for i, (frame, meta) in enumerate(zip(queue, raw_meta)):
            entry = {
                'scene_token': meta['scene_token'],
                'ego2global': meta['ego2global'],
                'timestamp': meta['timestamp'],
                'token': meta['token'],
            }
            ego2global = meta['ego2global']
            timestamp = meta['timestamp']

            if i == 0:
                entry['prev_bev_exists'] = False
                entry['ego_motion_delta'] = np.eye(4, dtype=np.float64)
                entry['time_delta'] = 0.0
            else:
                entry['prev_bev_exists'] = True
                entry['ego_motion_delta'] = (
                    np.linalg.inv(ego2global) @ prev_ego2global)
                entry['time_delta'] = float(timestamp - prev_timestamp)

            queue_metas[i] = entry
            prev_ego2global = ego2global
            prev_timestamp = timestamp

        sample = queue[-1]
        sample['inputs']['points'] = current_points
        sample['inputs']['history_points'] = history_points
        sample['data_samples'].set_metainfo(dict(
            queue_metas=queue_metas,
            queue_gt_instances_3d=queue_gt_instances_3d,
        ))
        if self.post_pipeline is not None:
            sample['_kl_history_data_samples'] = [
                frame['data_samples'] for frame in queue[:-1]
            ]
            sample = self.post_pipeline(sample)
            if sample is not None:
                sample.pop('_kl_history_data_samples', None)
        return sample
