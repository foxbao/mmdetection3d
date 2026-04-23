from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        temporal_fuser: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        motion_head: Optional[dict] = None,
        tracking_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg', None)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if voxelize_cfg is not None:
            self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        else:
            self.voxelize_reduce = False
            self.pts_voxel_layer = None

        self.pts_voxel_encoder = MODELS.build(
            pts_voxel_encoder) if pts_voxel_encoder is not None else None

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(
            pts_middle_encoder) if pts_middle_encoder is not None else None

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(
            pts_backbone) if pts_backbone is not None else None
        self.pts_neck = MODELS.build(
            pts_neck) if pts_neck is not None else None

        self.temporal_fuser = MODELS.build(
            temporal_fuser) if temporal_fuser is not None else None

        self.bbox_head = MODELS.build(bbox_head)
        self.tracking_head = MODELS.build(
            tracking_head) if tracking_head is not None else None
        self.motion_head = MODELS.build(
            motion_head) if motion_head is not None else None

        self.init_weights()

    def _cache_scene_context(self, feats):
        """Cache shared scene features for downstream tasks.

        This is the detector-agnostic interface we want future tracking,
        motion, and planning modules to consume.
        """
        bev_memory = feats[0] if isinstance(feats, (list, tuple)) else feats
        self._scene_context = dict(bev_memory=bev_memory)

    def _collect_scene_context(self):
        """Merge model-level scene features with detector-level context."""
        scene_context = dict(getattr(self, '_scene_context', {}))
        if hasattr(self.bbox_head, 'export_detection_context'):
            det_context = self.bbox_head.export_detection_context()
            for key, value in det_context.items():
                if value is not None:
                    scene_context[key] = value
        return scene_context

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = len(points)
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        temporal_kwargs = self._extract_temporal_kwargs(
            batch_inputs_dict, batch_input_metas)
        feats = self.extract_feat(
            batch_inputs_dict, batch_input_metas, **temporal_kwargs)
        self._cache_scene_context(feats)

        if self.with_bbox_head:
            if isinstance(self.bbox_head, CenterHead):
                outputs = self.bbox_head.predict(
                    [feats], batch_data_samples)
            else:
                outputs = self.bbox_head.predict(feats, batch_input_metas)

        scene_context = self._collect_scene_context()
        if self.tracking_head is not None:
            outputs = self.tracking_head.predict(outputs, scene_context)
        if self.motion_head is not None:
            self.bbox_head.predict_motion(
                outputs,
                self.motion_head,
                scene_context=scene_context)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def _extract_single_frame_bev(
        self,
        batch_inputs_dict,
        batch_input_metas,
    ):
        """Extract one-frame BEV features without temporal fusion."""
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        if self.pts_middle_encoder is not None:
            pts_feature = self.extract_pts_feat(batch_inputs_dict)
            features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        if self.pts_backbone is not None:
            x = self.pts_backbone(x)
        if self.pts_neck is not None:
            x = self.pts_neck(x)

        return x

    def _extract_prev_bev_from_inputs(self, batch_inputs_dict,
                                      batch_input_metas, current_bev):
        """Build a padded batch of prev_bev from per-sample prev_points."""
        prev_points = batch_inputs_dict.get('prev_points', None)
        batch_size = len(batch_input_metas)
        if prev_points is None:
            prev_bevs = current_bev.new_zeros(current_bev.shape)
            cold_mask = torch.ones(
                batch_size, dtype=torch.bool, device=current_bev.device)
            prev_ego2globals = [None] * batch_size
            prev_lidar_aug = [
                np.eye(4, dtype=np.float32) for _ in range(batch_size)
            ]
            prev_exists = torch.zeros(
                batch_size, dtype=torch.bool, device=current_bev.device)
            return (prev_bevs, cold_mask, prev_ego2globals, prev_lidar_aug,
                    prev_exists)

        if self.img_backbone is not None or batch_inputs_dict.get(
                'imgs', None) is not None:
            raise NotImplementedError(
                'prev_points-based prev_bev currently supports LiDAR-only '
                'BEVFusion.')

        valid_indices = []
        prev_ego2globals = []
        prev_lidar_aug = []
        for idx in range(batch_size):
            prev_pts = prev_points[idx]
            prev_e2g = batch_input_metas[idx].get('prev_ego2global', None)
            if prev_pts is None or prev_e2g is None:
                prev_ego2globals.append(None)
                prev_lidar_aug.append(np.eye(4, dtype=np.float32))
                continue
            valid_indices.append(idx)
            prev_ego2globals.append(prev_e2g)
            prev_lidar_aug.append(
                batch_input_metas[idx].get('lidar_aug_matrix', np.eye(4)))

        prev_bevs = current_bev.new_zeros(current_bev.shape)
        cold_mask = torch.ones(
            batch_size, dtype=torch.bool, device=current_bev.device)
        prev_exists = torch.zeros(
            batch_size, dtype=torch.bool, device=current_bev.device)
        if len(valid_indices) == 0:
            return (prev_bevs, cold_mask, prev_ego2globals, prev_lidar_aug,
                    prev_exists)

        prev_batch = {'points': [prev_points[idx] for idx in valid_indices]}
        prev_metas = []
        for idx in valid_indices:
            prev_meta = dict(batch_input_metas[idx])
            prev_meta['ego2global'] = batch_input_metas[idx]['prev_ego2global']
            prev_metas.append(prev_meta)

        with torch.no_grad():
            prev_bev = self._extract_single_frame_bev(prev_batch, prev_metas)

        if isinstance(prev_bev, (list, tuple)):
            prev_bev = prev_bev[0]

        prev_bevs[valid_indices] = prev_bev.detach()
        cold_mask[valid_indices] = False
        prev_exists[valid_indices] = True
        if any('prev_bev_exists' in meta for meta in batch_input_metas):
            meta_prev_exists = torch.tensor(
                [bool(meta.get('prev_bev_exists', False))
                 for meta in batch_input_metas],
                device=current_bev.device,
                dtype=torch.bool)
            prev_exists = prev_exists & meta_prev_exists
            cold_mask = cold_mask | (~prev_exists)
        return (prev_bevs, cold_mask, prev_ego2globals, prev_lidar_aug,
                prev_exists)

    def _extract_history_bev_from_queue_inputs(self, batch_inputs_dict,
                                               batch_input_metas, current_bev):
        """Recursively roll a short prev-points queue into one history BEV.

        Expected queue layout after pseudo-collate:
          prev_points_queue: [num_prev][batch]
        while the matching ego poses / existence flags stay sample-major in
        ``batch_input_metas`` as Python lists.
        """
        raw_queue = batch_inputs_dict.get('prev_points_queue', None)
        if raw_queue is None:
            return None

        if self.img_backbone is not None or batch_inputs_dict.get(
                'imgs', None) is not None:
            raise NotImplementedError(
                'prev_points_queue-based history BEV currently supports '
                'LiDAR-only BEVFusion.')

        num_prev = len(raw_queue)
        batch_size = len(batch_input_metas)
        if num_prev == 0 or batch_size == 0:
            return None

        device = current_bev.device
        lidar_aug_matrices = [
            meta.get('lidar_aug_matrix', np.eye(4))
            for meta in batch_input_metas
        ]
        lidar_coord_frame = (
            batch_input_metas[0].get('lidar_coord_frame', 'FLU')
            if batch_input_metas else 'FLU')

        history_bev = None
        history_ego2global = [None] * batch_size
        history_exists = torch.zeros(
            batch_size, dtype=torch.bool, device=device)

        for frame_idx in range(num_prev):
            frame_points = raw_queue[frame_idx]
            frame_ego2global = []
            frame_exists = []
            valid_indices = []
            for sample_idx in range(batch_size):
                meta = batch_input_metas[sample_idx]
                ego2global_queue = meta.get('prev_ego2global_queue', [])
                exists_queue = meta.get('prev_bev_exists_queue', [])
                frame_e2g = (ego2global_queue[frame_idx]
                             if frame_idx < len(ego2global_queue) else None)
                frame_exist = bool(exists_queue[frame_idx]) if (
                    frame_idx < len(exists_queue)) else False
                frame_pts = frame_points[sample_idx]
                if frame_pts is None or frame_e2g is None or not frame_exist:
                    frame_ego2global.append(None)
                    frame_exists.append(False)
                    continue
                frame_ego2global.append(frame_e2g)
                frame_exists.append(True)
                valid_indices.append(sample_idx)

            frame_bev = current_bev.new_zeros(current_bev.shape)
            frame_exists_tensor = torch.tensor(
                frame_exists, dtype=torch.bool, device=device)
            if valid_indices:
                frame_batch = dict(
                    points=[
                        frame_points[idx].to(device).contiguous()
                        for idx in valid_indices
                    ])
                frame_metas = []
                for idx in valid_indices:
                    frame_meta = dict(batch_input_metas[idx])
                    frame_meta['ego2global'] = frame_ego2global[idx]
                    frame_metas.append(frame_meta)
                with torch.no_grad():
                    valid_bev = self._extract_single_frame_bev(
                        frame_batch, frame_metas)
                if isinstance(valid_bev, (list, tuple)):
                    valid_bev = valid_bev[0]
                frame_bev[valid_indices] = valid_bev.detach()

            if history_bev is None:
                history_bev = frame_bev
                history_ego2global = frame_ego2global
                history_exists = frame_exists_tensor
                continue

            history_bev = self.temporal_fuser(
                curr_bev=frame_bev,
                prev_bev=history_bev,
                curr_ego2global=frame_ego2global,
                prev_ego2global=history_ego2global,
                curr_lidar_aug_matrix=lidar_aug_matrices,
                prev_lidar_aug_matrix=lidar_aug_matrices,
                cold_mask=~history_exists,
                prev_bev_exists=history_exists,
                lidar_coord_frame=lidar_coord_frame)
            history_ego2global = frame_ego2global
            history_exists = frame_exists_tensor

        if history_bev is None:
            return None

        cold_mask = ~history_exists
        prev_aug = [np.eye(4, dtype=np.float32) for _ in range(batch_size)]
        for idx in range(batch_size):
            if history_exists[idx]:
                prev_aug[idx] = np.asarray(
                    lidar_aug_matrices[idx], dtype=np.float32)
        return (history_bev, cold_mask, history_ego2global, prev_aug,
                history_exists)

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        x = self._extract_single_frame_bev(
            batch_inputs_dict, batch_input_metas)

        # ---------- temporal fusion (optional) ----------
        if self.temporal_fuser is not None:
            was_list = isinstance(x, (list, tuple))
            x_tensor = x[0] if was_list else x
            fuser_mode = getattr(self.temporal_fuser, 'fuser_mode',
                                 'history_queue')

            if fuser_mode == 'history_queue':
                adj_bevs = kwargs.get('adj_bevs', [])
                ego_motions = kwargs.get('ego_motions', [])
                lidar_coord_frame = kwargs.get('lidar_coord_frame', 'FLU')
                if adj_bevs:
                    adj_tensors = [
                        a[0] if isinstance(a, (list, tuple)) else a
                        for a in adj_bevs
                    ]
                    x_tensor = self.temporal_fuser(
                        x_tensor, adj_tensors, ego_motions,
                        lidar_coord_frame=lidar_coord_frame)
                    x = [x_tensor] if was_list else x_tensor
            elif fuser_mode == 'prev_bev':
                ego2globals = [
                    meta.get('ego2global', np.eye(4))
                    for meta in batch_input_metas
                ]
                lidar_aug_matrices = [
                    meta.get('lidar_aug_matrix', np.eye(4))
                    for meta in batch_input_metas
                ]
                lidar_coord_frame = (
                    batch_input_metas[0].get('lidar_coord_frame', 'FLU')
                    if batch_input_metas else 'FLU')

                prev_state = self._extract_history_bev_from_queue_inputs(
                    batch_inputs_dict, batch_input_metas, x_tensor)
                if prev_state is None:
                    prev_state = self._extract_prev_bev_from_inputs(
                        batch_inputs_dict, batch_input_metas, x_tensor)
                prev_bevs, cold_mask, prev_e2g, prev_aug, prev_exists = (
                    prev_state)
                x_tensor = self.temporal_fuser(
                    curr_bev=x_tensor,
                    prev_bev=prev_bevs,
                    curr_ego2global=ego2globals,
                    prev_ego2global=prev_e2g,
                    curr_lidar_aug_matrix=lidar_aug_matrices,
                    prev_lidar_aug_matrix=prev_aug,
                    cold_mask=cold_mask,
                    prev_bev_exists=prev_exists,
                    lidar_coord_frame=lidar_coord_frame)
                x = [x_tensor] if was_list else x_tensor
            else:
                raise ValueError(
                    f'Unsupported temporal_fuser mode: {fuser_mode!r}')

        return x

    def _extract_temporal_kwargs(
        self, batch_inputs_dict, batch_input_metas
    ) -> dict:
        """Build adj_bevs / ego_motions for temporal fusion if data exists.

        After mmengine pseudo_collate, nested lists are **transposed**
        (zip(*batch)), so the layout is **frame-major**:
          adj_points:      [num_adj][batch] — list of Tensor or None
          adj_ego_motions: [num_adj][batch] — list of Tensor(4,4)
        """
        if self.temporal_fuser is None:
            return {}

        if getattr(self.temporal_fuser, 'fuser_mode', 'history_queue'
                   ) != 'history_queue':
            return {}

        raw_pts = batch_inputs_dict.get('adj_points', None)
        raw_mot = batch_inputs_dict.get('adj_ego_motions', None)
        if raw_pts is None or raw_mot is None:
            return {}

        # pseudo_collate transposes [batch][num_adj] → [num_adj][batch]
        num_adj = len(raw_pts)
        batch_size = len(raw_pts[0]) if num_adj > 0 else 0
        if num_adj == 0 or batch_size == 0:
            return {}

        adj_bevs = []
        ego_motions = []
        device = next(self.parameters()).device

        for frame_idx in range(num_adj):
            # Ego-motion for this adj frame across the batch → (B, 4, 4)
            motions = torch.stack([
                raw_mot[frame_idx][b].to(device)
                for b in range(batch_size)
            ])
            ego_motions.append(motions)

            # Points for this adj frame across the batch
            frame_pts = raw_pts[frame_idx]  # already [batch] after transpose
            valid_indices = [
                idx for idx, pts in enumerate(frame_pts) if pts is not None
            ]
            if len(valid_indices) == 0:
                adj_bevs.append(None)
                continue

            # Extract BEV for this adj frame (no temporal fusion, no grad)
            # Missing history is handled per sample: extract only valid
            # samples, then scatter them back into a zero-padded full batch.
            adj_batch = dict(batch_inputs_dict)
            adj_batch['points'] = [
                frame_pts[idx].to(device).contiguous()
                for idx in valid_indices
            ]
            adj_metas = [batch_input_metas[idx] for idx in valid_indices]

            with torch.no_grad():
                adj_bev = self._extract_single_frame_bev(
                    adj_batch, adj_metas)
            if isinstance(adj_bev, (list, tuple)):
                padded_bev = []
                for tensor in adj_bev:
                    padded = tensor.new_zeros(
                        (batch_size, ) + tuple(tensor.shape[1:]))
                    padded[valid_indices] = tensor
                    padded_bev.append(padded.detach())
                adj_bev = padded_bev
            else:
                padded = adj_bev.new_zeros(
                    (batch_size, ) + tuple(adj_bev.shape[1:]))
                padded[valid_indices] = adj_bev
                adj_bev = padded.detach()
            adj_bevs.append(adj_bev)

        # Source the lidar coord frame from pkl metainfo (carried in
        # data_sample.metainfo via Pack3DDetInputs default meta_keys).
        lidar_coord_frame = (
            batch_input_metas[0].get('lidar_coord_frame', 'FLU')
            if batch_input_metas else 'FLU')

        return dict(adj_bevs=adj_bevs, ego_motions=ego_motions,
                    lidar_coord_frame=lidar_coord_frame)

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        temporal_kwargs = self._extract_temporal_kwargs(
            batch_inputs_dict, batch_input_metas)
        feats = self.extract_feat(
            batch_inputs_dict, batch_input_metas, **temporal_kwargs)
        self._cache_scene_context(feats)

        losses = dict()
        if self.with_bbox_head:
            if isinstance(self.bbox_head, CenterHead):
                bbox_loss = self.bbox_head.loss([feats], batch_data_samples)
            else:
                bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        scene_context = self._collect_scene_context()
        if self.tracking_head is not None:
            tracking_loss = self.tracking_head.loss(
                scene_context, batch_data_samples)
            losses.update(tracking_loss)
        if self.motion_head is not None:
            motion_loss = self.bbox_head.loss_motion(
                batch_data_samples,
                self.motion_head,
                scene_context=scene_context)
            losses.update(motion_loss)

        return losses
