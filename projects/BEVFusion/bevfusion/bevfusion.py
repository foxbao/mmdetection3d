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

        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process."""
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network."""
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
        """bool: Whether the detector has a segmentation head."""
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
        batch_size, num_cams, channels, height, width = x.size()
        x = x.view(batch_size * num_cams, channels, height, width).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        batch_cam, channels, height, width = x.size()
        x = x.view(batch_size, int(batch_cam / batch_size), channels, height,
                   width)

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
        for idx, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                feats_i, coords_i, num_points = ret
            else:
                assert len(ret) == 2
                feats_i, coords_i = ret
                num_points = None
            feats.append(feats_i)
            coords.append(F.pad(coords_i, (1, 0), mode='constant', value=idx))
            if num_points is not None:
                sizes.append(num_points)

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
        """Forward of testing."""
        del kwargs
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            if isinstance(self.bbox_head, CenterHead):
                outputs = self.bbox_head.predict(
                    [feats], batch_data_samples)
            else:
                outputs = self.bbox_head.predict(feats, batch_input_metas)

        return self.add_pred_to_datasample(batch_data_samples, outputs)

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
            for meta in batch_input_metas:
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
        """Recursively roll a short prev-points queue into one history BEV."""
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
    ):
        x = self._extract_single_frame_bev(
            batch_inputs_dict, batch_input_metas)

        if self.temporal_fuser is None:
            return x

        fuser_mode = getattr(self.temporal_fuser, 'fuser_mode', None)
        if fuser_mode != 'prev_bev':
            raise ValueError(f'Unsupported temporal_fuser mode: {fuser_mode!r}')

        was_list = isinstance(x, (list, tuple))
        x_tensor = x[0] if was_list else x
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
        prev_bevs, cold_mask, prev_e2g, prev_aug, prev_exists = prev_state
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
        return [x_tensor] if was_list else x_tensor

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        del kwargs
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            if isinstance(self.bbox_head, CenterHead):
                bbox_loss = self.bbox_head.loss([feats], batch_data_samples)
            else:
                bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)
        return losses
