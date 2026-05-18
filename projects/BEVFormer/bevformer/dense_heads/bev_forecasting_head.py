"""Per-detection trajectory regression head from BEV features (Phase 3 / B)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVForecastingHead(BaseModule):
    """Bilinear-sample BEV at instance centers, MLP-decode future trajectory.

    Outputs per-step (dx, dy) deltas; ``cumsum`` to match the
    ``gt_forecasting_locs`` cumulative-displacement convention.

    Loss is masked SmoothL1 with per-instance weighting by GT final-step
    magnitude (clamped). Without this, the long-tail KL distribution
    (~50% of instances stationary) makes plain L1 collapse to "predict
    zero everywhere" — see project_kl_data_quirks.md.
    """

    def __init__(self,
                 embed_dims: int = 512,
                 hidden_dims: int = 256,
                 num_steps: int = 6,
                 num_classes: int = 15,
                 dropout: float = 0.1,
                 pc_range=(-80.0, -48.0, -2.0, 80.0, 48.0, 6.0),
                 use_velocity: bool = True,
                 use_class_embed: bool = True,
                 motion_weight_clamp=(0.5, 5.0),
                 smooth_l1_beta: float = 0.5,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)
        self.use_velocity = bool(use_velocity)
        self.use_class_embed = bool(use_class_embed)
        self.motion_weight_clamp = (float(motion_weight_clamp[0]),
                                    float(motion_weight_clamp[1]))
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.loss_weight = float(loss_weight)

        pc_range_t = torch.as_tensor(pc_range, dtype=torch.float32)
        assert pc_range_t.numel() == 6, 'pc_range must be 6-element'
        self.register_buffer('pc_range', pc_range_t)

        in_dim = embed_dims
        if self.use_velocity:
            in_dim += 2
        if self.use_class_embed:
            in_dim += self.num_classes

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, num_steps * 2),
        )

    def _to_grid_norm(self, centers_xy: Tensor) -> Tensor:
        """Map lidar (x, y) to grid_sample-normalized coords [-1, 1]."""
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        x_n = 2.0 * (centers_xy[..., 0] - x_min) / (x_max - x_min) - 1.0
        y_n = 2.0 * (centers_xy[..., 1] - y_min) / (y_max - y_min) - 1.0
        return torch.stack([x_n, y_n], dim=-1)

    def _sample_bev(self, bev_feat: Tensor, centers_xy: Tensor) -> Tensor:
        """Bilinear-sample one batch slice's BEV at lidar (x, y) points.

        bev_feat: (1, C, H, W) — single batch sample
        centers_xy: (N, 2) lidar coords
        returns: (N, C)
        """
        if centers_xy.numel() == 0:
            return bev_feat.new_zeros(0, bev_feat.shape[1])
        norm = self._to_grid_norm(centers_xy.float())  # (N, 2)
        grid = norm.view(1, -1, 1, 2)  # grid_sample: (B, H_out, W_out, 2)
        sampled = F.grid_sample(
            bev_feat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)  # (1, C, N, 1)
        return sampled.squeeze(-1).squeeze(0).transpose(0, 1).contiguous()

    def _build_query_input(self, sampled_feat: Tensor,
                           velocities: Optional[Tensor],
                           labels: Optional[Tensor]) -> Tensor:
        parts = [sampled_feat]
        N = sampled_feat.shape[0]
        if self.use_velocity:
            if velocities is None:
                vel = sampled_feat.new_zeros(N, 2)
            else:
                vel = velocities.to(sampled_feat.dtype)
            parts.append(vel)
        if self.use_class_embed:
            if labels is None:
                onehot = sampled_feat.new_zeros(N, self.num_classes)
            else:
                idx = labels.long().clamp(0, self.num_classes - 1)
                onehot = F.one_hot(idx, num_classes=self.num_classes).to(
                    sampled_feat.dtype)
            parts.append(onehot)
        return torch.cat(parts, dim=-1)

    def forward(self,
                bev_feat: Tensor,
                centers_list: List[Tensor],
                velocities_list: Optional[List[Tensor]] = None,
                labels_list: Optional[List[Tensor]] = None) -> List[Tensor]:
        """Predict per-sample trajectories.

        Args:
            bev_feat: (B, C, H, W) BEV feature map.
            centers_list: list of length B, each (N_i, 2) lidar (x, y).
            velocities_list: optional list of (N_i, 2) current velocities.
            labels_list: optional list of (N_i,) class indices.

        Returns:
            List of length B, each (N_i, T, 2) cumulative displacement
            in current LiDAR (x, y) frame.
        """
        B = bev_feat.shape[0]
        assert len(centers_list) == B, \
            f'centers_list len={len(centers_list)} != batch={B}'
        outputs: List[Tensor] = []
        for b in range(B):
            centers = centers_list[b]
            if centers.shape[0] == 0:
                outputs.append(centers.new_zeros(0, self.num_steps, 2))
                continue
            sampled = self._sample_bev(bev_feat[b:b + 1], centers)
            vel = velocities_list[b] if velocities_list is not None else None
            lbl = labels_list[b] if labels_list is not None else None
            x = self._build_query_input(sampled, vel, lbl)
            deltas = self.mlp(x).view(-1, self.num_steps, 2)
            traj = deltas.cumsum(dim=1)
            outputs.append(traj)
        return outputs

    def loss(self,
             bev_feat: Tensor,
             centers_list: List[Tensor],
             velocities_list: Optional[List[Tensor]],
             labels_list: Optional[List[Tensor]],
             gt_locs_list: List[Tensor],
             gt_mask_list: List[Tensor]) -> dict:
        """Masked SmoothL1 with per-instance motion-magnitude weighting."""
        preds = self.forward(bev_feat, centers_list, velocities_list,
                             labels_list)

        all_preds, all_targets, all_masks, all_weights = [], [], [], []
        for pred, gt, mask in zip(preds, gt_locs_list, gt_mask_list):
            if pred.shape[0] == 0:
                continue
            gt_f = gt.to(device=pred.device, dtype=pred.dtype)
            mask_f = mask.to(device=pred.device, dtype=pred.dtype)
            all_preds.append(pred)
            all_targets.append(gt_f)
            all_masks.append(mask_f)
            final_mag = gt_f[:, -1].norm(dim=-1)
            w = final_mag.clamp(*self.motion_weight_clamp)
            all_weights.append(w)

        if not all_preds:
            return dict(loss_traj=bev_feat.new_zeros(()))

        pred = torch.cat(all_preds, dim=0)        # (N, T, 2)
        target = torch.cat(all_targets, dim=0)    # (N, T, 2)
        mask = torch.cat(all_masks, dim=0)        # (N, T)
        weight = torch.cat(all_weights, dim=0)    # (N,)

        elem = F.smooth_l1_loss(pred, target,
                                beta=self.smooth_l1_beta,
                                reduction='none')   # (N, T, 2)
        elem = elem.sum(-1)                          # (N, T) — sum xy
        elem = elem * mask
        elem = elem * weight.unsqueeze(-1)

        n_valid = mask.sum().clamp(min=1.0)
        loss = elem.sum() / n_valid * self.loss_weight
        return dict(loss_traj=loss)


@MODELS.register_module()
class TrackMotionHead(BaseModule):
    """MLP trajectory head driven by UniAD-style track query embeddings.

    This is an interface-alignment step toward UniAD motion: the primary
    per-agent feature is the track query embedding from ``outs_track``.
    Center, velocity and class are kept as lightweight auxiliary inputs.
    """

    uses_track_query = True

    def __init__(self,
                 embed_dims: int = 256,
                 hidden_dims: int = 256,
                 num_steps: int = 6,
                 num_classes: int = 15,
                 dropout: float = 0.1,
                 pc_range=(-80.0, -48.0, -2.0, 80.0, 48.0, 6.0),
                 use_center: bool = True,
                 use_velocity: bool = True,
                 use_class_embed: bool = True,
                 motion_weight_clamp=(0.5, 5.0),
                 smooth_l1_beta: float = 0.5,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)
        self.use_center = bool(use_center)
        self.use_velocity = bool(use_velocity)
        self.use_class_embed = bool(use_class_embed)
        self.motion_weight_clamp = (float(motion_weight_clamp[0]),
                                    float(motion_weight_clamp[1]))
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.loss_weight = float(loss_weight)

        pc_range_t = torch.as_tensor(pc_range, dtype=torch.float32)
        assert pc_range_t.numel() == 6, 'pc_range must be 6-element'
        self.register_buffer('pc_range', pc_range_t)

        in_dim = embed_dims
        if self.use_center:
            in_dim += 2
        if self.use_velocity:
            in_dim += 2
        if self.use_class_embed:
            in_dim += self.num_classes

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, num_steps * 2),
        )

    def _normalize_centers(self, centers_xy: Tensor) -> Tensor:
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        x_n = 2.0 * (centers_xy[..., 0] - x_min) / (x_max - x_min) - 1.0
        y_n = 2.0 * (centers_xy[..., 1] - y_min) / (y_max - y_min) - 1.0
        return torch.stack([x_n, y_n], dim=-1)

    def _build_query_input(self,
                           query_embeddings: Tensor,
                           centers: Optional[Tensor],
                           velocities: Optional[Tensor],
                           labels: Optional[Tensor]) -> Tensor:
        parts = [query_embeddings]
        N = query_embeddings.shape[0]
        dtype = query_embeddings.dtype
        if self.use_center:
            if centers is None:
                center_feat = query_embeddings.new_zeros(N, 2)
            else:
                center_feat = self._normalize_centers(centers).to(dtype)
            parts.append(center_feat)
        if self.use_velocity:
            if velocities is None:
                vel = query_embeddings.new_zeros(N, 2)
            else:
                vel = velocities.to(dtype)
            parts.append(vel)
        if self.use_class_embed:
            if labels is None:
                onehot = query_embeddings.new_zeros(N, self.num_classes)
            else:
                idx = labels.long().clamp(0, self.num_classes - 1)
                onehot = F.one_hot(idx, num_classes=self.num_classes).to(dtype)
            parts.append(onehot)
        return torch.cat(parts, dim=-1)

    def forward(self,
                query_embeddings_list: List[Tensor],
                centers_list: Optional[List[Tensor]] = None,
                velocities_list: Optional[List[Tensor]] = None,
                labels_list: Optional[List[Tensor]] = None) -> List[Tensor]:
        """Predict trajectories from per-agent track query embeddings."""
        outputs: List[Tensor] = []
        for b, query_embeddings in enumerate(query_embeddings_list):
            if query_embeddings.shape[0] == 0:
                outputs.append(
                    query_embeddings.new_zeros(0, self.num_steps, 2))
                continue
            centers = centers_list[b] if centers_list is not None else None
            vel = velocities_list[b] if velocities_list is not None else None
            lbl = labels_list[b] if labels_list is not None else None
            x = self._build_query_input(query_embeddings, centers, vel, lbl)
            deltas = self.mlp(x).view(-1, self.num_steps, 2)
            outputs.append(deltas.cumsum(dim=1))
        return outputs

    def loss(self,
             query_embeddings_list: List[Tensor],
             centers_list: Optional[List[Tensor]],
             velocities_list: Optional[List[Tensor]],
             labels_list: Optional[List[Tensor]],
             gt_locs_list: List[Tensor],
             gt_mask_list: List[Tensor]) -> dict:
        """Masked SmoothL1 loss for track-query motion predictions."""
        preds = self.forward(query_embeddings_list, centers_list,
                             velocities_list, labels_list)

        all_preds, all_targets, all_masks, all_weights = [], [], [], []
        for pred, gt, mask in zip(preds, gt_locs_list, gt_mask_list):
            if pred.shape[0] == 0:
                continue
            gt_f = gt.to(device=pred.device, dtype=pred.dtype)
            mask_f = mask.to(device=pred.device, dtype=pred.dtype)
            all_preds.append(pred)
            all_targets.append(gt_f)
            all_masks.append(mask_f)
            final_mag = gt_f[:, -1].norm(dim=-1)
            all_weights.append(final_mag.clamp(*self.motion_weight_clamp))

        if not all_preds:
            if query_embeddings_list:
                return dict(loss_traj=query_embeddings_list[0].new_zeros(()))
            return dict(loss_traj=torch.zeros(()))

        pred = torch.cat(all_preds, dim=0)
        target = torch.cat(all_targets, dim=0)
        mask = torch.cat(all_masks, dim=0)
        weight = torch.cat(all_weights, dim=0)

        elem = F.smooth_l1_loss(
            pred, target, beta=self.smooth_l1_beta, reduction='none')
        elem = elem.sum(-1)
        elem = elem * mask
        elem = elem * weight.unsqueeze(-1)

        n_valid = mask.sum().clamp(min=1.0)
        loss = elem.sum() / n_valid * self.loss_weight
        return dict(loss_traj=loss)
