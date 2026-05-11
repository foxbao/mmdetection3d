"""Transformer-based forecasting head (Phase 3 / B+ variant).

Upgrades the bilinear-sample MLP head (``BEVForecastingHead``) with two
transformer decoder layers so each instance's trajectory query can
*(1)* attend to other instances (self-attention → inter-object reasoning,
e.g. car-following, crane-trailer coordination, lane merging) and
*(2)* attend to the full BEV grid (cross-attention → global scene
context like roads, junctions, distant agents 30+ m away).

API mirrors ``BEVForecastingHead`` so detectors can swap heads via config.
Output and loss conventions are identical (cumulative displacement,
masked SmoothL1 with motion-magnitude weighting).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TransformerForecastingHead(BaseModule):
    """BEV bilinear sample → query → transformer decoder → trajectory."""

    def __init__(self,
                 bev_dim: int = 512,
                 embed_dims: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 ffn_dims: int = 512,
                 num_steps: int = 6,
                 num_classes: int = 15,
                 dropout: float = 0.1,
                 pc_range=(-80.0, -48.0, -2.0, 80.0, 48.0, 6.0),
                 use_velocity: bool = True,
                 use_class_embed: bool = True,
                 motion_weight_clamp=(0.5, 5.0),
                 smooth_l1_beta: float = 0.5,
                 loss_weight: float = 1.0,
                 sine_temperature: float = 10000.0) -> None:
        super().__init__()
        assert embed_dims % 2 == 0, 'embed_dims must be even (split y/x)'
        self.bev_dim = int(bev_dim)
        self.embed_dims = int(embed_dims)
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)
        self.use_velocity = bool(use_velocity)
        self.use_class_embed = bool(use_class_embed)
        self.motion_weight_clamp = (float(motion_weight_clamp[0]),
                                    float(motion_weight_clamp[1]))
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.loss_weight = float(loss_weight)
        self.sine_temperature = float(sine_temperature)

        pc_range_t = torch.as_tensor(pc_range, dtype=torch.float32)
        self.register_buffer('pc_range', pc_range_t)

        # BEV → embed_dims projection (for memory tokens of cross-attention)
        self.bev_proj = nn.Linear(self.bev_dim, self.embed_dims)

        # Query input dimension: bilinear-sampled BEV (raw 512) + extras
        in_query_dim = self.bev_dim
        if self.use_velocity:
            in_query_dim += 2
        if self.use_class_embed:
            in_query_dim += self.num_classes
        self.query_proj = nn.Sequential(
            nn.Linear(in_query_dim, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )

        # Query positional embedding from lidar (cx, cy) coords
        self.query_pos_mlp = nn.Sequential(
            nn.Linear(2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # Transformer decoder: self-attn (queries) + cross-attn (BEV) + FFN
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dims,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Trajectory regression
        self.reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, num_steps * 2),
        )

    # ------------------------- coord helpers -------------------------------

    def _to_grid_norm(self, centers_xy: Tensor) -> Tensor:
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        x_n = 2.0 * (centers_xy[..., 0] - x_min) / (x_max - x_min) - 1.0
        y_n = 2.0 * (centers_xy[..., 1] - y_min) / (y_max - y_min) - 1.0
        return torch.stack([x_n, y_n], dim=-1)

    def _sample_bev(self, bev_feat: Tensor, centers_xy: Tensor) -> Tensor:
        """Bilinear-sample raw BEV at lidar centers (returns bev_dim feats)."""
        if centers_xy.numel() == 0:
            return bev_feat.new_zeros(0, bev_feat.shape[1])
        norm = self._to_grid_norm(centers_xy.float())
        grid = norm.view(1, -1, 1, 2)
        sampled = F.grid_sample(
            bev_feat, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        return sampled.squeeze(-1).squeeze(0).transpose(0, 1).contiguous()

    def _bev_sine_pos(self, H: int, W: int,
                      device, dtype) -> Tensor:
        """2D sinusoidal positional encoding, half-channels for y, half for x."""
        num_feats = self.embed_dims // 2
        dim_t = torch.arange(num_feats, device=device, dtype=dtype)
        dim_t = self.sine_temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode='floor') / num_feats)

        y = torch.arange(H, device=device, dtype=dtype)
        x = torch.arange(W, device=device, dtype=dtype)

        pos_y = y[:, None] / dim_t[None, :]               # (H, num_feats)
        pos_y = torch.stack(
            [pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=-1
        ).flatten(-2)                                      # (H, num_feats)

        pos_x = x[:, None] / dim_t[None, :]               # (W, num_feats)
        pos_x = torch.stack(
            [pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=-1
        ).flatten(-2)                                      # (W, num_feats)

        pos = torch.zeros(H, W, self.embed_dims, device=device, dtype=dtype)
        pos[..., :num_feats] = pos_y[:, None, :].expand(H, W, num_feats)
        pos[..., num_feats:] = pos_x[None, :, :].expand(H, W, num_feats)
        return pos                                         # (H, W, C)

    # ------------------------- query construction --------------------------

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

    # ------------------------- forward / loss ------------------------------

    def forward(self,
                bev_feat: Tensor,
                centers_list: List[Tensor],
                velocities_list: Optional[List[Tensor]] = None,
                labels_list: Optional[List[Tensor]] = None) -> List[Tensor]:
        """Predict per-sample trajectories.

        Same interface as ``BEVForecastingHead.forward`` so the detector
        (or a downstream planner) can swap heads via config without
        touching call sites.
        """
        B, _, H, W = bev_feat.shape
        assert len(centers_list) == B

        # Memory tokens: project BEV to embed_dims, add 2D pos embed, flatten
        bev_perm = bev_feat.permute(0, 2, 3, 1).reshape(B, H * W, self.bev_dim)
        bev_kv = self.bev_proj(bev_perm)                   # (B, HW, C)
        pos_embed = self._bev_sine_pos(
            H, W, bev_feat.device, bev_kv.dtype)           # (H, W, C)
        bev_kv = bev_kv + pos_embed.reshape(H * W, self.embed_dims).unsqueeze(0)

        outputs: List[Tensor] = []
        for b in range(B):
            centers = centers_list[b]
            if centers.shape[0] == 0:
                outputs.append(centers.new_zeros(0, self.num_steps, 2))
                continue

            # Build query (per-instance feature + extras → embed_dims)
            sampled = self._sample_bev(bev_feat[b:b + 1], centers)
            vel = velocities_list[b] if velocities_list is not None else None
            lbl = labels_list[b] if labels_list is not None else None
            query_in = self._build_query_input(sampled, vel, lbl)
            query = self.query_proj(query_in)              # (N, C)

            # Add positional embedding from lidar coords
            qpos = self.query_pos_mlp(
                self._to_grid_norm(centers.float()))       # (N, C)
            query = query + qpos                           # (N, C)

            # Transformer decoder: self-attn(query) + cross-attn(BEV memory)
            query = query.unsqueeze(0)                     # (1, N, C)
            memory = bev_kv[b].unsqueeze(0)                # (1, HW, C)
            decoded = self.decoder(query, memory).squeeze(0)  # (N, C)

            # Regress per-step deltas, cumsum to cumulative displacement
            deltas = self.reg_branch(decoded).view(-1, self.num_steps, 2)
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
            gt_f = gt.to(pred.dtype)
            mask_f = mask.to(pred.dtype)
            all_preds.append(pred)
            all_targets.append(gt_f)
            all_masks.append(mask_f)
            final_mag = gt_f[:, -1].norm(dim=-1)
            w = final_mag.clamp(*self.motion_weight_clamp)
            all_weights.append(w)

        if not all_preds:
            return dict(loss_traj=bev_feat.new_zeros(()))

        pred = torch.cat(all_preds, dim=0)
        target = torch.cat(all_targets, dim=0)
        mask = torch.cat(all_masks, dim=0)
        weight = torch.cat(all_weights, dim=0)

        elem = F.smooth_l1_loss(pred, target,
                                beta=self.smooth_l1_beta,
                                reduction='none')           # (N, T, 2)
        elem = elem.sum(-1)                                  # (N, T)
        elem = elem * mask
        elem = elem * weight.unsqueeze(-1)

        n_valid = mask.sum().clamp(min=1.0)
        loss = elem.sum() / n_valid * self.loss_weight
        return dict(loss_traj=loss)
