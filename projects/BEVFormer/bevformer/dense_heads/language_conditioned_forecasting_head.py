"""Language-conditioned BEV forecasting head for KL template prompts."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LanguageConditionedForecastingHead(BaseModule):
    """Bilinear BEV query + template language tokens -> trajectories.

    This is the first-stage VLM bootstrap head. It keeps the same output
    convention as ``BEVForecastingHead`` but adds a small trainable text
    embedding table. The language target mask gates trajectory loss to the
    instances requested by the prompt and also supervises a selected-object
    score.
    """

    def __init__(self,
                 bev_dim: int = 512,
                 hidden_dims: int = 256,
                 text_embed_dims: int = 128,
                 vocab_size: int = 34,
                 pad_token_id: int = 0,
                 num_steps: int = 6,
                 num_classes: int = 15,
                 dropout: float = 0.1,
                 pc_range=(-80.0, -48.0, -2.0, 80.0, 48.0, 6.0),
                 use_velocity: bool = True,
                 use_class_embed: bool = True,
                 motion_weight_clamp=(0.5, 5.0),
                 smooth_l1_beta: float = 0.5,
                 loss_weight: float = 1.0,
                 selection_loss_weight: float = 0.2) -> None:
        super().__init__()
        self.bev_dim = int(bev_dim)
        self.text_embed_dims = int(text_embed_dims)
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)
        self.pad_token_id = int(pad_token_id)
        self.use_velocity = bool(use_velocity)
        self.use_class_embed = bool(use_class_embed)
        self.motion_weight_clamp = (float(motion_weight_clamp[0]),
                                    float(motion_weight_clamp[1]))
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.loss_weight = float(loss_weight)
        self.selection_loss_weight = float(selection_loss_weight)

        pc_range_t = torch.as_tensor(pc_range, dtype=torch.float32)
        assert pc_range_t.numel() == 6, 'pc_range must be 6-element'
        self.register_buffer('pc_range', pc_range_t)

        self.token_embed = nn.Embedding(
            int(vocab_size), self.text_embed_dims, padding_idx=pad_token_id)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_embed_dims, self.text_embed_dims),
            nn.LayerNorm(self.text_embed_dims),
            nn.ReLU(inplace=True),
        )

        in_dim = self.bev_dim + self.text_embed_dims
        if self.use_velocity:
            in_dim += 2
        if self.use_class_embed:
            in_dim += self.num_classes

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.traj_branch = nn.Linear(hidden_dims, num_steps * 2)
        self.selection_branch = nn.Linear(hidden_dims, 1)

    def _to_grid_norm(self, centers_xy: Tensor) -> Tensor:
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        x_n = 2.0 * (centers_xy[..., 0] - x_min) / (x_max - x_min) - 1.0
        y_n = 2.0 * (centers_xy[..., 1] - y_min) / (y_max - y_min) - 1.0
        return torch.stack([x_n, y_n], dim=-1)

    def _sample_bev(self, bev_feat: Tensor, centers_xy: Tensor) -> Tensor:
        if centers_xy.numel() == 0:
            return bev_feat.new_zeros(0, bev_feat.shape[1])
        norm = self._to_grid_norm(centers_xy.float())
        grid = norm.view(1, -1, 1, 2)
        sampled = F.grid_sample(
            bev_feat, grid, mode='bilinear', padding_mode='zeros',
            align_corners=False)
        return sampled.squeeze(-1).squeeze(0).transpose(0, 1).contiguous()

    def _pool_text(self,
                   token_ids: Optional[Tensor],
                   token_mask: Optional[Tensor],
                   device,
                   dtype) -> Tensor:
        if token_ids is None:
            return torch.zeros(self.text_embed_dims, device=device, dtype=dtype)
        token_ids = token_ids.to(device=device, dtype=torch.long).view(-1)
        embeds = self.token_embed(token_ids)
        if token_mask is None:
            mask = token_ids.ne(self.pad_token_id)
        else:
            mask = token_mask.to(device=device, dtype=torch.bool).view(-1)
        if mask.numel() != embeds.shape[0]:
            raise ValueError('language token mask length mismatch: '
                             f'{mask.numel()} vs {embeds.shape[0]}.')
        denom = mask.sum().clamp(min=1).to(embeds.dtype)
        pooled = (embeds * mask[:, None].to(embeds.dtype)).sum(0) / denom
        return self.text_proj(pooled.to(dtype)).to(dtype)

    def _build_query_input(self, sampled_feat: Tensor,
                           text_feat: Tensor,
                           velocities: Optional[Tensor],
                           labels: Optional[Tensor]) -> Tensor:
        n = sampled_feat.shape[0]
        parts = [sampled_feat, text_feat.expand(n, -1)]
        if self.use_velocity:
            if velocities is None:
                vel = sampled_feat.new_zeros(n, 2)
            else:
                vel = velocities.to(sampled_feat.dtype)
            parts.append(vel)
        if self.use_class_embed:
            if labels is None:
                onehot = sampled_feat.new_zeros(n, self.num_classes)
            else:
                idx = labels.long().clamp(0, self.num_classes - 1)
                onehot = F.one_hot(idx, num_classes=self.num_classes).to(
                    sampled_feat.dtype)
            parts.append(onehot)
        return torch.cat(parts, dim=-1)

    def _forward_one(self,
                     bev_feat: Tensor,
                     centers: Tensor,
                     velocities: Optional[Tensor],
                     labels: Optional[Tensor],
                     token_ids: Optional[Tensor],
                     token_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if centers.shape[0] == 0:
            traj = centers.new_zeros(0, self.num_steps, 2)
            scores = centers.new_zeros(0)
            return traj, scores
        sampled = self._sample_bev(bev_feat, centers)
        text_feat = self._pool_text(
            token_ids, token_mask, sampled.device, sampled.dtype)
        query = self._build_query_input(sampled, text_feat, velocities, labels)
        hidden = self.shared_mlp(query)
        deltas = self.traj_branch(hidden).view(-1, self.num_steps, 2)
        traj = deltas.cumsum(dim=1)
        selected_logits = self.selection_branch(hidden).squeeze(-1)
        return traj, selected_logits

    def forward(self,
                bev_feat: Tensor,
                centers_list: List[Tensor],
                velocities_list: Optional[List[Tensor]] = None,
                labels_list: Optional[List[Tensor]] = None,
                language_tokens_list: Optional[List[Tensor]] = None,
                language_token_mask_list: Optional[List[Tensor]] = None
                ) -> List[Tensor]:
        trajs, _ = self.forward_with_selection(
            bev_feat, centers_list, velocities_list, labels_list,
            language_tokens_list, language_token_mask_list)
        return trajs

    def forward_with_selection(
            self,
            bev_feat: Tensor,
            centers_list: List[Tensor],
            velocities_list: Optional[List[Tensor]] = None,
            labels_list: Optional[List[Tensor]] = None,
            language_tokens_list: Optional[List[Tensor]] = None,
            language_token_mask_list: Optional[List[Tensor]] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        batch_size = bev_feat.shape[0]
        assert len(centers_list) == batch_size
        trajs: List[Tensor] = []
        selected_logits: List[Tensor] = []
        for batch_idx in range(batch_size):
            vel = (velocities_list[batch_idx]
                   if velocities_list is not None else None)
            lbl = labels_list[batch_idx] if labels_list is not None else None
            tok = (language_tokens_list[batch_idx]
                   if language_tokens_list is not None else None)
            tok_mask = (language_token_mask_list[batch_idx]
                        if language_token_mask_list is not None else None)
            traj, logits = self._forward_one(
                bev_feat[batch_idx:batch_idx + 1], centers_list[batch_idx],
                vel, lbl, tok, tok_mask)
            trajs.append(traj)
            selected_logits.append(logits)
        return trajs, selected_logits

    def loss(self,
             bev_feat: Tensor,
             centers_list: List[Tensor],
             velocities_list: Optional[List[Tensor]],
             labels_list: Optional[List[Tensor]],
             gt_locs_list: List[Tensor],
             gt_mask_list: List[Tensor],
             language_tokens_list: Optional[List[Tensor]] = None,
             language_token_mask_list: Optional[List[Tensor]] = None,
             language_target_mask_list: Optional[List[Tensor]] = None) -> dict:
        preds, selected_logits = self.forward_with_selection(
            bev_feat, centers_list, velocities_list, labels_list,
            language_tokens_list, language_token_mask_list)

        traj_losses = []
        selection_losses = []
        for batch_idx, (pred, gt, mask) in enumerate(
                zip(preds, gt_locs_list, gt_mask_list)):
            if pred.shape[0] == 0:
                continue
            gt_f = gt.to(pred.dtype)
            mask_f = mask.to(pred.dtype)
            if (language_target_mask_list is None or
                    language_target_mask_list[batch_idx] is None):
                target_mask = pred.new_ones(pred.shape[0])
            else:
                target_mask = language_target_mask_list[batch_idx].to(
                    device=pred.device, dtype=pred.dtype)
            if target_mask.shape[0] != pred.shape[0]:
                raise ValueError('language target mask length mismatch: '
                                 f'{target_mask.shape[0]} vs {pred.shape[0]}.')

            elem = F.smooth_l1_loss(
                pred, gt_f, beta=self.smooth_l1_beta, reduction='none')
            elem = elem.sum(-1) * mask_f
            final_mag = gt_f[:, -1].norm(dim=-1)
            weights = final_mag.clamp(*self.motion_weight_clamp)
            elem = elem * weights.unsqueeze(-1)
            elem = elem * target_mask.unsqueeze(-1)
            denom = (mask_f * target_mask.unsqueeze(-1)).sum().clamp(min=1.0)
            traj_losses.append(elem.sum() / denom)

            logits = selected_logits[batch_idx]
            selection_losses.append(F.binary_cross_entropy_with_logits(
                logits, target_mask, reduction='mean'))

        if traj_losses:
            loss_traj = torch.stack(traj_losses).mean() * self.loss_weight
            loss_select = torch.stack(selection_losses).mean()
        else:
            loss_traj = bev_feat.new_zeros(())
            loss_select = bev_feat.new_zeros(())
        return dict(
            loss_lang_traj=loss_traj,
            loss_lang_select=loss_select * self.selection_loss_weight)
