"""Track-query interaction modules.

These are small, LiDAR-only ports of UniAD's track-query utilities. The base
track config only uses ``QueryInteractionModule``; MemoryBank is exposed for a
separate ablation config so existing runs are not affected.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .track_instance import Instances


class MemoryBank(nn.Module):
    """Temporal memory over track-query embeddings.

    This follows UniAD's MemoryBank: recent ``output_embedding`` vectors are
    saved per track and attended before QIM updates the next-frame query.
    """

    def __init__(self, args, dim_in: int, hidden_dim: int, dim_out: int):
        super().__init__()
        self.save_thresh = args.get('memory_bank_score_thresh', 0.0)
        self.save_period = args.get('memory_bank_save_period', 3)
        self.max_his_length = args.get('memory_bank_len', 4)

        self.save_proj = nn.Linear(dim_in, dim_in)
        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0.0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _forward_temporal_attn(self, ti: Instances) -> Instances:
        if len(ti) == 0:
            return ti

        key_padding_mask = ti.mem_padding_mask
        valid = key_padding_mask[:, -1] == 0
        embed = ti.output_embedding[valid]
        if len(embed) == 0:
            return ti

        prev_embed = ti.mem_bank[valid]
        key_padding_mask = key_padding_mask[valid]
        embed2 = self.temporal_attn(
            embed[None],
            prev_embed.transpose(0, 1),
            prev_embed.transpose(0, 1),
            key_padding_mask=key_padding_mask)[0][0]
        embed = self.temporal_norm1(embed + embed2)
        embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
        embed = self.temporal_norm2(embed + embed2)

        ti.output_embedding = ti.output_embedding.clone()
        ti.output_embedding[valid] = embed
        return ti

    def update(self, ti: Instances) -> None:
        embed = ti.output_embedding[:, None]
        scores = ti.scores
        save_period = ti.save_period
        mem_padding_mask = ti.mem_padding_mask
        device = embed.device

        if self.training:
            saved = scores > 0
        else:
            saved = (save_period == 0) & (scores > self.save_thresh)
            save_period[save_period > 0] -= 1
            save_period[saved] = self.save_period

        saved_embed = embed[saved]
        if len(saved_embed) == 0:
            return

        prev_embed = ti.mem_bank[saved]
        save_embed = self.save_proj(saved_embed)
        mem_padding_mask[saved] = torch.cat([
            mem_padding_mask[saved, 1:],
            torch.zeros(
                (len(saved_embed), 1), dtype=torch.bool, device=device)
        ], dim=1)
        ti.mem_bank = ti.mem_bank.clone()
        ti.mem_bank[saved] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)

    def forward(self, ti: Instances, update_bank: bool = True) -> Instances:
        ti = self._forward_temporal_attn(ti)
        if update_bank:
            self.update(ti)
        return ti


class QueryInteractionBase(nn.Module):
    """UniAD-style base class for track-query interaction modules."""

    def __init__(self, args, dim_in: int, hidden_dim: int, dim_out: int):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        raise NotImplementedError


class QueryInteractionModule(QueryInteractionBase):
    """Self-attention update over active track queries before next frame."""

    def __init__(self, args, dim_in: int, hidden_dim: int, dim_out: int):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.get('random_drop', 0.1)
        self.fp_ratio = args.get('fp_ratio', 0.3)
        self.update_query_pos = args.get('update_query_pos', False)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.get('merger_dropout', 0.0)
        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.get('update_query_pos', False):
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding(self, ti: Instances) -> Instances:
        """Self-attention over active tracks; rewrites query[:, dim//2:]."""
        if len(ti) == 0:
            return ti
        dim = ti.query.shape[1]
        out_embed = ti.output_embedding
        query_pos = ti.query[:, :dim // 2]
        query_feat = ti.query[:, dim // 2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            qp2 = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(qp2)
            query_pos = self.norm_pos(query_pos)
            ti.query = ti.query.clone()
            ti.query[:, :dim // 2] = query_pos

        qf2 = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(qf2)
        query_feat = self.norm_feat(query_feat)
        if not self.update_query_pos:
            ti.query = ti.query.clone()
        ti.query[:, dim // 2:] = query_feat
        return ti

    def _random_drop_tracks(self, ti: Instances) -> Instances:
        """Randomly drop tracks during training to prevent overfitting."""
        if self.random_drop > 0 and len(ti) > 0:
            keep = torch.rand_like(ti.scores) > self.random_drop
            ti = ti[keep]
        return ti

    def _add_fp_tracks(self, ti_all: Instances,
                       ti_active: Instances) -> Instances:
        """Inject inactive (presumed-FP) tracks as distractors during train."""
        inactive = ti_all[ti_all.obj_idxes < 0]
        if len(inactive) == 0 or len(ti_active) == 0:
            return ti_active
        fp_prob = torch.ones_like(ti_active.scores) * self.fp_ratio
        n_fp = int(torch.bernoulli(fp_prob).sum().item())
        if n_fp == 0:
            return ti_active
        if n_fp >= len(inactive):
            fp = inactive
        else:
            top = torch.argsort(inactive.scores)[-n_fp:]
            fp = inactive[top]
        return Instances.cat([ti_active, fp])

    def _select_active_tracks(self, data: dict) -> Instances:
        ti: Instances = data['track_instances']
        if self.training:
            active_idxes = (ti.obj_idxes >= 0) & (ti.iou > 0.5)
            active = ti[active_idxes]
            active = self._random_drop_tracks(active)
            if self.fp_ratio > 0:
                active = self._add_fp_tracks(ti, active)
        else:
            active = ti[ti.obj_idxes >= 0]
        return active

    def forward(self, data: dict) -> Instances:
        ti = data['track_instances']
        active = self._select_active_tracks(data)
        active = self._update_track_embedding(active)
        init_ti: Instances = data['init_track_instances']
        return Instances.cat([init_ti, active])
