import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TrackingHead(nn.Module):
    """Minimal tracking-head interface for future track-aware experiments.

    This head is intentionally lightweight. It establishes:

    - an explicit input contract: ``scene_context``
    - a training entry point: ``loss(scene_context, batch_data_samples)``
    - an inference entry point: ``predict(outputs, scene_context)``

    It does not yet implement cross-frame association. Instead, it exposes
    per-proposal tracking embeddings and bookkeeping tensors so later work can
    add association / identity supervision without changing the model wiring.
    """

    def __init__(
        self,
        query_channels: int = 128,
        hidden_dim: int = 128,
        use_bev_context: bool = False,
        bev_channels: int = 128,
        score_weight: float = 0.0,
    ):
        super().__init__()
        self.use_bev_context = use_bev_context
        self.score_weight = score_weight

        self.query_proj = nn.Linear(query_channels, hidden_dim)
        if use_bev_context:
            self.bev_proj = nn.Linear(bev_channels, hidden_dim)
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.bev_proj = None
            self.fuse = None
        self.score_head = nn.Linear(hidden_dim, 1)

    def _sample_bev_at_queries(self, bev_memory: torch.Tensor,
                               det_query_pos: torch.Tensor) -> torch.Tensor:
        """Sample BEV memory at proposal centers.

        Args:
            bev_memory: (B, C, H, W)
            det_query_pos: (B, N, 2) in BEV feature coordinates.

        Returns:
            (B, N, C) sampled BEV features.
        """
        batch_size, channels, height, width = bev_memory.shape
        num_queries = det_query_pos.shape[1]

        norm_x = det_query_pos[..., 0] / max(width - 1, 1) * 2.0 - 1.0
        norm_y = det_query_pos[..., 1] / max(height - 1, 1) * 2.0 - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)
        sampled = nn.functional.grid_sample(
            bev_memory,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous()

    def _extract_targets(self, scene_context, batch_data_samples,
                         device: torch.device):
        """Prepare placeholder per-proposal tracking targets.

        The current training data flow does not yet pack track ids into
        ``gt_instances_3d`` consistently, so this function returns optional
        targets and exposes where future association supervision should enter.
        """
        det_queries = scene_context['det_queries']
        batch_size, _, num_queries = det_queries.shape
        track_targets = torch.full(
            (batch_size, num_queries), -1, dtype=torch.long, device=device)
        proposal_mask = torch.zeros(
            (batch_size, num_queries), dtype=torch.float32, device=device)

        assign_result = scene_context.get('assign_result', None)
        if assign_result is not None:
            for batch_idx in range(batch_size):
                pos_inds, _ = assign_result[batch_idx]
                if len(pos_inds) == 0:
                    continue
                proposal_mask[batch_idx, pos_inds] = 1.0

        # Future work:
        # - read per-gt track ids from batch_data_samples
        # - scatter them onto matched proposals using assign_result
        # - add association / identity losses across time
        return track_targets, proposal_mask

    def forward(self, scene_context):
        det_queries = scene_context['det_queries']  # (B, C, N)
        x = self.query_proj(det_queries.permute(0, 2, 1))  # (B, N, D)

        if self.use_bev_context:
            bev_memory = scene_context['bev_memory']
            det_query_pos = scene_context['det_query_pos']
            bev_feat = self._sample_bev_at_queries(bev_memory, det_query_pos)
            bev_feat = self.bev_proj(bev_feat)
            x = self.fuse(torch.cat([x, bev_feat], dim=-1))

        track_scores = self.score_head(x).squeeze(-1)  # (B, N)
        return dict(track_embeddings=x, track_scores=track_scores)

    def loss(self, scene_context, batch_data_samples):
        outputs = self.forward(scene_context)
        device = outputs['track_embeddings'].device
        _, proposal_mask = self._extract_targets(
            scene_context, batch_data_samples, device)

        # Dummy zero-valued loss that still touches all tracking-head
        # parameters, so DDP wiring is valid before real association losses are
        # added.
        loss_track = (
            outputs['track_embeddings'].sum() +
            self.score_weight * outputs['track_scores'].sum()) * 0.0

        return dict(
            loss_track=loss_track,
            track_num_pos=proposal_mask.sum(),
        )

    def predict(self, outputs, scene_context):
        track_outputs = self.forward(scene_context)
        track_embeddings = track_outputs['track_embeddings']
        track_scores = track_outputs['track_scores'].sigmoid()

        for batch_idx, inst in enumerate(outputs):
            keep = inst._keep_inds
            num_keep = keep.shape[0]
            inst.track_query_feats = track_embeddings[batch_idx, keep]
            inst.track_scores_3d = track_scores[batch_idx, keep]
            inst.track_ids_3d = keep.new_full((num_keep,), -1, dtype=torch.long)

        return outputs
