import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class MotionDecoderHeadV2(nn.Module):
    """V2 motion decoder with local BEV context sampling."""

    use_bev_context = True

    def __init__(
        self,
        in_channels: int = 128,
        bev_channels: int = 128,
        hidden_dim: int = 256,
        forecast_steps: int = 6,
        num_modes: int = 3,
        num_attn_layers: int = 2,
        num_attn_heads: int = 8,
        dropout: float = 0.1,
        loss_weight: float = 0.5,
        cls_weight: float = 0.2,
        patch_radius: int = 1,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_modes = num_modes
        self.loss_weight = loss_weight
        self.cls_weight = cls_weight
        self.patch_radius = patch_radius

        self.query_proj = nn.Linear(in_channels, hidden_dim)
        self.bev_proj = nn.Linear(bev_channels, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attn_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation='relu') for _ in range(num_attn_layers)
        ])
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modes * forecast_steps * 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modes),
        )

    def _sample_local_bev(self, bev_feat: torch.Tensor,
                          query_pos: torch.Tensor) -> torch.Tensor:
        """Sample a small local BEV patch around each proposal center.

        Args:
            bev_feat: (B, C, H, W)
            query_pos: (B, N, 2) in BEV feature coordinates (x, y), center-based

        Returns:
            (B, N, C) pooled local BEV context
        """
        batch_size, channels, height, width = bev_feat.shape
        num_props = query_pos.shape[1]
        device = bev_feat.device
        dtype = bev_feat.dtype

        offsets = torch.arange(
            -self.patch_radius,
            self.patch_radius + 1,
            device=device,
            dtype=dtype)
        if offsets.numel() == 1:
            patch_xy = query_pos
        else:
            oy, ox = torch.meshgrid(offsets, offsets)
            offset_grid = torch.stack([ox, oy], dim=-1).view(1, 1, -1, 2)
            patch_xy = query_pos.unsqueeze(2) + offset_grid  # (B,N,P,2)

        norm_x = patch_xy[..., 0] / max(width - 1, 1) * 2.0 - 1.0
        norm_y = patch_xy[..., 1] / max(height - 1, 1) * 2.0 - 1.0
        sample_grid = torch.stack([norm_x, norm_y], dim=-1)
        sample_grid = sample_grid.view(batch_size, num_props, -1, 2)

        sampled = F.grid_sample(
            bev_feat,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        sampled = sampled.mean(dim=-1)  # (B, C, N)
        return sampled.permute(0, 2, 1).contiguous()

    def forward(self, query_feat: torch.Tensor, bev_feat: torch.Tensor,
                query_pos: torch.Tensor):
        """Decode multi-modal trajectories from proposal queries + BEV context.

        Args:
            query_feat: (B, Cq, N)
            bev_feat: (B, Cb, H, W)
            query_pos: (B, N, 2) in BEV feature coordinates
        """
        query = self.query_proj(query_feat.permute(0, 2, 1))  # (B,N,D)
        local_bev = self._sample_local_bev(bev_feat, query_pos)  # (B,N,Cb)
        local_bev = self.bev_proj(local_bev)
        x = self.fuse(torch.cat([query, local_bev], dim=-1))

        for layer in self.decoder_layers:
            x = layer(x)

        batch_size, num_props, _ = x.shape
        traj_preds = self.traj_head(x).view(batch_size, num_props,
                                            self.num_modes,
                                            self.forecast_steps, 2)
        mode_logits = self.cls_head(x)
        return traj_preds, mode_logits

    def loss(self, traj_preds: torch.Tensor, mode_logits: torch.Tensor,
             gt_locs: torch.Tensor, gt_mask: torch.Tensor,
             proposal_mask: torch.Tensor):
        _, _, num_modes, num_steps, _ = traj_preds.shape

        valid_steps = proposal_mask.unsqueeze(-1) * gt_mask.float()
        valid_count = valid_steps.sum(dim=-1)

        gt_expand = gt_locs.unsqueeze(2).expand(-1, -1, num_modes, -1, -1)
        mask_expand = valid_steps.unsqueeze(2).unsqueeze(-1)
        mask_expand = mask_expand.expand_as(traj_preds)

        l1 = F.l1_loss(
            traj_preds * mask_expand,
            gt_expand * mask_expand,
            reduction='none').sum(dim=(-1, -2))
        ade = l1 / (valid_count.unsqueeze(-1).clamp(min=1.0) * 2.0)
        best_mode = ade.argmin(dim=-1)

        gather_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gather_idx = gather_idx.expand(-1, -1, 1, num_steps, 2)
        best_traj = torch.gather(traj_preds, 2, gather_idx).squeeze(2)

        full_mask = valid_steps.unsqueeze(-1).expand_as(best_traj)
        num_valid = full_mask.sum().clamp(min=1.0)
        loss_traj = F.l1_loss(
            best_traj * full_mask,
            gt_locs * full_mask,
            reduction='sum') / num_valid

        cls_valid = (proposal_mask > 0) & (valid_count > 0)
        if cls_valid.any():
            loss_mode = F.cross_entropy(
                mode_logits[cls_valid], best_mode[cls_valid], reduction='mean')
        else:
            loss_mode = mode_logits.sum() * 0.0

        return dict(
            loss_traj=loss_traj * self.loss_weight,
            loss_mode=loss_mode * self.cls_weight)

    def predict(self, query_feat: torch.Tensor, bev_feat: torch.Tensor,
                query_pos: torch.Tensor):
        traj_preds, mode_logits = self.forward(query_feat, bev_feat, query_pos)
        best_mode = mode_logits.argmax(dim=-1)

        _, _, _, num_steps, _ = traj_preds.shape
        gather_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gather_idx = gather_idx.expand(-1, -1, 1, num_steps, 2)
        best_traj = torch.gather(traj_preds, 2, gather_idx).squeeze(2)
        return best_traj, traj_preds, mode_logits
