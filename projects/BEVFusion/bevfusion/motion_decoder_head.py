import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class MotionDecoderHead(nn.Module):

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 256,
        forecast_steps: int = 6,
        num_modes: int = 3,
        num_attn_layers: int = 2,
        num_attn_heads: int = 8,
        dropout: float = 0.1,
        loss_weight: float = 0.5,
        cls_weight: float = 0.2,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_modes = num_modes
        self.loss_weight = loss_weight
        self.cls_weight = cls_weight

        self.input_proj = nn.Linear(in_channels, hidden_dim)
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

    def forward(self, query_feat: torch.Tensor):
        """Decode multi-modal trajectories from proposal queries.

        Args:
            query_feat: (B, C, N)

        Returns:
            tuple:
                traj_preds: (B, N, K, T, 2)
                mode_logits: (B, N, K)
        """
        x = query_feat.permute(0, 2, 1)  # (B, N, C)
        x = self.input_proj(x)
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
        """Best-of-K motion loss with matched-proposal masking."""
        _, _, num_modes, num_steps, _ = traj_preds.shape

        valid_steps = proposal_mask.unsqueeze(-1) * gt_mask.float()  # (B,N,T)
        valid_count = valid_steps.sum(dim=-1)  # (B,N)

        gt_expand = gt_locs.unsqueeze(2).expand(-1, -1, num_modes, -1, -1)
        mask_expand = valid_steps.unsqueeze(2).unsqueeze(-1)
        mask_expand = mask_expand.expand_as(traj_preds)

        l1 = F.l1_loss(
            traj_preds * mask_expand,
            gt_expand * mask_expand,
            reduction='none').sum(dim=(-1, -2))  # (B,N,K)
        ade = l1 / (valid_count.unsqueeze(-1).clamp(min=1.0) * 2.0)
        best_mode = ade.argmin(dim=-1)  # (B,N)

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

    def predict(self, query_feat: torch.Tensor):
        """Return top-1 and all-mode trajectory predictions."""
        traj_preds, mode_logits = self.forward(query_feat)
        best_mode = mode_logits.argmax(dim=-1)

        _, _, _, num_steps, _ = traj_preds.shape
        gather_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gather_idx = gather_idx.expand(-1, -1, 1, num_steps, 2)
        best_traj = torch.gather(traj_preds, 2, gather_idx).squeeze(2)
        return best_traj, traj_preds, mode_logits
