import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class MotionHead(nn.Module):

    def __init__(
        self,
        in_channels: int = 128,
        forecast_steps: int = 6,
        hidden_channels: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        loss_weight: float = 0.5,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.loss_weight = loss_weight
        output_dim = forecast_steps * 2

        layers = []
        ch_in = in_channels
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(ch_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            ch_in = hidden_channels
        layers.append(nn.Linear(ch_in, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, query_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feat: (B, C, num_proposals) from TransFusionHead decoder.
        Returns:
            (B, num_proposals, forecast_steps, 2)
        """
        B, C, N = query_feat.shape
        x = query_feat.permute(0, 2, 1)
        x = self.mlp(x)
        return x.view(B, N, self.forecast_steps, 2)

    def loss(self, traj_preds, gt_locs, gt_mask, proposal_mask):
        """Masked L1 loss on trajectory predictions.

        Args:
            traj_preds: (B, N, 6, 2)
            gt_locs: (B, N, 6, 2)
            gt_mask: (B, N, 6) bool/float
            proposal_mask: (B, N) float — 1 for matched proposals
        """
        full_mask = (proposal_mask.unsqueeze(-1)
                     * gt_mask.float()).unsqueeze(-1)  # (B, N, 6, 1)
        full_mask = full_mask.expand_as(traj_preds)  # (B, N, 6, 2)

        num_valid = full_mask.sum().clamp(min=1.0)
        loss_traj = F.l1_loss(
            traj_preds * full_mask,
            gt_locs * full_mask,
            reduction='sum',
        ) / num_valid

        return dict(loss_traj=loss_traj * self.loss_weight)
