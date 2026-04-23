"""Prev-BEV temporal alignment and fusion for recurrent LiDAR BEV models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BaseTemporalBEVAligner(nn.Module):
    """Shared BEV alignment helpers for recurrent temporal fusion."""

    def __init__(
        self,
        in_channels: int = 512,
        num_adj_frames: int = 1,
        bev_xbound: list = [-80.0, 80.0, 0.4],
        bev_ybound: list = [-48.0, 48.0, 0.4],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_adj_frames = num_adj_frames
        self.x_min, self.x_max, self.x_res = bev_xbound
        self.y_min, self.y_max, self.y_res = bev_ybound

    def _invert_rigid_transform(self,
                                transform: torch.Tensor) -> torch.Tensor:
        """Invert a batch of rigid 4x4 transforms analytically."""
        rot = transform[:, :3, :3]
        trans = transform[:, :3, 3]

        rot_t = rot.transpose(1, 2).contiguous()
        inv = torch.eye(
            4, device=transform.device, dtype=transform.dtype).unsqueeze(0)
        inv = inv.repeat(transform.shape[0], 1, 1)
        inv[:, :3, :3] = rot_t
        inv[:, :3, 3] = -torch.bmm(rot_t, trans.unsqueeze(-1)).squeeze(-1)
        return inv

    def _ego_motion_to_affine2d(self, ego_motion: torch.Tensor,
                                height: int,
                                width: int) -> torch.Tensor:
        """Convert a 4x4 ego-motion matrix into affine_grid coordinates."""
        del height, width

        rot = ego_motion[:, :2, :2]
        trans = ego_motion[:, :2, 3]

        dtype = ego_motion.dtype
        device = ego_motion.device
        half_x = torch.tensor(
            (self.x_max - self.x_min) / 2.0, dtype=dtype, device=device)
        half_y = torch.tensor(
            (self.y_max - self.y_min) / 2.0, dtype=dtype, device=device)
        center = torch.tensor(
            [(self.x_min + self.x_max) / 2.0,
             (self.y_min + self.y_max) / 2.0],
            dtype=dtype,
            device=device)

        meter_to_grid = torch.zeros(2, 2, dtype=dtype, device=device)
        meter_to_grid[0, 1] = 1.0 / half_y
        meter_to_grid[1, 0] = 1.0 / half_x
        grid_to_meter = torch.zeros(2, 2, dtype=dtype, device=device)
        grid_to_meter[0, 1] = half_x
        grid_to_meter[1, 0] = half_y

        theta_rot = meter_to_grid @ rot @ grid_to_meter
        center_src = torch.einsum('bij,j->bi', rot, center) + trans
        theta_trans = (meter_to_grid @ (center_src - center).T).T

        return torch.cat([theta_rot, theta_trans.unsqueeze(-1)], dim=-1)

    def _ego_to_lidar_motion(self,
                             ego_motion: torch.Tensor,
                             lidar_coord_frame: str = 'FLU') -> torch.Tensor:
        """Convert ego-frame motion into the LiDAR frame used by BEV."""
        if lidar_coord_frame == 'RFU':
            lidar2ego = torch.eye(
                4, device=ego_motion.device, dtype=ego_motion.dtype)
            lidar2ego[0, 0] = 0
            lidar2ego[0, 1] = 1
            lidar2ego[1, 0] = -1
            lidar2ego[1, 1] = 0
            ego2lidar = lidar2ego.transpose(0, 1).contiguous()
            return ego2lidar @ ego_motion @ lidar2ego
        if lidar_coord_frame == 'FLU':
            return ego_motion
        raise ValueError(
            f'unknown lidar_coord_frame: {lidar_coord_frame!r}')

    def warp_bev(self, bev: torch.Tensor,
                 ego_motion: torch.Tensor,
                 lidar_coord_frame: str = 'FLU') -> torch.Tensor:
        """Warp a historical BEV feature to the current ego frame."""
        batch_size, channels, height, width = bev.shape
        lidar_motion = self._ego_to_lidar_motion(
            ego_motion, lidar_coord_frame=lidar_coord_frame)
        lidar_motion_inv = self._invert_rigid_transform(lidar_motion)
        theta = self._ego_motion_to_affine2d(
            lidar_motion_inv, height, width)
        grid = F.affine_grid(
            theta, [batch_size, channels, height, width],
            align_corners=False)
        return F.grid_sample(
            bev,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)


@MODELS.register_module()
class PrevBEVTemporalFuser(BaseTemporalBEVAligner):
    """BEVFormer-inspired recurrent prev_bev fuser for LiDAR BEV features."""

    def __init__(
        self,
        in_channels: int = 512,
        bev_xbound: list = [-80.0, 80.0, 0.4],
        bev_ybound: list = [-48.0, 48.0, 0.4],
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: float = 2.0,
        use_motion_embed: bool = True,
        use_can_bus_embed: bool = True,
        can_bus_norm: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            num_adj_frames=1,
            bev_xbound=bev_xbound,
            bev_ybound=bev_ybound)
        self.fuser_mode = 'prev_bev'
        self.use_motion_embed = use_motion_embed
        self.use_can_bus_embed = use_can_bus_embed
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)
        hidden_dim = int(in_channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels),
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.temporal_embed = nn.Parameter(torch.zeros(2, in_channels))
        nn.init.normal_(self.temporal_embed, std=0.02)

        if use_motion_embed:
            self.motion_mlp = nn.Sequential(
                nn.Linear(5, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, in_channels),
            )
        if use_can_bus_embed:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(6, in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // 2, in_channels),
                nn.ReLU(inplace=True),
            )
            if can_bus_norm:
                self.can_bus_mlp.add_module('norm', nn.LayerNorm(in_channels))

    @staticmethod
    def _hist_to_curr_motion(curr_ego2global,
                             prev_ego2global,
                             curr_lidar_aug_matrix,
                             prev_lidar_aug_matrix,
                             device,
                             dtype):
        """Build prev-augmented-ego to curr-augmented-ego transforms."""
        batch_size = len(curr_ego2global)
        motion = torch.eye(
            4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        for idx in range(batch_size):
            prev_global = prev_ego2global[idx]
            if prev_global is None:
                continue

            curr_global = torch.as_tensor(
                curr_ego2global[idx], device=device, dtype=dtype)
            prev_global = torch.as_tensor(
                prev_global, device=device, dtype=dtype)
            curr_aug = torch.as_tensor(
                curr_lidar_aug_matrix[idx], device=device, dtype=dtype)
            prev_aug = torch.as_tensor(
                prev_lidar_aug_matrix[idx], device=device, dtype=dtype)

            prev_to_curr = torch.linalg.inv(curr_global) @ prev_global
            motion[idx] = curr_aug @ prev_to_curr @ torch.linalg.inv(prev_aug)
        return motion

    def _decompose_bev_motion(self,
                              hist_to_curr: torch.Tensor,
                              lidar_coord_frame: str):
        """Decompose motion into BEV shift and rotation terms."""
        lidar_motion = self._ego_to_lidar_motion(
            hist_to_curr, lidar_coord_frame=lidar_coord_frame)
        curr_to_prev = self._invert_rigid_transform(lidar_motion)
        shift_m = curr_to_prev[:, :2, 3]
        yaw = torch.atan2(curr_to_prev[:, 1, 0], curr_to_prev[:, 0, 0])
        hist_yaw = torch.atan2(lidar_motion[:, 1, 0], lidar_motion[:, 0, 0])

        range_x = self.x_max - self.x_min
        range_y = self.y_max - self.y_min
        shift_norm = torch.stack([
            shift_m[:, 1] / range_y,
            shift_m[:, 0] / range_x,
        ], dim=-1)
        return curr_to_prev, shift_m, shift_norm, yaw, hist_yaw

    def _build_shift_rotate_grid(self,
                                 shift_m: torch.Tensor,
                                 yaw: torch.Tensor,
                                 height: int,
                                 width: int) -> torch.Tensor:
        """Build grid_sample coordinates from explicit BEV shift+rotate."""
        device = shift_m.device
        dtype = shift_m.dtype

        xs = torch.linspace(
            self.x_min + 0.5 * self.x_res,
            self.x_max - 0.5 * self.x_res,
            steps=height,
            device=device,
            dtype=dtype)
        ys = torch.linspace(
            self.y_min + 0.5 * self.y_res,
            self.y_max - 0.5 * self.y_res,
            steps=width,
            device=device,
            dtype=dtype)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        cos_yaw = torch.cos(yaw).view(-1, 1, 1)
        sin_yaw = torch.sin(yaw).view(-1, 1, 1)
        rot = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1),
        ], dim=-2)
        rotated = torch.matmul(rot, base.unsqueeze(-1)).squeeze(-1)
        shifted = rotated + shift_m.view(-1, 1, 1, 2)

        grid_col = 2.0 * (shifted[..., 1] - self.y_min) / (
            self.y_max - self.y_min) - 1.0
        grid_row = 2.0 * (shifted[..., 0] - self.x_min) / (
            self.x_max - self.x_min) - 1.0
        return torch.stack([grid_col, grid_row], dim=-1)

    def _warp_prev_bev_shift_rotate(self,
                                    prev_bev: torch.Tensor,
                                    hist_to_curr: torch.Tensor,
                                    lidar_coord_frame: str):
        """Explicit BEVFormer-style shift+rotate warp for prev_bev."""
        _, _, height, width = prev_bev.shape
        curr_to_prev, shift_m, shift_norm, yaw, hist_yaw = (
            self._decompose_bev_motion(
                hist_to_curr, lidar_coord_frame=lidar_coord_frame))
        grid = self._build_shift_rotate_grid(
            shift_m=shift_m, yaw=yaw, height=height, width=width)
        warped = F.grid_sample(
            prev_bev,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        return warped, curr_to_prev, shift_m, shift_norm, yaw, hist_yaw

    def _motion_embedding(self,
                          hist_to_curr: torch.Tensor,
                          shift_norm: torch.Tensor,
                          hist_yaw: torch.Tensor,
                          height: int,
                          width: int) -> torch.Tensor:
        """Encode explicit BEVFormer-style motion features."""
        motion = torch.stack([
            hist_to_curr[:, 0, 3],
            hist_to_curr[:, 1, 3],
            hist_yaw,
            shift_norm[:, 0],
            shift_norm[:, 1],
        ], dim=-1)
        motion_embed = self.motion_mlp(motion)
        motion_embed = motion_embed[:, None, None, :].expand(
            -1, height, width, -1)
        return motion_embed.reshape(-1, 1, self.in_channels)

    def _can_bus_embedding(self,
                           hist_to_curr: torch.Tensor,
                           shift_norm: torch.Tensor,
                           hist_yaw: torch.Tensor,
                           prev_bev_exists: torch.Tensor,
                           height: int,
                           width: int) -> torch.Tensor:
        """Encode BEVFormer-style can_bus-like motion for current queries."""
        can_bus = torch.stack([
            hist_to_curr[:, 0, 3],
            hist_to_curr[:, 1, 3],
            hist_yaw,
            shift_norm[:, 0],
            shift_norm[:, 1],
            prev_bev_exists.to(hist_to_curr.dtype),
        ], dim=-1)
        can_bus_embed = self.can_bus_mlp(can_bus)
        can_bus_embed = can_bus_embed[:, None, None, :].expand(
            -1, height, width, -1)
        return can_bus_embed.reshape(-1, 1, self.in_channels)

    def forward(self,
                curr_bev: torch.Tensor,
                prev_bev: torch.Tensor,
                curr_ego2global: list,
                prev_ego2global: list,
                curr_lidar_aug_matrix: list,
                prev_lidar_aug_matrix: list,
                cold_mask: torch.Tensor,
                prev_bev_exists: torch.Tensor,
                lidar_coord_frame: str = 'FLU') -> torch.Tensor:
        """Fuse a prev_bev into the current frame."""
        batch_size, channels, height, width = curr_bev.shape
        hist_to_curr = self._hist_to_curr_motion(
            curr_ego2global=curr_ego2global,
            prev_ego2global=prev_ego2global,
            curr_lidar_aug_matrix=curr_lidar_aug_matrix,
            prev_lidar_aug_matrix=prev_lidar_aug_matrix,
            device=curr_bev.device,
            dtype=curr_bev.dtype)
        warped_prev, _, _, shift_norm, _, hist_yaw = (
            self._warp_prev_bev_shift_rotate(
                prev_bev,
                hist_to_curr=hist_to_curr,
                lidar_coord_frame=lidar_coord_frame))
        if cold_mask.any():
            warped_prev = warped_prev.masked_fill(
                cold_mask.view(-1, 1, 1, 1), 0.0)

        curr_tokens = curr_bev.permute(0, 2, 3, 1).reshape(
            batch_size * height * width, 1, channels)
        if self.use_can_bus_embed:
            curr_tokens = curr_tokens + self._can_bus_embedding(
                hist_to_curr=hist_to_curr,
                shift_norm=shift_norm,
                hist_yaw=hist_yaw,
                prev_bev_exists=prev_bev_exists,
                height=height,
                width=width)
        prev_tokens = warped_prev.permute(0, 2, 3, 1).reshape(
            batch_size * height * width, 1, channels)

        memory_tokens = torch.cat([prev_tokens, curr_tokens], dim=1)
        memory_tokens = memory_tokens + self.temporal_embed.unsqueeze(0)
        if self.use_motion_embed:
            motion_embed = self._motion_embedding(
                hist_to_curr=hist_to_curr,
                shift_norm=shift_norm,
                hist_yaw=hist_yaw,
                height=height,
                width=width)
            memory_tokens[:, :1, :] = memory_tokens[:, :1, :] + motion_embed

        attn_out, _ = self.temporal_attn(
            query=curr_tokens,
            key=memory_tokens,
            value=memory_tokens)
        out = self.norm1(curr_tokens + attn_out)
        out = self.norm2(out + self.ffn(out))
        out = out.view(batch_size, height, width, channels).permute(0, 3, 1, 2)

        if cold_mask.any():
            out = torch.where(cold_mask.view(-1, 1, 1, 1), curr_bev, out)
        return out.contiguous()
