"""Temporal BEV Fuser — align and fuse multi-frame BEV features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BaseTemporalBEVAligner(nn.Module):
    """Shared BEV alignment helpers for temporal fusion modules.

    This base class only owns the ego-motion-to-grid conversion and BEV warp
    utilities. Concrete fusers should implement their own fusion layers and
    ``forward`` methods.

    Args:
        in_channels (int): Channels of a single-frame BEV feature.
        num_adj_frames (int): Number of historical frames to fuse.
        bev_xbound (list): [xmin, xmax, resolution] of BEV grid (meters).
        bev_ybound (list): [ymin, ymax, resolution] of BEV grid (meters).
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_adj_frames: int = 2,
        bev_xbound: list = [-80.0, 80.0, 0.4],
        bev_ybound: list = [-48.0, 48.0, 0.4],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_adj_frames = num_adj_frames

        # BEV grid physical extents
        self.x_min, self.x_max, self.x_res = bev_xbound
        self.y_min, self.y_max, self.y_res = bev_ybound

    # ------------------------------------------------------------------

    def _invert_rigid_transform(self,
                                transform: torch.Tensor) -> torch.Tensor:
        """Invert a batch of rigid 4x4 transforms without MAGMA.

        The ego-motion matrices used here are SE(3)-style rigid transforms,
        so the inverse can be formed analytically:

            T = [R t]
                [0 1]

            T^{-1} = [R^T  -R^T t]
                     [ 0      1  ]
        """
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
                                H: int, W: int) -> torch.Tensor:
        """Convert a 4×4 ego-motion matrix to a 2×3 affine matrix that
        operates on normalised BEV grid coordinates (as required by
        ``F.affine_grid``).

        ``ego_motion`` maps an output-frame point to the source-frame point
        that should be sampled:
            p_src = ego_motion @ p_out  (in LiDAR metres).

        BEVFusionSparseEncoder stores BEV tensors as (B, C, H, W), where
        H follows physical X and W follows physical Y.  ``affine_grid`` uses
        normalized grid coordinates ordered as (column, row), so the grid
        vector is [norm_y, norm_x], not [norm_x, norm_y].

        Args:
            ego_motion: (B, 4, 4) float tensor.
            H, W: spatial size of BEV feature map.

        Returns:
            theta: (B, 2, 3) affine matrix for ``F.affine_grid``.
        """
        # Extract 2D rotation and translation in physical [x, y] metres.
        R = ego_motion[:, :2, :2]   # (B, 2, 2)
        t = ego_motion[:, :2, 3]    # (B, 2)    [tx, ty] in metres

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

        # q = meter_to_grid @ (p - center), where
        # p = [x, y] and q = [grid_col(norm_y), grid_row(norm_x)].
        meter_to_grid = torch.zeros(2, 2, dtype=dtype, device=device)
        meter_to_grid[0, 1] = 1.0 / half_y
        meter_to_grid[1, 0] = 1.0 / half_x
        grid_to_meter = torch.zeros(2, 2, dtype=dtype, device=device)
        grid_to_meter[0, 1] = half_x
        grid_to_meter[1, 0] = half_y

        # Convert p_src = R @ p_out + t from metre coordinates to normalized
        # grid coordinates.  The center term keeps non-zero-centered BEV ranges
        # correct as well.
        theta_R = meter_to_grid @ R @ grid_to_meter
        center_src = torch.einsum('bij,j->bi', R, center) + t
        theta_t = (meter_to_grid @ (center_src - center).T).T

        theta = torch.cat(
            [theta_R, theta_t.unsqueeze(-1)], dim=-1)  # (B, 2, 3)
        return theta

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
        """Warp a historical BEV feature to the current ego frame.

        Args:
            bev: (B, C, H, W) historical BEV feature.
            ego_motion: (B, 4, 4) transform from hist ego → current ego,
                expressed in the EGO frame (X=forward, Y=left).
            lidar_coord_frame: 'RFU' or 'FLU'. Read from pkl metainfo
                (`lidar_coord_frame`) by the parent detector and forwarded
                here. 'RFU' applies an extra ego↔lidar 90° rotation; 'FLU'
                uses the ego motion directly.

        Returns:
            Warped BEV: (B, C, H, W).
        """
        B, C, H, W = bev.shape

        lidar_motion = self._ego_to_lidar_motion(
            ego_motion, lidar_coord_frame=lidar_coord_frame)

        # F.affine_grid semantics: for each output pixel p_curr, the source
        # sampling position is theta @ p_curr.  We need curr→hist direction,
        # i.e. inv(lidar_motion), because lidar_motion maps hist→curr.
        lidar_motion_inv = self._invert_rigid_transform(lidar_motion)
        theta = self._ego_motion_to_affine2d(lidar_motion_inv, H, W)
        grid = F.affine_grid(theta, [B, C, H, W], align_corners=False)
        warped = F.grid_sample(bev, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
        return warped


@MODELS.register_module()
class TemporalBEVFuser(BaseTemporalBEVAligner):
    """Fuse current-frame BEV features with ego-motion-aligned history BEV.

    Pipeline:
        1. For each historical BEV, warp it to the current ego frame using
           a 2D affine transform derived from the 3D ego-motion matrix.
        2. Concatenate [current, aligned_hist_1, aligned_hist_2, ...].
        3. 1×1 conv to reduce channel dim back to ``out_channels``.

    Args:
        in_channels (int): Channels of a single-frame BEV feature.
        num_adj_frames (int): Number of historical frames to fuse.
        bev_xbound (list): [xmin, xmax, resolution] of BEV grid (meters).
        bev_ybound (list): [ymin, ymax, resolution] of BEV grid (meters).
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_adj_frames: int = 2,
        bev_xbound: list = [-80.0, 80.0, 0.4],
        bev_ybound: list = [-48.0, 48.0, 0.4],
    ):
        super().__init__(
            in_channels=in_channels,
            num_adj_frames=num_adj_frames,
            bev_xbound=bev_xbound,
            bev_ybound=bev_ybound)
        self.fuser_mode = 'history_queue'

        # 1×1 conv: concat (1 + num_adj) frames → out_channels
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels * (1 + num_adj_frames), in_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, current_bev: torch.Tensor,
                adj_bevs: list, ego_motions: list,
                lidar_coord_frame: str = 'FLU') -> torch.Tensor:
        """
        Args:
            current_bev: (B, C, H, W) current-frame BEV feature.
            adj_bevs: list of (B, C, H, W) tensors, length = num_adj_frames.
                      Entry can be None if the historical frame is unavailable
                      (scene boundary).
            ego_motions: list of (B, 4, 4) tensors, same length.
            lidar_coord_frame: 'RFU' or 'FLU'. Sourced from pkl metainfo by
                the parent detector (no config knob to avoid drift).

        Returns:
            fused_bev: (B, C, H, W) temporally fused BEV feature.
        """
        aligned = [current_bev]
        for bev, motion in zip(adj_bevs, ego_motions):
            if bev is None:
                # Scene boundary — just duplicate current frame
                aligned.append(torch.zeros_like(current_bev))
            else:
                aligned.append(self.warp_bev(bev, motion, lidar_coord_frame))

        # concat along channel dim then reduce
        x = torch.cat(aligned, dim=1)  # (B, C*(1+N), H, W)
        return self.reduce_conv(x)


@MODELS.register_module()
class PerCellTemporalFuser(BaseTemporalBEVAligner):
    """Per-cell temporal attention fuser over ego-motion-aligned BEV history.

    For each BEV cell, the current-frame token attends across the temporal
    dimension to a sequence of [current, hist_1, ..., hist_N] tokens taken
    from the same cell. A learnable temporal embedding is added to K/V to
    let the attention distinguish timesteps; without it (or without putting
    `current` into K/V), with `num_adj_frames=1` the attention degenerates to
    softmax(1)=identity and the layer collapses to a fixed `current+hist_1`
    fusion.

    Note: this is *not* a faithful reproduction of BEVFormer TSA — there is no
    deformable attention and no spatial cross-cell mixing. It is a per-cell
    temporal Transformer block, closer in spirit to SOLOFusion / StreamPETR's
    time-axis attention. Use the :class:`TemporalBEVFuser` for the simpler
    concat+conv fusion baseline.
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_adj_frames: int = 1,
        bev_xbound: list = [-80.0, 80.0, 0.4],
        bev_ybound: list = [-48.0, 48.0, 0.4],
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: float = 2.0,
    ):
        super().__init__(
            in_channels=in_channels,
            num_adj_frames=num_adj_frames,
            bev_xbound=bev_xbound,
            bev_ybound=bev_ybound)
        self.fuser_mode = 'history_queue'
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

        # One learnable embedding per K/V slot: index 0 = current, 1.. = hist.
        # Required so attention can tell timesteps apart; otherwise with
        # num_adj_frames=1 the softmax degenerates to identity.
        self.temporal_embed = nn.Parameter(
            torch.zeros(1 + num_adj_frames, in_channels))
        nn.init.normal_(self.temporal_embed, std=0.02)

    def forward(self, current_bev: torch.Tensor,
                adj_bevs: list, ego_motions: list,
                lidar_coord_frame: str = 'FLU') -> torch.Tensor:
        """Fuse history BEV memories with per-cell temporal attention.

        Args:
            current_bev: (B, C, H, W)
            adj_bevs: list of (B, C, H, W) or None
            ego_motions: list of (B, 4, 4)
            lidar_coord_frame: 'RFU' or 'FLU'

        Returns:
            (B, C, H, W) temporally updated current BEV.
        """
        batch_size, channels, height, width = current_bev.shape

        aligned_hist = []
        for bev, motion in zip(adj_bevs, ego_motions):
            if bev is None:
                aligned_hist.append(torch.zeros_like(current_bev))
            else:
                aligned_hist.append(
                    self.warp_bev(bev, motion, lidar_coord_frame))

        # Current query tokens: (B*H*W, 1, C)
        current_tokens = current_bev.permute(0, 2, 3, 1).reshape(
            batch_size * height * width, 1, channels)

        # K/V = [current, *aligned_hist] → (B, T, C, H, W) → (B*H*W, T, C),
        # where T = 1 + num_adj_frames.
        all_frames = torch.stack([current_bev] + aligned_hist, dim=1)
        num_tokens = all_frames.shape[1]
        memory_tokens = all_frames.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, num_tokens, channels)
        memory_tokens = memory_tokens + self.temporal_embed.unsqueeze(0)

        attn_out, _ = self.temporal_attn(
            query=current_tokens,
            key=memory_tokens,
            value=memory_tokens)
        x = self.norm1(current_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return x.contiguous()


@MODELS.register_module()
class PrevBEVTemporalFuser(BaseTemporalBEVAligner):
    """BEVFormer-inspired recurrent prev_bev fuser for LiDAR BEV features.

    This module consumes a single ``prev_bev`` per sample, explicitly
    decomposes the relative motion into BEV-plane ``shift + rotate``, warps
    the previous BEV into the current frame, then updates the current BEV with
    per-cell temporal attention over ``[prev_bev, current]``.

    Unlike BEVFormer's encoder-integrated TSA, this stays on top of an
    existing LiDAR BEV backbone. The goal is to preserve the detector while
    switching the temporal contract from an explicit multi-frame queue to a
    recurrent ``prev_bev`` memory.
    """

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
        """Build prev-augmented-ego -> curr-augmented-ego transforms."""
        batch_size = len(curr_ego2global)
        motion = torch.eye(
            4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(batch_size):
            prev_global = prev_ego2global[i]
            if prev_global is None:
                continue

            curr_global = torch.as_tensor(
                curr_ego2global[i], device=device, dtype=dtype)
            prev_global = torch.as_tensor(
                prev_global, device=device, dtype=dtype)
            curr_aug = torch.as_tensor(
                curr_lidar_aug_matrix[i], device=device, dtype=dtype)
            prev_aug = torch.as_tensor(
                prev_lidar_aug_matrix[i], device=device, dtype=dtype)

            prev_to_curr = torch.linalg.inv(curr_global) @ prev_global
            motion[i] = curr_aug @ prev_to_curr @ torch.linalg.inv(prev_aug)
        return motion

    def _decompose_bev_motion(self,
                              hist_to_curr: torch.Tensor,
                              lidar_coord_frame: str,
                              height: int,
                              width: int):
        """Decompose motion into explicit BEV rotation and shift terms.

        Returns:
            curr_to_prev (Tensor): (B, 4, 4) source-sampling motion.
            shift_m (Tensor): (B, 2) XY translation in meters, curr -> prev.
            shift_norm (Tensor): (B, 2) normalized BEV-grid shift
                ordered as [grid_x(y-axis), grid_y(x-axis)].
            yaw (Tensor): (B,) rotation angle in radians, curr -> prev.
            hist_yaw (Tensor): (B,) rotation angle in radians, prev -> curr.
        """
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
        """Build grid_sample coordinates from explicit BEV shift + rotate."""
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
        rotated = torch.matmul(
            rot, base.unsqueeze(-1)).squeeze(-1)
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
                hist_to_curr,
                lidar_coord_frame=lidar_coord_frame,
                height=height,
                width=width))
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
        """Fuse a ``prev_bev`` into the current frame."""
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


@MODELS.register_module()
class ConvTemporalEncoder(BaseTemporalBEVAligner):
    """Recurrent BEV memory fuser (BEVDet4D-style concat + conv).

    Designed to consume a single recurrent ``prev_bev`` cached by the parent
    detector across iterations. The cached BEV is warped from its own ego
    frame into the current ego frame, concatenated channel-wise with the
    current BEV, and reduced back to ``in_channels`` by a small conv block.
    A residual connection lets cold-start slots (no history available) pass
    the current BEV through unchanged.

    Args:
        in_channels: Channels of one BEV feature map.
        bev_xbound / bev_ybound: BEV physical extents (meters).
        hidden_channels: Channels in the intermediate conv. Defaults to
            ``in_channels``.
    """

    def __init__(
        self,
        in_channels: int = 512,
        bev_xbound: list = [-80.0, 80.0, 0.4],
        bev_ybound: list = [-48.0, 48.0, 0.4],
        hidden_channels: int = None,
    ):
        super().__init__(
            in_channels=in_channels,
            num_adj_frames=1,
            bev_xbound=bev_xbound,
            bev_ybound=bev_ybound)
        hidden_channels = hidden_channels or in_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    @staticmethod
    def _relative_motion(curr_ego2global, prev_ego2global, device, dtype):
        """Build per-sample (hist-ego → curr-ego) transforms.

        Returns a ``(B, 4, 4)`` tensor; cold slots get identity (won't be
        consumed because their warped_prev is zeroed downstream anyway).
        """
        B = len(curr_ego2global)
        out = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(
            B, 1, 1)
        for i, (curr, prev) in enumerate(
                zip(curr_ego2global, prev_ego2global)):
            if prev is None:
                continue
            curr_t = torch.as_tensor(curr, device=device, dtype=dtype)
            prev_t = torch.as_tensor(prev, device=device, dtype=dtype)
            out[i] = torch.linalg.inv(curr_t) @ prev_t
        return out

    def forward(self,
                curr_bev: torch.Tensor,
                prev_bev: torch.Tensor,
                cold_mask: torch.Tensor,
                curr_ego2global: list,
                prev_ego2global: list,
                lidar_coord_frame: str = 'FLU') -> torch.Tensor:
        """
        Args:
            curr_bev: (B, C, H, W) current-frame BEV after pts_neck.
            prev_bev: (B, C, H, W) cached previous BEV (zeros at cold slots).
            cold_mask: (B,) bool, True where the slot has no valid history.
            curr_ego2global: list of length B, np.ndarray (4,4) per sample.
            prev_ego2global: list of length B, np.ndarray (4,4) or None.
            lidar_coord_frame: 'RFU' or 'FLU'; passed through to ``warp_bev``.

        Returns:
            (B, C, H, W) — fused BEV. Cold slots return ``curr_bev``
            unchanged via the residual path (the conv branch sees zeros).
        """
        relative = self._relative_motion(
            curr_ego2global, prev_ego2global,
            device=curr_bev.device, dtype=curr_bev.dtype)
        warped_prev = self.warp_bev(prev_bev, relative, lidar_coord_frame)

        # Zero out warped history at cold slots so the conv path can't
        # leak whatever was in `prev_bev` (it should already be zeros, but
        # be defensive — a stale cache entry under a different scene_token
        # would otherwise contaminate the cold output).
        if cold_mask.any():
            warped_prev = warped_prev.masked_fill(
                cold_mask.view(-1, 1, 1, 1), 0.0)

        fused = self.fuse(torch.cat([curr_bev, warped_prev], dim=1))
        out = curr_bev + fused
        # Cold slots bypass the fused branch entirely so the first frame
        # of any scene is exactly the single-frame BEV — no random conv
        # response to the zero half of `[curr, 0]` leaks into them.
        if cold_mask.any():
            out = torch.where(
                cold_mask.view(-1, 1, 1, 1), curr_bev, out)
        return out


# Backwards-compat alias for configs that still reference the old name.
MODELS.register_module(
    name='BEVFormerStyleTemporalFuser', module=PerCellTemporalFuser)
