"""Temporal BEV Fuser — align and fuse multi-frame BEV features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TemporalBEVFuser(nn.Module):
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
        bev_xbound: list = [-48.0, 48.0, 0.4],
        bev_ybound: list = [-80.0, 80.0, 0.4],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_adj_frames = num_adj_frames

        # BEV grid physical extents
        self.x_min, self.x_max, self.x_res = bev_xbound
        self.y_min, self.y_max, self.y_res = bev_ybound

        # 1×1 conv: concat (1 + num_adj) frames → out_channels
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels * (1 + num_adj_frames), in_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------

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

        if lidar_coord_frame == 'RFU':
            # LiDAR uses RFU (col-0=Right, col-1=Forward, col-2=Up) while
            # ego is FLU (X=Forward, Y=Left, Z=Up). Convert ego_motion to
            # the LiDAR frame:
            #     lidar_motion = inv(lidar2ego) @ ego_motion @ lidar2ego
            # with lidar2ego_R = [[0,1,0],[-1,0,0],[0,0,1]] (90° about Z).
            lidar2ego = torch.eye(
                4, device=ego_motion.device, dtype=ego_motion.dtype)
            lidar2ego[0, 0] = 0; lidar2ego[0, 1] =  1
            lidar2ego[1, 0] = -1; lidar2ego[1, 1] = 0
            ego2lidar = torch.inverse(lidar2ego)
            lidar_motion = ego2lidar @ ego_motion @ lidar2ego
        elif lidar_coord_frame == 'FLU':
            lidar_motion = ego_motion
        else:
            raise ValueError(
                f'unknown lidar_coord_frame: {lidar_coord_frame!r}')

        # F.affine_grid semantics: for each output pixel p_curr, the source
        # sampling position is theta @ p_curr.  We need curr→hist direction,
        # i.e. inv(lidar_motion), because lidar_motion maps hist→curr.
        lidar_motion_inv = torch.inverse(lidar_motion)
        theta = self._ego_motion_to_affine2d(lidar_motion_inv, H, W)
        grid = F.affine_grid(theta, [B, C, H, W], align_corners=False)
        warped = F.grid_sample(bev, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
        return warped

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
