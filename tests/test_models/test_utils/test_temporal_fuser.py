# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch

from projects.BEVFusion.bevfusion.temporal_fuser import PrevBEVTemporalFuser


def _make_se2(dx: float, dy: float, yaw_deg: float) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    transform = torch.eye(4, dtype=torch.float32)
    transform[0, 0] = cos_yaw
    transform[0, 1] = -sin_yaw
    transform[1, 0] = sin_yaw
    transform[1, 1] = cos_yaw
    transform[0, 3] = dx
    transform[1, 3] = dy
    return transform


def test_prev_bev_shift_rotate_matches_affine_flu():
    fuser = PrevBEVTemporalFuser(
        in_channels=8,
        bev_xbound=[-4.0, 4.0, 1.0],
        bev_ybound=[-4.0, 4.0, 1.0],
        use_motion_embed=False)
    prev_bev = torch.randn(2, 8, 8, 8)
    curr_ego2global = [torch.eye(4), torch.eye(4)]
    prev_ego2global = [
        _make_se2(0.8, -0.4, 12.0),
        _make_se2(-1.2, 0.5, -8.0),
    ]
    lidar_aug = [torch.eye(4), torch.eye(4)]
    hist_to_curr = fuser._hist_to_curr_motion(
        curr_ego2global=curr_ego2global,
        prev_ego2global=prev_ego2global,
        curr_lidar_aug_matrix=lidar_aug,
        prev_lidar_aug_matrix=lidar_aug,
        device=prev_bev.device,
        dtype=prev_bev.dtype)

    warped_sr, _, _, _, _, _ = fuser._warp_prev_bev_shift_rotate(
        prev_bev, hist_to_curr=hist_to_curr, lidar_coord_frame='FLU')
    warped_affine = fuser.warp_bev(
        prev_bev, hist_to_curr, lidar_coord_frame='FLU')

    assert torch.allclose(warped_sr, warped_affine, atol=1e-4, rtol=1e-4)


def test_prev_bev_shift_rotate_matches_affine_rfu():
    fuser = PrevBEVTemporalFuser(
        in_channels=8,
        bev_xbound=[-3.0, 3.0, 1.0],
        bev_ybound=[-3.0, 3.0, 1.0],
        use_motion_embed=False)
    prev_bev = torch.randn(1, 8, 6, 6)
    curr_ego2global = [torch.eye(4)]
    prev_ego2global = [_make_se2(0.5, 0.7, 15.0)]
    lidar_aug = [torch.eye(4)]
    hist_to_curr = fuser._hist_to_curr_motion(
        curr_ego2global=curr_ego2global,
        prev_ego2global=prev_ego2global,
        curr_lidar_aug_matrix=lidar_aug,
        prev_lidar_aug_matrix=lidar_aug,
        device=prev_bev.device,
        dtype=prev_bev.dtype)

    warped_sr, _, _, _, _, _ = fuser._warp_prev_bev_shift_rotate(
        prev_bev, hist_to_curr=hist_to_curr, lidar_coord_frame='RFU')
    warped_affine = fuser.warp_bev(
        prev_bev, hist_to_curr, lidar_coord_frame='RFU')

    assert torch.allclose(warped_sr, warped_affine, atol=1e-4, rtol=1e-4)
