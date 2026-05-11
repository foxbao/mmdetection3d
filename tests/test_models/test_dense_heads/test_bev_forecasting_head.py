"""Unit tests for BEVForecastingHead (Phase 3 Step 1).

Covers:
  * shape correctness (batch + variable N per sample)
  * empty N=0 graceful handling
  * cumulative-displacement output (cumsum convention matches GT)
  * lidar→grid coord mapping (boundary + center)
  * bilinear-sample correctness (constant feature → constant sample)
  * mask gating (masked-out steps don't contribute to loss)
  * dtype safety (float64 GT doesn't poison float32 output)
  * gradient flow (loss.backward populates MLP grads)
  * motion weighting (stationary vs fast obj weights match clamp bounds)
  * use_velocity / use_class_embed extends MLP input dim correctly
"""

from __future__ import annotations

import torch

from projects.BEVFormer.bevformer.dense_heads import BEVForecastingHead


PC_RANGE = (-80.0, -48.0, -2.0, 80.0, 48.0, 6.0)


def _make_head(**kwargs):
    defaults = dict(
        embed_dims=8, hidden_dims=16, num_steps=6, num_classes=3,
        dropout=0.0, pc_range=PC_RANGE,
        use_velocity=False, use_class_embed=False)
    defaults.update(kwargs)
    return BEVForecastingHead(**defaults)


# ----------------------------- forward shapes -------------------------------

def test_forward_shapes_per_sample():
    head = _make_head()
    bev = torch.randn(2, 8, 12, 20)
    centers = [torch.randn(5, 2) * 30.0,
               torch.randn(3, 2) * 30.0]
    out = head(bev, centers)
    assert len(out) == 2
    assert out[0].shape == (5, 6, 2)
    assert out[1].shape == (3, 6, 2)


def test_empty_batch_sample_no_crash():
    head = _make_head()
    bev = torch.randn(2, 8, 12, 20)
    centers = [torch.zeros(0, 2), torch.randn(2, 2) * 30.0]
    out = head(bev, centers)
    assert out[0].shape == (0, 6, 2)
    assert out[1].shape == (2, 6, 2)


# ----------------------------- output convention ----------------------------

def test_cumsum_output_convention():
    """Force MLP last layer to emit constant +0.1 per step → cumsum = ramp."""
    head = _make_head()
    last = head.mlp[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(0.1)
    bev = torch.randn(1, 8, 12, 20)
    centers = [torch.zeros(1, 2)]
    out = head(bev, centers)[0]  # (1, 6, 2)
    expected = torch.linspace(0.1, 0.6, steps=6).view(1, 6, 1).expand(1, 6, 2)
    assert torch.allclose(out, expected, atol=1e-5)


# ----------------------------- coord conversion -----------------------------

def test_lidar_to_grid_norm_boundaries():
    head = _make_head()
    centers = torch.tensor([
        [0.0, 0.0],          # center → (0, 0)
        [80.0, 48.0],        # +x +y boundary → (1, 1)
        [-80.0, -48.0],      # -x -y boundary → (-1, -1)
        [40.0, 0.0],         # half +x → (0.5, 0)
    ])
    norm = head._to_grid_norm(centers)
    expected = torch.tensor([
        [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [0.5, 0.0]])
    assert torch.allclose(norm, expected, atol=1e-6)


def test_bilinear_sample_constant_feature():
    """Constant BEV feature → sampled value equals that constant everywhere."""
    head = _make_head()
    bev = torch.full((1, 8, 12, 20), fill_value=2.5)
    centers = torch.tensor([[0.0, 0.0], [10.0, 5.0], [-30.0, -20.0]])
    sampled = head._sample_bev(bev, centers)
    assert sampled.shape == (3, 8)
    assert torch.allclose(sampled, torch.full((3, 8), 2.5), atol=1e-5)


# ----------------------------- loss behaviour -------------------------------

def test_loss_all_masked_out_gives_zero():
    """If every step is masked False, loss must be 0 — no contribution from
    GT magnitudes that the head was told to ignore."""
    head = _make_head(motion_weight_clamp=(1.0, 1.0))
    bev = torch.zeros(1, 8, 12, 20)
    centers = [torch.zeros(1, 2)]
    gt = torch.full((1, 6, 2), 100.0)  # huge — but masked out
    mask = torch.zeros(1, 6, dtype=torch.bool)
    losses = head.loss(bev, centers, None, None, [gt], [mask])
    assert losses['loss_traj'].item() == 0.0


def test_loss_nonzero_when_mask_active():
    head = _make_head(motion_weight_clamp=(1.0, 1.0))
    bev = torch.zeros(1, 8, 12, 20)
    centers = [torch.zeros(1, 2)]
    gt = torch.full((1, 6, 2), 1.0)
    mask = torch.ones(1, 6, dtype=torch.bool)
    losses = head.loss(bev, centers, None, None, [gt], [mask])
    assert losses['loss_traj'].item() > 0


def test_loss_dtype_no_upcast_from_float64_gt():
    """KL pkl stores float64 forecasting tensors — must not poison float32 head."""
    head = _make_head().float()
    bev = torch.randn(1, 8, 12, 20).float()
    centers = [torch.zeros(1, 2)]
    gt = torch.zeros(1, 6, 2, dtype=torch.float64)  # GT is float64
    mask = torch.ones(1, 6, dtype=torch.bool)
    losses = head.loss(bev, centers, None, None, [gt], [mask])
    assert losses['loss_traj'].dtype == torch.float32


def test_grad_flows_to_mlp_weights():
    head = _make_head(motion_weight_clamp=(1.0, 1.0))
    bev = torch.randn(1, 8, 12, 20)
    centers = [torch.zeros(1, 2)]
    gt = torch.full((1, 6, 2), 2.0)
    mask = torch.ones(1, 6, dtype=torch.bool)
    losses = head.loss(bev, centers, None, None, [gt], [mask])
    losses['loss_traj'].backward()
    last = head.mlp[-1]
    assert last.weight.grad is not None
    assert last.weight.grad.abs().sum() > 0


def test_motion_weighting_clamp_bounds():
    """Stationary obj weighted at lower clamp; fast obj at upper clamp."""
    head = _make_head(motion_weight_clamp=(0.5, 5.0))
    bev = torch.zeros(1, 8, 12, 20)
    centers = [torch.zeros(2, 2)]

    gt = torch.zeros(2, 6, 2)
    gt[1, -1] = torch.tensor([10.0, 10.0])  # final disp ≈ 14.14, clamps to 5
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, -1] = True

    final_mag = gt[:, -1].float().norm(dim=-1)
    weights = final_mag.clamp(*head.motion_weight_clamp)
    assert weights[0].item() == 0.5  # stationary clamped up to lower bound
    assert weights[1].item() == 5.0  # fast clamped down to upper bound


def test_use_velocity_and_class_embed_extends_input_dim():
    head = BEVForecastingHead(
        embed_dims=8, hidden_dims=16, num_steps=6, num_classes=3,
        pc_range=PC_RANGE, use_velocity=True, use_class_embed=True,
        dropout=0.0)
    # 8 (BEV) + 2 (vel) + 3 (class one-hot) = 13
    assert head.mlp[0].in_features == 13

    bev = torch.randn(1, 8, 12, 20)
    centers = [torch.tensor([[0.0, 0.0]])]
    velocities = [torch.tensor([[1.5, -0.3]])]
    labels = [torch.tensor([1])]
    out = head(bev, centers, velocities, labels)
    assert out[0].shape == (1, 6, 2)


def test_out_of_range_center_uses_zero_padding():
    """Centers outside pc_range get zero-padded BEV samples (don't crash, don't
    fall back to nearest-pixel garbage)."""
    head = _make_head()
    bev = torch.full((1, 8, 12, 20), fill_value=7.0)  # uniform 7.0
    centers = torch.tensor([[200.0, 0.0]])  # well outside x_max=80
    sampled = head._sample_bev(bev, centers)
    # padding_mode='zeros' → far-out center sees 0
    assert torch.allclose(sampled, torch.zeros_like(sampled), atol=1e-5)
