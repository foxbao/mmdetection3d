"""Unit tests for TransformerForecastingHead (Phase 3 / B+).

Mirrors B's test coverage where applicable, plus B+ specific:
  * self-attention actually couples queries (changing one query's input
    must shift another query's output)
  * cross-attention reads global BEV (placing motion signal far from a
    query's local cell still reaches it)
  * 2D positional encoding is non-trivial (different positions →
    different memory tokens)
"""

from __future__ import annotations

import torch

from projects.BEVFormer.bevformer.dense_heads import TransformerForecastingHead


PC_RANGE = (-80.0, -48.0, -2.0, 80.0, 48.0, 6.0)


def _make_head(**kwargs):
    defaults = dict(
        bev_dim=8, embed_dims=16, num_layers=1, num_heads=2,
        ffn_dims=32, num_steps=6, num_classes=3,
        dropout=0.0, pc_range=PC_RANGE,
        use_velocity=False, use_class_embed=False)
    defaults.update(kwargs)
    return TransformerForecastingHead(**defaults)


# ----------------------------- shapes -------------------------------------

def test_forward_shapes_per_sample():
    head = _make_head().eval()
    bev = torch.randn(2, 8, 12, 20)
    centers = [torch.randn(5, 2) * 30.0,
               torch.randn(3, 2) * 30.0]
    out = head(bev, centers)
    assert len(out) == 2
    assert out[0].shape == (5, 6, 2)
    assert out[1].shape == (3, 6, 2)


def test_empty_batch_sample_no_crash():
    head = _make_head().eval()
    bev = torch.randn(2, 8, 12, 20)
    centers = [torch.zeros(0, 2), torch.randn(2, 2) * 30.0]
    out = head(bev, centers)
    assert out[0].shape == (0, 6, 2)
    assert out[1].shape == (2, 6, 2)


def test_output_is_cumsum():
    """Per-step deltas → cumsum convention: last step magnitude ≥ first step
    on average over many random outputs (cumulative grows). Smoke test only."""
    head = _make_head(num_layers=1).eval()
    torch.manual_seed(0)
    bev = torch.randn(1, 8, 12, 20)
    centers = [torch.randn(20, 2) * 30.0]
    out = head(bev, centers)[0]   # (20, 6, 2)
    last = out[:, -1].abs().sum()
    first = out[:, 0].abs().sum()
    # Cumsum doesn't strictly guarantee growth, but with random deltas the
    # last cumulative magnitude should typically exceed the first.
    assert last >= first * 0.5  # very loose smoke


# ------------------------- coord conversion --------------------------------

def test_lidar_to_grid_norm_boundaries():
    head = _make_head()
    centers = torch.tensor([
        [0.0, 0.0], [80.0, 48.0], [-80.0, -48.0]])
    norm = head._to_grid_norm(centers)
    expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    assert torch.allclose(norm, expected, atol=1e-6)


# ------------------------- BEV positional encoding ------------------------

def test_bev_pos_embed_distinguishes_positions():
    """Different (H, W) cells get different sinusoidal embeddings."""
    head = _make_head(embed_dims=16)
    pos = head._bev_sine_pos(H=12, W=20, device='cpu', dtype=torch.float32)
    assert pos.shape == (12, 20, 16)
    # Adjacent cells should differ
    assert not torch.allclose(pos[0, 0], pos[0, 1])
    assert not torch.allclose(pos[0, 0], pos[1, 0])
    # Far cells should differ more (rough check)
    diff_close = (pos[0, 0] - pos[0, 1]).abs().sum()
    diff_far = (pos[0, 0] - pos[11, 19]).abs().sum()
    assert diff_far > diff_close


# ------------------------- self-attention coupling ------------------------

def test_self_attention_couples_queries():
    """Two queries in the same batch are processed together — changing
    one query's input MUST shift the other query's output (proves
    self-attention is wired and not a no-op)."""
    head = _make_head(num_layers=1, dropout=0.0).eval()
    bev = torch.zeros(1, 8, 12, 20)  # zero BEV: forces signal to come from queries

    centers = torch.tensor([[10.0, 0.0], [-10.0, 0.0]])

    # Force a class-embed difference so the query inputs differ; otherwise
    # both queries land on the same zero feature and same query-pos hash.
    head_cls = _make_head(num_layers=1, dropout=0.0,
                          use_class_embed=True).eval()

    # Run with both queries having class 0
    labels_a = [torch.tensor([0, 0])]
    out_a = head_cls(bev, [centers], labels_list=labels_a)[0]

    # Now flip query 1's class — query 0's output should change
    # (only possible via self-attention since BEV is zero)
    labels_b = [torch.tensor([0, 2])]
    out_b = head_cls(bev, [centers], labels_list=labels_b)[0]

    # Query 0 output must shift even though its OWN inputs are unchanged
    delta_q0 = (out_a[0] - out_b[0]).abs().sum()
    assert delta_q0 > 1e-4, \
        f'query 0 output unchanged when query 1 class flipped — ' \
        f'self-attention not wired? delta={delta_q0:.6f}'


# ------------------------- cross-attention to BEV --------------------------

def test_cross_attention_reads_global_bev():
    """Single-query forward: changing BEV at a cell FAR from the query must
    still affect the output (proves cross-attention spans the full grid)."""
    head = _make_head(num_layers=1, dropout=0.0).eval()
    centers = [torch.tensor([[0.0, 0.0]])]   # query at lidar origin → BEV center

    bev_a = torch.zeros(1, 8, 12, 20)
    out_a = head(bev_a, centers)[0]

    # Place a strong feature at a corner of BEV (lidar (-80, -48), far from origin)
    bev_b = torch.zeros(1, 8, 12, 20)
    bev_b[0, :, 0, 0] = 5.0
    out_b = head(bev_b, centers)[0]

    delta = (out_a - out_b).abs().sum()
    assert delta > 1e-3, \
        f'corner BEV change did not affect center query — cross-attn dead? ' \
        f'delta={delta:.6f}'


# --------------------------- loss + grad ----------------------------------

def test_loss_all_masked_out_gives_zero():
    head = _make_head(motion_weight_clamp=(1.0, 1.0)).eval()
    bev = torch.zeros(1, 8, 12, 20)
    centers = [torch.zeros(1, 2)]
    gt = torch.full((1, 6, 2), 100.0)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    losses = head.loss(bev, centers, None, None, [gt], [mask])
    assert losses['loss_traj'].item() == 0.0


def test_loss_grad_flows_through_decoder():
    head = _make_head(motion_weight_clamp=(1.0, 1.0)).train()
    bev = torch.randn(1, 8, 12, 20)
    centers = [torch.zeros(2, 2)]
    gt = torch.full((2, 6, 2), 1.0)
    mask = torch.ones(2, 6, dtype=torch.bool)
    losses = head.loss(bev, centers, None, None, [gt], [mask])
    losses['loss_traj'].backward()
    # Decoder layer params should have grad
    decoder_params = list(head.decoder.parameters())
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in decoder_params)
    assert has_grad, 'no grad reached transformer decoder'


def test_dtype_no_upcast_from_float64_gt():
    head = _make_head().float().eval()
    bev = torch.randn(1, 8, 12, 20).float()
    centers = [torch.zeros(1, 2)]
    gt = torch.zeros(1, 6, 2, dtype=torch.float64)
    mask = torch.ones(1, 6, dtype=torch.bool)
    losses = head.loss(bev, centers, None, None, [gt], [mask])
    assert losses['loss_traj'].dtype == torch.float32


def test_use_velocity_and_class_embed_extends_query_dim():
    head = TransformerForecastingHead(
        bev_dim=8, embed_dims=16, num_layers=1, num_heads=2, ffn_dims=32,
        num_steps=6, num_classes=3, pc_range=PC_RANGE,
        use_velocity=True, use_class_embed=True, dropout=0.0)
    # query proj input dim: 8 (bev) + 2 (vel) + 3 (class) = 13
    assert head.query_proj[0].in_features == 13

    bev = torch.randn(1, 8, 12, 20)
    centers = [torch.tensor([[0.0, 0.0]])]
    velocities = [torch.tensor([[1.5, -0.3]])]
    labels = [torch.tensor([1])]
    out = head(bev, centers, velocities, labels)
    assert out[0].shape == (1, 6, 2)
