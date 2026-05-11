"""Tests for template language-conditioned BEV forecasting."""

from __future__ import annotations

import torch

from projects.BEVFormer.bevformer.dense_heads import \
    LanguageConditionedForecastingHead


PC_RANGE = (-80.0, -48.0, -2.0, 80.0, 48.0, 6.0)


def _make_head(**kwargs):
    defaults = dict(
        bev_dim=8,
        hidden_dims=16,
        text_embed_dims=4,
        vocab_size=34,
        num_steps=6,
        num_classes=3,
        dropout=0.0,
        pc_range=PC_RANGE,
        use_velocity=False,
        use_class_embed=False)
    defaults.update(kwargs)
    return LanguageConditionedForecastingHead(**defaults)


def test_forward_shapes_with_language_tokens():
    head = _make_head()
    bev = torch.randn(2, 8, 12, 20)
    centers = [torch.randn(5, 2) * 10.0, torch.randn(3, 2) * 10.0]
    tokens = [torch.tensor([2, 4, 5, 0]), torch.tensor([2, 7, 5, 0])]
    token_masks = [torch.tensor([1, 1, 1, 0], dtype=torch.bool),
                   torch.tensor([1, 1, 1, 0], dtype=torch.bool)]

    trajs, selected_logits = head.forward_with_selection(
        bev, centers, language_tokens_list=tokens,
        language_token_mask_list=token_masks)

    assert trajs[0].shape == (5, 6, 2)
    assert trajs[1].shape == (3, 6, 2)
    assert selected_logits[0].shape == (5, )
    assert selected_logits[1].shape == (3, )


def test_language_target_mask_gates_trajectory_loss():
    head = _make_head(motion_weight_clamp=(1.0, 1.0))
    with torch.no_grad():
        head.traj_branch.weight.zero_()
        head.traj_branch.bias.zero_()

    bev = torch.zeros(1, 8, 12, 20)
    centers = [torch.zeros(2, 2)]
    gt = torch.zeros(2, 6, 2)
    gt[0] = 100.0
    mask = torch.ones(2, 6, dtype=torch.bool)
    language_target = [torch.tensor([False, True])]
    tokens = [torch.tensor([2, 4, 5, 0])]
    token_masks = [torch.tensor([1, 1, 1, 0], dtype=torch.bool)]

    losses = head.loss(
        bev, centers, None, None, [gt], [mask],
        language_tokens_list=tokens,
        language_token_mask_list=token_masks,
        language_target_mask_list=language_target)

    assert losses['loss_lang_traj'].item() == 0.0
    assert losses['loss_lang_select'].item() > 0.0


def test_grad_flows_to_text_embedding():
    head = _make_head(motion_weight_clamp=(1.0, 1.0))
    bev = torch.randn(1, 8, 12, 20)
    centers = [torch.zeros(1, 2)]
    gt = torch.ones(1, 6, 2)
    mask = torch.ones(1, 6, dtype=torch.bool)
    tokens = [torch.tensor([2, 7, 5, 0])]
    token_masks = [torch.tensor([1, 1, 1, 0], dtype=torch.bool)]
    language_target = [torch.tensor([True])]

    losses = head.loss(
        bev, centers, None, None, [gt], [mask],
        language_tokens_list=tokens,
        language_token_mask_list=token_masks,
        language_target_mask_list=language_target)
    total_loss = losses['loss_lang_traj'] + losses['loss_lang_select']
    total_loss.backward()

    grad = head.token_embed.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0
