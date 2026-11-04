import pytest
import torch

from tokenworld.dynamics.model import DynamicsWorldModel


def _make_model(**overrides):
    kwargs = dict(
        dim=32,
        dim_latent=4,
        num_latent_tokens=6,
        num_spatial_tokens=6,
        spatial_grid_size=(2, 3),
        num_agents=1,
        num_video_views=2,
        video_tokenizer=None,
        depth=6,
        time_block_every=3,
        num_discrete_actions=(2, 2),
        num_continuous_actions=12,
        dim_proprio=None,
        max_steps=8,
        attn_heads=4,
        attn_dim_head=8,
        use_time_rnn=False,
    )
    kwargs.update(overrides)
    return DynamicsWorldModel(**kwargs)


def test_native_tokens_remain_one_to_one_spatial_tokens():
    model = _make_model()
    latents = torch.randn(2, 5, 2, 6, 4)

    spatial_tokens = model.latents_to_spatial_tokens(latents)

    assert spatial_tokens.shape == (2, 5, 2, 6, 32)
    assert model.to_latent_pred(spatial_tokens).shape == latents.shape


def test_factorized_2d_positions_follow_row_major_grid():
    model = _make_model()

    positions = model.get_spatial_pos_emb()

    assert positions.shape == (6, 32)
    torch.testing.assert_close(
        positions[0] - positions[1],
        model.spatial_col_pos_emb[0] - model.spatial_col_pos_emb[1],
    )
    torch.testing.assert_close(
        positions[0] - positions[3],
        model.spatial_row_pos_emb[0] - model.spatial_row_pos_emb[1],
    )


def test_four_temporal_blocks_for_recommended_schedule():
    model = _make_model(depth=12, time_block_every=3)
    assert model.transformer.is_time == [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_spatial_grid_must_match_spatial_token_count():
    with pytest.raises(AssertionError, match="must contain exactly 6 tokens"):
        _make_model(spatial_grid_size=(2, 4))


def test_single_view_native_axis_is_not_added_twice():
    model = _make_model(
        num_video_views=1,
        dim_proprio=14,
        prob_no_shortcut_train=1.0,
    )
    loss, *_ = model(
        vlm_latents=torch.randn(1, 4, 1, 6, 4),
        discrete_actions=torch.zeros(1, 3, 2, dtype=torch.long),
        continuous_actions=torch.zeros(1, 3, 12),
        proprio=torch.zeros(1, 4, 14),
        rewards=torch.zeros(1, 4),
        train_decoder=False,
        training_mode="world_model",
    )
    assert torch.isfinite(loss)
    loss.backward()
