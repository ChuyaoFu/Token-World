from unittest.mock import patch

import torch

from tokenworld.dynamics.model import DynamicsWorldModel
from tokenworld.evaluation.rollout import WorldModelRollout


def test_resumed_rollout_positions_and_short_block():
    model = DynamicsWorldModel(dim=32, dim_latent=4, num_latent_tokens=6, num_spatial_tokens=6,
                               spatial_grid_size=(2, 3), dim_proprio=14, depth=3, time_block_every=3,
                               num_discrete_actions=(2, 2), num_continuous_actions=12,
                               max_steps=8, attn_heads=4, attn_dim_head=8, use_time_rnn=True).eval()
    api = object.__new__(WorldModelRollout)
    api.world_model, api.num_steps, api.no_shortcut = model, 2, False
    z = torch.randn(1, 1, 6, 4)
    p = torch.zeros(1, 14)
    actions = torch.zeros(1, 4, 12)
    grips = torch.zeros(1, 4, 2, dtype=torch.long)
    prefix = api.rollout(z, grips, actions, p, return_time_cache=True)
    assert prefix.time_cache.next_kv_cache.shape[-2] == 4
    assert prefix.time_cache.next_rnn_hiddens is not None

    rotary = model.transformer.time_rotary
    with patch.object(rotary, 'forward', wraps=rotary.forward) as spy:
        resumed = api.rollout(prefix.latents[:, -1], grips[:, :2], actions[:, :2], prefix.proprio[:, -1],
                              time_cache=prefix.time_cache, return_time_cache=True)
    assert all(call.kwargs['offset'] == 4 for call in spy.call_args_list)
    assert resumed.time_cache.next_kv_cache.shape[-2] == 5
    assert prefix.time_cache.next_kv_cache.shape[-2] == 4
    torch.testing.assert_close(resumed.latents[:, 0], prefix.latents[:, -1])
    assert api.rollout(z, grips[:, :2], actions[:, :2], p).time_cache is None
