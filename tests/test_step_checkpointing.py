from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenworld.dynamics.model import DynamicsWorldModel
from tokenworld.dynamics.trainer import BehaviorCloneTrainer


class _TinyWorldModelDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {
            "vlm_latents": torch.randn(4, 1, 6, 4, generator=generator),
            "continuous_actions": torch.zeros(3, 12),
            "discrete_actions": torch.zeros(3, 2, dtype=torch.long),
            "proprio": torch.zeros(4, 14),
            "rewards": torch.zeros(4),
        }


def test_step_limit_and_step_checkpoint_interval(tmp_path):
    model = DynamicsWorldModel(
        dim=32,
        dim_latent=4,
        num_latent_tokens=6,
        num_spatial_tokens=6,
        spatial_grid_size=(2, 3),
        num_agents=1,
        num_video_views=1,
        dim_proprio=14,
        depth=3,
        time_block_every=3,
        num_discrete_actions=(2, 2),
        num_continuous_actions=12,
        max_steps=8,
        attn_heads=4,
        attn_dim_head=8,
        use_time_rnn=False,
        prob_no_shortcut_train=1.0,
    )
    trainer = BehaviorCloneTrainer(
        model,
        dataset=_TinyWorldModelDataset(),
        val_dataset=None,
        batch_size=1,
        num_epochs=1,
        max_train_steps=3,
        checkpoint_every=None,
        checkpoint_every_steps=2,
        checkpoint_folder=str(tmp_path),
        train_decoder=False,
        find_unused_parameters=False,
        cpu=True,
    )

    saves = []

    def record_save(save_path, **metadata):
        saves.append((Path(save_path).name, metadata["global_step"]))

    trainer.save_checkpoint = record_save
    trainer.forward()

    assert saves == [
        ("checkpoint_step_0000002.pt", 2),
        ("checkpoint_step_final.pt", 3),
    ]
