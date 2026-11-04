import json
from pathlib import Path

import pytest
import torch

from tokenworld.data.dataset import (
    RobotwinSVAETokenDataset,
    load_robotwin_svae_train_val,
    resolve_latent_normalization,
)


LATENT_METHOD = "symmetric_qwen"
CONTINUOUS_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_cache(tmp_path: Path, *, export_frame_stride: int = 1) -> Path:
    root = tmp_path / "cache"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir(parents=True)

    video_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    _write_json(
        root / "cache_spec.json",
        {
            "latent_method": LATENT_METHOD,
            "export_frame_stride": export_frame_stride,
            "video_keys": video_keys,
            "latent_specs": {
                LATENT_METHOD: {"shape_per_frame": [3, 4, 2]},
            },
            "codec_config": {
                "grid_size": [2, 2],
                "latent_grid_size": [2, 2],
            },
        },
    )

    records = []
    for split, outcome in (("train", "success"), ("val", "failure")):
        num_frames = 8
        latents = torch.empty(num_frames, 3, 4, 2)
        for frame in range(num_frames):
            for view in range(3):
                latents[frame, view, :, 0] = 1.0 + 2.0 * (frame + view)
                latents[frame, view, :, 1] = 10.0 + 5.0 * (frame + view + 0.5)

        actions = torch.arange(num_frames * 14, dtype=torch.float32).reshape(
            num_frames, 14
        )
        actions[:, 6] = torch.linspace(0.0, 1.0, num_frames)
        actions[:, 13] = torch.linspace(1.0, 0.0, num_frames)
        payload = {
            "latents": {LATENT_METHOD: latents},
            "actions": actions,
            "states": actions.clone(),
        }
        relative_path = f"{split}/episode_000000.pt"
        torch.save(payload, root / relative_path)
        records.append(
            {
                "split": split,
                "task": "adjust_bottle",
                "outcome": outcome,
                "num_frames": num_frames,
                "path": relative_path,
            }
        )

    _write_json(root / "index.json", {"records": records})
    torch.save(
        {
            "version": 1,
            LATENT_METHOD: {
                "mean": torch.tensor([1.0, 10.0]),
                "std": torch.tensor([2.0, 5.0]),
                "count": 1,
            },
            "latent_by_view": {
                LATENT_METHOD: {
                    key: {
                        "mean": torch.tensor([1.0, 10.0]),
                        "std": torch.tensor([2.0, 5.0]),
                        "count": 1,
                    }
                    for key in video_keys
                }
            },
            "action": {
                "mean": torch.zeros(14),
                "std": torch.ones(14),
                "count": 1,
            },
            "state": {
                "mean": torch.zeros(14),
                "std": torch.ones(14),
                "count": 1,
            },
        },
        root / "train_stats.pt",
    )
    return root


def test_native_tokens_normalization_views_and_destination_actions(tmp_path):
    root = _make_cache(tmp_path)
    dataset = RobotwinSVAETokenDataset(
        root,
        split="train",
        num_hist=2,
        num_pred=2,
        views=["cam_right_wrist", "cam_high"],
        random_window_per_episode=False,
        sequence_interval=2,
        window_stride=8,
    )

    sample = dataset[0]
    assert sample["vlm_latents"].shape == (4, 2, 4, 2)
    assert sample["continuous_actions"].shape == (3, 12)
    assert sample["discrete_actions"].shape == (3, 2)
    assert sample["proprio"].shape == (4, 14)
    assert sample["rewards"].shape == (4,)

    # Selected view order is preserved; normalized frame-0 values are view 2 then 0.
    torch.testing.assert_close(
        sample["vlm_latents"][0, :, 0],
        torch.tensor([[2.0, 2.5], [0.0, 0.5]]),
    )

    payload = torch.load(root / "train/episode_000000.pt", weights_only=True)
    destination_actions = payload["actions"][[2, 4, 6]]
    torch.testing.assert_close(
        sample["continuous_actions"],
        destination_actions[:, CONTINUOUS_INDICES],
    )
    torch.testing.assert_close(sample["proprio"], payload["states"][[0, 2, 4, 6]])
    torch.testing.assert_close(
        sample["discrete_actions"],
        (destination_actions[:, [6, 13]] > 0.5).long(),
    )


def test_loader_builds_random_train_and_deterministic_validation(tmp_path):
    root = _make_cache(tmp_path)
    datasets, trajectory_datasets = load_robotwin_svae_train_val(
        cache_dir=root,
        num_hist=2,
        num_pred=2,
        tasks=["adjust_bottle"],
        outcomes=["success", "failure"],
        random_stride_min=1,
        random_stride_max=2,
        val_sequence_interval=1,
        val_window_stride=4,
    )

    assert datasets is trajectory_datasets
    assert len(datasets["train"]) == 1  # one resampled window per train episode
    assert len(datasets["valid"]) == 2  # starts 0 and 4, including tail alignment
    assert datasets["train"][0]["vlm_latents"].shape == (4, 3, 4, 2)


def test_contract_checks_action_state_identity_lazily(tmp_path):
    root = _make_cache(tmp_path)
    path = root / "train/episode_000000.pt"
    payload = torch.load(path, weights_only=True)
    payload["states"][0, 0] += 1.0
    torch.save(payload, path)

    dataset = RobotwinSVAETokenDataset(
        root,
        split="train",
        num_hist=2,
        num_pred=2,
        random_window_per_episode=False,
    )
    with pytest.raises(ValueError, match="action must equal observation.state"):
        dataset[0]


def test_random_stride_requires_full_rate_cache(tmp_path):
    root = _make_cache(tmp_path, export_frame_stride=5)
    with pytest.raises(ValueError, match="export_frame_stride=1"):
        RobotwinSVAETokenDataset(
            root,
            split="train",
            num_hist=2,
            num_pred=2,
            random_window_per_episode=True,
        )


def test_per_view_stats_are_selected_in_requested_view_order(tmp_path):
    root = _make_cache(tmp_path)
    stats = torch.load(root / "train_stats.pt", weights_only=True)
    mean, std, scope, keys = resolve_latent_normalization(
        stats,
        latent_method=LATENT_METHOD,
        video_keys=[
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        view_indices=[2, 0],
        scope="per_view",
    )
    assert scope == "per_view"
    assert keys == [
        "observation.images.cam_right_wrist",
        "observation.images.cam_high",
    ]
    assert mean.shape == (1, 2, 1, 2)
    assert std.shape == (1, 2, 1, 2)
