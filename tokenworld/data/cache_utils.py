# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
from __future__ import annotations
from pathlib import Path
import json
import torch
from tokenworld.data.video import _decode_selected_frames

def _episode_data_path(data_root: Path, record: dict) -> Path:
    dataset_dir = data_root / record["dataset"]
    with open(dataset_dir / "meta/info.json", "r") as handle:
        info = json.load(handle)
    episode_index = int(record["episode_index"])
    fields = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // int(info["chunks_size"]),
    }
    path = dataset_dir / str(info["data_path"]).format(**fields)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _running_stats_update(accumulator: dict, values: torch.Tensor) -> None:
    values = values.double().reshape(-1, values.shape[-1])
    if "sum" not in accumulator:
        accumulator["sum"] = torch.zeros(values.shape[-1], dtype=torch.float64)
        accumulator["square_sum"] = torch.zeros_like(accumulator["sum"])
        accumulator["count"] = 0
    accumulator["sum"] += values.sum(dim=0)
    accumulator["square_sum"] += values.square().sum(dim=0)
    accumulator["count"] += values.shape[0]


def _finish_stats(accumulator: dict) -> dict:
    count = max(int(accumulator["count"]), 1)
    mean = accumulator["sum"] / count
    variance = accumulator["square_sum"] / count - mean.square()
    return {"mean": mean.float(), "std": variance.clamp_min(1.0e-12).sqrt().float(), "count": count}

