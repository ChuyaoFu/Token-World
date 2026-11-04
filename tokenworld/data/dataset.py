"""TokenWorld dataset adapter for IRASim RoboTwin S-VAE sequence caches.

IRASim stores one file per episode with semantic latents shaped
``[frame, view, token, channel]``.  Its native dataset packs those values into
an image-like grid for the IRASim DiT.  TokenWorld consumes the original token
layout instead, so this adapter deliberately reads the cache before that grid
packing step and returns model-ready TokenWorld batches.  When requested, it
also streams raw RobotWin RGB frames from LeRobot mp4 videos and keeps a
temporary on-disk latent cache for standalone decoder training.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

from einops import rearrange
import torch
from torch.utils.data import Dataset

try:  # optional at import time; remote training uses decord
    import decord
except Exception:  # pragma: no cover
    decord = None

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def resolve_latent_normalization(
    stats: dict,
    *,
    latent_method: str,
    video_keys: Sequence[str],
    view_indices: Sequence[int],
    scope: str = "per_view",
) -> tuple[torch.Tensor, torch.Tensor, str, list[str]]:
    """Resolve train-only latent statistics with explicit view provenance.

    Cached latents are ``[T,V,N,D]``. Per-view statistics are returned as
    ``[1,V,1,D]`` so each camera is normalized independently and unambiguously.
    ``auto`` is retained for older global-stat caches; new single-view runs
    should use the explicit ``per_view`` scope.
    """
    scope = str(scope).lower().strip()
    if scope not in {"per_view", "global", "auto"}:
        raise ValueError(
            "latent_stats_scope must be one of per_view/global/auto, "
            f"got {scope!r}"
        )

    global_stats = stats.get(latent_method)
    per_view_root = stats.get("latent_by_view", {}).get(latent_method, {})
    selected_keys = [str(video_keys[int(index)]) for index in view_indices]
    missing = [key for key in selected_keys if key not in per_view_root]

    use_per_view = scope == "per_view" or (scope == "auto" and not missing)
    if use_per_view:
        if missing:
            raise KeyError(
                f"Missing per-view {latent_method!r} statistics for {missing}; "
                "regenerate train_stats.pt or explicitly request global scope"
            )
        means = torch.stack(
            [per_view_root[key]["mean"].float() for key in selected_keys], dim=0
        )
        stds = torch.stack(
            [per_view_root[key]["std"].float() for key in selected_keys], dim=0
        ).clamp_min(1.0e-6)
        return means[None, :, None, :], stds[None, :, None, :], "per_view", selected_keys

    if global_stats is None:
        raise KeyError(f"Missing global statistics for latent method {latent_method!r}")
    mean = global_stats["mean"].float().reshape(1, 1, 1, -1)
    std = global_stats["std"].float().reshape(1, 1, 1, -1).clamp_min(1.0e-6)
    return mean, std, "global", [f"global:{latent_method}"]


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


class RobotwinSVAETokenDataset(Dataset):
    """Read normalized native S-VAE tokens and aligned absolute-qpos actions.

    Returned samples can be passed directly to ``DynamicsWorldModel.forward``:

    - ``vlm_latents``: ``[T, V, N, D]``
    - ``continuous_actions``: ``[T - 1, 12]``
    - ``discrete_actions``: ``[T - 1, 2]``
    - ``proprio``: ``[T, 14]``

    RoboTwin cache actions use the environment order
    ``[left6, left_gripper, right6, right_gripper]``.  Actions are paired with
    destination frames, while proprioception remains aligned with every visual
    frame.  Latent and action statistics always come from the cache's training
    split statistics.
    """

    VIEW_ALIASES = {
        "cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }
    CONTINUOUS_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
    GRIPPER_INDICES = (6, 13)

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        raw_data_path: str | Path | None = None,
        latent_cache_dir: str | Path | None = None,
        split: str,
        num_hist: int = 4,
        num_pred: int = 16,
        latent_method: str = "symmetric_qwen",
        tasks: Sequence[str] | str | None = None,
        outcomes: Sequence[str] | str | None = None,
        views: Sequence[int | str] | None = None,
        normalize_latents: bool = True,
        latent_stats_scope: str = "per_view",
        normalize_action: bool = True,
        verify_action_equals_state: bool = True,
        load_raw_images: bool = False,
        random_window_per_episode: bool = False,
        random_stride_min: int = 1,
        random_stride_max: int = 8,
        sequence_interval: int = 1,
        window_stride: int | None = None,
        tail_align: bool = True,
        lru_episodes: int = 8,
        action_schema: str = "robotwin14",
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if num_hist < 1 or num_pred < 1:
            raise ValueError("num_hist and num_pred must both be positive")

        self.cache_root = Path(cache_dir).expanduser().resolve()
        self.raw_data_root = (
            Path(raw_data_path).expanduser().resolve()
            if raw_data_path is not None
            else None
        )
        self.latent_cache_root = (
            Path(latent_cache_dir).expanduser().resolve()
            if latent_cache_dir is not None
            else None
        )
        self.split = split
        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.num_frames = self.num_hist + self.num_pred
        self.latent_method = str(latent_method)
        self.normalize_latents = bool(normalize_latents)
        self.latent_stats_scope_requested = str(latent_stats_scope)
        self.normalize_action = bool(normalize_action)
        self.verify_action_equals_state = bool(verify_action_equals_state)
        self.load_raw_images = bool(load_raw_images)
        self.random_window_per_episode = bool(random_window_per_episode)
        self.random_stride_min = int(random_stride_min)
        self.random_stride_max = int(random_stride_max)
        self.sequence_interval = int(sequence_interval)
        self.window_stride = (
            self.num_frames if window_stride is None else int(window_stride)
        )
        self.tail_align = bool(tail_align)
        self.lru_episodes = max(int(lru_episodes), 1)
        self.action_schema = str(action_schema)
        self._episode_cache: OrderedDict[int, dict] = OrderedDict()
        self._raw_dataset_info_cache: dict[str, dict] = {}
        self._raw_video_cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()

        if self.random_stride_min < 1 or self.random_stride_max < self.random_stride_min:
            raise ValueError("random stride bounds must satisfy 1 <= min <= max")
        if self.sequence_interval < 1 or self.window_stride < 1:
            raise ValueError("sequence_interval and window_stride must be positive")

        with open(self.cache_root / "cache_spec.json", "r", encoding="utf-8") as handle:
            self.spec = json.load(handle)
        with open(self.cache_root / "index.json", "r", encoding="utf-8") as handle:
            index = json.load(handle)
        self.stats = _torch_load(self.cache_root / "train_stats.pt")

        if self.spec.get("latent_method") != self.latent_method:
            raise ValueError(
                f"Cache latent method is {self.spec.get('latent_method')!r}, "
                f"requested {self.latent_method!r}"
            )
        latent_spec = self.spec["latent_specs"][self.latent_method]
        per_frame = list(latent_spec["shape_per_frame"])
        if len(per_frame) != 3:
            raise ValueError(f"Expected cache shape [views,tokens,channels], got {per_frame}")
        self.total_num_views, self.tokens_per_view, self.latent_channels = map(
            int, per_frame
        )
        self.video_keys = list(self.spec["video_keys"])
        if len(self.video_keys) != self.total_num_views:
            raise ValueError("Cache view count does not match video_keys")
        self.view_indices = self._resolve_views(views)
        self.num_views = len(self.view_indices)

        if self.load_raw_images and self.raw_data_root is None:
            raise ValueError("raw_data_path is required when load_raw_images=True")

        codec_config = self.spec["codec_config"]
        self.decoder_feature_dim = int(codec_config.get("input_dim", self.latent_channels))
        self.decoder_cache_version = 2
        self.decoder_cache_metadata: dict | None = None
        latent_grid = codec_config.get("latent_grid_size")
        if latent_grid is None:
            latent_grid = codec_config["grid_size"]
        self.grid_height, self.grid_width = map(int, latent_grid)
        if self.grid_height * self.grid_width != self.tokens_per_view:
            if (self.total_num_views > 1 and self.grid_height * self.grid_width == self.total_num_views * self.tokens_per_view and self.grid_width % self.total_num_views == 0):
                self.grid_width //= self.total_num_views
            else:
                raise ValueError(f"Grid {self.grid_height}x{self.grid_width} does not match {self.tokens_per_view} tokens")

        task_filter = self._as_filter(tasks)
        outcome_filter = self._as_filter(outcomes)
        self.records = []
        for record in index["records"]:
            if record["split"] != self.split:
                continue
            if task_filter is not None and record["task"] not in task_filter:
                continue
            if outcome_filter is not None and record["outcome"] not in outcome_filter:
                continue
            copied = dict(record)
            copied["absolute_path"] = self.cache_root / copied["path"]
            self.records.append(copied)
        if not self.records:
            raise ValueError(
                f"No records for split={split!r}, tasks={tasks!r}, outcomes={outcomes!r}"
            )

        if self.random_window_per_episode:
            if self.split != "train":
                raise ValueError("random_window_per_episode is only valid for training")
            if self.spec.get("export_frame_stride") != 1:
                raise ValueError(
                    "Random temporal stride requires a full-rate cache with "
                    f"export_frame_stride=1, got {self.spec.get('export_frame_stride')}"
                )
            minimum_span = (self.num_frames - 1) * self.random_stride_min + 1
            self.samples = [
                {"record_index": record_index}
                for record_index, record in enumerate(self.records)
                if int(record["num_frames"]) >= minimum_span
            ]
        else:
            self.samples = self._build_fixed_windows()
        if not self.samples:
            raise ValueError(f"No valid {self.num_frames}-frame windows in {split!r} split")

        (
            self.latent_mean,
            self.latent_std,
            self.latent_stats_scope,
            self.latent_stats_keys,
        ) = resolve_latent_normalization(
            self.stats,
            latent_method=self.latent_method,
            video_keys=self.video_keys,
            view_indices=self.view_indices,
            scope=self.latent_stats_scope_requested,
        )
        self.action_mean = self.stats["action"]["mean"].float()
        self.action_std = self.stats["action"]["std"].float().clamp_min(1.0e-6)
        # Per-view normalization keeps an explicit camera axis
        # ``[1, V, 1, D]``; validate only the channel dimension rather than
        # multiplying it by the number of selected views.
        if self.latent_mean.shape[-1] != self.latent_channels:
            raise ValueError("Latent statistics do not match cache channel dimension")
        if self.action_schema == "muka7":
            if self.action_mean.numel() != 7: raise ValueError(f"Expected 7-D Muka actions, got {self.action_mean.numel()}")
            self.action_dim = self.proprio_dim = self.state_dim = 7
        else:
            if self.action_mean.numel() != 14: raise ValueError(f"Expected 14-D RoboTwin actions, got {self.action_mean.numel()}")
            self.action_dim = self.proprio_dim = self.state_dim = 14

        # Compatibility attributes used by TokenWorld's existing loaders/evaluators.
        self.vlm_token_shape = (self.tokens_per_view, self.latent_channels)
        self.decoder_vlm_token_shape = (self.tokens_per_view, self.decoder_feature_dim)
        # schema-specific dimensions were set above

    def normalization_metadata(self) -> dict:
        """Return normalization provenance for checkpoints and evaluation."""
        return {
            "latent_method": self.latent_method,
            "latent_stats_scope_requested": self.latent_stats_scope_requested,
            "latent_stats_scope": self.latent_stats_scope,
            "latent_stats_keys": list(self.latent_stats_keys),
            "view_indices": list(self.view_indices),
            "video_keys": [self.video_keys[index] for index in self.view_indices],
            "normalize_latents": self.normalize_latents,
            "normalize_action": self.normalize_action,
            "stats_path": str(self.cache_root / "train_stats.pt"),
        }

    @staticmethod
    def _as_filter(values: Sequence[str] | str | None) -> set[str] | None:
        if values is None:
            return None
        if isinstance(values, str):
            values = values.split(",")
        return {str(value).strip() for value in values if str(value).strip()}

    def _resolve_views(self, views: Sequence[int | str] | None) -> list[int]:
        if views is None:
            return list(range(self.total_num_views))
        indices = []
        for view in views:
            if isinstance(view, int):
                index = view
            else:
                key = self.VIEW_ALIASES.get(str(view), str(view))
                if key not in self.video_keys:
                    raise ValueError(f"Unknown view {view!r}; available={self.video_keys}")
                index = self.video_keys.index(key)
            if index < 0 or index >= self.total_num_views:
                raise ValueError(
                    f"View index {index} is outside [0,{self.total_num_views})"
                )
            if index in indices:
                raise ValueError(f"Duplicate view index {index}")
            indices.append(index)
        if not indices:
            raise ValueError("At least one view must be selected")
        return indices

    def _raw_dataset_root(self, record: dict) -> Path:
        if self.raw_data_root is None:
            raise ValueError("raw_data_root is not configured")
        dataset_name = str(record["dataset"])
        candidate = self.raw_data_root / dataset_name
        if candidate.is_dir():
            return candidate
        if (self.raw_data_root / "meta" / "info.json").is_file():
            return self.raw_data_root
        raise FileNotFoundError(
            f"Could not resolve raw dataset root for dataset={dataset_name!r} "
            f"under {self.raw_data_root}"
        )

    def _load_raw_dataset_info(self, record: dict) -> dict:
        dataset_name = str(record["dataset"])
        if dataset_name in self._raw_dataset_info_cache:
            return self._raw_dataset_info_cache[dataset_name]
        dataset_root = self._raw_dataset_root(record)
        info_path = dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")
        with open(info_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
        self._raw_dataset_info_cache[dataset_name] = info
        return info

    def _resolve_raw_video_path(self, record: dict, view_idx: int) -> Path:
        dataset_root = self._raw_dataset_root(record)
        info = self._load_raw_dataset_info(record)
        episode_index = int(record["episode_index"])
        chunks_size = int(info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunks_size
        video_key = self.video_keys[view_idx]
        relative = info["video_path"].format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        video_path = dataset_root / relative
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing raw video: {video_path}")
        return video_path

    def _get_raw_video_entry(self, record_index: int, view_idx: int) -> dict:
        record = self.records[record_index]
        cache_key = (str(record["dataset"]), int(record["episode_index"]), int(view_idx))
        if cache_key in self._raw_video_cache:
            entry = self._raw_video_cache.pop(cache_key)
            self._raw_video_cache[cache_key] = entry
            return entry

        video_path = self._resolve_raw_video_path(record, view_idx)
        reader = None
        if decord is not None:
            try:
                reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            except Exception:
                reader = None
        entry = {"path": video_path, "reader": reader}
        self._raw_video_cache[cache_key] = entry
        while len(self._raw_video_cache) > self.lru_episodes * max(self.num_views, 1):
            self._raw_video_cache.popitem(last=False)
        return entry

    def _raw_episode_length(self, record_index: int) -> int:
        video_entry = self._get_raw_video_entry(record_index, self.view_indices[0])
        cached_length = video_entry.get("length")
        if cached_length is not None:
            return int(cached_length)

        reader = video_entry.get("reader")
        if reader is not None:
            length = len(reader)
        else:
            if cv2 is None:
                raise ImportError(
                    "Neither decord nor cv2 is available to inspect RobotWin mp4 length"
                )
            capture = cv2.VideoCapture(str(video_entry["path"]))
            if not capture.isOpened():
                raise FileNotFoundError(f"Could not open video: {video_entry['path']}")
            try:
                length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()

        video_entry["length"] = int(length)
        return int(length)

    def _load_video_frames(self, video_entry: dict, frames=None) -> torch.Tensor:
        frame_indices = None if frames is None else [int(frame) for frame in frames]
        reader = video_entry.get("reader")
        if reader is not None:
            if frame_indices is None:
                frame_indices = list(range(len(reader)))
            if len(frame_indices) == 0:
                return torch.empty((0, 0, 0, 3), dtype=torch.uint8)
            if min(frame_indices) < 0 or max(frame_indices) >= len(reader):
                raise IndexError(
                    f"Frame indices {frame_indices[:3]}... are outside video length {len(reader)}"
                )
            frames_np = reader.get_batch(frame_indices).asnumpy()
            return torch.from_numpy(frames_np)

        if cv2 is None:
            raise ImportError(
                "Neither decord nor cv2 is available to decode RobotWin mp4 videos"
            )

        video_path = str(video_entry["path"])
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        try:
            if frame_indices is None:
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_indices = list(range(total_frames))
            images = []
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not read frame {frame_index} from {video_path}"
                    )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(torch.from_numpy(frame.copy()))
        finally:
            capture.release()

        if not images:
            return torch.empty((0, 0, 0, 3), dtype=torch.uint8)
        return torch.stack(images, dim=0)

    def _latent_cache_path(self, record: dict) -> Path | None:
        if self.latent_cache_root is None:
            return None
        view_tag = "-".join(map(str, self.view_indices))
        return (
            self.latent_cache_root
            / record["split"]
            / str(record.get("dataset", "default"))
            / record["task"]
            / str(record.get("outcome", "unknown"))
            / (
                f"episode_{int(record['episode_index']):06d}"
                f"_views-{view_tag}"
                "_decoded_vlm.pt"
            )
        )

    def _build_fixed_windows(self) -> list[dict]:
        samples = []
        span = (self.num_frames - 1) * self.sequence_interval + 1
        for record_index, record in enumerate(self.records):
            last_start = int(record["num_frames"]) - span
            if last_start < 0:
                continue
            starts = list(range(0, last_start + 1, self.window_stride))
            if self.tail_align and starts[-1] != last_start:
                starts.append(last_start)
            samples.extend(
                {"record_index": record_index, "start_frame_id": start}
                for start in starts
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_episode(self, record_index: int) -> dict:
        if record_index in self._episode_cache:
            payload = self._episode_cache.pop(record_index)
            self._episode_cache[record_index] = payload
            return payload
        record = self.records[record_index]
        payload = _torch_load(record["absolute_path"])
        if self.verify_action_equals_state:
            actions = payload.get("actions")
            states = payload.get("states")
            if (
                actions is None
                or states is None
                or actions.shape != states.shape
                or not torch.equal(actions.float(), states.float())
            ):
                raise ValueError(
                    "Cache contract violation: action must equal observation.state; "
                    f"episode={record['absolute_path']}"
                )
        self._episode_cache[record_index] = payload
        while len(self._episode_cache) > self.lru_episodes:
            self._episode_cache.popitem(last=False)
        return payload

    def _load_decoder_episode(self, record_index: int) -> dict:
        record = self.records[record_index]
        cache_path = self._latent_cache_path(record)
        if cache_path is None:
            raise ValueError(
                "Decoder training requires data.latent_cache_dir so decoded VLM "
                "features can be reused from disk"
            )
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Decoded VLM cache is missing. Run cache materialization before "
                f"building the decoder DataLoader: {cache_path}"
            )
        payload = _torch_load(cache_path)
        if payload.get("cache_kind") != "robotwin_svae_decoded_vlm":
            raise ValueError(f"Unexpected decoder cache payload: {cache_path}")
        if int(payload.get("cache_version", -1)) != self.decoder_cache_version:
            raise ValueError(f"Unsupported decoder cache version: {cache_path}")
        if payload.get("latent_method") != self.latent_method:
            raise ValueError(f"Latent method mismatch in cached payload: {cache_path}")
        if list(payload.get("views", [])) != list(self.view_indices):
            raise ValueError(f"View mismatch in cached payload: {cache_path}")
        if int(payload.get("feature_dim", -1)) != self.decoder_feature_dim:
            raise ValueError(f"Decoded feature dimension mismatch: {cache_path}")
        if self.decoder_cache_metadata is not None:
            cached_codec = payload.get("codec", {})
            for key in ("checkpoint", "checkpoint_size", "checkpoint_mtime_ns"):
                if cached_codec.get(key) != self.decoder_cache_metadata.get(key):
                    raise ValueError(
                        f"Codec checkpoint mismatch in cached payload: {cache_path}"
                    )
        return payload

    def materialize_decoder_episode(
        self,
        record_index: int,
        codec_decoder: torch.nn.Module,
        device: torch.device | str,
        *,
        decode_batch_size: int = 32,
        cache_dtype: torch.dtype = torch.bfloat16,
        cache_metadata: dict | None = None,
        autocast_context=None,
    ) -> Path:
        """Decode one episode in small GPU batches and persist it on disk."""

        cache_path = self._latent_cache_path(self.records[record_index])
        if cache_path is None:
            raise ValueError("latent_cache_dir is required for decoder cache materialization")
        if cache_path.is_file():
            self._load_decoder_episode(record_index)
            return cache_path

        record = self.records[record_index]
        source = self._load_episode(record_index)
        compressed = source["latents"][self.latent_method]
        expected = (
            int(record["num_frames"]),
            self.total_num_views,
            self.tokens_per_view,
            self.latent_channels,
        )
        if tuple(compressed.shape) != expected:
            raise ValueError(f"Expected cached latents {expected}, got {tuple(compressed.shape)}")

        selected = compressed[:, self.view_indices]
        flat_selected = selected.reshape(-1, self.tokens_per_view, self.latent_channels)
        decoded = torch.empty(
            (flat_selected.shape[0], self.tokens_per_view, self.decoder_feature_dim),
            dtype=cache_dtype,
        )
        decode_batch_size = max(int(decode_batch_size), 1)
        for start in range(0, flat_selected.shape[0], decode_batch_size):
            batch = flat_selected[start : start + decode_batch_size].to(device, non_blocking=True)
            context = autocast_context() if autocast_context is not None else nullcontext()
            with torch.inference_mode(), context:
                decoded_batch = codec_decoder(batch)
            expected_batch = (batch.shape[0], self.tokens_per_view, self.decoder_feature_dim)
            if tuple(decoded_batch.shape) != expected_batch:
                raise ValueError(
                    f"Codec returned {tuple(decoded_batch.shape)}, expected {expected_batch}"
                )
            decoded[start : start + decoded_batch.shape[0]].copy_(decoded_batch.to("cpu", dtype=cache_dtype))

        payload = {
            "cache_kind": "robotwin_svae_decoded_vlm",
            "cache_version": self.decoder_cache_version,
            "latent_method": self.latent_method,
            "views": list(self.view_indices),
            "feature_dim": self.decoder_feature_dim,
            "vlm_latents": decoded.reshape(
                int(record["num_frames"]),
                self.num_views,
                self.tokens_per_view,
                self.decoder_feature_dim,
            ),
            "actions": source["actions"].float(),
            "states": source["states"].float(),
        }
        if cache_metadata:
            payload["codec"] = dict(cache_metadata)
        _atomic_torch_save(payload, cache_path)
        self._episode_cache.pop(record_index, None)
        return cache_path

    def _load_raw_images(self, record_index: int, record: dict, frames=None):
        all_view_images = []
        for view_idx in self.view_indices:
            video_entry = self._get_raw_video_entry(
                record_index=record_index, view_idx=view_idx
            )
            images = self._load_video_frames(video_entry, frames)
            all_view_images.append(images)
        raw_images = torch.stack(all_view_images, dim=0)
        raw_images = raw_images.permute(1, 0, 2, 3, 4)
        return raw_images

    def _frame_ids(self, sample: dict, episode_length: int) -> torch.Tensor:
        if self.random_window_per_episode:
            max_feasible_stride = (int(episode_length) - 1) // (self.num_frames - 1)
            stride_upper = min(self.random_stride_max, max_feasible_stride)
            if stride_upper < self.random_stride_min:
                raise ValueError(f"Episode is too short for {self.num_frames} frames")
            temporal_stride = int(
                torch.randint(self.random_stride_min, stride_upper + 1, ()).item()
            )
            max_start = int(episode_length) - 1 - (
                self.num_frames - 1
            ) * temporal_stride
            start = int(torch.randint(0, max_start + 1, ()).item())
        else:
            temporal_stride = self.sequence_interval
            start = int(sample["start_frame_id"])
            span = (self.num_frames - 1) * temporal_stride + 1
            max_start = max(int(episode_length) - span, 0)
            start = min(start, max_start)
        return torch.arange(
            start,
            start + self.num_frames * temporal_stride,
            temporal_stride,
            dtype=torch.long,
        )

    def __getitem__(self, index: int):
        sample = self.samples[index]
        record_index = int(sample["record_index"])
        record = self.records[record_index]
        if not self.load_raw_images:
            payload = self._load_episode(record_index)
            episode_length = int(payload["latents"][self.latent_method].shape[0])
            frame_ids = self._frame_ids(sample, episode_length)
            latents = payload["latents"][self.latent_method][frame_ids].float()
            expected = (
                self.num_frames,
                self.total_num_views,
                self.tokens_per_view,
                self.latent_channels,
            )
            if tuple(latents.shape) != expected:
                raise ValueError(
                    f"Expected cached latents {expected}, got {tuple(latents.shape)}"
                )
            latents = latents[:, self.view_indices]
            if self.normalize_latents:
                latents = (latents - self.latent_mean) / self.latent_std

            # Muka stores action[t] = state[t+1]; pair the preceding action with each destination frame.
            if self.action_schema == "muka7":
                actions = payload["states"][frame_ids[1:]].float(); proprio = payload["states"][frame_ids].float()
                if actions.shape[-1] != 7 or proprio.shape[-1] != 7: raise ValueError("Muka action/state tensors must be 7-D")
                continuous_actions = actions[..., :6]
                if self.normalize_action:
                    continuous_actions = (continuous_actions - self.action_mean[:6]) / self.action_std[:6]; proprio = (proprio - self.action_mean) / self.action_std
                discrete_actions = (actions[..., 6] > 0.5).long().unsqueeze(-1)
            else:
                actions = payload["actions"][frame_ids[1:]].float(); proprio = payload["states"][frame_ids].float()
                if actions.shape[-1] != 14 or proprio.shape[-1] != 14: raise ValueError("RoboTwin action/state tensors must both be 14-D")
                continuous_actions = actions[..., self.CONTINUOUS_INDICES]
                if self.normalize_action:
                    continuous_mean = self.action_mean[list(self.CONTINUOUS_INDICES)]; continuous_std = self.action_std[list(self.CONTINUOUS_INDICES)]
                    continuous_actions = (continuous_actions - continuous_mean) / continuous_std; proprio = (proprio - self.action_mean) / self.action_std
                discrete_actions = (actions[..., self.GRIPPER_INDICES] > 0.5).long()

            return {
                "vlm_latents": latents,
                "continuous_actions": continuous_actions,
                "discrete_actions": discrete_actions,
                "proprio": proprio,
                "rewards": torch.zeros(self.num_frames, dtype=torch.float32),
            }

        payload = self._load_decoder_episode(record_index)
        episode_length = int(payload["vlm_latents"].shape[0])
        if self.view_indices:
            episode_length = min(episode_length, self._raw_episode_length(record_index))
        frame_ids = self._frame_ids(sample, episode_length)
        latents = payload["vlm_latents"][frame_ids]
        expected = (
            self.num_frames,
            self.num_views,
            self.tokens_per_view,
            self.decoder_feature_dim,
        )
        if tuple(latents.shape) != expected:
            raise ValueError(f"Expected decoder latents {expected}, got {tuple(latents.shape)}")

        actions = payload["actions"][frame_ids[1:]].float()
        states = payload["states"][frame_ids].float()
        if actions.shape[-1] != 14 or states.shape[-1] != 14:
            raise ValueError("RoboTwin action/state tensors must both be 14-D")
        continuous_actions = actions[..., self.CONTINUOUS_INDICES]
        if self.normalize_action:
            continuous_mean = self.action_mean[list(self.CONTINUOUS_INDICES)]
            continuous_std = self.action_std[list(self.CONTINUOUS_INDICES)]
            continuous_actions = (continuous_actions - continuous_mean) / continuous_std
            states = (states - self.action_mean) / self.action_std
        discrete_actions = (actions[..., self.GRIPPER_INDICES] > 0.5).long()
        proprio = states.clone()

        raw_images = self._load_raw_images(record_index, record, frame_ids)
        image = raw_images.float().div_(255.0)
        image = rearrange(image, "t v h w c -> t v c h w")

        return (
            {
                "visual": latents,
                "proprio": proprio,
            },
            actions,
            proprio,
            torch.zeros(self.num_frames, dtype=torch.float32),
            image,
        )


def load_robotwin_svae_train_val(
    *,
    cache_dir: str | Path,
    raw_data_path: str | Path | None = None,
    latent_cache_dir: str | Path | None = None,
    num_hist: int,
    num_pred: int,
    latent_method: str = "symmetric_qwen",
    tasks: Sequence[str] | str | None = None,
    outcomes: Sequence[str] | str | None = None,
    views: Sequence[int | str] | None = None,
    normalize_latents: bool = True,
    latent_stats_scope: str = "per_view",
    normalize_action: bool = True,
    verify_action_equals_state: bool = True,
    load_raw_images: bool = False,
    random_window_per_episode: bool = True,
    random_stride_min: int = 1,
    random_stride_max: int = 8,
    train_sequence_interval: int = 1,
    train_window_stride: int = 1,
    val_sequence_interval: int = 1,
    val_window_stride: int | None = None,
    tail_align: bool = True,
    lru_episodes: int = 8,
    action_schema: str = "robotwin14",
):
    common = dict(
        cache_dir=cache_dir,
        raw_data_path=raw_data_path,
        latent_cache_dir=latent_cache_dir,
        num_hist=num_hist,
        num_pred=num_pred,
        latent_method=latent_method,
        tasks=tasks,
        outcomes=outcomes,
        views=views,
        normalize_latents=normalize_latents,
        latent_stats_scope=latent_stats_scope,
        normalize_action=normalize_action,
        verify_action_equals_state=verify_action_equals_state,
        load_raw_images=load_raw_images,
        random_stride_min=random_stride_min,
        random_stride_max=random_stride_max,
        tail_align=tail_align,
        lru_episodes=lru_episodes,
        action_schema=action_schema,
    )
    train_dataset = RobotwinSVAETokenDataset(
        split="train",
        random_window_per_episode=random_window_per_episode,
        sequence_interval=train_sequence_interval,
        window_stride=train_window_stride,
        **common,
    )
    val_dataset = RobotwinSVAETokenDataset(
        split="val",
        random_window_per_episode=False,
        sequence_interval=val_sequence_interval,
        window_stride=val_window_stride,
        **common,
    )
    datasets = {"train": train_dataset, "valid": val_dataset}
    return datasets, datasets
