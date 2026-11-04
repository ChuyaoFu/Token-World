import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tokenworld.codec import train
from tokenworld.data import video


class FakeVideo:
    def __init__(self, camera=0):
        self.camera = camera
        self.streams = SimpleNamespace(video=[SimpleNamespace(thread_type=None)])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def decode(self, **kwargs):
        for index in range(3):
            image = Image.new("RGB", (4, 4), (index, self.camera, 0))
            yield SimpleNamespace(to_image=lambda image=image: image)


@pytest.mark.parametrize("num_views", [1, 3])
def test_video_decoder_uses_supplied_camera_count(monkeypatch, num_views):
    monkeypatch.setattr(video.av, "open", lambda path: FakeVideo(int(Path(path).stem)))
    decoded = video._decode_selected_frames(
        [Path(f"{i}.mp4") for i in range(num_views)], [0, 2], (2, 2)
    )
    assert [[image.getpixel((0, 0)) for image in row] for row in decoded] == [
        [(frame, camera, 0) for camera in range(num_views)] for frame in [0, 2]
    ]


def test_training_stream_reads_only_cam_high_with_resume_and_tail_padding(tmp_path, monkeypatch):
    dataset = tmp_path / "robotwin_success"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }))
    path = dataset / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    path.parent.mkdir(parents=True)
    path.touch()
    opened = []

    def open_video(name):
        opened.append(Path(name))
        assert Path(name) == path
        return FakeVideo()

    monkeypatch.setattr(video.av, "open", open_video)
    record = {"dataset": "robotwin_success", "episode_index": 0, "frame_indices": [0, 1, 2]}
    assert train._record_views(record) == 3
    batches = list(train.iter_image_batches(tmp_path, [record], 2, (2, 2), 2, 5))
    assert [[image.getpixel((0, 0))[0] for image in batch] for batch in batches] == [[2, 0], [1]]
    assert opened == [path, path]


@pytest.mark.parametrize("preset,num_views", [("cam_high", 1), ("all", 3)])
def test_selected_camera_stream_resumes_inside_frame_and_pads(tmp_path, monkeypatch, preset, num_views):
    keys = train.CAMERA_PRESETS[preset]
    monkeypatch.setattr(train, "VIDEO_KEYS", keys)
    dataset = tmp_path / "robotwin_success"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }))
    paths = []
    for key in keys:
        p = dataset / f"videos/chunk-000/{key}/episode_000000.mp4"
        p.parent.mkdir(parents=True); p.touch(); paths.append(p)
    opened = []
    def open_video(name):
        p = Path(name); opened.append(p)
        return FakeVideo(paths.index(p))
    monkeypatch.setattr(video.av, "open", open_video)
    record = {"dataset": "robotwin_success", "episode_index": 0, "frame_indices": [0, 1, 2]}
    assert train._record_views(record) == 3 * num_views
    start = num_views + 1
    total = 3 * num_views + 2
    batches = list(train.iter_image_batches(tmp_path, [record], 2, (2, 2), start, total))
    observed = [image.getpixel((0, 0))[:2] for batch in batches for image in batch]
    full = [(frame, camera) for frame in range(3) for camera in range(num_views)]
    assert observed == (full * 2)[start:total]
    assert set(opened) == set(paths)
