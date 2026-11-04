# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
from __future__ import annotations
from pathlib import Path
from PIL import Image
import av

def _decode_selected_frames(
    video_paths: list[Path],
    frame_indices: list[int],
    image_size: tuple[int, int],
) -> list[list[Image.Image]]:
    selected = set(frame_indices)
    height, width = image_size
    per_view = []
    for path in video_paths:
        images = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in selected:
                    images.append(
                        frame.to_image()
                        .convert("RGB")
                        .resize((width, height), Image.Resampling.BILINEAR)
                    )
                if frame_index >= frame_indices[-1]:
                    break
        if len(images) != len(frame_indices):
            raise RuntimeError(
                f"Video {path} yielded {len(images)}/{len(frame_indices)} selected frames; "
                f"last requested index={frame_indices[-1]}"
            )
        per_view.append(images)
    return [
        [per_view[view_index][frame_index] for view_index in range(len(per_view))]
        for frame_index in range(len(frame_indices))
    ]

