"""Video -> model-input conversion.

Mirrors the training notebook's `frames_extraction`:
  * 16 frames, spread evenly over the clip (step = total_frames // 16)
  * RGB colour order (imageio in the notebook returns RGB; OpenCV returns BGR, so we convert)
  * resized to 64x64
Normalisation (/255) is applied later, right before inference.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import (
    IMAGE_HEIGHT, IMAGE_WIDTH, MAX_SEGMENTS, MIN_FRAMES_PER_SEGMENT, SEQUENCE_LENGTH,
)


class VideoError(Exception):
    """Raised when a video cannot be decoded or is unusable."""


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    total_frames: int

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0


def read_meta(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open the video. The file may be corrupted or use an unsupported codec.")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:  # some containers (e.g. webm) don't report a count -> count manually
            total = 0
            while cap.grab():
                total += 1
    finally:
        cap.release()
    return VideoMeta(width, height, fps if fps > 0 else 25.0, total)


def sample_indices(start: int, end: int, seq: int = SEQUENCE_LENGTH) -> list[int]:
    """Same maths as the notebook: step = length // seq, take the first `seq` frames."""
    length = end - start
    step = max(length // seq, 1)
    return list(range(start, end, step))[:seq]


def plan_segments(total_frames: int) -> list[tuple[int, int]]:
    """Split the video into up to MAX_SEGMENTS equal parts (each long enough to sample 16 frames)."""
    n = min(MAX_SEGMENTS, total_frames // MIN_FRAMES_PER_SEGMENT)
    if n < 2:
        return []
    bounds = np.linspace(0, total_frames, n + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n)]


def _read_frames(path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode the video once, keeping only the requested frame indices (RGB, 64x64, uint8)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open the video.")
    frames: dict[int, np.ndarray] = {}
    last = max(wanted)
    idx = 0
    try:
        while idx <= last:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames[idx] = cv2.resize(rgb, (IMAGE_HEIGHT, IMAGE_WIDTH))
            idx += 1
    finally:
        cap.release()
    return frames


def _stack(indices: list[int], frames: dict[int, np.ndarray]) -> np.ndarray:
    """Build one (16,64,64,3) sequence; if a frame is missing use the nearest decoded one."""
    available = sorted(frames)
    out = []
    for i in indices:
        if i in frames:
            out.append(frames[i])
        else:
            nearest = min(available, key=lambda a: abs(a - i))
            out.append(frames[nearest])
    return np.stack(out)


def build_sequences(path: str):
    """Returns (meta, full_video_sequence, [(start, end, sequence), ...])."""
    meta = read_meta(path)
    if meta.total_frames < SEQUENCE_LENGTH:
        raise VideoError(
            f"Video is too short: it has {meta.total_frames} frames, at least {SEQUENCE_LENGTH} are required."
        )

    whole_idx = sample_indices(0, meta.total_frames)
    segments = plan_segments(meta.total_frames)
    seg_idx = [sample_indices(s, e) for s, e in segments]

    wanted = set(whole_idx).union(*map(set, seg_idx)) if seg_idx else set(whole_idx)
    frames = _read_frames(path, wanted)
    if not frames:
        raise VideoError("No frames could be decoded from this video.")

    whole = _stack(whole_idx, frames)
    segs = [(s, e, _stack(ix, frames)) for (s, e), ix in zip(segments, seg_idx)]
    return meta, whole, segs
