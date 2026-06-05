"""VideoReader - seekable frame access with a small decode cache.

Random seeking with cv2 is expensive, so we keep a tiny LRU of recently decoded
frames and decode sequentially when stepping forward (the common case during
playback / scrubbing).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2


class VideoReader:
    def __init__(self, path: Path | str, cache: int = 64):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 29.97
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._pos = 0                      # 0-based index of NEXT frame to read
        self._cache: OrderedDict[int, "any"] = OrderedDict()
        self._cap_max = cache

    def __len__(self):
        return self.n_frames

    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def _store(self, idx, frame):
        self._cache[idx] = frame
        self._cache.move_to_end(idx)
        while len(self._cache) > self._cap_max:
            self._cache.popitem(last=False)

    def frame(self, idx: int):
        """Return frame `idx` (1-based, matching the tracks CSV) as BGR, or None."""
        idx = max(1, min(self.n_frames, int(idx)))
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx].copy()
        target0 = idx - 1                  # 0-based
        if target0 != self._pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target0)
            self._pos = target0
        ok, fr = self.cap.read()
        if not ok:
            return None
        self._pos += 1
        self._store(idx, fr)
        return fr.copy()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()
