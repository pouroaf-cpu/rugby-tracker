"""Per-clip landmark cache + temporal smoothing.

This is what makes the overlay "get clearer the longer it runs". A background
pass fills the cache with the raw per-frame BlazePose landmarks; rendering then
pulls a temporally SMOOTHED estimate for a frame by averaging its cached
neighbours (weighted by how close they are and how confident each one is). The
more neighbours are cached, the smoother and steadier the skeleton - so as the
refine pass sweeps the clip, the line stops jittering and replay gets faster
(cache reads instead of re-running the model).

Landmarks are stored as (33, 3): normalised (x, y) plus visibility. A frame that
was processed but had no detection is marked done with visibility 0.
"""
from __future__ import annotations

import numpy as np


class PoseTrack:
    def __init__(self, n_frames: int):
        self.n = max(1, int(n_frames))
        self.data = np.full((self.n, 33, 3), np.nan, dtype=np.float32)
        self.done = np.zeros(self.n, dtype=bool)

    def set(self, idx: int, arr) -> None:
        """Store the (33,3) array for 0-based frame idx (None => processed, no
        detection)."""
        if not (0 <= idx < self.n):
            return
        if arr is None:
            self.data[idx, :, :2] = np.nan
            self.data[idx, :, 2] = 0.0
        else:
            self.data[idx] = arr
        self.done[idx] = True

    def has(self, idx: int) -> bool:
        return 0 <= idx < self.n and bool(self.done[idx])

    def progress(self) -> float:
        return float(self.done.mean())

    def smoothed_at(self, idx: int, w: int = 5) -> "np.ndarray | None":
        """Visibility-weighted, distance-weighted average of cached frames in
        [idx-w, idx+w]. None if nothing in that window is cached yet."""
        if not (0 <= idx < self.n):
            return None
        lo = max(0, idx - w)
        hi = min(self.n, idx + w + 1)
        dmask = self.done[lo:hi]
        if not dmask.any():
            return None
        block = self.data[lo:hi][dmask]                     # (k,33,3)
        frames = np.arange(lo, hi)[dmask]                   # (k,)
        wdist = (w + 1 - np.abs(frames - idx)).clip(min=1).astype(np.float32)  # (k,)
        vis = np.nan_to_num(block[:, :, 2])                 # (k,33)
        wt = wdist[:, None] * vis                           # (k,33)
        wsum = wt.sum(0)                                    # (33,)
        out = np.zeros((33, 3), np.float32)
        good = wsum > 1e-6
        xy = np.nan_to_num(block[:, :, :2]) * wt[:, :, None]
        xy = xy.sum(0)                                      # (33,2)
        out[good, :2] = xy[good] / wsum[good, None]
        out[:, 2] = vis.mean(0)                             # mean visibility
        out[~good, 2] = 0.0
        return out
