"""Detection cache IO (Phase 3 output) as Parquet.

Columns: frame, x1, y1, x2, y2, conf, team_score, kept
  team_score : best colour-match score against the tracked team profile(s)
  kept       : passed the colour + height filters (the candidate pool for tracking)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["frame", "x1", "y1", "x2", "y2", "conf", "team_score", "kept"]


def write_detections(path: Path | str, rows) -> Path:
    """rows: iterable of dicts (or tuples in COLUMNS order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows), columns=COLUMNS) if rows and isinstance(
        next(iter(rows)), (list, tuple)) else pd.DataFrame(list(rows))
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = 0
    df = df[COLUMNS]
    df.to_parquet(path, index=False)
    return path


def read_detections(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def kept_by_frame(df: pd.DataFrame):
    """dict frame -> list of (x1,y1,x2,y2,conf) for kept detections."""
    out = {}
    k = df[df["kept"]] if "kept" in df.columns else df
    for fr, g in k.groupby("frame"):
        out[int(fr)] = [(r.x1, r.y1, r.x2, r.y2, r.conf) for r in g.itertuples()]
    return out


def all_by_frame(df: pd.DataFrame):
    """dict frame -> list of (x1,y1,x2,y2,conf) for ALL detections (kept or not).

    For the interactive tracker, the ROI is the selector, so every detected
    player should be available to snap to -- not just the team-filtered set."""
    out = {}
    for fr, g in df.groupby("frame"):
        out[int(fr)] = [(r.x1, r.y1, r.x2, r.y2, r.conf) for r in g.itertuples()]
    return out
