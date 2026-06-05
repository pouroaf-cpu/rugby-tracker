"""Target-track CSV IO (Phase 5 output).

Schema: frame, target_id, x1, y1, x2, y2, source, confidence
  source     : where the box came from - 'detection' | 'csrt' | 'manual' | 'sam2'
  confidence : per-frame tracking confidence 0..1 (optional; defaults to 1.0 for
               older CSVs without the column). Drives the confidence-adaptive
               marker (tight ellipse high-conf -> growing -> search circle low-conf).
"""
from __future__ import annotations

import csv
from pathlib import Path

HEADER = ["frame", "target_id", "x1", "y1", "x2", "y2", "source", "confidence"]

SOURCE_DETECTION = "detection"
SOURCE_CSRT = "csrt"
SOURCE_MANUAL = "manual"
SOURCE_SAM2 = "sam2"


def write_tracks(path: Path | str, rows) -> Path:
    """rows: iterable of dicts with HEADER keys (confidence optional, default 1.0)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in sorted(rows, key=lambda d: d["frame"]):
            conf = r.get("confidence", 1.0)
            conf = 1.0 if conf is None else float(conf)
            w.writerow([r["frame"], r["target_id"],
                        f'{r["x1"]:.2f}', f'{r["y1"]:.2f}',
                        f'{r["x2"]:.2f}', f'{r["y2"]:.2f}', r["source"],
                        f'{conf:.3f}'])
    return path


def read_tracks(path: Path | str):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            c = r.get("confidence")
            try:
                conf = float(c) if c not in (None, "") else 1.0
            except ValueError:
                conf = 1.0
            rows.append({"frame": int(r["frame"]), "target_id": int(r["target_id"]),
                         "x1": float(r["x1"]), "y1": float(r["y1"]),
                         "x2": float(r["x2"]), "y2": float(r["y2"]),
                         "source": r["source"], "confidence": conf})
    return rows
