"""Body-region crops from a player bbox.

Shared by calibration (Phase 2) and pre-render (Phase 3) so the regions a team
profile is LEARNED from are identical to the regions it is MATCHED against.
Same split as v1 filter_by_team.
"""
from __future__ import annotations


def _crop(frame, b, ytop, ybot, xpad=0.20):
    x1, y1, x2, y2 = (int(v) for v in b)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    cx1 = x1 + int(xpad * w); cx2 = x1 + int((1 - xpad) * w)
    ry1 = y1 + int(ytop * h); ry2 = y1 + int(ybot * h)
    patch = frame[max(0, ry1):ry2, max(0, cx1):cx2]
    return patch if patch.size else None


def torso_patch(frame, b):
    """Upper-central region: jersey primary/secondary colour."""
    return _crop(frame, b, 0.18, 0.48)


def shorts_patch(frame, b):
    """Lower-central region: shorts colour."""
    return _crop(frame, b, 0.50, 0.78)
