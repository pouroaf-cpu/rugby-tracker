"""Bounding-box math. Boxes are (x1, y1, x2, y2) in pixel coords."""
from __future__ import annotations


def wh(b):
    return b[2] - b[0], b[3] - b[1]


def area(b):
    w, h = wh(b)
    return max(0.0, w) * max(0.0, h)


def center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def contains(b, x, y):
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def clamp(b, w, h):
    return (max(0, min(b[0], w - 1)), max(0, min(b[1], h - 1)),
            max(0, min(b[2], w - 1)), max(0, min(b[3], h - 1)))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (area(a) + area(b) - inter)


def dist2(a, b):
    """Squared distance between two box centres."""
    ax, ay = center(a)
    bx, by = center(b)
    return (ax - bx) ** 2 + (ay - by) ** 2


def nearest(boxes, x, y):
    """Index of the box whose centre is nearest to (x, y), or None."""
    best, bestd = None, None
    for i, b in enumerate(boxes):
        cx, cy = center(b)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if bestd is None or d < bestd:
            bestd, best = d, i
    return best
