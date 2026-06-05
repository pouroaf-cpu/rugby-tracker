"""Field-of-play mask: a per-frame pitch region used as a SECONDARY filter to
drop detections that are clearly off the grass (crowd, stands, beyond the
touchline). Recomputed every frame, so it follows the Veo camera's pan/tilt/zoom.

Designed to err GREEDY: it must never reject a real on-pitch player, only catch
detections outside the field. (Staff standing ON the grass are left to the
colour filter.) Signs/goalposts aren't people, so YOLO never proposes them.
"""
from __future__ import annotations

import cv2
import numpy as np

GRASS_LO = np.array([33, 40, 30])
GRASS_HI = np.array([90, 255, 255])


def field_mask(frame, min_area_frac: float = 0.04):
    """Binary mask (255 = on pitch) = convex hull of the largest grass region
    that reaches the foreground. Hull fills the holes players punch in the grass
    and keeps the mask greedy so no on-pitch player is ever excluded."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
    # denoise + sever thin links to background grass/hills
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g)
    if n <= 1:
        return np.zeros((h, w), np.uint8)
    # largest component that extends into the lower 40% (the foreground pitch)
    best, best_area = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        if area >= min_area_frac * h * w and bottom >= 0.6 * h and area > best_area:
            best, best_area = i, area
    if best is None:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(np.vstack(cnts))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def feet_in_field(mask, box, pad: int = 10) -> bool:
    """True if the box's feet (bottom-centre) are on the pitch, with a small
    tolerance neighbourhood."""
    x1, y1, x2, y2 = (int(v) for v in box)
    fx = max(0, min((x1 + x2) // 2, mask.shape[1] - 1))
    fy = max(0, min(y2, mask.shape[0] - 1))
    y0, x0 = max(0, fy - pad), max(0, fx - pad)
    patch = mask[y0:fy + 1, x0:min(mask.shape[1], fx + pad + 1)]
    return bool(patch.size) and int(patch.max()) > 0
