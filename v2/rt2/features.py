"""Perceptual team fingerprints.

Learned HSV signatures fail to separate teams at Veo resolution because the
dominant colour in the torso/shorts regions is SKIN, which both teams share.
Instead we describe each player box by a small vector of skin-immune perceptual
colour fractions (white/black shorts, navy/red/gold shirt, ...) and give each
team the MEAN vector over its calibration samples (its "fingerprint"). A
detection is classified to the nearest fingerprint -- the colours common to
both teams (skin) cancel in the comparison, so the discriminators decide.
"""
from __future__ import annotations

import cv2
import numpy as np

from rt2 import regions

# Skin-immune perceptual colour gates (OpenCV HSV). white = low S; black = low V;
# the hued gates require real saturation so warm skin doesn't trip them much.
COLORS = ["white", "black", "red", "navy", "gold"]
REGIONS = ["torso", "shorts"]
FEATURE_NAMES = [f"{r}_{c}" for r in REGIONS for c in COLORS]


def _color_frac(hsv, name) -> float:
    if name == "white":
        m = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([179, 50, 255]))
    elif name == "black":
        m = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 120, 55]))
    elif name == "red":
        m = (cv2.inRange(hsv, np.array([0, 90, 40]), np.array([10, 255, 230]))
             | cv2.inRange(hsv, np.array([170, 90, 40]), np.array([179, 255, 230])))
    elif name == "navy":
        m = cv2.inRange(hsv, np.array([95, 40, 15]), np.array([130, 255, 165]))
    elif name == "gold":
        m = cv2.inRange(hsv, np.array([16, 80, 90]), np.array([34, 255, 255]))
    else:
        return 0.0
    return float(m.sum()) / 255.0 / (hsv.shape[0] * hsv.shape[1])


def _region_features(patch):
    if patch is None or patch.size == 0:
        return [0.0] * len(COLORS)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return [_color_frac(hsv, c) for c in COLORS]


def feature_vector(frame, box):
    """Length-10 perceptual feature vector for a player box (torso then shorts)."""
    vec = []
    vec.extend(_region_features(regions.torso_patch(frame, box)))
    vec.extend(_region_features(regions.shorts_patch(frame, box)))
    return vec


def learn_fingerprint(vectors):
    """Mean feature vector over sample boxes (a team's fingerprint), or None."""
    arr = np.array([v for v in vectors if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return None
    return [float(x) for x in arr.mean(axis=0)]


def distance(vec, fingerprint) -> float:
    """L1 distance between a box's features and a team fingerprint."""
    return float(np.abs(np.array(vec) - np.array(fingerprint)).sum())


def classify(vec, teams):
    """Return (best_tracked_dist, best_other_dist) L1 distances to the nearest
    tracked and nearest non-tracked team fingerprints (inf if none)."""
    bt = bo = float("inf")
    for t in teams:
        fp = getattr(t, "fingerprint", None)
        if not fp:
            continue
        d = distance(vec, fp)
        if getattr(t, "track", False):
            bt = min(bt, d)
        else:
            bo = min(bo, d)
    return bt, bo
