"""
cmc.py - Camera-Motion Compensation for PANNING/ZOOMING footage.

THE PROBLEM
  Field-accurate overlays (offside lines, metre-radius ground circles, minimap)
  need a pixel<->pitch homography H. On Veo auto-follow footage the camera pans
  and zooms, so a homography clicked on ONE frame is wrong on the next. We can't
  re-click every frame.

THE FIX (v1)
  Calibrate ONCE on an anchor frame (H0 from rt2.homography). Then estimate the
  frame-to-frame CAMERA motion (a 2D homography between consecutive frames, from
  matched background features) and COMPOSE it, so the pitch mapping slides along
  with the pan/zoom. Drift accumulates over long spans -> we expose a drift signal
  so the app can prompt a quick re-anchor. Pure cv2/numpy (NO torch) - safe to
  import anywhere, headless-testable.

KEY OBJECTS
  estimate_pair(prevG, curG) -> (H_prev_to_cur, n_inliers)
      ORB features + RANSAC homography mapping points in prevG -> curG.
  CameraTracker
      anchor(gray); update(gray) each frame; keeps the cumulative transform
      ANCHOR_image -> CURRENT_image (from_anchor) and its inverse (to_anchor),
      plus a drift/health signal.
  LiveHomography(pitch_h, tracker)
      wraps a PitchHomography (the anchored H0) + a CameraTracker so
      image_to_pitch / pitch_to_image / foot_point work on the CURRENT frame.

PLAYERS ARE OUTLIERS
  Moving players bias the motion estimate; RANSAC rejects most, and the static
  background (lines, stands, posts) dominates. update() accepts an optional
  `ignore_boxes` list to additionally mask player regions out of the features.
"""
from __future__ import annotations

import numpy as np


def _to_gray(img):
    import cv2
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _mask_for(shape, ignore_boxes):
    """255 everywhere except inside ignore_boxes (player regions), or None."""
    if not ignore_boxes:
        return None
    import cv2
    h, w = shape[:2]
    m = np.full((h, w), 255, np.uint8)
    for b in ignore_boxes:
        x1, y1, x2, y2 = (int(v) for v in b[:4])
        cv2.rectangle(m, (max(0, x1), max(0, y1)),
                      (min(w - 1, x2), min(h - 1, y2)), 0, -1)
    return m


def estimate_pair(prev_gray, cur_gray, max_features=1200, ignore_boxes=None):
    """Homography mapping points in prev_gray -> cur_gray (camera motion), and the
    RANSAC inlier count. Returns (H 3x3 float64, n_inliers) or (None, 0). Never
    raises."""
    try:
        import cv2
        pg, cg = _to_gray(prev_gray), _to_gray(cur_gray)
        if pg is None or cg is None:
            return None, 0
        mask = _mask_for(pg.shape, ignore_boxes)
        orb = cv2.ORB_create(max_features)
        k1, d1 = orb.detectAndCompute(pg, mask)
        k2, d2 = orb.detectAndCompute(cg, mask)
        if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
            return None, 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if len(matches) < 12:
            return None, 0
        matches = sorted(matches, key=lambda m: m.distance)[:400]
        src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0
        return H.astype(np.float64), int(inl.sum()) if inl is not None else 0
    except Exception:
        return None, 0


def _apply_H(H, x, y):
    """Apply a 3x3 homography to a point. Returns (x', y') or None."""
    v = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-9:
        return None
    return (float(v[0] / v[2]), float(v[1] / v[2]))


class CameraTracker:
    """Tracks cumulative camera motion from an ANCHOR frame to the CURRENT frame.

    `from_anchor(x,y)` maps an anchor-frame image point to where it is NOW (for
    DRAWING a calibrated overlay). `to_anchor(x,y)` maps a current-frame point
    back to the anchor frame (to look it up through the anchored homography)."""

    def __init__(self, min_inliers=25):
        self.min_inliers = int(min_inliers)
        self._prev = None
        self.A2C = np.eye(3, dtype=np.float64)   # anchor_image -> current_image
        self.C2A = np.eye(3, dtype=np.float64)   # current_image -> anchor_image
        self.last_inliers = 0
        self.frames_since_anchor = 0
        self.healthy = True

    def anchor(self, gray):
        """(Re)set the anchor to this frame; reset the cumulative transform."""
        self._prev = _to_gray(gray)
        self.A2C = np.eye(3, dtype=np.float64)
        self.C2A = np.eye(3, dtype=np.float64)
        self.last_inliers = 0
        self.frames_since_anchor = 0
        self.healthy = True
        return self

    def update(self, gray, ignore_boxes=None):
        """Fold one new frame's motion into the cumulative transform. Returns True
        if the motion was estimated healthily, False if it was weak (the previous
        transform is held and `healthy` goes False so the app can prompt a
        re-anchor)."""
        g = _to_gray(gray)
        if self._prev is None:
            return self.anchor(g) and True
        H_pc, n = estimate_pair(self._prev, g, ignore_boxes=ignore_boxes)
        self.last_inliers = n
        if H_pc is None or n < self.min_inliers:
            self.healthy = False
            self._prev = g
            self.frames_since_anchor += 1
            return False
        # prev->cur composes onto anchor->prev to give anchor->cur
        self.A2C = H_pc @ self.A2C
        try:
            self.C2A = np.linalg.inv(self.A2C)
        except np.linalg.LinAlgError:
            self.healthy = False
            return False
        self._prev = g
        self.frames_since_anchor += 1
        self.healthy = True
        return True

    def from_anchor(self, x, y):
        return _apply_H(self.A2C, x, y)

    def to_anchor(self, x, y):
        return _apply_H(self.C2A, x, y)


class LiveHomography:
    """A PitchHomography anchored at one frame + a CameraTracker = a pixel<->pitch
    map valid on the CURRENT (panned) frame. All methods return None when the
    pitch homography isn't ok or a transform fails."""

    def __init__(self, pitch_h, tracker: "CameraTracker"):
        self.pitch_h = pitch_h
        self.tracker = tracker

    @property
    def ok(self):
        return bool(self.pitch_h and getattr(self.pitch_h, "ok", False))

    @property
    def healthy(self):
        return bool(self.ok and self.tracker.healthy)

    def image_to_pitch(self, x, y):
        if not self.ok:
            return None
        a = self.tracker.to_anchor(x, y)       # current image -> anchor image
        if a is None:
            return None
        return self.pitch_h.image_to_pitch(a[0], a[1])

    def pitch_to_image(self, X, Y):
        if not self.ok:
            return None
        a = self.pitch_h.pitch_to_image(X, Y)  # pitch -> anchor image
        if a is None:
            return None
        return self.tracker.from_anchor(a[0], a[1])   # anchor image -> current

    def foot_point(self, box):
        if not self.ok:
            return None
        x = 0.5 * (box[0] + box[2])
        y = max(box[1], box[3])
        return self.image_to_pitch(x, y)


def _selftest():
    print("[cmc-selftest] camera-motion compensation")
    import cv2
    # a textured synthetic frame, then a known TRANSLATION of it
    rng = np.random.default_rng(0)
    base = (rng.random((300, 400)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    dx, dy = 17, -9
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(base, M, (400, 300))

    H, n = estimate_pair(base, shifted)
    assert H is not None and n >= 25, f"weak motion estimate (n={n})"
    # a point at (200,150) in base should map ~ (200+dx,150+dy) in shifted
    p = _apply_H(H, 200, 150)
    assert abs(p[0] - (200 + dx)) < 2 and abs(p[1] - (150 + dy)) < 2, p
    print(f"[cmc-selftest] estimate_pair recovers translation ({dx},{dy}) "
          f"-> {tuple(round(v,1) for v in p)} (inliers {n})  OK")

    # CameraTracker: anchor, then a couple of shifts; from_anchor follows them
    t = CameraTracker()
    t.anchor(base)
    cur = base
    total = 0
    for _ in range(3):
        M = np.float32([[1, 0, 10], [0, 1, 5]])
        cur = cv2.warpAffine(cur, M, (400, 300))
        total += 10
        ok = t.update(cur)
        assert ok, "healthy shift should track"
    moved = t.from_anchor(100, 100)        # anchor point now sits +30,+15 later
    assert abs(moved[0] - (100 + 30)) < 4 and abs(moved[1] - (100 + 15)) < 4, moved
    back = t.to_anchor(*moved)
    assert abs(back[0] - 100) < 4 and abs(back[1] - 100) < 4, back
    print(f"[cmc-selftest] CameraTracker tracks 3 shifts; from/to_anchor "
          f"round-trip  OK")

    # LiveHomography degrades safely with a not-ok pitch homography
    class _NotOk:
        ok = False
    lh = LiveHomography(_NotOk(), CameraTracker())
    assert lh.image_to_pitch(1, 2) is None and lh.pitch_to_image(1, 2) is None
    assert not lh.ok and not lh.healthy
    print("[cmc-selftest] LiveHomography inert when pitch H not ok  OK")
    print("[cmc-selftest] PASS")


if __name__ == "__main__":
    _selftest()
