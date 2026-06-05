"""homography.py - pitch-plane homography for the rugby tracker.

A single planar homography H maps IMAGE pixels (x, y) of the broadcast frame to
REAL PITCH METRES (X, Y) on the ground plane, and H^-1 maps back. With it, a
player's feet (the bottom-centre of their box) localise to a true pitch position
in metres -- far sharper than the noisy normalised frame-zone coords the
candidate shortlist uses today -- and we can draw a top-down minimap.

COORDINATE FRAME (the one the calibrate tool's template uses)
  Origin (0, 0)      = the corner where the LEFT try-line meets the BOTTOM
                       touchline (the near-left in-goal corner).
  +X  (metres)       = along the touchline towards the RIGHT try-line
                       (0 .. PITCH_LENGTH). The in-goal areas are included, so
                       the field-of-play try-lines sit at X = IN_GOAL and
                       X = IN_GOAL + PLAY_LENGTH.
  +Y  (metres)       = across the pitch from the BOTTOM touchline to the TOP
                       touchline (0 .. PITCH_WIDTH).
  Distances are real metres; this matches "wingers play wide / fullback deep /
  forwards central" directly (wide = small or large Y; the attacking/defending
  thirds are X bands).

Pure numpy + cv2, fully headless. Nothing here imports torch / ultralytics /
pyarrow. Every public method is robust: a degenerate / insufficient calibration
yields an object whose `.ok` is False and whose transforms return None, and
`load()` never raises -- it returns None on any problem.

Run `python v2/rt2/homography.py` for the self-test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:                                   # cv2 is opencv-contrib in this venv
    import cv2
except Exception:                      # pragma: no cover - cv2 is always present here
    cv2 = None


# --- standard rugby pitch extents, in metres ---------------------------------
# A full pitch is the field of play (try-line to try-line) plus an in-goal at
# each end. Field of play is ~94-100 m long x ~68-70 m wide; in-goals are
# 6-22 m deep. We pick representative regulation-ish values; the actual mapping
# comes from the clicked correspondences, these constants just give the template
# named points sensible coordinates and size the minimap.
PLAY_LENGTH = 100.0      # try-line to try-line (field of play), metres
IN_GOAL = 10.0           # in-goal depth at each end, metres
PITCH_LENGTH = PLAY_LENGTH + 2 * IN_GOAL    # 120 m total, try-line area incl.
PITCH_WIDTH = 70.0       # touchline to touchline, metres

# X of the two try-lines and the key cross-pitch lines (metres from origin).
_TRY_L = IN_GOAL                         # left try-line  = 10
_TRY_R = IN_GOAL + PLAY_LENGTH           # right try-line = 110
_22_L = _TRY_L + 22.0                    # left 22m line  = 32
_22_R = _TRY_R - 22.0                    # right 22m line = 88
_HALF = IN_GOAL + PLAY_LENGTH / 2.0      # halfway        = 60
_Y_BOT = 0.0                             # bottom touchline
_Y_TOP = PITCH_WIDTH                     # top touchline
_Y_MID = PITCH_WIDTH / 2.0               # mid-width (goalposts sit here)
_10_L = _HALF - 10.0                     # left 10m line  = 50 ("40m from L goal")
_10_R = _HALF + 10.0                     # right 10m line = 70 ("40m from R goal")
_Y_5B = 5.0                              # 5m dash line off the BOTTOM touch
_Y_15B = 15.0                            # 15m dash line off the BOTTOM touch
_Y_15T = PITCH_WIDTH - 15.0              # = 55 (15m off the TOP touch)
_Y_5T = PITCH_WIDTH - 5.0                # = 65 (5m off the TOP touch)
_POST_HALF = 2.8                         # union goalposts are 5.6 m apart
_POST_B = _Y_MID - _POST_HALF            # = 32.2 (lower upright base)
_POST_T = _Y_MID + _POST_HALF            # = 37.8 (upper upright base)

# Calibration landmark template = every VERTICAL line crossed with every painted
# CROSS line (the 5m/15m dashes + the two touchlines), mirrored both ends. These
# are the clean, clickable intersections the user calibrates from as the camera
# pans (only a couple are visible per frame; CMC accumulates them).
_VERTS = [("L try", _TRY_L), ("L 22m", _22_L), ("L 10m", _10_L),
          ("halfway", _HALF),
          ("R 10m", _10_R), ("R 22m", _22_R), ("R try", _TRY_R)]
_CROSS = [("low flag", _Y_BOT), ("5m low", _Y_5B), ("15m low", _Y_15B),
          ("15m up", _Y_15T), ("5m up", _Y_5T), ("up flag", _Y_TOP)]
_NAMED_POINTS = {f"{vn} / {cn}": (float(X), float(Y))
                 for vn, X in _VERTS for cn, Y in _CROSS}
_NAMED_POINTS["halfway / centre (kickoff)"] = (_HALF, _Y_MID)
_NAMED_POINTS["L post base (lower)"] = (_TRY_L, _POST_B)
_NAMED_POINTS["L post base (upper)"] = (_TRY_L, _POST_T)
_NAMED_POINTS["R post base (lower)"] = (_TRY_R, _POST_B)
_NAMED_POINTS["R post base (upper)"] = (_TRY_R, _POST_T)


class _Pitch:
    """Standard pitch description: extents + named reference points (metres).

    `named_points` maps a human label -> (X, Y) in metres, offered by the
    calibrate tool as a template so the user clicks each landmark in the frame
    and we already know its real-world coordinate. Goalpost bases sit on the
    try-lines at mid-width.
    """

    length = PITCH_LENGTH
    width = PITCH_WIDTH
    play_length = PLAY_LENGTH
    in_goal = IN_GOAL

    named_points = _NAMED_POINTS

    def names(self):
        """Template landmark labels, in a sensible click order."""
        return list(self.named_points.keys())


PITCH = _Pitch()


def _as_xy_array(points):
    """Coerce a list of (x, y) pairs to a float32 (N, 2) array, or None."""
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 2:
        return None
    return arr


class PitchHomography:
    """A planar image<->pitch homography built from >=4 point correspondences.

    Construct with matched lists `image_points` (pixels) and `pitch_points`
    (metres). On success `.ok` is True and `.H` / `.H_inv` are 3x3 float64
    arrays. On any failure (too few points, degenerate config, cv2 missing) the
    object is still usable but `.ok` is False and every transform returns None.
    """

    def __init__(self, image_points=None, pitch_points=None, H=None,
                 anchor_frame=None):
        self.image_points = [tuple(map(float, p)) for p in (image_points or [])]
        self.pitch_points = [tuple(map(float, p)) for p in (pitch_points or [])]
        # The video frame index the pitch points were clicked on. CMC must
        # anchor at exactly this frame for the live homography to be valid.
        try:
            self.anchor_frame = None if anchor_frame is None else int(anchor_frame)
        except (TypeError, ValueError):
            self.anchor_frame = None
        self.H = None
        self.H_inv = None
        self.ok = False
        self.reproj_error = None

        if H is not None:
            self._set_H(np.asarray(H, dtype=np.float64))
            # still recompute error if we have correspondences to check against
            if self.ok and len(self.image_points) >= 4:
                self.reproj_error = self._compute_reproj_error()
            return

        img = _as_xy_array(self.image_points)
        pit = _as_xy_array(self.pitch_points)
        if img is None or pit is None:
            return
        if img.shape[0] < 4 or pit.shape[0] < 4 or img.shape[0] != pit.shape[0]:
            return
        if cv2 is None:
            return
        try:
            H_est, _mask = cv2.findHomography(img, pit, method=0)
        except cv2.error:
            H_est = None
        if H_est is None:
            return
        self._set_H(np.asarray(H_est, dtype=np.float64))
        if self.ok:
            self.reproj_error = self._compute_reproj_error()

    # ---- internals -------------------------------------------------------
    def _set_H(self, H):
        """Validate + store H and its inverse; sets `.ok`."""
        if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
            self.ok = False
            return
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.ok = False
            return
        if not np.all(np.isfinite(H_inv)):
            self.ok = False
            return
        self.H = H
        self.H_inv = H_inv
        self.ok = True

    @staticmethod
    def _apply(M, x, y):
        """Apply 3x3 M to point (x, y) with the homogeneous divide; None if the
        point maps to (near) the line at infinity."""
        if M is None:
            return None
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            return None
        v = M @ np.array([xf, yf, 1.0], dtype=np.float64)
        w = v[2]
        if not np.isfinite(w) or abs(w) < 1e-12:
            return None
        out = (v[0] / w, v[1] / w)
        if not (np.isfinite(out[0]) and np.isfinite(out[1])):
            return None
        return out

    def _compute_reproj_error(self):
        """RMS reprojection error (metres) of image_points -> pitch vs the given
        pitch_points. None if not computable."""
        if not self.ok:
            return None
        errs = []
        for (px, py), (mx, my) in zip(self.image_points, self.pitch_points):
            mapped = self._apply(self.H, px, py)
            if mapped is None:
                continue
            errs.append((mapped[0] - mx) ** 2 + (mapped[1] - my) ** 2)
        if not errs:
            return None
        return float(np.sqrt(np.mean(errs)))

    # ---- public transforms ----------------------------------------------
    def image_to_pitch(self, x, y):
        """Image pixel (x, y) -> pitch (X_m, Y_m), or None if not ok / degenerate."""
        if not self.ok:
            return None
        return self._apply(self.H, x, y)

    def pitch_to_image(self, X, Y):
        """Pitch (X_m, Y_m) -> image pixel (x_px, y_px), or None."""
        if not self.ok:
            return None
        return self._apply(self.H_inv, X, Y)

    def foot_point(self, box):
        """Pitch (X, Y) of a player's FEET: the bottom-centre of an (x1,y1,x2,y2)
        box (the contact point with the ground plane, where the homography is
        valid). None on a bad box or if not ok."""
        if not self.ok or box is None:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        fx = (x1 + x2) * 0.5
        fy = max(y1, y2)           # bottom edge (image y grows downward)
        return self.image_to_pitch(fx, fy)

    # ---- persistence -----------------------------------------------------
    def to_dict(self):
        return {
            "image_points": [list(p) for p in self.image_points],
            "pitch_points": [list(p) for p in self.pitch_points],
            "H": self.H.tolist() if self.H is not None else None,
            "reproj_error": self.reproj_error,
            "anchor_frame": self.anchor_frame,
            "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH,
                      "play_length": PLAY_LENGTH, "in_goal": IN_GOAL},
        }

    def save(self, path):
        """Write the homography to JSON. Returns the Path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        """Load a homography from JSON. TOLERANT: returns None on a missing,
        unreadable, or garbled file, or if it does not yield an ok homography.
        Never raises.

        Prefers the stored H (exact round-trip with how it was computed); falls
        back to re-deriving from the stored correspondences."""
        try:
            path = Path(path)
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        img = data.get("image_points")
        pit = data.get("pitch_points")
        H = data.get("H")
        af = data.get("anchor_frame")              # tolerant: may be absent -> None
        try:
            obj = cls(image_points=img, pitch_points=pit, H=H, anchor_frame=af)
        except Exception:                          # noqa: BLE001 - never raise on load
            return None
        if not obj.ok:
            # last resort: try re-deriving from correspondences alone
            try:
                obj = cls(image_points=img, pitch_points=pit, anchor_frame=af)
            except Exception:                      # noqa: BLE001
                return None
        return obj if obj.ok else None


# =============================================================================
# Self-test
# =============================================================================
def _selftest():
    print("[homography-selftest] building from synthetic correspondences")
    assert cv2 is not None, "cv2 (opencv-contrib) must be available"

    # A known projective map metres = M . image (homogeneous). We invent M,
    # generate image points, push them through M to get the 'true' pitch points,
    # then check PitchHomography recovers the map.
    M_true = np.array([
        [0.05, 0.002, -3.0],
        [0.001, 0.06, -2.0],
        [1e-4, 2e-4, 1.0],
    ], dtype=np.float64)

    def push(M, x, y):
        v = M @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    img_pts = [(100.0, 800.0), (1800.0, 820.0), (1750.0, 300.0),
               (250.0, 320.0), (960.0, 540.0)]
    pit_pts = [push(M_true, x, y) for (x, y) in img_pts]

    h = PitchHomography(image_points=img_pts, pitch_points=pit_pts)
    assert h.ok, "should build with 5 good correspondences"

    # round-trip: image -> pitch -> image within tolerance
    for (x, y) in img_pts:
        XY = h.image_to_pitch(x, y)
        assert XY is not None
        back = h.pitch_to_image(*XY)
        assert back is not None
        assert abs(back[0] - x) < 1e-3 and abs(back[1] - y) < 1e-3, (x, y, back)

    # image -> pitch matches the true map within tolerance
    for (x, y), (mx, my) in zip(img_pts, pit_pts):
        XY = h.image_to_pitch(x, y)
        assert abs(XY[0] - mx) < 1e-4 and abs(XY[1] - my) < 1e-4, (XY, (mx, my))
    assert h.reproj_error is not None and h.reproj_error < 1e-4, h.reproj_error
    print(f"[homography-selftest] round-trip OK (reproj {h.reproj_error:.2e} m)")

    # foot_point uses the BOTTOM-CENTRE of the box
    box = (100.0, 200.0, 300.0, 600.0)   # x1,y1,x2,y2
    expect = h.image_to_pitch((100.0 + 300.0) / 2.0, 600.0)
    got = h.foot_point(box)
    assert got is not None and expect is not None
    assert abs(got[0] - expect[0]) < 1e-9 and abs(got[1] - expect[1]) < 1e-9, (got, expect)
    # and it is NOT the box centre (sanity: a y-centred point differs)
    centre = h.image_to_pitch(200.0, 400.0)
    assert abs(got[1] - centre[1]) > 1e-6
    print("[homography-selftest] foot_point uses bottom-centre OK")

    # anchor_frame: default None, coerced to int, round-trips through save/load
    assert h.anchor_frame is None, "anchor_frame defaults to None"
    h_af = PitchHomography(image_points=img_pts, pitch_points=pit_pts,
                           anchor_frame=137)
    assert h_af.anchor_frame == 137 and isinstance(h_af.anchor_frame, int)
    # garbage anchor_frame -> None, never raises
    assert PitchHomography(image_points=img_pts, pitch_points=pit_pts,
                           anchor_frame="nope").anchor_frame is None
    print("[homography-selftest] anchor_frame coercion OK")

    # save / load round-trip (tmp file)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.pitch.json"
        h.save(p)
        h2 = PitchHomography.load(p)
        assert h2 is not None and h2.ok
        # transforms agree
        for (x, y) in img_pts:
            a = h.image_to_pitch(x, y)
            b = h2.image_to_pitch(x, y)
            assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
        # anchor_frame round-trips through save/load (and absence -> None)
        assert h2.anchor_frame is None, "no anchor_frame stored -> None on load"
        paf = Path(td) / "af.pitch.json"
        h_af.save(paf)
        h_af2 = PitchHomography.load(paf)
        assert h_af2 is not None and h_af2.anchor_frame == 137, h_af2.anchor_frame
        # tolerant load: missing + garbled never raise, return None
        assert PitchHomography.load(Path(td) / "nope.json") is None
        bad = Path(td) / "bad.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        assert PitchHomography.load(bad) is None
        empty = Path(td) / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        assert PitchHomography.load(empty) is None
    print("[homography-selftest] save/load + tolerant load OK")

    # insufficient points -> not ok, transforms return None, never raises
    bad = PitchHomography(image_points=[(0, 0), (1, 1), (2, 2)],
                          pitch_points=[(0, 0), (1, 0), (0, 1)])
    assert not bad.ok
    assert bad.image_to_pitch(5, 5) is None
    assert bad.pitch_to_image(5, 5) is None
    assert bad.foot_point((0, 0, 10, 10)) is None
    # mismatched lengths -> not ok
    assert not PitchHomography(image_points=[(0, 0)] * 4,
                               pitch_points=[(0, 0)] * 3).ok
    # empty -> not ok
    assert not PitchHomography().ok
    print("[homography-selftest] degenerate/insufficient -> not ok + None OK")

    # PITCH template sanity
    assert len(PITCH.names()) >= 8
    for name in PITCH.names():
        X, Y = PITCH.named_points[name]
        assert 0.0 <= X <= PITCH_LENGTH and 0.0 <= Y <= PITCH_WIDTH
    assert PITCH.named_points["halfway / centre (kickoff)"] == (60.0, 35.0)
    assert PITCH.named_points["R try / up flag"] == (110.0, 70.0)
    assert PITCH.named_points["R 22m / 5m low"] == (88.0, 5.0)
    assert PITCH.named_points["R post base (upper)"] == (110.0, 37.8)
    print(f"[homography-selftest] PITCH template {len(PITCH.names())} named points OK")

    print("[homography-selftest] PASS")


if __name__ == "__main__":
    _selftest()
