"""BlazePose skeleton overlay via MediaPipe Tasks PoseLandmarker.

Why this model: the heavy 33-landmark BlazePose model is the most accurate option
for limb / ankle form analysis because it includes the FEET (heel + foot-index),
which the YOLO-pose 17-keypoint model does not have at all.

Running mode is IMAGE (stateless per-frame) on purpose: the training UI scrubs
backwards and forwards freely, and VIDEO mode requires monotonically increasing
timestamps which a scrubbing UI violates.

The overlay is FORM-FOCUSED, not a full skeleton:
  * legs + feet (hips→knees→ankles→heel→toe)  - AMBER, the priority for squats/lunges
  * back / spine (mid-shoulder→mid-hip)        - GREEN
  * head + neck (head marker + neck line)      - MAGENTA, a separate read on head position
  * arms (shoulder→elbow→wrist)                - dim GREY, de-emphasised
  * fingers / hands and face landmarks are NOT drawn (noise for form work)

Landmarks are passed around as a plain numpy array of shape (33, 3): normalised
(x, y) in 0..1 plus visibility. This lets the same draw path render either a live
detection or a temporally-smoothed track from posecache.PoseTrack.

Model file: models/pose_landmarker_heavy.task (downloaded once, gitignored).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mpp
from mediapipe.tasks.python import vision as _vision

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_heavy.task"

# --- BlazePose landmark indices we use ---
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32
# Extra spine bend points (populated by SynthPose only; BlazePose leaves them
# empty and the back falls back to a straight shoulder->hip line).
SP_C7, SP_THORACIC, SP_LUMBAR = 33, 34, 35
N_KP = 36                      # 33 BlazePose slots + 3 spine markers

# Bone groups (face landmarks 0..10 and hand landmarks 17..22 are intentionally absent).
LEG_BONES = [(L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN),
             (L_AN, L_HEEL), (L_HEEL, L_TOE), (L_AN, L_TOE),
             (R_AN, R_HEEL), (R_HEEL, R_TOE), (R_AN, R_TOE)]
ARM_BONES = [(L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR)]
PELVIS = (L_HIP, R_HIP)
SHOULDERS = (L_SH, R_SH)

# Joints to mark with a dot (no fingers, no face).
JOINTS = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
          L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_HEEL, R_HEEL, L_TOE, R_TOE]
EMPH_JOINTS = {L_KN, R_KN, L_AN, R_AN}     # knees + ankles drawn bigger

# BGR colours - all bright/saturated so every segment pops
C_LEG = (0, 150, 255)     # orange - legs / feet (priority)
C_BACK = (0, 255, 0)      # green  - spine / back
C_HEAD = (255, 0, 255)    # magenta- head / neck (separate read)
C_ARM = (255, 255, 0)     # cyan   - arms
C_JOINT = (0, 0, 255)     # red    - knee/ankle reference dots
C_BORDER = (15, 15, 15)   # near-black outline so crossing limbs stay distinct
VIS_MIN = 0.3


def to_array(landmarks) -> np.ndarray:
    """MediaPipe NormalizedLandmark list -> (N_KP,3) array of (x, y, visibility).
    BlazePose fills the first 33; spine slots 33-35 stay empty."""
    out = np.zeros((N_KP, 3), np.float32)
    for i, lm in enumerate(landmarks):
        out[i] = (lm.x, lm.y, lm.visibility if lm.visibility is not None else 1.0)
    return out


def _px(arr, i, w, h):
    return int(arr[i, 0] * w), int(arr[i, 1] * h)


def _mid(arr, i, j, w, h):
    return (int((arr[i, 0] + arr[j, 0]) * 0.5 * w),
            int((arr[i, 1] + arr[j, 1]) * 0.5 * h))


def _capsule(frame, p1, p2, color, hw):
    """A thick oriented rectangle + rounded end caps, all in one colour."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L >= 1:
        ux, uy = dx / L, dy / L
        nx, ny = -uy * hw, ux * hw          # perpendicular, length = half-width
        quad = np.array([[p1[0] + nx, p1[1] + ny], [p2[0] + nx, p2[1] + ny],
                         [p2[0] - nx, p2[1] - ny], [p1[0] - nx, p1[1] - ny]], np.int32)
        cv2.fillConvexPoly(frame, quad, color, cv2.LINE_AA)
    r = max(1, int(round(hw)))
    cv2.circle(frame, (int(p1[0]), int(p1[1])), r, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(p2[0]), int(p2[1])), r, color, -1, cv2.LINE_AA)


def _bone(frame, p1, p2, color, hw, border):
    """Draw a limb segment as a thick ORIENTED RECTANGLE with a dark OUTLINE.
    The outline is drawn first at hw+border, the bright fill on top at hw - so
    when a later bone crosses an earlier one its dark border separates them and
    you can tell which limb is in front."""
    _capsule(frame, p1, p2, C_BORDER, hw + border)
    _capsule(frame, p1, p2, color, hw)


def _dot(frame, p, color, r, border):
    cv2.circle(frame, p, int(r + border), C_BORDER, -1, cv2.LINE_AA)
    cv2.circle(frame, p, int(r), color, -1, cv2.LINE_AA)


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(frame, p, text, sc):
    """A small numbered badge (dark disc + white text) marking a limb's ID. The
    ID is the SAME in both side-by-side videos, so you can match a limb across the
    two camera angles."""
    f = 0.5 * sc
    th = max(1, int(round(sc)))
    (tw, thh), _ = cv2.getTextSize(text, _FONT, f, th)
    cx, cy = int(p[0]), int(p[1])
    rad = int(max(tw, thh) * 0.6 + 5 * sc)
    cv2.circle(frame, (cx, cy), rad, C_BORDER, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (cx - tw // 2, cy + thh // 2), _FONT, f,
                (255, 255, 255), th, cv2.LINE_AA)


def draw_skeleton(frame_bgr, arr, *, leg=13, back=12, arm=10, head=8):
    """Draw the form-focused skeleton from a (N_KP,3) array IN PLACE as thick,
    bright, dark-outlined oriented rectangles. The leg/back/arm/head values are
    half-widths (px at ~1000px wide, auto-scaled to the frame). The back follows
    real spine bend points (C7/thoracic/lumbar) when present, else a straight
    shoulder->hip line. No-op if arr is None; each element drawn only when its
    landmark visibility passes VIS_MIN."""
    if arr is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    vis = arr[:, 2]
    n = arr.shape[0]
    sc = max(1.0, w / 1000.0)
    hw_leg, hw_arm, hw_back, hw_head = leg * sc, arm * sc, back * sc, head * sc
    bd = max(2, int(round(sc * 2.5)))      # outline thickness

    def ok(i):
        return i < n and vis[i] >= VIS_MIN

    def P(i):
        return _px(arr, i, w, h)

    # Build the list of bones to draw. Each carries a DEPTH SCORE = mean endpoint
    # visibility; we draw low-score (further / self-occluded) first so the nearer,
    # higher-confidence limb ends up ON TOP - correct for side views. `label` is a
    # per-limb ID, identical in both videos so limbs marry up across angles.
    bones = []   # (pa, pb, color, hw, score, label)

    def add(a, b, color, hw, label=None):
        if ok(a) and ok(b):
            bones.append((P(a), P(b), color, hw, (vis[a] + vis[b]) * 0.5, label))

    def add_pt(pa, pb, color, hw, score, label=None):
        if pa is not None and pb is not None:
            bones.append((pa, pb, color, hw, score, label))

    # arms (cyan)
    add(L_SH, L_EL, C_ARM, hw_arm, "7"); add(L_EL, L_WR, C_ARM, hw_arm, "9")
    add(R_SH, R_EL, C_ARM, hw_arm, "8"); add(R_EL, R_WR, C_ARM, hw_arm, "10")
    # legs + feet (orange)
    add(L_HIP, L_KN, C_LEG, hw_leg, "1"); add(L_KN, L_AN, C_LEG, hw_leg, "3")
    add(R_HIP, R_KN, C_LEG, hw_leg, "2"); add(R_KN, R_AN, C_LEG, hw_leg, "4")
    add(L_AN, L_TOE, C_LEG, hw_leg, "5"); add(L_AN, L_HEEL, C_LEG, hw_leg)
    add(L_HEEL, L_TOE, C_LEG, hw_leg)
    add(R_AN, R_TOE, C_LEG, hw_leg, "6"); add(R_AN, R_HEEL, C_LEG, hw_leg)
    add(R_HEEL, R_TOE, C_LEG, hw_leg)

    # back (green): shoulder + pelvis cross-bars + a FLEXIBLE spine through
    # whatever bend points exist (C7 -> thoracic -> lumbar -> pelvis), else a
    # straight shoulder->hip line.
    mid_sh = _mid(arr, L_SH, R_SH, w, h) if (ok(L_SH) and ok(R_SH)) else None
    mid_hip = _mid(arr, L_HIP, R_HIP, w, h) if (ok(L_HIP) and ok(R_HIP)) else None
    sh_sc = (vis[L_SH] + vis[R_SH]) * 0.5 if (ok(L_SH) and ok(R_SH)) else 0.5
    hip_sc = (vis[L_HIP] + vis[R_HIP]) * 0.5 if (ok(L_HIP) and ok(R_HIP)) else 0.5
    if mid_sh:
        add_pt(P(L_SH), P(R_SH), C_BACK, hw_back, sh_sc)
    if mid_hip:
        add_pt(P(L_HIP), P(R_HIP), C_BACK, hw_back, hip_sc)
    spine = [P(SP_C7) if ok(SP_C7) else mid_sh]
    if ok(SP_THORACIC):
        spine.append(P(SP_THORACIC))
    if ok(SP_LUMBAR):
        spine.append(P(SP_LUMBAR))
    spine.append(mid_hip)
    spine = [p for p in spine if p is not None]
    for a, b in zip(spine, spine[1:]):
        add_pt(a, b, C_BACK, hw_back, (sh_sc + hip_sc) * 0.5, "S" if a is spine[0] else None)

    # head marker + neck block (nose, else ear-midpoint)
    head_pt = None
    if ok(NOSE):
        head_pt = P(NOSE)
    elif ok(L_EAR) and ok(R_EAR):
        head_pt = _mid(arr, L_EAR, R_EAR, w, h)
    neck_to = (P(SP_C7) if ok(SP_C7) else mid_sh)
    head_sc = vis[NOSE] if ok(NOSE) else 0.9

    # DEPTH SORT: draw furthest (lowest confidence) first, nearest last/on top.
    bones.sort(key=lambda e: e[4])
    for pa, pb, color, hw, _s, _lab in bones:
        _bone(frame_bgr, pa, pb, color, hw, bd)

    # head drawn near the top of the stack
    if head_pt:
        if neck_to:
            _bone(frame_bgr, head_pt, neck_to, C_HEAD, hw_head, bd)
        rr = int(hw_head * 2.2)
        cv2.circle(frame_bgr, head_pt, rr + bd, C_BORDER, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, head_pt, rr, C_HEAD, -1, cv2.LINE_AA)

    # knees + ankles reference dots, then limb ID badges, on top so they read.
    for i in EMPH_JOINTS:
        if ok(i):
            _dot(frame_bgr, P(i), C_JOINT, max(3, int(sc * 4)), bd)
    for pa, pb, _c, _hw, _s, label in bones:
        if label:
            _label(frame_bgr, ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2), label, sc)
    return frame_bgr


class PoseOverlay:
    """BlazePose estimator. IMAGE mode is stateless, so ONE instance can annotate
    frames from several clips in any order. NOT thread-safe - use a separate
    instance per thread (e.g. the refine worker creates its own)."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL, min_conf: float = 0.3,
                 mode: str = "image"):
        """mode="image": stateless per-frame, safe for random-access scrubbing.
        mode="video": temporal tracking - each frame is guided by the previous
        one, far steadier THROUGH MOVEMENT, but requires strictly increasing
        timestamps (so only use it on a sequential forward pass, e.g. Refine)."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"BlazePose model not found: {model_path}\n"
                "Download it once with:\n"
                "  Invoke-WebRequest "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task "
                f"-OutFile '{model_path}'")
        self.mode = mode
        rm = _vision.RunningMode.VIDEO if mode == "video" else _vision.RunningMode.IMAGE
        opts = _vision.PoseLandmarkerOptions(
            base_options=_mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=rm,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = _vision.PoseLandmarker.create_from_options(opts)

    def landmarks_array(self, frame_bgr, t_ms: "int | None" = None) -> "np.ndarray | None":
        """Return a (33,3) array of (x, y, visibility) for the most prominent
        person, or None if no pose is found. In video mode pass a monotonically
        increasing t_ms."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            res = self._lm.detect_for_video(image, int(t_ms if t_ms is not None else 0))
        else:
            res = self._lm.detect(image)
        if not res.pose_landmarks:
            return None
        return to_array(res.pose_landmarks[0])

    # backwards-compatible aliases
    def draw(self, frame_bgr, arr, **kw):
        return draw_skeleton(frame_bgr, arr, **kw)

    def annotate(self, frame_bgr):
        out = frame_bgr.copy()
        return draw_skeleton(out, self.landmarks_array(out))

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass


if __name__ == "__main__":
    est = PoseOverlay()
    blank = np.zeros((480, 640, 3), np.uint8)
    print("[pose] model loaded OK; pose in blank frame:",
          est.landmarks_array(blank) is not None)
    est.close()
