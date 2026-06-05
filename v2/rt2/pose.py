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

# BGR colours
C_LEG = (0, 215, 255)     # amber  - legs / feet (priority)
C_BACK = (80, 255, 80)    # green  - spine / back
C_HEAD = (255, 0, 255)    # magenta- head / neck (separate read)
C_ARM = (150, 150, 150)   # grey   - arms (de-emphasised)
C_JOINT = (0, 0, 255)     # red    - joints
VIS_MIN = 0.3


def to_array(landmarks) -> np.ndarray:
    """MediaPipe NormalizedLandmark list -> (33,3) array of (x, y, visibility)."""
    out = np.zeros((33, 3), np.float32)
    for i, lm in enumerate(landmarks):
        out[i] = (lm.x, lm.y, lm.visibility if lm.visibility is not None else 1.0)
    return out


def _px(arr, i, w, h):
    return int(arr[i, 0] * w), int(arr[i, 1] * h)


def _mid(arr, i, j, w, h):
    return (int((arr[i, 0] + arr[j, 0]) * 0.5 * w),
            int((arr[i, 1] + arr[j, 1]) * 0.5 * h))


def _bone(frame, p1, p2, color, hw):
    """Draw a limb segment as a thick ORIENTED RECTANGLE (with rounded end caps so
    bent joints connect cleanly) - the rectangle's direction shows which way the
    segment is pointing, which a thin line can't convey."""
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


def draw_skeleton(frame_bgr, arr, *, leg=9, back=9, arm=5, head=5):
    """Draw the form-focused skeleton from a (33,3) array IN PLACE as thick
    oriented rectangles. The leg/back/arm/head values are half-widths (px at
    ~1000px wide, auto-scaled to the frame). No-op if arr is None; each element
    is drawn only when its landmark visibility passes VIS_MIN."""
    if arr is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    vis = arr[:, 2]
    sc = max(1.0, w / 1000.0)
    hw_leg, hw_arm, hw_back, hw_head = leg * sc, arm * sc, back * sc, head * sc

    def ok(i):
        return vis[i] >= VIS_MIN

    def P(i):
        return _px(arr, i, w, h)

    # arms first (so legs/back/head draw on top)
    for a, b in ARM_BONES:
        if ok(a) and ok(b):
            _bone(frame_bgr, P(a), P(b), C_ARM, hw_arm)

    # back: shoulders, pelvis, and the central spine block
    mid_sh = mid_hip = None
    if ok(L_SH) and ok(R_SH):
        _bone(frame_bgr, P(L_SH), P(R_SH), C_BACK, hw_back)
        mid_sh = _mid(arr, L_SH, R_SH, w, h)
    if ok(L_HIP) and ok(R_HIP):
        _bone(frame_bgr, P(L_HIP), P(R_HIP), C_BACK, hw_back)
        mid_hip = _mid(arr, L_HIP, R_HIP, w, h)
    if mid_sh and mid_hip:
        _bone(frame_bgr, mid_sh, mid_hip, C_BACK, hw_back)

    # legs + feet (priority - thick amber)
    for a, b in LEG_BONES:
        if ok(a) and ok(b):
            _bone(frame_bgr, P(a), P(b), C_LEG, hw_leg)

    # head: a dedicated marker + neck block (nose, else ear-midpoint)
    head_pt = None
    if ok(NOSE):
        head_pt = P(NOSE)
    elif ok(L_EAR) and ok(R_EAR):
        head_pt = _mid(arr, L_EAR, R_EAR, w, h)
    if head_pt:
        if mid_sh:
            _bone(frame_bgr, head_pt, mid_sh, C_HEAD, hw_head)
        cv2.circle(frame_bgr, head_pt, int(hw_head * 2.2), C_HEAD, max(2, int(sc * 2)), cv2.LINE_AA)

    # knees + ankles: a contrasting dot for a precise reference point
    for i in EMPH_JOINTS:
        if ok(i):
            cv2.circle(frame_bgr, P(i), max(2, int(sc * 3)), C_JOINT, -1, cv2.LINE_AA)
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
