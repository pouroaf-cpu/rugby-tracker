"""BlazePose skeleton overlay via MediaPipe Tasks PoseLandmarker.

Why this model: the heavy 33-landmark BlazePose model is the most accurate option
for limb / ankle form analysis because it includes the FEET (heel + foot-index),
which the YOLO-pose 17-keypoint model does not have at all. Lower-body angles
(ankle dorsiflexion, knee tracking) need those foot points.

Running mode is IMAGE (stateless per-frame) on purpose: the training UI lets you
scrub backwards and forwards freely, and the VIDEO mode requires monotonically
increasing timestamps which a scrubbing UI violates. IMAGE mode runs the same
detector on each frame independently, so seeking can't desync it.

Model file: models/pose_landmarker_heavy.task (downloaded once, gitignored).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mpp
from mediapipe.tasks.python import vision as _vision

# Project-root/models/pose_landmarker_heavy.task  (this file is v2/rt2/pose.py)
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_heavy.task"

# BlazePose 33-landmark BODY skeleton (face landmarks 0..10 deliberately omitted -
# only the body matters for exercise form). Indices follow the BlazePose spec.
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),     # shoulders + arms
    (11, 23), (12, 24), (23, 24),                         # torso
    (23, 25), (25, 27), (24, 26), (26, 28),               # upper + lower legs
    (27, 29), (29, 31), (27, 31),                         # left ankle-heel-toe
    (28, 30), (30, 32), (28, 32),                         # right ankle-heel-toe
]
# Lower body (hips, legs, feet) is highlighted in a separate colour because it's
# the focus for squats / lunges.
_LOWER = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
_VIS_MIN = 0.3   # hide landmarks the model isn't confident are visible

# BGR colours
_C_LOWER = (0, 215, 255)   # amber  - legs / feet
_C_UPPER = (80, 255, 80)   # green  - torso / arms
_C_JOINT = (0, 0, 255)     # red    - joints


class PoseOverlay:
    """Single reusable BlazePose estimator. IMAGE mode is stateless, so ONE
    instance can annotate frames from several clips in any order."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL, min_conf: float = 0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"BlazePose model not found: {model_path}\n"
                "Download it once with:\n"
                "  Invoke-WebRequest "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task "
                f"-OutFile '{model_path}'")
        opts = _vision.PoseLandmarkerOptions(
            base_options=_mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = _vision.PoseLandmarker.create_from_options(opts)

    def landmarks(self, frame_bgr):
        """Return the 33 normalised landmarks (x,y in 0..1) for the most
        prominent person, or None if no pose is found."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect(image)
        return res.pose_landmarks[0] if res.pose_landmarks else None

    def draw(self, frame_bgr, lms, *, bone: int = 3, joint: int = 5):
        """Draw the skeleton onto frame_bgr IN PLACE and return it. No-op if
        lms is None."""
        if lms is None:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        pts = []
        for lm in lms:
            vis = lm.visibility if lm.visibility is not None else 1.0
            pts.append((int(lm.x * w), int(lm.y * h), vis))
        for a, b in POSE_CONNECTIONS:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va < _VIS_MIN or vb < _VIS_MIN:
                continue
            col = _C_LOWER if (a in _LOWER and b in _LOWER) else _C_UPPER
            cv2.line(frame_bgr, (xa, ya), (xb, yb), col, bone, cv2.LINE_AA)
        for i, (x, y, v) in enumerate(pts):
            if i < 11 or v < _VIS_MIN:   # skip face landmarks 0..10
                continue
            cv2.circle(frame_bgr, (x, y), joint, _C_JOINT, -1, cv2.LINE_AA)
        return frame_bgr

    def annotate(self, frame_bgr):
        """Convenience: detect + draw on a COPY, return the annotated copy."""
        out = frame_bgr.copy()
        return self.draw(out, self.landmarks(out))

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Smoke test: load the model and run it on a blank frame (expects 0 poses).
    est = PoseOverlay()
    blank = np.zeros((480, 640, 3), np.uint8)
    print("[pose] model loaded OK; poses in blank frame:",
          0 if est.landmarks(blank) is None else 1)
    est.close()
