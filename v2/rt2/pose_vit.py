"""GPU pose engine: YOLO person box + SynthPose (ViTPose, HuggingFace).

A drop-in alternative to rt2.pose.PoseOverlay that runs on the GPU and returns the
SAME (33, 3) BlazePose-layout array, so the existing form-focused overlay
(rt2.pose.draw_skeleton), cache and smoothing all work unchanged.

Model: stanfordmimi/synthpose-vitpose-base-hf - a ViTPose fine-tuned for
biomechanics / kinematic analysis. It predicts 52 anatomical keypoints including
real FOOT detail (heel/calcaneus + big toe), which is exactly what matters for
ankle/lower-limb form and is why it beats both YOLO-pose (no feet) and the plain
ViTPose body-17 checkpoint. Loaded via transformers VitPoseForPoseEstimation.

Pipeline (both stages on CUDA):
  1. YOLO (ultralytics, models/yolo11m.pt) finds the person box.
  2. SynthPose returns 52 keypoints; we map the ones we draw into the 33 slots.

Top-down + stateless per frame (safe for scrubbing); t_ms is accepted only for
interface parity with PoseOverlay.
"""
from __future__ import annotations

import numpy as np
import cv2

_MODEL_ID = "stanfordmimi/synthpose-vitpose-base-hf"

# SynthPose 52-keypoint index -> our 33-slot BlazePose layout (see rt2.pose).
# 0-16 are standard COCO; 44/45 big toes, 46/47 calcaneus (heels).
_SYNTH_TO_BLAZE = {
    0: 0,                  # nose
    3: 7, 4: 8,            # ears
    5: 11, 6: 12,          # shoulders
    7: 13, 8: 14,          # elbows
    9: 15, 10: 16,         # wrists
    11: 23, 12: 24,        # hips
    13: 25, 14: 26,        # knees
    15: 27, 16: 28,        # ankles
    46: 29, 47: 30,        # heels   (l_calc -> L heel, r_calc -> R heel)
    45: 31, 44: 32,        # toes    (l_big_toe -> L foot, r_big_toe -> R foot)
}


class ViTPoseOverlay:
    def __init__(self, det_model: str = "models/yolo11m.pt", min_conf: float = 0.3,
                 mode: str = "image"):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        from ultralytics import YOLO

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.min_conf = min_conf
        self.det = YOLO(det_model)
        self.processor = AutoProcessor.from_pretrained(_MODEL_ID)
        self.model = VitPoseForPoseEstimation.from_pretrained(_MODEL_ID).to(self.device).eval()
        n = getattr(self.model.config, "num_labels", None)
        if n != 52:
            print(f"[pose_vit] WARNING: expected 52 SynthPose keypoints, got {n}; "
                  "foot mapping may be wrong.")

    def _person_box(self, frame_bgr):
        """Largest person box as [x, y, w, h] (COCO), or None."""
        res = self.det.predict(frame_bgr, classes=[0], conf=self.min_conf,
                               verbose=False, device=0 if self.device == "cuda" else "cpu")
        if not res:
            return None
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        x1, y1, x2, y2 = xyxy[int(areas.argmax())]
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def landmarks_array(self, frame_bgr, t_ms=None) -> "np.ndarray | None":
        box = self._person_box(frame_bgr)
        if box is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        torch = self._torch
        inputs = self.processor(rgb, boxes=[[box]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_pose_estimation(outputs, boxes=[[box]])
        if not results or not results[0]:
            return None
        person = results[0][0]
        kpts = person["keypoints"]
        scores = person["scores"]
        kpts = kpts.cpu().numpy() if hasattr(kpts, "cpu") else np.asarray(kpts)
        scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)

        arr = np.zeros((33, 3), np.float32)
        for si, bl in _SYNTH_TO_BLAZE.items():
            if si < len(kpts):
                arr[bl] = (kpts[si][0] / w, kpts[si][1] / h, float(scores[si]))
        return arr

    def close(self):
        try:
            del self.model
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    eng = ViTPoseOverlay()
    print("[pose_vit] device:", eng.device)
    if len(sys.argv) > 1:
        cap = cv2.VideoCapture(sys.argv[1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
        ok, fr = cap.read()
        cap.release()
        if ok:
            a = eng.landmarks_array(fr)
            if a is None:
                print("no person")
            else:
                feet = (a[[29, 30, 31, 32], 2] > 0.3).sum()
                print(f"33-array | visible joints={(a[:,2]>0.3).sum()} | foot kpts visible={feet}")
    eng.close()
