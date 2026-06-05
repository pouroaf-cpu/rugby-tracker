"""
sam3track.py - SAM 3 VIDEO-PROPAGATION segment tracker (the "ruck lock").

WHAT THIS IS
  When normal CSRT/snap tracking jumps onto the wrong player in a busy passage
  (a ruck/maul), you mark that short window and SAM 3 RE-TRACKS it: seeded by one
  ROI box on the first frame, SAM 3's video predictor PROPAGATES that identity
  forward with memory, so it holds the SAME player through the mess instead of
  teleporting. It returns a clean per-frame box track for the window, which the
  app merges back into the target track.

NOT LIVE - it is a BACKGROUND / OFFLINE repair tool
  Measured on the RTX 3060: ~0.5 fps (~2 s/frame) including the first-frame
  encode. So this is for SHORT flagged windows (a few seconds), run as a
  background job, NOT for real-time playback. A 5 s ruck (~150 frames) ~= 5 min.

GATED MODEL
  Needs the Meta-gated SAM 3 checkpoint at models/sam3.pt (request access on
  Hugging Face). available(model_path) is False without it -> every call is a
  safe no-op returning {} (never raises).

LAZY/SAFE IMPORTS
  ultralytics/torch are imported ONLY inside track_segment (after cv2), so this
  module is import-safe everywhere (torch-before-pyarrow segfault avoidance, as
  in rt2/sam2track).

KNOWN ULTRALYTICS QUIRK (handled)
  The SAM3 interactive/video path passes the bool `compile` straight through as
  the torch.compile MODE, so the default compile=False raises
  "Unrecognized mode=False". We pass compile=None to disable compilation.
"""
from __future__ import annotations

import pathlib
import tempfile


def available(model_path) -> bool:
    """True iff ultralytics imports AND the SAM3 checkpoint file exists. Cached
    per path is unnecessary; this is cheap. NEVER raises."""
    try:
        if not model_path or not pathlib.Path(model_path).exists():
            return False
        import ultralytics  # noqa: F401
        from ultralytics.models.sam.predict import SAM3VideoPredictor  # noqa: F401
        return True
    except Exception:
        return False


# CROP optimization: SAM3 is ~2s/frame mostly because it encodes the full 1080p
# frame. Crop a FIXED window around the seed box (the player stays roughly local
# over a short re-track window, esp. a ruck) -> far fewer pixels -> several times
# faster, and sharper. The window is fixed for the whole segment so the video
# predictor's memory stays consistent; results are offset back to full coords.
SAM3_CROP_PAD_FRAC = 1.5            # pad each side by this fraction of the box
SAM3_CROP_MIN_FRAME_FRAC = 0.14     # ...and at least this fraction of the frame


def _seg_crop(box, W, H):
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    padx = max(bw * SAM3_CROP_PAD_FRAC, W * SAM3_CROP_MIN_FRAME_FRAC)
    pady = max(bh * SAM3_CROP_PAD_FRAC, H * SAM3_CROP_MIN_FRAME_FRAC)
    cx1 = int(max(0, x1 - padx)); cy1 = int(max(0, y1 - pady))
    cx2 = int(min(W, x2 + padx)); cy2 = int(min(H, y2 + pady))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return 0, 0, int(W), int(H)
    return cx1, cy1, cx2, cy2


def _mask_to_box(mask_bool):
    """Tight (x1,y1,x2,y2) ints from a 2-D boolean mask, or None if empty."""
    try:
        import numpy as np
        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    except Exception:
        return None


class Sam3Segment:
    """Propagate ONE ROI box through a short video window with SAM 3."""

    def __init__(self, model_path, device: str = "0", imgsz: int = 512,
                 crop: bool = True):
        # imgsz=512 (not 1024) is the speed lever: with crop=True the player fills
        # the cropped window, so 512 keeps detail while ~3x faster (measured ~1 fps
        # vs ~0.35 fps at 1024 on the 3060). crop alone doesn't help (SAM resizes
        # any input to imgsz); crop is what makes the lower imgsz lossless.
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = imgsz
        self.crop = crop

    def track_segment(self, video_path, start_frame, n_frames, box,
                      on_progress=None):
        """Re-track the window [start_frame, start_frame+n_frames) (1-based frame
        numbers, matching the tracks CSV / VideoReader) starting from `box` on
        the FIRST window frame.

        Returns dict {frame_number -> [x1,y1,x2,y2]} in ORIGINAL frame coords/
        numbers. Empty dict on any failure / unavailable model. NEVER raises.
        `on_progress(i, n)` is called per propagated frame if given.
        """
        if not available(self.model_path):
            return {}
        try:
            import cv2
            import numpy as np
            start_frame = int(start_frame)
            n_frames = int(n_frames)
            if n_frames <= 0:
                return {}
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            if x2 - x1 < 2 or y2 - y1 < 2:
                return {}
        except Exception:
            return {}

        seg_path = None
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {}
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            # seek to start (1-based -> 0-based)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
            frames = []
            for _ in range(n_frames):
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                return {}
            h, w = frames[0].shape[:2]
            # CROP a fixed window around the seed (speed + sharpness); offset back.
            if self.crop:
                cx1, cy1, cx2, cy2 = _seg_crop((x1, y1, x2, y2), w, h)
            else:
                cx1, cy1, cx2, cy2 = 0, 0, w, h
            frames = [f[cy1:cy2, cx1:cx2] for f in frames]
            ch, cw = frames[0].shape[:2]
            seed = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
            # SAM3 video predictor requires a real VIDEO source -> temp mp4
            fd, seg_name = tempfile.mkstemp(suffix=".mp4", prefix="rt2_sam3_")
            import os
            os.close(fd)
            seg_path = pathlib.Path(seg_name)
            vw = cv2.VideoWriter(str(seg_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
            for f in frames:
                vw.write(f)
            vw.release()

            # ultralytics/torch imported HERE (after cv2/data) - segfault-safe.
            from ultralytics.models.sam.predict import SAM3VideoPredictor
            pred = SAM3VideoPredictor(overrides=dict(
                model=self.model_path, mode="predict", imgsz=self.imgsz,
                verbose=False, save=False,
                compile=None))   # compile=None avoids the mode=False bug
            out = {}
            results = pred(source=str(seg_path), bboxes=[seed], stream=True)
            for i, r in enumerate(results):
                bb = None
                if getattr(r, "masks", None) is not None and len(r.masks) > 0:
                    m = r.masks.data[0].detach().cpu().numpy() > 0.5
                    bb = _mask_to_box(m)
                if bb is not None:
                    # crop coords -> full-frame coords
                    out[start_frame + i] = [bb[0] + cx1, bb[1] + cy1,
                                            bb[2] + cx1, bb[3] + cy1]
                if on_progress is not None:
                    try:
                        on_progress(i + 1, len(frames))
                    except Exception:
                        pass
            return out
        except Exception:
            return {}
        finally:
            if seg_path is not None:
                try:
                    seg_path.unlink()
                except OSError:
                    pass


def _selftest():
    print("[sam3track-selftest] inert + mask->box logic")
    import numpy as np
    # available() is False for a missing checkpoint (inert), never raises
    assert available(None) is False
    assert available("definitely_not_here_sam3.pt") is False
    # a Sam3Segment with a bogus path is a safe no-op
    s = Sam3Segment("definitely_not_here_sam3.pt")
    assert s.track_segment("nope.mp4", 1, 10, [0, 0, 20, 40]) == {}
    print("[sam3track-selftest] inert contract (no model -> {}; no raise)  OK")
    # mask -> box
    m = np.zeros((100, 100), bool)
    m[20:60, 30:50] = True
    assert _mask_to_box(m) == [30, 20, 49, 59], _mask_to_box(m)
    assert _mask_to_box(np.zeros((10, 10), bool)) is None
    print("[sam3track-selftest] _mask_to_box tight bbox  OK")
    # bad box -> {}
    assert s.track_segment("x.mp4", 1, 10, [5, 5, 6, 6]) == {}
    print("[sam3track-selftest] PASS")


if __name__ == "__main__":
    _selftest()
