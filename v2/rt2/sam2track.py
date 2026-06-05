"""
sam2track.py - OPTIONAL SAM 2 box-refinement for the live tracker.

WHAT THIS IS
  A tiny wrapper around ultralytics' SAM 2 (already shipped in the installed
  `ultralytics` package - NO separate facebookresearch/sam2 install, NO CUDA
  compile). Given the player's current box it returns a TIGHT box derived from
  SAM 2's segmentation MASK of that player. Used by track.py as an optional
  per-frame "snap to the clean mask" refinement of the CSRT/snapped box - it
  hugs the actual body instead of the loose ROI, which also gives a better
  height-residual and a cleaner marker.

WHY A LAZY IMPORT (this matters)
  Importing ultralytics pulls in torch. On this machine importing torch AFTER
  pyarrow is fine, but torch BEFORE pyarrow SEGFAULTS the process. track.py
  loads pyarrow (the detections parquet) at startup, so as long as we only
  import ultralytics LATER (on first use, when the user enables SAM2) the order
  is safe. Therefore this module imports ultralytics/torch ONLY inside _ensure(),
  never at module import time.

GRACEFULLY INERT
  Like rt2/ocr.py: every public entry point is safe with no model / no GPU.
    * available()             -> bool (cached; never raises)
    * Sam2Box(...).refine(..) -> a refined [x1,y1,x2,y2] or None (never raises)
  So track.py can import and call this unconditionally; without ultralytics it
  simply gets None and does nothing.

PERF (RTX 3060, sam2.1_t.pt): ~7s first load, then ~130 ms / refine. Fine for
careful tracking / re-anchoring, not for full-speed playback.
"""
from __future__ import annotations

# Cached availability probe (None = not yet probed).
_AVAILABLE = None

# Default model: the TINY SAM2.1 checkpoint - fast (~0.13s/frame on a 3060) so the
# LIVE per-frame body silhouette stays responsive. SAM 3 (rt2/sam3track) is the
# slow high-quality pass that runs behind. Swap up with --sam2-model sam2.1_l.pt
# (large) for the tightest live masks if you can take ~0.3-0.5s/frame.
#   sizes: sam2.1_t.pt (tiny) < sam2.1_s.pt (small) < sam2.1_b.pt (base+) < sam2.1_l.pt (large)
DEFAULT_MODEL = "sam2.1_t.pt"

# A refined mask box is REJECTED (returns None) if it is implausible vs the
# prompt box - SAM2 occasionally grabs the whole pitch or nothing.
_MIN_AREA_RATIO = 0.10      # mask box must be >= 10% of the prompt area
_MAX_AREA_RATIO = 4.0       # ... and <= 4x the prompt area
_MIN_OVERLAP = 0.10         # ... and its centre region must overlap the prompt


def available() -> bool:
    """True iff ultralytics (with SAM) can be imported. Cached; NEVER raises.

    NOTE: this triggers the torch import, so only call it AFTER pyarrow has been
    imported (see the module docstring). In track.py that is always the case
    (the detections parquet is loaded at startup, long before SAM2 is enabled)."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    _AVAILABLE = False
    try:
        from ultralytics import SAM  # noqa: F401
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def _clamp(b, w, h):
    return [max(0, min(int(b[0]), w - 1)), max(0, min(int(b[1]), h - 1)),
            max(0, min(int(b[2]), w - 1)), max(0, min(int(b[3]), h - 1))]


# CROP optimization: run SAM on a padded window around the player instead of the
# full 1080p frame. ~7x fewer pixels to encode -> much faster AND sharper (the
# player fills more of the model input = higher effective resolution, no
# hallucination). Results are mapped back to full-frame coords by the crop offset.
CROP_PAD_FRAC = 0.8        # pad each side by this fraction of the box size
CROP_MIN_SIDE = 256        # never crop smaller than this (SAM needs context)


def _crop_for(box, W, H, pad_frac=CROP_PAD_FRAC, min_side=CROP_MIN_SIDE):
    """Padded crop window (cx1,cy1,cx2,cy2) around `box`, clamped to the frame and
    at least min_side on each axis where possible."""
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    padx = max(bw * pad_frac, (min_side - bw) / 2.0)
    pady = max(bh * pad_frac, (min_side - bh) / 2.0)
    cx1 = int(max(0, x1 - padx)); cy1 = int(max(0, y1 - pady))
    cx2 = int(min(W, x2 + padx)); cy2 = int(min(H, y2 + pady))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:        # degenerate -> whole frame
        return 0, 0, int(W), int(H)
    return cx1, cy1, cx2, cy2


class Sam2Box:
    """Lazy SAM 2 segmenter that refines a player box to its tight mask box."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "0",
                 crop: bool = True):
        self.model_name = model
        self.device = device
        self.crop = crop          # crop to a window around the box (faster+sharper)
        self._model = None
        self._failed = False

    def _run(self, frame, box):
        """Run SAM on a (optionally cropped) window around `box`. Returns
        (result0_or_None, ox, oy) where (ox,oy) is the crop origin to add back to
        get full-frame coords."""
        H, W = frame.shape[:2]
        if self.crop:
            cx1, cy1, cx2, cy2 = _crop_for(box, W, H)
        else:
            cx1, cy1, cx2, cy2 = 0, 0, W, H
        sub = frame[cy1:cy2, cx1:cx2]
        bx = [box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1]
        res = self._model(sub, bboxes=[bx], device=self.device, verbose=False)
        return (res[0] if res else None), cx1, cy1

    # ---- lazy load (imports torch HERE, not at module import) ----
    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if self._failed or not available():
            return False
        try:
            from ultralytics import SAM
            self._model = SAM(self.model_name)
            return True
        except Exception:
            self._failed = True
            self._model = None
            return False

    def ready(self) -> bool:
        return self._model is not None

    def mask_polys(self, frame, box, max_pts: int = 120):
        """Return the player's SAM2 mask as a list of polygons (each a list of
        (x, y) points in FRAME coords, largest first), or None if unavailable /
        implausible / anything fails. Used to draw a body-TIGHT silhouette and
        dim the background. NEVER raises."""
        if frame is None or box is None:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            return None
        pw, ph = x2 - x1, y2 - y1
        if pw <= 1 or ph <= 1:
            return None
        if not self._ensure():
            return None
        try:
            import numpy as np
            r, ox, oy = self._run(frame, (x1, y1, x2, y2))
            if r is None or r.masks is None or len(r.masks) == 0:
                return None
            # mask polygon(s) come back in CROP coords -> shift to full-frame
            xy = getattr(r.masks, "xy", None)
            polys = []
            if xy is not None and len(xy) > 0:
                for arr in xy:
                    a = np.asarray(arr, dtype=float)
                    if a.ndim == 2 and a.shape[0] >= 3:
                        a = a + np.array([ox, oy], dtype=float)
                        polys.append(a)
            if not polys:
                return None
            # keep the largest polygon (by its bbox area) and sanity-check it
            def _bbox_area(a):
                w = a[:, 0].max() - a[:, 0].min()
                h = a[:, 1].max() - a[:, 1].min()
                return max(0.0, w) * max(0.0, h)
            polys.sort(key=_bbox_area, reverse=True)
            big = polys[0]
            bx1, by1 = float(big[:, 0].min()), float(big[:, 1].min())
            bx2, by2 = float(big[:, 0].max()), float(big[:, 1].max())
            ratio = ((bx2 - bx1) * (by2 - by1)) / max(1.0, pw * ph)
            if ratio < _MIN_AREA_RATIO or ratio > _MAX_AREA_RATIO:
                return None
            ix1, iy1 = max(bx1, x1), max(by1, y1)
            ix2, iy2 = min(bx2, x2), min(by2, y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / max(1.0, (bx2 - bx1) * (by2 - by1)) < _MIN_OVERLAP:
                return None
            # downsample very dense contours to keep painting cheap
            out = []
            for a in polys[:2]:                       # at most 2 parts
                if len(a) > max_pts:
                    step = int(len(a) / max_pts) + 1
                    a = a[::step]
                out.append([(float(px), float(py)) for px, py in a])
            return out or None
        except Exception:
            return None

    def refine(self, frame, box):
        """Return a tight [x1,y1,x2,y2] from SAM2's mask of the player in `box`,
        or None if SAM2 is unavailable / the segmentation is implausible /
        anything goes wrong. NEVER raises."""
        if frame is None or box is None:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            return None
        pw, ph = x2 - x1, y2 - y1
        if pw <= 1 or ph <= 1:
            return None
        if not self._ensure():
            return None
        try:
            import numpy as np
            H, W = frame.shape[:2]
            r, ox, oy = self._run(frame, (x1, y1, x2, y2))
            if r is None or r.masks is None or len(r.masks) == 0:
                return None
            msk = r.masks.data[0].detach().cpu().numpy() > 0.5
            ys, xs = np.where(msk)
            if xs.size == 0:
                return None
            mb = [int(xs.min()) + ox, int(ys.min()) + oy,
                  int(xs.max()) + ox, int(ys.max()) + oy]
            mw, mh = mb[2] - mb[0], mb[3] - mb[1]
            if mw <= 1 or mh <= 1:
                return None
            # plausibility vs the prompt box
            ratio = (mw * mh) / max(1.0, pw * ph)
            if ratio < _MIN_AREA_RATIO or ratio > _MAX_AREA_RATIO:
                return None
            # overlap: the mask box must intersect the prompt box meaningfully
            ix1, iy1 = max(mb[0], x1), max(mb[1], y1)
            ix2, iy2 = min(mb[2], x2), min(mb[3], y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / max(1.0, mw * mh) < _MIN_OVERLAP:
                return None
            return _clamp(mb, W, H)
        except Exception:
            return None


def _selftest():
    """Headless. Asserts the inert contract always holds; if ultralytics is
    present, asserts SAM2 refines a synthetic blob to a sane box. Prints PASS."""
    import numpy as np
    avail = available()
    print(f"[sam2 selftest] ultralytics/SAM available: {avail}")

    # inert contract holds regardless: bad inputs -> None, no raise
    box = Sam2Box()
    assert box.refine(None, [0, 0, 10, 10]) is None
    assert box.refine(np.zeros((20, 20, 3), np.uint8), None) is None
    assert box.refine(np.zeros((20, 20, 3), np.uint8), [5, 5, 5, 5]) is None
    print("[sam2 selftest] inert contract (bad input -> None, no raise)  OK")

    if not avail:
        # with no ultralytics, refine is always a safe no-op
        assert box.refine(np.zeros((64, 64, 3), np.uint8), [10, 10, 40, 40]) is None
        print("[sam2 selftest] PASS (inert: no ultralytics -> refine None)")
        return

    # ultralytics present: segment a bright rectangle on a dark field
    import torch
    dev = "0" if torch.cuda.is_available() else "cpu"
    img = np.zeros((256, 256, 3), np.uint8)
    img[80:180, 110:150] = (220, 220, 220)        # a "player" blob
    prompt = [105, 75, 155, 185]                  # a loose box around it
    sb = Sam2Box(device=dev)
    out = sb.refine(img, prompt)
    assert out is not None, "expected a refined box for a clear blob"
    x1, y1, x2, y2 = out
    assert 0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256, out
    # the refined box should sit roughly over the blob, tighter in x than the prompt
    assert x1 >= 100 and x2 <= 160 and (x2 - x1) <= (prompt[2] - prompt[0]) + 5, out
    print(f"[sam2 selftest] refined synthetic blob -> {out} (device={dev})  OK")

    # mask_polys: a silhouette polygon whose bbox hugs the blob
    polys = sb.mask_polys(img, prompt)
    assert polys and len(polys) >= 1, "expected a mask polygon for a clear blob"
    pts = polys[0]
    xs = [px for px, _ in pts]; ys = [py for _, py in pts]
    assert min(xs) >= 100 and max(xs) <= 160, (min(xs), max(xs))
    assert len(pts) >= 3
    assert sb.mask_polys(None, prompt) is None
    print(f"[sam2 selftest] mask_polys -> {len(pts)}-pt silhouette in "
          f"x[{min(xs):.0f},{max(xs):.0f}]  OK")
    print("[sam2 selftest] PASS")


if __name__ == "__main__":
    _selftest()
