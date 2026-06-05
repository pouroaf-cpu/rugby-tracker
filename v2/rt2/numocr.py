"""
numocr.py - jersey-number reader with a LEARNED recognizer (PARSeq) + a closed-
set constraint, falling back to tesseract (rt2.ocr).

WHY
  The research-standard jersey pipeline is legibility -> torso crop -> a learned
  scene-text recognizer -> sequence aggregation. tesseract (rt2.ocr) is the
  generic fallback; PARSeq (a transformer scene-text model) reads small/blurry
  digits much better. This module hides which backend is used and ALWAYS applies
  the closed-set constraint (1..MAX_JERSEY) so impossible reads are dropped.

BACKENDS (chosen lazily, on first use; cached)
  * PARSeq via torch.hub('baudm/parseq') if importable (needs pytorch_lightning,
    timm, nltk - already installed). Runs on GPU if available.
  * else tesseract via rt2.ocr (if the binary is present).
  * else INERT: read_number -> None.

USED OFFLINE (apps/mark_all --ocr, the profiler): heavy reading where torch is
already loaded. The live tracker keeps the lighter tesseract path. LAZY torch
import (only inside _ensure), so importing this module is always safe.

EVERY public function is inert-safe and NEVER raises.
"""
from __future__ import annotations

MAX_JERSEY = 23                 # closed-set: rugby matchday numbers 1..23
_PARSEQ = None                  # lazy model (or False once we know it failed)
_BACKEND = None                 # "parseq" | "tesseract" | None (cached)


def _digits_only(s):
    return "".join(ch for ch in str(s) if ch.isdigit())


def _valid(num_str):
    """A 1-2 digit string in 1..MAX_JERSEY, else None."""
    d = _digits_only(num_str)
    if not (1 <= len(d) <= 2):
        return None
    return d if 1 <= int(d) <= MAX_JERSEY else None


def _ensure_parseq():
    """Lazy-load PARSeq once. Returns the model or False. NEVER raises."""
    global _PARSEQ
    if _PARSEQ is not None:
        return _PARSEQ
    try:
        import torch
        m = torch.hub.load("baudm/parseq", "parseq",
                           pretrained=True, trust_repo=True).eval()
        if torch.cuda.is_available():
            m = m.cuda()
        _PARSEQ = m
    except Exception:
        _PARSEQ = False
    return _PARSEQ


def backend():
    """Which backend is active: 'parseq' | 'tesseract' | None. Cached."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if _ensure_parseq():
        _BACKEND = "parseq"
    else:
        try:
            from rt2 import ocr
            _BACKEND = "tesseract" if ocr.available() else None
        except Exception:
            _BACKEND = None
    return _BACKEND


def available() -> bool:
    return backend() is not None


def _read_parseq(bgr_crop):
    """PARSeq read of a BGR crop -> (digits, conf) constrained to 1..MAX_JERSEY,
    or None. NEVER raises."""
    m = _ensure_parseq()
    if not m:
        return None
    try:
        import torch
        import numpy as np
        import cv2
        from PIL import Image
        from torchvision import transforms as T
        if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
            return None
        pil = Image.fromarray(cv2.cvtColor(np.asarray(bgr_crop), cv2.COLOR_BGR2RGB))
        tf = T.Compose([T.Resize((32, 128)), T.ToTensor(), T.Normalize(0.5, 0.5)])
        x = tf(pil).unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            probs = m(x).softmax(-1)
            labels, confs = m.tokenizer.decode(probs)
        raw = labels[0]
        valid = _valid(raw)
        if valid is None:
            return None
        try:
            conf = float(confs[0].mean()) if len(confs) else 0.5
        except Exception:
            conf = 0.5
        return (valid, max(0.0, min(1.0, conf)))
    except Exception:
        return None


def _read_tesseract(bgr_crop):
    try:
        from rt2 import ocr
        if not ocr.available():
            return None
        res = ocr.read_number(bgr_crop)
        if not res:
            return None
        valid = _valid(res[0])
        return (valid, float(res[1])) if valid else None
    except Exception:
        return None


def read_number(bgr_crop):
    """Read a jersey number (1..MAX_JERSEY) from a BGR crop.

    Tries PARSeq FIRST (strong on 2-digit numbers), then falls back to tesseract
    (better on isolated SINGLE digits, e.g. a prop's #1/#3, which PARSeq tends to
    mangle). Always applies the closed-set constraint. Returns (digits, conf) or
    None. NEVER raises."""
    if _ensure_parseq():
        r = _read_parseq(bgr_crop)
        if r is not None:
            return r
    return _read_tesseract(bgr_crop)        # single-digit + no-PARSeq path


def _selftest():
    print("[numocr-selftest] backend:", backend())
    import numpy as np
    # inert-safe: bad input -> None, never raises
    assert read_number(None) is None
    assert _valid("30") is None and _valid("14") == "14" and _valid("0") is None
    assert _valid("7") == "7" and _valid("123") is None
    print("[numocr-selftest] closed-set constraint (1..23, 1-2 digits)  OK")
    if backend() is None:
        print("[numocr-selftest] PASS (inert: no PARSeq + no tesseract)")
        return
    # render synthetic numbers and check the backend reads the valid ones
    import cv2
    got = {}
    for n in ("14", "7", "19"):
        img = np.zeros((80, 120, 3), np.uint8)
        cv2.putText(img, n, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    (255, 255, 255), 5)
        r = read_number(img)
        got[n] = r[0] if r else None
    print(f"[numocr-selftest] synthetic reads: {got}")
    assert got.get("14") == "14" or got.get("19") == "19", \
        "expected at least one synthetic 2-digit number to read back"
    print("[numocr-selftest] PASS")


if __name__ == "__main__":
    _selftest()
