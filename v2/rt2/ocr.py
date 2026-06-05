"""
ocr.py - OPTIONAL jersey-number OCR soft signal (a BONUS confidence cue).

WHAT THIS IS
  A tiny, self-contained reader that tries to OCR a 1-2 digit jersey number off
  a player crop (the upper back / torso, where rugby numbers sit). It is used by
  track.py purely as a SOFT confidence cue when the player's back faces the
  camera - it never drives selection on its own and is DEFAULT OFF.

GRACEFULLY INERT WITHOUT TESSERACT
  This module requires BOTH the `pytesseract` python package AND a usable
  tesseract BINARY. `pytesseract` is pip-installed in this project, but the
  binary is a SEPARATE install. When the binary is missing (or anything else
  goes wrong) every public function here is INERT:
    * available()      -> False   (cached; never raises)
    * read_number(...)  -> None    (never raises)
  So importing and calling this module is always safe even on a machine with no
  tesseract: the caller (track.py) simply gets None and does nothing.

  Install the binary (Windows) with:
      winget install UB-Mannheim.TesseractOCR
  The default install path C:\\Program Files\\Tesseract-OCR\\tesseract.exe is
  probed automatically; otherwise tesseract must be on PATH.

API
  available() -> bool
      True only if pytesseract imports AND the tesseract binary answers a
      version probe. Result is CACHED (probed once). Never raises.

  read_number(bgr_crop) -> (digits: str, conf: float) | None
      OCR a 1-2 digit number from a BGR crop. Returns the best confident token
      as (digits, conf in 0..1), or None when unavailable / no good read /
      invalid crop. Never raises.

  _selftest()
      Asserts the inert path (read_number is None when not available()); if the
      binary IS present, renders a synthetic "7" and asserts it reads back.
      Prints a clear PASS line. Also runs under `python rt2/ocr.py`.
"""
from __future__ import annotations

import numpy as np

# Tunables for small-digit OCR.
_OCR_UPSCALE = 4               # upscale factor (INTER_CUBIC) - jersey crops are tiny
_OCR_MIN_CONF = 40.0           # tesseract per-word confidence floor (0..100)
_TESSERACT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"

# Cached availability probe result (None = not yet probed).
_AVAILABLE = None


def available() -> bool:
    """True iff pytesseract imports AND a tesseract binary answers a version
    probe. Cached after the first call. NEVER raises."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    _AVAILABLE = False
    try:
        import os
        import pytesseract
        # Prefer the standard Windows install location if it exists; otherwise
        # fall back to whatever is on PATH.
        try:
            if os.path.exists(_TESSERACT_WIN_PATH):
                pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN_PATH
        except Exception:
            pass
        # The actual binary probe - raises if the binary is missing / broken.
        pytesseract.get_tesseract_version()
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def _preprocess(bgr_crop):
    """Build OCR-friendly binary images from a small BGR digit crop.

    Upscales ~4x (INTER_CUBIC), greyscales, applies CLAHE, then Otsu-thresholds
    in BOTH polarities (normal + inverted) since a jersey number may be dark on
    light OR light on dark. Returns a list of single-channel uint8 images, or an
    empty list if the crop is unusable. Never raises (caller wraps too)."""
    import cv2
    if bgr_crop is None:
        return []
    arr = np.asarray(bgr_crop)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return []
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        return []
    big = cv2.resize(arr, (w * _OCR_UPSCALE, h * _OCR_UPSCALE),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thr_inv = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return [thr, thr_inv]


def read_number(bgr_crop):
    """OCR a 1-2 digit jersey number from a BGR crop.

    Returns (digits, conf) where digits is a 1-2 char string and conf is in
    [0, 1], or None when: OCR is unavailable, the crop is invalid, or no token
    clears the confidence floor. NEVER raises - any failure returns None.
    """
    if not available():
        return None
    try:
        import cv2  # noqa: F401  (used by _preprocess; import here keeps it lazy)
        import pytesseract

        imgs = _preprocess(bgr_crop)
        if not imgs:
            return None

        best = None       # (digits, conf_0_1)
        for img in imgs:
            data = pytesseract.image_to_data(
                img, config=_OCR_CONFIG,
                output_type=pytesseract.Output.DICT)
            texts = data.get("text", [])
            confs = data.get("conf", [])
            for txt, cf in zip(texts, confs):
                token = "".join(ch for ch in str(txt) if ch.isdigit())
                if not (1 <= len(token) <= 2):
                    continue
                try:
                    cval = float(cf)
                except (TypeError, ValueError):
                    continue
                if cval < _OCR_MIN_CONF:
                    continue
                if best is None or cval > best[1] * 100.0:
                    best = (token, cval / 100.0)
        return best
    except Exception:
        return None


def _selftest():
    """Assert the inert path always holds; if the binary is present, assert a
    synthetic '7' reads back. Prints a PASS line. Safe with NO binary."""
    avail = available()
    print(f"[ocr selftest] tesseract available: {avail}")

    if not avail:
        # INERT PATH: with no usable binary, read_number must be a safe no-op.
        assert read_number(None) is None, "read_number(None) must be None"
        dummy = np.zeros((40, 30, 3), dtype=np.uint8)
        assert read_number(dummy) is None, (
            "read_number must be None when OCR is unavailable")
        print("[ocr selftest] PASS (inert: no tesseract binary -> read_number None)")
        return

    # BINARY PRESENT: render a synthetic white-on-dark "7" and read it back.
    import cv2
    img = np.zeros((80, 60, 3), dtype=np.uint8)
    cv2.putText(img, "7", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                (255, 255, 255), 5, cv2.LINE_AA)
    res = read_number(img)
    assert res is not None, "expected a read for a synthetic '7'"
    digits, conf = res
    assert digits == "7", f"expected '7', got {digits!r}"
    assert 0.0 <= conf <= 1.0, conf
    # invalid crop is still None even when the binary is present
    assert read_number(None) is None
    print(f"[ocr selftest] PASS (binary present: read '7' conf {conf:.2f})")


if __name__ == "__main__":
    _selftest()
