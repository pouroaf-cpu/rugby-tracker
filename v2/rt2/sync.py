"""Audio-based clip synchronisation via ffmpeg + cross-correlation.

Two camera angles of the same moment can be married up automatically if both
recorded the same sound (a clap, a whistle, the thud of a rep). We extract a
mono low-rate waveform from each clip with ffmpeg, rectify it to an amplitude
envelope, and cross-correlate the two to find the time shift that lines up their
shared transients.

Returns the offset in SECONDS, defined as:

    offset = (event time in B) - (event time in A)

so the caller aligns them with:  b_time = a_time + offset.

Returns None when there's no usable audio (no audio stream / ffmpeg missing /
weak correlation), in which case the UI falls back to the manual offset slider -
which is also the path used to compare yourself against a pro reference clip,
since two unrelated recordings share no audio.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_envelope(path: Path | str, sr: int, max_seconds: float) -> "np.ndarray | None":
    """Decode mono PCM via ffmpeg and return a mean-removed, unit-norm amplitude
    envelope. None if ffmpeg fails or the clip has no audio."""
    if not have_ffmpeg():
        return None
    cmd = [
        "ffmpeg", "-v", "quiet", "-nostdin",
        "-i", str(path),
        "-t", str(max_seconds),
        "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-",          # raw float32 little-endian to stdout
    ]
    try:
        res = subprocess.run(cmd, capture_output=True)
    except Exception:
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    sig = np.frombuffer(res.stdout, dtype=np.float32)
    if sig.size < sr // 4:           # < 0.25s of audio - not usable
        return None
    env = np.abs(sig)
    env = env - env.mean()
    norm = np.linalg.norm(env)
    if norm == 0:
        return None
    return env / norm


def audio_offset_seconds(
    path_a: Path | str,
    path_b: Path | str,
    sr: int = 8000,
    max_seconds: float = 120.0,
    min_peak: float = 0.05,
) -> "float | None":
    """Best-fit offset (seconds) such that b_time = a_time + offset.
    None if it can't be determined from audio."""
    ea = _extract_envelope(path_a, sr, max_seconds)
    eb = _extract_envelope(path_b, sr, max_seconds)
    if ea is None or eb is None:
        return None

    from scipy.signal import fftconvolve
    # cross-correlation of b against a == fftconvolve(b, reversed a)
    corr = fftconvolve(eb, ea[::-1], mode="full")
    k = int(np.argmax(corr))
    peak = float(corr[k])
    if peak < min_peak:              # no clear shared transient
        return None
    lag = k - (len(ea) - 1)          # >0 => B's event is later than A's
    return lag / sr


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        off = audio_offset_seconds(sys.argv[1], sys.argv[2])
        print("offset (s):", off)
    else:
        print("ffmpeg available:", have_ffmpeg())
        print("usage: python -m rt2.sync <clipA> <clipB>")
