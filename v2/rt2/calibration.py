"""Calibration model: learned colour signatures + a height-vs-position model.

A team is identified by colour signatures LEARNED from sample boxes the user
drops (not hardcoded gates), plus a height model that accounts for perspective:
on a wide Veo shot a player's pixel height shrinks the higher (further) they are
in the frame, so we model height as a linear function of the box-centre y.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

# Grass-green HSV band, dropped before learning jersey colour.
_GRASS_LO = np.array([35, 40, 30])
_GRASS_HI = np.array([85, 255, 255])
# Skin/warm-tone band, also dropped: it dominates torso/leg pixels and is common
# to BOTH teams, so it drowns out the team-specific jersey/shorts colours. The
# band spares WHITE shorts (low S), BLACK shorts (low V) and NAVY shirts (hue ~110).
_SKIN_LO = np.array([0, 40, 60])
_SKIN_HI = np.array([28, 185, 245])


def _pool_hsv(patches, drop_bg=True):
    """Stack HSV pixels from BGR patches into (N,3), dropping grass + skin so the
    learner sees the team-specific jersey/shorts colour, not the common bg."""
    rows = []
    for p in patches:
        if p is None or p.size == 0:
            continue
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        if drop_bg:
            arr = hsv.reshape(-1, 1, 3)
            grass = cv2.inRange(arr, _GRASS_LO, _GRASS_HI).ravel() > 0
            skin = cv2.inRange(arr, _SKIN_LO, _SKIN_HI).ravel() > 0
            keep = ~(grass | skin)
            if keep.sum() > 5:        # don't drop everything on a tiny/odd patch
                hsv = hsv[keep]
        rows.append(hsv)
    if not rows:
        return np.empty((0, 3), np.float32)
    return np.vstack(rows).astype(np.float32)


def _dominant_mode(px):
    """Mean + spread of the DOMINANT colour cluster in pooled HSV pixels.

    A torso patch mixes jersey + numbers + skin + shadow; a single percentile
    band over all of it is loose and non-discriminative. Clustering and keeping
    the largest cluster isolates the actual dominant jersey colour, giving a
    tight band. Falls back to a robust IQR spread if clustering is unavailable.
    """
    try:
        from sklearn.cluster import KMeans
        k = int(min(3, max(1, len(px) // 200)))
        if k <= 1:
            return px.mean(0), px.std(0)
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(px)
        lab, cnt = np.unique(km.labels_, return_counts=True)
        dom = lab[int(np.argmax(cnt))]
        c = px[km.labels_ == dom]
        return c.mean(0), c.std(0)
    except Exception:
        med = np.median(px, axis=0)
        iqr = (np.percentile(px, 75, axis=0) - np.percentile(px, 25, axis=0)) / 1.35
        return med, iqr


@dataclass
class ColorSignature:
    """Tight HSV band around the dominant colour mode of the sampled pixels."""
    lo: list  # [h, s, v]
    hi: list

    @staticmethod
    def learn(patches, tight: float = 1.6) -> "ColorSignature | None":
        px = _pool_hsv(patches)
        if len(px) < 30:
            return None
        center, spread = _dominant_mode(px)
        pad = np.array([3, 18, 18])           # floor so the band is never razor-thin
        lo = np.clip(center - tight * spread - pad, [0, 0, 0], [179, 255, 255])
        hi = np.clip(center + tight * spread + pad, [0, 0, 0], [179, 255, 255])
        return ColorSignature(lo=[float(x) for x in lo], hi=[float(x) for x in hi])

    def fraction(self, patch_bgr) -> float:
        """Fraction of pixels in `patch_bgr` that fall in this HSV band."""
        if patch_bgr is None or patch_bgr.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(self.lo), np.array(self.hi))
        return float(m.sum()) / 255.0 / (patch_bgr.shape[0] * patch_bgr.shape[1])


@dataclass
class HeightModel:
    """height ~= a*y_center + b, with a residual std for plausibility gating."""
    a: float
    b: float
    resid_std: float
    y_min: float
    y_max: float

    @staticmethod
    def learn(samples) -> "HeightModel | None":
        """samples: iterable of (y_center, box_height)."""
        pts = np.array([(float(y), float(h)) for y, h in samples], dtype=np.float64)
        if len(pts) < 4:
            return None
        y, h = pts[:, 0], pts[:, 1]
        a, b = np.polyfit(y, h, 1)
        resid = h - (a * y + b)
        return HeightModel(a=float(a), b=float(b),
                           resid_std=float(np.std(resid) or 1.0),
                           y_min=float(y.min()), y_max=float(y.max()))

    def predict(self, y_center: float) -> float:
        return self.a * y_center + self.b

    def plausible(self, y_center: float, height: float, tol: float = 3.0) -> bool:
        """True if `height` is within tol*resid_std of the expected height at y."""
        exp = self.predict(y_center)
        band = max(tol * self.resid_std, 0.35 * max(exp, 1.0))
        return abs(height - exp) <= band


@dataclass
class TeamProfile:
    name: str
    track: bool = True
    primary: ColorSignature | None = None
    secondary: ColorSignature | None = None
    shorts: ColorSignature | None = None
    height: HeightModel | None = None
    n_samples: int = 0
    fingerprint: list | None = None   # perceptual feature vector (rt2.features)

    def to_dict(self):
        d = {"name": self.name, "track": self.track, "n_samples": self.n_samples,
             "primary": asdict(self.primary) if self.primary else None,
             "secondary": asdict(self.secondary) if self.secondary else None,
             "shorts": asdict(self.shorts) if self.shorts else None,
             "height": asdict(self.height) if self.height else None,
             "fingerprint": list(self.fingerprint) if self.fingerprint else None}
        return d

    @staticmethod
    def from_dict(d):
        def cs(x): return ColorSignature(**x) if x else None
        hm = HeightModel(**d["height"]) if d.get("height") else None
        return TeamProfile(name=d["name"], track=d.get("track", True),
                           n_samples=d.get("n_samples", 0),
                           primary=cs(d.get("primary")), secondary=cs(d.get("secondary")),
                           shorts=cs(d.get("shorts")), height=hm,
                           fingerprint=d.get("fingerprint"))


@dataclass
class MatchCalibration:
    video: str
    teams: list = field(default_factory=list)

    def tracked_teams(self):
        return [t for t in self.teams if t.track]

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {"video": self.video, "teams": [t.to_dict() for t in self.teams]}
        path.write_text(json.dumps(obj, indent=2))
        return path

    @staticmethod
    def load(path: Path | str) -> "MatchCalibration":
        obj = json.loads(Path(path).read_text())
        return MatchCalibration(video=obj["video"],
                                teams=[TeamProfile.from_dict(t) for t in obj["teams"]])
