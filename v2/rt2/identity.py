"""identity.py - ONLINE "me vs not-me" identity classifier (Phase 6).

A self-contained, dependency-light (numpy-only; NO sklearn) ONLINE logistic-
regression model that learns the TARGET player from the user's CONFIRMATIONS
(positives) AND CORRECTIONS (hard negatives = "rule out this player"). Confidence
tightens over rewatches because every confirmed/ruled-out detection feeds one SGD
step. It is pure logic / Qt-free so it can be exercised headlessly.

WHY THIS EXISTS
  The motion/appearance TargetModel picks the best candidate INSIDE the ROI, but
  it has no notion of "this looks like a player I have explicitly said is NOT me".
  IdentityModel folds the user's explicit feedback into a learned decision
  boundary over a small FIXED-LENGTH feature vector (team-fingerprint distances,
  height residual, position, motion-consistency, size, appearance similarity).

HOW IT LEARNS
  * add_positive(vec)  - a CONFIRMED target frame (snap / manual / held-with-det).
  * add_negative(vec)  - the OTHER in-frame detections (WEAK, weight ~0.3) AND the
                         user's explicit "Rule out (not me)" corrections (HARD,
                         weight ~2.0). The hard negatives are the key signal: they
                         teach the boundary to push known-wrong players away.
  Features are STANDARDIZED online via running Welford mean/variance, then one
  weighted-logloss SGD step updates weights w and bias b.

HOW IT IS USED (in apps/track.py)
  Once ready() (>= MIN_POS positives AND >= MIN_NEG negatives) the per-detection
  prob(vec) in [0,1] is folded into the per-frame CONFIDENCE blend and used as a
  candidate-selection tiebreaker (prefer higher prob among the eligible pool). It
  NEVER hard-excludes - only the explicit rule-out + team + height do that.

PERSISTENCE
  save(path)/load(path) round-trip the whole model as JSON (dim, counts, mean,
  var, w, b). load tolerates missing/old/corrupt files and NEVER raises.

ROBUSTNESS
  Any NaN/inf/None/wrong-length vector is silently IGNORED (no update, prob 0.5).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

# readiness thresholds: need a few of each side before the boundary is trusted
MIN_POS = 8
MIN_NEG = 8

# online logistic-regression SGD step size (on STANDARDIZED features)
LR = 0.10
# L2 weight decay per step (tiny - keeps weights from blowing up over a long game)
L2 = 1e-4


def _sigmoid(x: float) -> float:
    # numerically stable logistic
    if x >= 0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)


class IdentityModel:
    """Online logistic-regression "me vs not-me" classifier over a fixed-length
    feature vector. Standardizes features with running Welford stats, then takes
    one weighted-logloss SGD step per added sample."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        # running standardizer (Welford) over the RAW feature vectors
        self._n = 0
        self._mean = np.zeros(self.dim, dtype=np.float64)
        self._M2 = np.zeros(self.dim, dtype=np.float64)   # sum of squared deviations
        # logistic-regression parameters (on STANDARDIZED features)
        self.w = np.zeros(self.dim, dtype=np.float64)
        self.b = 0.0
        # label counts (for ready())
        self.n_pos = 0
        self.n_neg = 0

    # ---- validation ----
    def _clean(self, vec):
        """Return a finite float64 vector of length self.dim, or None if the input
        is None / wrong-length / contains any NaN or inf."""
        if vec is None:
            return None
        try:
            v = np.asarray(vec, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
        if v.shape[0] != self.dim:
            return None
        if not np.all(np.isfinite(v)):
            return None
        return v

    # ---- standardizer ----
    def _update_standardizer(self, v):
        """Fold one raw vector into the running Welford mean/variance."""
        self._n += 1
        delta = v - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (v - self._mean)

    def _std(self):
        """Per-feature population std (>= a small floor so we never divide by 0)."""
        if self._n < 2:
            return np.ones(self.dim, dtype=np.float64)
        var = self._M2 / self._n
        std = np.sqrt(np.maximum(var, 0.0))
        return np.where(std < 1e-6, 1.0, std)

    def _standardize(self, v):
        return (v - self._mean) / self._std()

    # ---- learning ----
    def add(self, vec, label, weight: float = 1.0):
        """Update the standardizer with `vec`, then take ONE weighted-logloss SGD
        step of logistic regression on the standardized vector.

        label : 1 for "me" (positive), 0 for "not me" (negative).
        weight: per-sample loss weight (hard negatives use a bigger weight).
        Silently no-ops on an invalid vec / non-finite weight."""
        v = self._clean(vec)
        if v is None:
            return
        try:
            w_s = float(weight)
            lab = float(label)
        except (TypeError, ValueError):
            return
        if not np.isfinite(w_s) or w_s <= 0.0:
            return
        lab = 1.0 if lab >= 0.5 else 0.0
        # update standardizer FIRST so this sample is reflected in z
        self._update_standardizer(v)
        z = self._standardize(v)
        p = _sigmoid(float(np.dot(self.w, z) + self.b))
        # gradient of weighted logloss wrt the linear pre-activation = w_s*(p - y)
        g = w_s * (p - lab)
        self.w -= LR * (g * z + L2 * self.w)
        self.b -= LR * g
        if lab >= 0.5:
            self.n_pos += 1
        else:
            self.n_neg += 1

    def add_positive(self, vec, weight: float = 1.0):
        """Convenience: add `vec` as a "me" sample (label 1)."""
        self.add(vec, 1, weight)

    def add_negative(self, vec, weight: float = 1.0):
        """Convenience: add `vec` as a "not me" sample (label 0)."""
        self.add(vec, 0, weight)

    # ---- inference ----
    def prob(self, vec) -> float:
        """P(me | vec) in [0,1]. Returns the neutral 0.5 when not ready() or when
        `vec` is invalid, so the caller can always blend it safely."""
        if not self.ready():
            return 0.5
        v = self._clean(vec)
        if v is None:
            return 0.5
        z = self._standardize(v)
        return _sigmoid(float(np.dot(self.w, z) + self.b))

    def ready(self) -> bool:
        """True once enough of BOTH classes have been seen to trust the boundary."""
        return self.n_pos >= MIN_POS and self.n_neg >= MIN_NEG

    def counts(self):
        """(n_pos, n_neg) samples folded in (for status display)."""
        return (self.n_pos, self.n_neg)

    # ---- persistence (JSON; tolerant; never raises) ----
    def save(self, path):
        """Write the whole model to JSON. Returns the path, or None on failure
        (never raises)."""
        try:
            path = pathlib.Path(path)
            data = {
                "version": 1,
                "dim": int(self.dim),
                "n": int(self._n),
                "n_pos": int(self.n_pos),
                "n_neg": int(self.n_neg),
                "mean": self._mean.astype(float).tolist(),
                "M2": self._M2.astype(float).tolist(),
                "w": self.w.astype(float).tolist(),
                "b": float(self.b),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
            return path
        except Exception:
            return None

    def load(self, path):
        """Load a model from JSON into this instance. Returns True on a successful
        load of a DIM-MATCHING model, else False. Tolerates missing / old /
        malformed files and NEVER raises (the model is left untouched on any
        problem)."""
        try:
            path = pathlib.Path(path)
            if not path.exists():
                return False
            data = json.loads(path.read_text(encoding="utf-8"))
            dim = int(data.get("dim", -1))
            if dim != self.dim:
                return False
            mean = np.asarray(data.get("mean", []), dtype=np.float64).ravel()
            M2 = np.asarray(data.get("M2", []), dtype=np.float64).ravel()
            w = np.asarray(data.get("w", []), dtype=np.float64).ravel()
            if (mean.shape[0] != self.dim or M2.shape[0] != self.dim
                    or w.shape[0] != self.dim):
                return False
            b = float(data.get("b", 0.0))
            n = int(data.get("n", 0))
            n_pos = int(data.get("n_pos", 0))
            n_neg = int(data.get("n_neg", 0))
            if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(M2))
                    and np.all(np.isfinite(w)) and np.isfinite(b)):
                return False
            self._mean = mean
            self._M2 = np.maximum(M2, 0.0)
            self.w = w
            self.b = b
            self._n = max(0, n)
            self.n_pos = max(0, n_pos)
            self.n_neg = max(0, n_neg)
            return True
        except Exception:
            return False


def _selftest():
    print("[identity-selftest] online logistic me/not-me classifier")
    rng = np.random.default_rng(0)
    dim = 5
    pos_centre = np.array([3.0, 3.0, 3.0, 0.0, 0.0])
    neg_centre = np.array([-3.0, -3.0, -3.0, 0.0, 0.0])

    m = IdentityModel(dim=dim)
    # not ready -> neutral 0.5
    assert m.prob(pos_centre) == 0.5, "empty model must return neutral 0.5"
    assert not m.ready()

    # feed two well-separated clusters
    for _ in range(40):
        m.add_positive(pos_centre + rng.normal(0, 0.4, dim))
        m.add_negative(neg_centre + rng.normal(0, 0.4, dim))

    assert m.ready(), "model should be ready after >= MIN_POS/MIN_NEG of each"
    p_pos = m.prob(pos_centre)
    p_neg = m.prob(neg_centre)
    assert p_pos > 0.7, f"prob(pos-centre)={p_pos:.3f} should exceed 0.7"
    assert p_neg < 0.3, f"prob(neg-centre)={p_neg:.3f} should be below 0.3"
    print(f"[identity-selftest] separation: prob(pos)={p_pos:.3f} > 0.7, "
          f"prob(neg)={p_neg:.3f} < 0.3  OK")

    # weighted HARD negative still updates counts + boundary, no raise
    m.add_negative(neg_centre, weight=2.0)
    assert m.counts()[1] >= MIN_NEG

    # robustness: NaN / inf / None / wrong-length are ignored (no count change)
    before = m.counts()
    m.add_positive(None)
    m.add_positive([float("nan")] * dim)
    m.add_positive([float("inf")] * dim)
    m.add_positive([1.0, 2.0])                      # wrong length
    assert m.counts() == before, "invalid vecs must not change counts"
    assert m.prob(None) in (0.5,) or 0.0 <= m.prob(None) <= 1.0
    print("[identity-selftest] robust to NaN/inf/None/wrong-length  OK")

    # save / load round-trip preserves predictions (measured AFTER all updates)
    import tempfile, os
    p_pos2 = m.prob(pos_centre)
    p_neg2 = m.prob(neg_centre)
    tmp = pathlib.Path(tempfile.gettempdir()) / "rt2_identity_selftest.json"
    saved = m.save(tmp)
    assert saved is not None and tmp.exists()
    m2 = IdentityModel(dim=dim)
    ok = m2.load(tmp)
    assert ok, "load of a dim-matching model should succeed"
    assert abs(m2.prob(pos_centre) - p_pos2) < 1e-9, "round-trip changed prob(pos)"
    assert abs(m2.prob(neg_centre) - p_neg2) < 1e-9, "round-trip changed prob(neg)"
    assert m2.ready()
    # dim mismatch / missing / corrupt all return False, never raise
    assert IdentityModel(dim=dim + 1).load(tmp) is False
    assert IdentityModel(dim=dim).load(tmp.parent / "rt2_does_not_exist.json") is False
    bad = tmp.parent / "rt2_identity_bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert IdentityModel(dim=dim).load(bad) is False
    try:
        os.unlink(tmp); os.unlink(bad)
    except OSError:
        pass
    print("[identity-selftest] save/load round-trip preserves predictions; "
          "tolerates missing/old/corrupt  OK")

    print("[identity-selftest] PASS")


if __name__ == "__main__":
    _selftest()
