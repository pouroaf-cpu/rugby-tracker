"""profiles.py - per-player "case files" built by the overnight batch profiler.

THE IDEA (the user's "the more it watches the more it learns" vision)
  Run the profiler unattended over the whole game, over and over. It links the
  team-filtered detections into short tracklets, and folds each tracklet's
  evidence into a per-player CASE FILE:
    * jersey NUMBER  - read by OCR across every clip; votes accumulate, so a
                       number glimpsed once in clip 3 + once in clip 7 can cross
                       the confidence line that neither glimpse could alone.
    * height residual - pixel-height vs the team perspective model. The ONE
                       resolution-robust trait that actually individuates
                       teammates at Veo distance (appearance/colour can't).
    * play zone       - where on the pitch they live (wide+deep vs tight). Used
                       as a soft cross-check on the number's expected role.
    * speed profile   - rough px/frame dynamics.
    * colour fingerprint - the team perceptual vector (confirms team, not person).

  Once a player's NUMBER is known, their ROLE is known (rugby numbering is
  positional: 11/14 wings, 6/7 flankers, ...). That is exactly the user's payoff:
  "you see their jersey number, realise they're a winger and not a flanker, so
  you can rule out another character confusion -- which puts the confidence up on
  the flanker." `rule_out_for(my_number)` returns every OTHER identified
  teammate as a confident "not me" seed for the live tracker's rule-out set.

This module is PURE (no cv2 / no Qt / no torch) so it is headless-testable and
reusable by apps/track.py. The heavy per-frame scanning lives in
apps/profile_game.py; here we only hold + combine + persist the evidence.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Rugby union jersey numbering -> role + forward/back group ----------------
ROLE_BY_NUMBER = {
    1: "loosehead prop", 2: "hooker", 3: "tighthead prop",
    4: "lock", 5: "lock", 6: "blindside flanker",
    7: "openside flanker", 8: "number 8",
    9: "halfback", 10: "first-five-eighth", 11: "left wing",
    12: "inside centre", 13: "outside centre", 14: "right wing",
    15: "fullback",
}
FORWARDS = set(range(1, 9))
BACKS = set(range(9, 16))

# Largest valid jersey number. Reads outside [1, MAX_JERSEY] are impossible
# (rugby matchday numbers run 1..23) and are dropped as OCR misreads - this is
# the user's "there are only ~25 numbers" constraint, which removes a lot of
# garbage two-digit reads (43, 58, 73, ...). The profiler may override it.
MAX_JERSEY = 23

# Roles whose expected play-zone is WIDE (near a touchline) and/or DEEP. Used
# only as a soft sanity cross-check against an OCR'd number -- never to override
# the number itself. Coords are normalized: x 0=left touch..1=right touch,
# y 0=far/top..1=near/bottom of the wide Veo frame.
WIDE_DEEP_NUMBERS = {11, 14, 15}     # back three hug the touchlines / play deep
TIGHT_NUMBERS = set(range(1, 9))     # forwards live central, in tight channels


def role_for_number(n):
    """(role, group) for a jersey number, or (None, None) if unknown/invalid."""
    if n is None:
        return (None, None)
    try:
        n = int(n)
    except (TypeError, ValueError):
        return (None, None)
    role = ROLE_BY_NUMBER.get(n)
    if role is None:
        return (None, None)
    return (role, "forward" if n in FORWARDS else "back")


def _welford_combine(nA, meanA, M2A, nB, meanB, M2B):
    """Exact parallel combine of two Welford accumulators -> (n, mean, M2)."""
    if nB <= 0:
        return nA, meanA, M2A
    if nA <= 0:
        return nB, meanB, M2B
    n = nA + nB
    delta = meanB - meanA
    mean = meanA + delta * (nB / n)
    M2 = M2A + M2B + delta * delta * (nA * nB / n)
    return n, mean, M2


def _std(n, M2):
    return math.sqrt(M2 / n) if n > 1 else 0.0


@dataclass
class PlayerProfile:
    """One identified (or candidate) player's accumulated case file."""
    pid: str
    number: int | None = None
    number_conf: float = 0.0
    number_votes: dict = field(default_factory=dict)   # "digits" -> total weight
    role: str | None = None
    group: str | None = None
    role_zone_ok: bool | None = None                   # number-vs-zone cross-check

    # height residual vs the team perspective height model (the individuating trait)
    h_n: int = 0
    h_mean: float = 0.0
    h_M2: float = 0.0
    # normalized play zone (x and y tracked independently)
    zx_n: int = 0
    zx_mean: float = 0.0
    zx_M2: float = 0.0
    zy_n: int = 0
    zy_mean: float = 0.0
    zy_M2: float = 0.0
    # speed (px/frame)
    sp_n: int = 0
    sp_mean: float = 0.0
    sp_M2: float = 0.0
    # team perceptual colour fingerprint (running mean)
    fp_n: int = 0
    fp_sum: list | None = None
    # bookkeeping
    n_tracklets: int = 0
    n_frames: int = 0
    clips: list = field(default_factory=list)

    # ---- folding evidence in -------------------------------------------------
    def vote_number(self, digits, weight: float = 1.0):
        """Cast a weighted OCR vote for a jersey number string (e.g. "11")."""
        if digits is None:
            return
        d = str(digits).strip()
        if not (d.isdigit() and 1 <= len(d) <= 2):
            return
        # the user's closed-set constraint: impossible numbers are misreads
        if not (1 <= int(d) <= MAX_JERSEY):
            return
        try:
            w = float(weight)
        except (TypeError, ValueError):
            return
        if not math.isfinite(w) or w <= 0:
            return
        self.number_votes[d] = self.number_votes.get(d, 0.0) + w

    def fold_height(self, n, mean, M2):
        self.h_n, self.h_mean, self.h_M2 = _welford_combine(
            self.h_n, self.h_mean, self.h_M2, int(n), float(mean), float(M2))

    def fold_zone(self, nx, mx, M2x, ny, my, M2y):
        self.zx_n, self.zx_mean, self.zx_M2 = _welford_combine(
            self.zx_n, self.zx_mean, self.zx_M2, int(nx), float(mx), float(M2x))
        self.zy_n, self.zy_mean, self.zy_M2 = _welford_combine(
            self.zy_n, self.zy_mean, self.zy_M2, int(ny), float(my), float(M2y))

    def fold_speed(self, n, mean, M2):
        self.sp_n, self.sp_mean, self.sp_M2 = _welford_combine(
            self.sp_n, self.sp_mean, self.sp_M2, int(n), float(mean), float(M2))

    def fold_fingerprint(self, vec, n: int = 1):
        if vec is None:
            return
        v = [float(x) for x in vec]
        if any(not math.isfinite(x) for x in v):
            return
        if self.fp_sum is None:
            self.fp_sum = [0.0] * len(v)
        if len(v) != len(self.fp_sum):
            return
        for i, x in enumerate(v):
            self.fp_sum[i] += x * n
        self.fp_n += int(n)

    def note_clip(self, clip: str):
        if clip and clip not in self.clips:
            self.clips.append(clip)

    # ---- derived readouts ----------------------------------------------------
    @property
    def fingerprint(self):
        if not self.fp_sum or self.fp_n <= 0:
            return None
        return [s / self.fp_n for s in self.fp_sum]

    @property
    def height_resid(self):
        """(mean, std) of the height residual, or (None, None)."""
        if self.h_n <= 0:
            return (None, None)
        return (self.h_mean, _std(self.h_n, self.h_M2))

    @property
    def zone(self):
        """(x, y) mean normalized play position, or (None, None)."""
        if self.zx_n <= 0 or self.zy_n <= 0:
            return (None, None)
        return (self.zx_mean, self.zy_mean)

    @property
    def speed(self):
        if self.sp_n <= 0:
            return (None, None)
        return (self.sp_mean, _std(self.sp_n, self.sp_M2))

    def finalize(self, min_number_weight: float = 2.0):
        """Pick the winning number from the votes, derive role + group, and run
        the soft number-vs-zone cross-check. Idempotent."""
        if self.number_votes:
            best = max(self.number_votes.items(), key=lambda kv: kv[1])
            total = sum(self.number_votes.values())
            if best[1] >= min_number_weight and total > 0:
                self.number = int(best[0])
                self.number_conf = best[1] / total
            else:
                self.number = None
                self.number_conf = 0.0
        self.role, self.group = role_for_number(self.number)
        self.role_zone_ok = self._zone_supports_number()
        return self

    def _zone_supports_number(self):
        """True/False if the play zone agrees with the number's expected zone;
        None when we can't tell (no number / no zone)."""
        if self.number is None:
            return None
        zx, zy = self.zone
        if zx is None:
            return None
        wide = (zx < 0.30 or zx > 0.70)      # near either touchline
        deep = (zy < 0.45)                   # upper / far third of the frame
        if self.number in WIDE_DEEP_NUMBERS:
            return bool(wide or deep)
        if self.number in TIGHT_NUMBERS:
            return bool(0.25 <= zx <= 0.75)  # central channels
        return None                          # midfield backs: no strong expectation

    def evidence_score(self) -> float:
        """0..1 rough confidence that this is a real, distinct player worth
        trusting -- driven by how much was seen and whether a number was read."""
        seen = 1.0 - math.exp(-self.n_frames / 120.0)       # saturates ~ a few s
        spread = min(1.0, len(self.clips) / 3.0)             # seen across clips
        named = 0.0
        if self.number is not None:
            named = 0.5 + 0.5 * self.number_conf
            if self.role_zone_ok:
                named = min(1.0, named + 0.1)
        return round(0.45 * seen + 0.20 * spread + 0.35 * named, 3)

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        hr = self.height_resid
        d["height_resid_mean"] = hr[0]
        d["height_resid_std"] = hr[1]
        d["zone"] = list(self.zone) if self.zone[0] is not None else None
        d["speed_mean"] = self.speed[0]
        d["evidence"] = self.evidence_score()
        return d

    @staticmethod
    def from_dict(d):
        keep = {f.name for f in PlayerProfile.__dataclass_fields__.values()}
        return PlayerProfile(**{k: v for k, v in d.items() if k in keep})


@dataclass
class GameProfiles:
    """Container of all per-player case files for one match (across clips)."""
    game: str = ""
    fps: float = 29.97
    frame_w: int = 0
    frame_h: int = 0
    team: str = ""
    n_clips: int = 0
    players: list = field(default_factory=list)   # list[PlayerProfile]

    def numbered(self):
        return [p for p in self.players if p.number is not None]

    def by_number(self, n):
        for p in self.players:
            if p.number == int(n):
                return p
        return None

    def rule_out_for(self, my_number):
        """Every OTHER identified (numbered) teammate -> a confident "not me"
        seed for the live tracker. This is the payoff: knowing #11 is the winger
        lets us rule them out so the flanker we actually want gains confidence.

        Returns list of dicts: {number, role, group, pid, fingerprint,
        height_resid_mean, height_resid_std, evidence}.
        """
        out = []
        try:
            me = int(my_number) if my_number is not None else None
        except (TypeError, ValueError):
            me = None
        for p in self.numbered():
            if me is not None and p.number == me:
                continue
            hr = p.height_resid
            out.append({
                "number": p.number, "role": p.role, "group": p.group,
                "pid": p.pid, "fingerprint": p.fingerprint,
                "height_resid_mean": hr[0], "height_resid_std": hr[1],
                "evidence": p.evidence_score(),
            })
        out.sort(key=lambda r: (r["number"] is None, r["number"] or 0))
        return out

    def summary_lines(self):
        lines = []
        nums = self.numbered()
        lines.append(f"{len(self.players)} candidate player(s); "
                     f"{len(nums)} with a confident jersey number.")
        for p in sorted(self.players,
                        key=lambda q: (q.number is None, q.number or 0, q.pid)):
            hr = p.height_resid
            hr_s = f"{hr[0]:+.0f}px" if hr[0] is not None else "  ?  "
            zx, zy = p.zone
            zone_s = f"({zx:.2f},{zy:.2f})" if zx is not None else "(  ?  )"
            if p.number is not None:
                tag = f"#{p.number:>2} {p.role or '?'}"
                if p.role_zone_ok is False:
                    tag += " [zone!]"          # number's zone looks off - flag it
            else:
                tag = f"{p.pid} (unnumbered)"
            lines.append(f"  {tag:<34} h{hr_s} zone{zone_s} "
                         f"frames={p.n_frames:>5} clips={len(p.clips)} "
                         f"ev={p.evidence_score():.2f}")
        return lines

    def to_dict(self):
        return {
            "game": self.game, "fps": self.fps,
            "frame_w": self.frame_w, "frame_h": self.frame_h,
            "team": self.team, "n_clips": self.n_clips,
            "players": [p.to_dict() for p in self.players],
        }

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path | str) -> "GameProfiles":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        gp = GameProfiles(
            game=obj.get("game", ""), fps=obj.get("fps", 29.97),
            frame_w=obj.get("frame_w", 0), frame_h=obj.get("frame_h", 0),
            team=obj.get("team", ""), n_clips=obj.get("n_clips", 0),
            players=[PlayerProfile.from_dict(p) for p in obj.get("players", [])])
        return gp


def write_ruleout_seed(path: Path | str, my_number, game_profiles: "GameProfiles"):
    """Persist the rule-out seed file the live tracker consumes to pre-exclude
    the OTHER identified teammates. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "my_number": (int(my_number) if my_number is not None else None),
        "my_role": role_for_number(my_number)[0],
        "rule_out": game_profiles.rule_out_for(my_number),
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ selftest --
def _selftest():
    print("[profiles-selftest] role mapping + accumulation + rule-out")

    assert role_for_number(7) == ("openside flanker", "forward")
    assert role_for_number(11) == ("left wing", "back")
    assert role_for_number(None) == (None, None)
    assert role_for_number("nope") == (None, None)

    # Welford combine must equal a single-pass accumulation.
    import random
    rng = random.Random(0)
    xs = [rng.gauss(5, 2) for _ in range(50)]

    def welford(seq):
        n = 0; mean = 0.0; M2 = 0.0
        for x in seq:
            n += 1
            d = x - mean
            mean += d / n
            M2 += d * (x - mean)
        return n, mean, M2
    a = welford(xs[:20]); b = welford(xs[20:])
    nC, mC, M2C = _welford_combine(*a, *b)
    nT, mT, M2T = welford(xs)
    assert nC == nT and abs(mC - mT) < 1e-9 and abs(M2C - M2T) < 1e-6, "welford combine"
    print("[profiles-selftest] welford combine == single pass  OK")

    # A numbered player: votes accumulate across "clips" and cross the line.
    p = PlayerProfile(pid="p1")
    p.vote_number("11", 1.0)         # one weak glimpse, clip A
    p.vote_number("11", 1.5)         # another glimpse, clip C -> now over 2.0
    p.vote_number("1", 0.4)          # a stray misread
    na, ma, M2a = welford([150.0, 152.0, 148.0, 151.0])
    p.fold_height(na, ma, M2a)
    p.fold_zone(4, 0.08, 0.001, 4, 0.30, 0.002)   # wide + deep -> wing zone
    p.fold_fingerprint([0.1, 0.2, 0.0], 3)
    p.n_frames = 200; p.note_clip("A"); p.note_clip("C"); p.n_tracklets = 2
    p.finalize(min_number_weight=2.0)
    assert p.number == 11, p.number
    assert p.role == "left wing"
    assert p.role_zone_ok is True, "wing in a wide/deep zone should cross-check OK"
    assert abs(p.fingerprint[1] - 0.2) < 1e-9
    assert p.evidence_score() > 0.5
    print(f"[profiles-selftest] #{p.number} {p.role} "
          f"conf={p.number_conf:.2f} ev={p.evidence_score():.2f} "
          f"(votes accumulated across clips)  OK")

    # An under-evidenced number stays unnamed.
    q = PlayerProfile(pid="p2")
    q.vote_number("3", 0.5)
    q.finalize(min_number_weight=2.0)
    assert q.number is None, "a single weak vote must not name a player"

    # A forward seen central; zone cross-check agrees.
    f = PlayerProfile(pid="p3")
    f.vote_number("7", 3.0)
    f.fold_zone(3, 0.50, 0.001, 3, 0.60, 0.001)
    f.finalize()
    assert f.number == 7 and f.role_zone_ok is True

    # GameProfiles + rule_out_for: my number is excluded, others returned.
    gp = GameProfiles(game="testmatch", players=[p, f])
    ro = gp.rule_out_for(7)          # I am the #7 flanker
    assert [r["number"] for r in ro] == [11], "should rule out the #11, not myself"
    assert ro[0]["role"] == "left wing"
    print("[profiles-selftest] rule_out_for(#7) -> rule out #11 wing  OK")

    # Round-trip save/load.
    import tempfile, os
    tmp = Path(tempfile.gettempdir()) / "rt2_profiles_selftest.json"
    gp.save(tmp)
    gp2 = GameProfiles.load(tmp)
    assert len(gp2.players) == 2
    assert gp2.by_number(11).role == "left wing"
    assert abs(gp2.by_number(11).height_resid[0] - p.height_resid[0]) < 1e-6
    seed = Path(tempfile.gettempdir()) / "rt2_ruleout_selftest.json"
    write_ruleout_seed(seed, 7, gp)
    seed_obj = json.loads(seed.read_text())
    assert seed_obj["my_number"] == 7 and seed_obj["my_role"] == "openside flanker"
    assert seed_obj["rule_out"][0]["number"] == 11
    for f_ in (tmp, seed):
        try:
            os.unlink(f_)
        except OSError:
            pass
    print("[profiles-selftest] save/load + rule-out seed round-trip  OK")

    print("[profiles-selftest] PASS")


if __name__ == "__main__":
    _selftest()
