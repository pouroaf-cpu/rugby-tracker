"""candidates.py - "which of these tracks could be ME?" shortlisting.

The crop/SAM3 work needs to know WHERE to look without the user pointing at every
segment. Given the overnight `mark_all` tracks (every player, with team + zone +
height + any read number) and the user's own profile (their number/role, height,
team), this scores each track:

  * RULED-OUT  - confidently NOT you: the opposition; a confidently-read DIFFERENT
                 jersey number; or a position/role contradiction (you're a FORWARD
                 (#1-8) but this track lives out wide & deep like a winger/fullback).
  * ME-LIKELY  - strong match: your team + your number read, OR your team + role-
                 consistent zone + consistent height and nothing contradicting.
  * POSSIBLE   - your team, nothing rules it out, but not confirmed.

So you only choose among a handful of ME-LIKELY/POSSIBLE tracks, and every
teammate you later confirm/rule-out removes more of the field. Pure logic (no cv2/
Qt/torch) so it is headless-testable and reusable by the GUI overlay.
"""
from __future__ import annotations

import sys as _sys
import pathlib as _pathlib
from dataclasses import dataclass, field

# allow `python v2/rt2/candidates.py` standalone AND `import rt2.candidates`
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from rt2.profiles import role_for_number, MAX_JERSEY

# Zone gates (normalized: x 0=left touch..1=right touch, y 0=far/top..1=near/bottom).
_WIDE = 0.26          # within this of either touchline = "wide"
_DEEP = 0.42          # above this (smaller y) = "deep / back field"

# Scoring thresholds
_ME_LIKELY = 0.66
_POSSIBLE = 0.34
_NUMBER_MIN_WEIGHT = 1.5     # vote weight to trust a track's read number
_HEIGHT_TOL = 18.0           # px residual deviation beyond which height contradicts


@dataclass
class MeProfile:
    """What we know about the user (the target)."""
    number: int | None = None
    height_resid_mean: float | None = None
    team: str | None = None          # the user's team name (tracked team)

    @property
    def group(self):                  # "forward" | "back" | None
        return role_for_number(self.number)[1]


@dataclass
class TrackAgg:
    """Aggregated signals for one mark_all track (obj_id)."""
    obj_id: int
    n_frames: int = 0
    team_counts: dict = field(default_factory=dict)   # team_label -> count
    zx_mean: float | None = None
    zy_mean: float | None = None
    height_resid_mean: float | None = None
    number_votes: dict = field(default_factory=dict)  # "digits" -> weight

    def dominant_team(self):
        if not self.team_counts:
            return None
        return max(self.team_counts.items(), key=lambda kv: kv[1])[0]

    def read_number(self):
        """Best whitelisted number for this track, or None."""
        v = {d: w for d, w in self.number_votes.items()
             if str(d).isdigit() and 1 <= int(d) <= MAX_JERSEY}
        if not v:
            return None
        d, w = max(v.items(), key=lambda kv: kv[1])
        return int(d) if w >= _NUMBER_MIN_WEIGHT else None


@dataclass
class Verdict:
    obj_id: int
    status: str            # "me-likely" | "possible" | "ruled-out"
    score: float
    reasons: list = field(default_factory=list)


def _is_wide_deep(zx, zy):
    """True if a track lives out wide AND/OR deep - back-three (winger/fullback)
    territory rather than a forward's tight central channels."""
    if zx is None or zy is None:
        return False
    wide = (zx < _WIDE) or (zx > 1.0 - _WIDE)
    deep = zy < _DEEP
    return wide and deep


def classify_track(agg: "TrackAgg", me: "MeProfile") -> "Verdict":
    """Score one track against the user's profile. See module docstring."""
    reasons = []

    # ---- HARD rule-outs ----
    dom = agg.dominant_team()
    if dom == "opp":
        return Verdict(agg.obj_id, "ruled-out", 0.0, ["opposition team"])
    # a CONFIDENT team label that isn't yours = the opposition, even when it's the
    # team's NAME (e.g. "hautapu") rather than the literal "opp". "unsure"/blank
    # stay eligible so the target is never wrongly ruled out.
    if (me.team and dom and dom not in ("unsure", "")
            and dom.lower() != me.team.lower()):
        return Verdict(agg.obj_id, "ruled-out", 0.0,
                       [f"opposition team ({dom})"])

    tnum = agg.read_number()
    if tnum is not None and me.number is not None and tnum != me.number:
        return Verdict(agg.obj_id, "ruled-out", 0.0,
                       [f"reads #{tnum}, you are #{me.number}"])

    # role/zone contradiction: a FORWARD that consistently plays wide+deep isn't you
    if me.group == "forward" and _is_wide_deep(agg.zx_mean, agg.zy_mean):
        return Verdict(agg.obj_id, "ruled-out", 0.0,
                       ["plays wide & deep (back three); you're a forward"])

    # ---- soft scoring ----
    score = 0.40                       # baseline for an un-contradicted same-team track
    if tnum is not None and tnum == me.number:
        score += 0.45
        reasons.append(f"reads YOUR #{me.number}")
    if me.team is not None and dom == me.team:
        score += 0.10
        reasons.append("your team")
    elif dom == "unsure" or dom is None:
        reasons.append("team unsure")

    # role/zone consistency (soft): forwards central; back-three wide/deep
    if me.group == "forward":
        if agg.zx_mean is not None and 0.30 <= agg.zx_mean <= 0.70:
            score += 0.08
            reasons.append("central (forward-like)")
    elif me.group == "back":
        if _is_wide_deep(agg.zx_mean, agg.zy_mean):
            score += 0.06
            reasons.append("wide/deep (back-like)")

    # height-residual consistency (soft)
    if me.height_resid_mean is not None and agg.height_resid_mean is not None:
        dev = abs(agg.height_resid_mean - me.height_resid_mean)
        if dev <= _HEIGHT_TOL * 0.5:
            score += 0.10
            reasons.append("matching height")
        elif dev > _HEIGHT_TOL:
            score -= 0.20
            reasons.append("height off")

    score = max(0.0, min(1.0, score))
    if score >= _ME_LIKELY:
        status = "me-likely"
    elif score >= _POSSIBLE:
        status = "possible"
    else:
        status = "ruled-out"
    return Verdict(agg.obj_id, status, round(score, 3), reasons)


def shortlist(aggs, me: "MeProfile", min_frames: int = 8):
    """Classify all tracks; return (verdicts_sorted, summary).

    verdicts_sorted: by status (me-likely, possible, ruled-out) then score desc.
    summary: {'me_likely': n, 'possible': n, 'ruled_out': n, 'total': n}.
    Tracks shorter than min_frames are skipped (noise)."""
    order = {"me-likely": 0, "possible": 1, "ruled-out": 2}
    verdicts = [classify_track(a, me) for a in aggs if a.n_frames >= min_frames]
    verdicts.sort(key=lambda v: (order[v.status], -v.score))
    summary = {"me_likely": sum(v.status == "me-likely" for v in verdicts),
               "possible": sum(v.status == "possible" for v in verdicts),
               "ruled_out": sum(v.status == "ruled-out" for v in verdicts),
               "total": len(verdicts)}
    return verdicts, summary


def _selftest():
    print("[candidates-selftest] role/zone/number/team shortlisting")
    me = MeProfile(number=3, height_resid_mean=-10.0, team="blue&gold")  # a prop = forward
    assert me.group == "forward"

    # opposition -> ruled out
    opp = TrackAgg(1, n_frames=200, team_counts={"opp": 180, "unsure": 20})
    assert classify_track(opp, me).status == "ruled-out"

    # a winger (wide+deep) on my team -> ruled out (I'm a forward)
    wing = TrackAgg(2, n_frames=200, team_counts={"blue&gold": 200},
                    zx_mean=0.08, zy_mean=0.30, height_resid_mean=-9)
    v = classify_track(wing, me)
    assert v.status == "ruled-out" and "wide & deep" in v.reasons[0], v

    # someone reading a different number -> ruled out
    other = TrackAgg(3, n_frames=200, team_counts={"blue&gold": 200},
                     zx_mean=0.5, zy_mean=0.6, number_votes={"11": 3.0})
    assert classify_track(other, me).status == "ruled-out"

    # a central same-team track reading MY number, matching height -> me-likely
    mine = TrackAgg(4, n_frames=200, team_counts={"blue&gold": 200},
                    zx_mean=0.52, zy_mean=0.6, height_resid_mean=-11,
                    number_votes={"3": 3.0})
    vm = classify_track(mine, me)
    assert vm.status == "me-likely" and vm.score >= _ME_LIKELY, vm
    assert any("YOUR #3" in r for r in vm.reasons)

    # a central same-team track with NO number -> possible (not confirmed)
    poss = TrackAgg(5, n_frames=200, team_counts={"blue&gold": 200},
                    zx_mean=0.5, zy_mean=0.62, height_resid_mean=-12)
    assert classify_track(poss, me).status in ("possible", "me-likely")

    verdicts, summ = shortlist([opp, wing, other, mine, poss], me)
    assert summ["total"] == 5 and summ["ruled_out"] == 3 and summ["me_likely"] >= 1
    assert verdicts[0].obj_id == 4         # me-likely first
    print(f"[candidates-selftest] shortlist {summ} (me-likely first = obj "
          f"{verdicts[0].obj_id})  OK")

    # with NO number known for me, zone/team still shortlist (don't over-rule-out)
    me2 = MeProfile(number=None, team="blue&gold")
    v2 = classify_track(poss, me2)
    assert v2.status in ("possible", "me-likely")
    # but a known back-three winger is NOT ruled out when MY role is unknown
    assert classify_track(wing, me2).status != "ruled-out"
    print("[candidates-selftest] role-unknown stays permissive  OK")

    print("[candidates-selftest] PASS")


if __name__ == "__main__":
    _selftest()
