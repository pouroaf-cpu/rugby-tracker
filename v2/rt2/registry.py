"""registry.py - the PERSISTENT, CROSS-MATCH PLAYER REGISTRY (long-term memory).

THE IDEA (the user's "the more it watches the more it learns" vision, made durable)
  profiles.py builds ONE match's worth of per-player case files. This module is
  the layer above: a single SQLite DB that REMEMBERS players match-over-match.
  Every clip the system watches folds its evidence into a STABLE per-player
  identity, so the knowledge of each player gets richer over a whole season.

THE CRUCIAL DOMAIN TRUTH (we design WITH it, never against it)
  At Veo distance (players 40-150px, both teams same kit) APPEARANCE cannot tell
  teammates apart. The only reliable individuating signals are:
    (1) JERSEY NUMBER when readable, and
    (2) HEIGHT RESIDUAL vs the perspective model (a stable physical trait).
  Therefore:
    * identity is keyed on a stable UUID -- NEVER on the jersey number (a number
      can change match to match), so jersey number is stored as HISTORY.
    * when LINKING an incoming observation to an existing player we trust NUMBER
      first, HEIGHT RESIDUAL second, and treat the colour fingerprint as weak
      corroboration only (it can break a height tie, never link on its own).

STORAGE
  One SQLite file at output/registry.sqlite (path via ProjectPaths). Small vectors
  (a fingerprint is ~10 floats) are stored as JSON TEXT -- parquet is overkill.

PURITY / TESTABILITY
  stdlib only (sqlite3, json, uuid, datetime, math) -- no torch, no cv2, no Qt,
  no numpy. Ingests profiles.PlayerProfile / GameProfiles objects. Fully headless
  selftest under `python v2/rt2/registry.py`.
"""
from __future__ import annotations

import datetime
import json
import math
import sqlite3
import uuid as _uuid
from pathlib import Path

# rt2 sibling imports (this module lives in rt2/). When this file is run directly
# (python v2/rt2/registry.py) it loads as __main__ and `rt2` isn't yet importable,
# so put v2/ on the path first.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # adds v2/
from rt2.profiles import PlayerProfile, GameProfiles, role_for_number  # noqa: F401

SCHEMA_VERSION = 1

# --- linking policy constants -------------------------------------------------
# Max |height-residual| difference (px) at which we'll accept a HEIGHT link, and
# the nearest candidate must beat the runner-up by at least HEIGHT_MARGIN px to
# be a "clear minimum" (otherwise it's an ambiguous tie -> we may use the
# fingerprint to break it, or fall back to creating a new player).
HEIGHT_LINK_TOL = 12.0      # px: outside this, height is NOT considered a match
HEIGHT_MARGIN = 3.0         # px: required gap to runner-up for a confident height link
# When two height candidates are within HEIGHT_MARGIN of each other we call it a
# tie and let the colour fingerprint break it -- but ONLY if the fingerprints are
# clearly separable (best cosine-distance gap >= FP_TIE_MARGIN).
FP_TIE_MARGIN = 0.05        # min cosine-distance gap to trust a fp tie-break

# --- drift-bounded signature-update constants ---------------------------------
# A persistent identity must not be yanked by one noisy match. We cap the
# effective sample window (so old evidence keeps weight), and clamp how far any
# single update may move the stored height-residual centroid.
MAX_N = 400                 # cap on the running sample count used for weighting
MAX_DRIFT = 6.0             # px: max movement of height_resid_mean per update
FP_MAX_N = 400              # same idea for the fingerprint running count


def registry_db(pp) -> Path:
    """The canonical DB path for a ProjectPaths instance."""
    return Path(pp.output) / "registry.sqlite"


def _now(now=None) -> str:
    """ISO timestamp; pass an explicit `now` for deterministic tests."""
    if now is not None:
        return str(now)
    return datetime.datetime.now().isoformat(timespec="seconds")


def _jloads(text, default):
    if not text:
        return default
    try:
        v = json.loads(text)
        return v if v is not None else default
    except (ValueError, TypeError):
        return default


def _jdumps(obj) -> str:
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps(None)


def _finite(x):
    """float(x) if finite, else None."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_vec(vec):
    """A list of finite floats, or None if vec is None/empty/non-finite/bad."""
    if vec is None:
        return None
    try:
        v = [float(x) for x in vec]
    except (TypeError, ValueError):
        return None
    if not v or any(not math.isfinite(x) for x in v):
        return None
    return v


def _cos_dist(a, b):
    """Cosine distance in [0,2] between two equal-length vectors, or None."""
    a = _clean_vec(a)
    b = _clean_vec(b)
    if a is None or b is None or len(a) != len(b):
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot / (na * nb)


class PlayerRegistry:
    """Persistent cross-match player registry over a single SQLite DB.

    Context-manager friendly:
        with PlayerRegistry() as reg:
            reg.ingest_game_profiles(gp, ...)
    """

    def __init__(self, db_path=None):
        if db_path is None:
            try:
                from rt2.paths import ProjectPaths
                db_path = registry_db(ProjectPaths())
            except Exception:
                db_path = Path("registry.sqlite")
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error:
            pass
        self._migrate()

    # ---- context manager -----------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    # ---- schema --------------------------------------------------------------
    def _migrate(self):
        """Create tables if absent; safe to call on every open."""
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS players (
                uuid TEXT PRIMARY KEY,
                display_name TEXT,
                is_teammate INTEGER DEFAULT 1,
                is_me INTEGER DEFAULT 0,
                status TEXT,
                current_number INTEGER,
                jersey_history TEXT,
                role TEXT,
                n_frames INTEGER DEFAULT 0,
                n_clips INTEGER DEFAULT 0,
                n_matches INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS player_signatures (
                uuid TEXT PRIMARY KEY REFERENCES players(uuid),
                height_resid_mean REAL,
                height_resid_std REAL,
                height_n INTEGER DEFAULT 0,
                fingerprint TEXT,
                fingerprint_n INTEGER DEFAULT 0,
                zone_x REAL,
                zone_y REAL,
                speed_mean REAL,
                number_votes TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS match_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT,
                match_id TEXT,
                clip TEXT,
                number_read INTEGER,
                number_conf REAL,
                height_resid_mean REAL,
                n_frames INTEGER,
                evidence REAL,
                link_reason TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                video TEXT,
                status TEXT,
                pipeline_version TEXT,
                n_players INTEGER,
                processed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_obs_uuid_match
                ON match_observations(uuid, match_id);
            CREATE INDEX IF NOT EXISTS idx_players_number
                ON players(current_number);
            """
        )
        cur = c.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            c.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)",
                      (str(SCHEMA_VERSION),))
        self.conn.commit()

    # ---- low-level helpers ---------------------------------------------------
    def _new_uuid(self) -> str:
        return _uuid.uuid4().hex

    def _next_temp_name(self) -> str:
        cur = self.conn.execute("SELECT COUNT(*) FROM players")
        n = (cur.fetchone()[0] or 0) + 1
        return f"Player_{n:03d}"

    def _ensure_signature_row(self, uuid, now):
        self.conn.execute(
            "INSERT OR IGNORE INTO player_signatures(uuid, updated_at) "
            "VALUES(?,?)", (uuid, now))

    # ---- jersey-history bookkeeping ------------------------------------------
    @staticmethod
    def _bump_jersey_history(history, number, now):
        """Return (history, current_number). history is a list of
        {number, matches, last_seen}. A None number leaves things unchanged."""
        if number is None:
            cur = history[-1]["number"] if history else None
            return history, cur
        try:
            number = int(number)
        except (TypeError, ValueError):
            cur = history[-1]["number"] if history else None
            return history, cur
        for entry in history:
            if entry.get("number") == number:
                entry["matches"] = int(entry.get("matches", 0)) + 1
                entry["last_seen"] = now
                return history, number
        history.append({"number": number, "matches": 1, "last_seen": now})
        return history, number

    # ---- create / mark players -----------------------------------------------
    def set_me(self, display_name, number=None, now=None) -> str:
        """Create or update the single designated "me" player (is_me=1,
        is_teammate=1). Idempotent: only ever one me."""
        now = _now(now)
        cur = self.conn.execute("SELECT uuid FROM players WHERE is_me=1")
        row = cur.fetchone()
        if row is not None:
            uuid = row["uuid"]
            self.rename(uuid, display_name, now=now)
            if number is not None:
                self.set_number(uuid, number, now=now)
            return uuid
        uuid = self.add_player(display_name=display_name, is_teammate=True,
                               number=number, now=now)
        self.conn.execute(
            "UPDATE players SET is_me=1, updated_at=? WHERE uuid=?", (now, uuid))
        self.conn.commit()
        return uuid

    def add_player(self, display_name=None, is_teammate=True, number=None,
                   now=None) -> str:
        """Explicitly create a player; returns the new uuid."""
        now = _now(now)
        uuid = self._new_uuid()
        if not display_name:
            display_name = self._next_temp_name()
        history = []
        history, current = self._bump_jersey_history(history, number, now)
        role = role_for_number(current)[0]
        self.conn.execute(
            "INSERT INTO players(uuid, display_name, is_teammate, is_me, status, "
            "current_number, jersey_history, role, n_frames, n_clips, n_matches, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid, display_name, 1 if is_teammate else 0, 0, "active",
             current, _jdumps(history), role, 0, 0, 0, now, now))
        self._ensure_signature_row(uuid, now)
        self.conn.commit()
        return uuid

    # ---- linking (read-only: no mutation) ------------------------------------
    def link(self, profile, now=None, block_uuids=None):
        """Find the best existing player for `profile` WITHOUT mutating anything.

        Policy, in order:
          1. confident number -> player with current_number==number OR whose
             jersey_history contains it.                      reason="number"
          2. teammate whose height_resid_mean is within HEIGHT_LINK_TOL and is the
             clear minimum (beats runner-up by HEIGHT_MARGIN). reason="height"
             (fingerprint may break an otherwise-tied height match -> "height+fp")
          3. no match -> (None, "new")

        block_uuids: uuids that have ALREADY been claimed by another profile in
        THIS SAME match ingest -- they're a different person right now, so we
        never height-link to them (two confident numbers in one match = two
        people; but a player's number CAN differ across matches, which is why we
        block per-match-claimed uuids rather than blocking by number globally).
        """
        block = set(block_uuids or ())
        number = getattr(profile, "number", None)
        my_n = None

        # 1) NUMBER (the strongest individuating signal)
        if number is not None:
            try:
                my_n = int(number)
            except (TypeError, ValueError):
                my_n = None
            if my_n is not None:
                hit = self.find_by_number(my_n)
                if hit is not None and hit["uuid"] not in block:
                    return hit["uuid"], "number"

        # 2/3) HEIGHT RESIDUAL among teammates with a stored height signature.
        hr = getattr(profile, "height_resid", (None, None))
        obs_h = _finite(hr[0]) if hr else None
        if obs_h is None:
            return None, "new"

        rows = self.conn.execute(
            "SELECT p.uuid, p.current_number AS cn, s.height_resid_mean AS h, "
            "s.fingerprint AS fp FROM players p "
            "JOIN player_signatures s ON p.uuid=s.uuid "
            "WHERE p.is_teammate=1 AND s.height_resid_mean IS NOT NULL "
            "AND s.height_n > 0"
        ).fetchall()
        cands = []
        for r in rows:
            h = _finite(r["h"])
            if h is None:
                continue
            # already claimed by a different profile in THIS match -> not me now
            if r["uuid"] in block:
                continue
            cands.append((abs(h - obs_h), r["uuid"], r["fp"]))
        if not cands:
            return None, "new"
        cands.sort(key=lambda t: t[0])
        best_d, best_uuid, best_fp = cands[0]
        if best_d > HEIGHT_LINK_TOL:
            return None, "new"
        if len(cands) == 1:
            return best_uuid, "height"
        runner_d = cands[1][0]
        if runner_d - best_d >= HEIGHT_MARGIN:
            return best_uuid, "height"

        # ambiguous height tie -> try the colour fingerprint ONLY as a tie-break
        obs_fp = _clean_vec(getattr(profile, "fingerprint", None))
        if obs_fp is not None:
            scored = []
            for d, u, fp in cands:
                if d - best_d >= HEIGHT_MARGIN:
                    break  # only the genuinely-tied cluster
                cd = _cos_dist(obs_fp, _jloads(fp, None))
                if cd is not None:
                    scored.append((cd, u))
            if len(scored) >= 2:
                scored.sort(key=lambda t: t[0])
                if scored[1][0] - scored[0][0] >= FP_TIE_MARGIN:
                    return scored[0][1], "height+fp"
        # genuinely ambiguous and no fp separation -> safest is a NEW player
        return None, "new"

    # ---- drift-bounded signature update --------------------------------------
    @staticmethod
    def _drift_mean(old_mean, old_n, obs_mean, obs_n):
        """Weighted, window-capped, drift-clamped centroid update -> new_mean."""
        obs_mean = _finite(obs_mean)
        if obs_mean is None or obs_n <= 0:
            return old_mean
        if old_mean is None or old_n <= 0:
            return obs_mean
        denom = min(MAX_N, old_n + obs_n)
        if denom <= 0:
            return old_mean
        new_mean = old_mean + (obs_mean - old_mean) * (obs_n / denom)
        # clamp single-update movement so one bad match can't corrupt an identity
        delta = new_mean - old_mean
        if delta > MAX_DRIFT:
            new_mean = old_mean + MAX_DRIFT
        elif delta < -MAX_DRIFT:
            new_mean = old_mean - MAX_DRIFT
        return new_mean

    def _update_signature(self, uuid, profile, now, replace_n=0):
        """Fold a profile's evidence into the stored signature (drift-bounded).

        replace_n: height_n already counted for this (uuid, match) being replaced;
        subtract it from the stored window first so re-ingest doesn't double-count.
        """
        cur = self.conn.execute(
            "SELECT * FROM player_signatures WHERE uuid=?", (uuid,)).fetchone()
        if cur is None:
            self._ensure_signature_row(uuid, now)
            cur = self.conn.execute(
                "SELECT * FROM player_signatures WHERE uuid=?", (uuid,)).fetchone()

        old_h_mean = _finite(cur["height_resid_mean"])
        old_h_n = int(cur["height_n"] or 0)
        if replace_n:
            old_h_n = max(0, old_h_n - int(replace_n))

        hr = getattr(profile, "height_resid", (None, None))
        obs_h_mean = _finite(hr[0]) if hr else None
        obs_h_std = _finite(hr[1]) if hr else None
        obs_h_n = int(getattr(profile, "h_n", 0) or 0)

        new_h_mean = self._drift_mean(old_h_mean, old_h_n, obs_h_mean, obs_h_n)
        new_h_n = min(MAX_N, old_h_n + max(0, obs_h_n))
        # height std: keep the latest observed std when we have one (cheap + good
        # enough -- the mean is the load-bearing trait, the std is advisory).
        new_h_std = obs_h_std if obs_h_std is not None else _finite(cur["height_resid_std"])

        # fingerprint: window-capped running mean (weak corroboration only)
        old_fp = _clean_vec(_jloads(cur["fingerprint"], None))
        old_fp_n = int(cur["fingerprint_n"] or 0)
        obs_fp = _clean_vec(getattr(profile, "fingerprint", None))
        obs_fp_n = int(getattr(profile, "fp_n", 0) or 0)
        new_fp, new_fp_n = self._blend_fp(old_fp, old_fp_n, obs_fp, obs_fp_n)

        # zone + speed: cheap latest-wins-blended scalars (advisory only)
        zone = getattr(profile, "zone", (None, None))
        new_zx = _finite(zone[0]) if zone else None
        new_zy = _finite(zone[1]) if zone else None
        if new_zx is None:
            new_zx = _finite(cur["zone_x"])
        if new_zy is None:
            new_zy = _finite(cur["zone_y"])
        spd = getattr(profile, "speed", (None, None))
        new_sp = _finite(spd[0]) if spd else None
        if new_sp is None:
            new_sp = _finite(cur["speed_mean"])

        # number votes: accumulate the JSON tally across matches
        votes = _jloads(cur["number_votes"], {})
        for d, w in (getattr(profile, "number_votes", {}) or {}).items():
            wv = _finite(w)
            if wv is None:
                continue
            votes[str(d)] = float(votes.get(str(d), 0.0)) + wv

        self.conn.execute(
            "UPDATE player_signatures SET height_resid_mean=?, height_resid_std=?, "
            "height_n=?, fingerprint=?, fingerprint_n=?, zone_x=?, zone_y=?, "
            "speed_mean=?, number_votes=?, updated_at=? WHERE uuid=?",
            (new_h_mean, new_h_std, new_h_n,
             _jdumps(new_fp) if new_fp is not None else None, new_fp_n,
             new_zx, new_zy, new_sp, _jdumps(votes), now, uuid))
        return obs_h_n

    @staticmethod
    def _blend_fp(old_fp, old_n, obs_fp, obs_n):
        """Window-capped running-mean blend of two fingerprint vectors."""
        if obs_fp is None or obs_n <= 0:
            return old_fp, min(MAX_N, max(0, old_n))
        if old_fp is None or old_n <= 0:
            return obs_fp, min(FP_MAX_N, obs_n)
        if len(old_fp) != len(obs_fp):
            return old_fp, min(MAX_N, old_n)  # incompatible -> keep the established one
        denom = min(FP_MAX_N, old_n + obs_n)
        w = obs_n / denom if denom > 0 else 0.0
        new_fp = [o + (x - o) * w for o, x in zip(old_fp, obs_fp)]
        return new_fp, min(FP_MAX_N, old_n + obs_n)

    # ---- ingest --------------------------------------------------------------
    def ingest_profile(self, profile, match_id, clip="", now=None,
                       block_uuids=None) -> str:
        """Link `profile` to an existing player (or create one), then do a
        drift-bounded signature update + bookkeeping + an audit observation row.
        Re-ingesting the same (player, match_id) REPLACES rather than double-counts.
        Returns the player's uuid.

        block_uuids: uuids already claimed by another profile in this same match
        (forwarded to link() so two distinct players in one match never collide).
        """
        now = _now(now)
        uuid, reason = self.link(profile, now=now, block_uuids=block_uuids)
        if uuid is None or reason == "new":
            number = getattr(profile, "number", None)
            uuid = self.add_player(display_name=None, is_teammate=True,
                                   number=number, now=now)
            reason = "new"

        # Was this (uuid, match_id) already ingested? If so we REPLACE: subtract
        # its prior frame/height contribution so re-runs stay idempotent-ish.
        prev = self.conn.execute(
            "SELECT id, n_frames FROM match_observations WHERE uuid=? AND match_id=?",
            (uuid, match_id)).fetchone()
        replace_n = 0
        prev_frames = 0
        if prev is not None:
            prev_frames = int(prev["n_frames"] or 0)
            # the height_n we previously folded == the obs h_n at that time; we
            # approximate it with the profile's current h_n window for subtraction
            replace_n = int(getattr(profile, "h_n", 0) or 0)
            self.conn.execute("DELETE FROM match_observations WHERE id=?",
                              (prev["id"],))

        obs_h_n = self._update_signature(uuid, profile, now, replace_n=replace_n)

        # bookkeeping on the players row
        prow = self.conn.execute(
            "SELECT * FROM players WHERE uuid=?", (uuid,)).fetchone()
        history = _jloads(prow["jersey_history"], [])
        history, current = self._bump_jersey_history(
            history, getattr(profile, "number", None), now)
        role = role_for_number(current)[0] or prow["role"]

        new_frames = int(getattr(profile, "n_frames", 0) or 0)
        n_frames = int(prow["n_frames"] or 0) - prev_frames + new_frames
        clips = list(getattr(profile, "clips", []) or [])
        n_clips = int(prow["n_clips"] or 0)
        if prev is None:
            n_clips += max(1, len(clips))
        n_matches = int(prow["n_matches"] or 0) + (0 if prev is not None else 1)

        self.conn.execute(
            "UPDATE players SET current_number=?, jersey_history=?, role=?, "
            "n_frames=?, n_clips=?, n_matches=?, updated_at=? WHERE uuid=?",
            (current, _jdumps(history), role, max(0, n_frames), n_clips,
             n_matches, now, uuid))

        # audit observation row
        ev = None
        try:
            ev = float(profile.evidence_score())
        except Exception:
            ev = None
        hr = getattr(profile, "height_resid", (None, None))
        self.conn.execute(
            "INSERT INTO match_observations(uuid, match_id, clip, number_read, "
            "number_conf, height_resid_mean, n_frames, evidence, link_reason, "
            "created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (uuid, match_id, clip or "",
             getattr(profile, "number", None),
             _finite(getattr(profile, "number_conf", None)),
             _finite(hr[0]) if hr else None, new_frames, ev, reason, now))
        self.conn.commit()
        return uuid

    def ingest_game_profiles(self, gp, match_id=None, video="",
                             pipeline_version="", now=None) -> dict:
        """MAIN entry point. Upsert a matches row, ingest every PlayerProfile.

        Numbered/me players are ingested FIRST so their numbers anchor the linking
        of any unnumbered players in the same match. Re-ingesting the same match_id
        is idempotent-ish (observations are replaced, frames not double-counted).
        """
        now = _now(now)
        if match_id is None:
            match_id = self._derive_match_id(video or getattr(gp, "game", ""))

        players = list(getattr(gp, "players", []) or [])
        # ingest numbered (strongest signal) first, then the rest
        ordered = sorted(
            players,
            key=lambda p: (getattr(p, "number", None) is None,
                           getattr(p, "number", None) or 0))

        # which players were brand-new BEFORE this match (for n_new accounting)?
        uuids = []
        n_new = 0
        n_linked = 0
        # detect whether this match was processed before (re-ingest)
        seen_before = self.conn.execute(
            "SELECT 1 FROM matches WHERE match_id=?", (match_id,)).fetchone()

        claimed = set()   # uuids taken by an earlier profile in THIS match
        for p in ordered:
            _, reason = self.link(p, now=now, block_uuids=claimed)
            uuid = self.ingest_profile(p, match_id, clip="", now=now,
                                       block_uuids=claimed)
            uuids.append(uuid)
            claimed.add(uuid)
            if reason == "new":
                n_new += 1
            else:
                n_linked += 1

        self.conn.execute(
            "INSERT INTO matches(match_id, video, status, pipeline_version, "
            "n_players, processed_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(match_id) DO UPDATE SET video=excluded.video, "
            "status=excluded.status, pipeline_version=excluded.pipeline_version, "
            "n_players=excluded.n_players, processed_at=excluded.processed_at",
            (match_id, video, "processed", pipeline_version, len(players), now))
        self.conn.commit()
        return {
            "match_id": match_id,
            "n_ingested": len(uuids),
            "n_new": n_new,
            "n_linked": n_linked,
            "reingest": bool(seen_before),
            "uuids": uuids,
        }

    def _derive_match_id(self, video) -> str:
        stem = Path(str(video)).stem or "match"
        base = stem
        i = 0
        while True:
            cand = base if i == 0 else f"{base}_{i}"
            row = self.conn.execute(
                "SELECT 1 FROM matches WHERE match_id=?", (cand,)).fetchone()
            if row is None:
                return cand
            i += 1

    # ---- reads ---------------------------------------------------------------
    def get_player(self, uuid):
        row = self.conn.execute(
            "SELECT * FROM players WHERE uuid=?", (uuid,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["jersey_history"] = _jloads(d.get("jersey_history"), [])
        return d

    def list_players(self, teammates_only=False):
        q = "SELECT * FROM players"
        if teammates_only:
            q += " WHERE is_teammate=1"
        q += " ORDER BY is_me DESC, current_number IS NULL, current_number, display_name"
        out = []
        for row in self.conn.execute(q).fetchall():
            d = dict(row)
            d["jersey_history"] = _jloads(d.get("jersey_history"), [])
            out.append(d)
        return out

    def get_signature(self, uuid):
        row = self.conn.execute(
            "SELECT * FROM player_signatures WHERE uuid=?", (uuid,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["fingerprint"] = _jloads(d.get("fingerprint"), None)
        d["number_votes"] = _jloads(d.get("number_votes"), {})
        return d

    def find_by_number(self, n):
        """Player whose CURRENT number is n, else one whose jersey_history has it,
        else None. Returns the players-row dict."""
        try:
            n = int(n)
        except (TypeError, ValueError):
            return None
        row = self.conn.execute(
            "SELECT * FROM players WHERE current_number=? "
            "ORDER BY is_me DESC, updated_at DESC LIMIT 1", (n,)).fetchone()
        if row is not None:
            d = dict(row)
            d["jersey_history"] = _jloads(d.get("jersey_history"), [])
            return d
        # fall back to jersey history
        for row in self.conn.execute("SELECT * FROM players").fetchall():
            hist = _jloads(row["jersey_history"], [])
            if any(e.get("number") == n for e in hist):
                d = dict(row)
                d["jersey_history"] = hist
                return d
        return None

    # ---- the payoff: rule-out seeds ------------------------------------------
    def _resolve_uuid(self, uuid_or_number):
        """Accept a uuid OR a jersey number; return (uuid, number) or (None,None)."""
        p = self.get_player(uuid_or_number) if isinstance(uuid_or_number, str) else None
        if p is not None and len(str(uuid_or_number)) >= 8 and not str(uuid_or_number).isdigit():
            return p["uuid"], p.get("current_number")
        # try as a number
        try:
            n = int(uuid_or_number)
        except (TypeError, ValueError):
            if p is not None:
                return p["uuid"], p.get("current_number")
            return None, None
        hit = self.find_by_number(n)
        if hit is not None:
            return hit["uuid"], hit.get("current_number")
        return None, n

    def rule_out_for(self, uuid_or_number):
        """Every OTHER teammate as a confident "not me" rule-out seed (mirrors
        GameProfiles.rule_out_for). Each dict carries the individuating traits the
        live tracker uses to exclude that player."""
        me_uuid, me_number = self._resolve_uuid(uuid_or_number)
        out = []
        rows = self.conn.execute(
            "SELECT p.uuid, p.display_name, p.current_number, p.role, "
            "s.height_resid_mean AS h, s.height_resid_std AS hs, "
            "s.fingerprint AS fp FROM players p "
            "LEFT JOIN player_signatures s ON p.uuid=s.uuid "
            "WHERE p.is_teammate=1").fetchall()
        for r in rows:
            if me_uuid is not None and r["uuid"] == me_uuid:
                continue
            ev = self._evidence_for(r["uuid"])
            out.append({
                "uuid": r["uuid"],
                "display_name": r["display_name"],
                "number": r["current_number"],
                "role": r["role"],
                "height_resid_mean": _finite(r["h"]),
                "height_resid_std": _finite(r["hs"]),
                "fingerprint": _jloads(r["fp"], None),
                "evidence": ev,
            })
        out.sort(key=lambda d: (d["number"] is None, d["number"] or 0,
                                d["display_name"] or ""))
        return out

    def _evidence_for(self, uuid):
        """A 0..1 rough confidence reconstructed from stored bookkeeping, using the
        same shape as PlayerProfile.evidence_score()."""
        p = self.conn.execute(
            "SELECT n_frames, n_clips, current_number FROM players WHERE uuid=?",
            (uuid,)).fetchone()
        if p is None:
            return 0.0
        sig = self.conn.execute(
            "SELECT number_votes FROM player_signatures WHERE uuid=?",
            (uuid,)).fetchone()
        n_frames = int(p["n_frames"] or 0)
        n_clips = int(p["n_clips"] or 0)
        seen = 1.0 - math.exp(-n_frames / 120.0)
        spread = min(1.0, n_clips / 3.0)
        named = 0.0
        if p["current_number"] is not None:
            conf = 0.0
            votes = _jloads(sig["number_votes"], {}) if sig else {}
            if votes:
                total = sum(float(v) for v in votes.values())
                best = max(float(v) for v in votes.values())
                conf = (best / total) if total > 0 else 0.0
            named = 0.5 + 0.5 * conf
        return round(0.45 * seen + 0.20 * spread + 0.35 * named, 3)

    def export_ruleout_seed(self, uuid_or_number, path, now=None):
        """Write the JSON shape profiles.write_ruleout_seed writes:
        {my_number, my_role, rule_out:[...]}."""
        me_uuid, me_number = self._resolve_uuid(uuid_or_number)
        my_role = role_for_number(me_number)[0]
        obj = {
            "my_number": me_number,
            "my_role": my_role,
            "rule_out": self.rule_out_for(uuid_or_number),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return path

    # ---- edits ---------------------------------------------------------------
    def rename(self, uuid, display_name, now=None):
        now = _now(now)
        if not display_name:
            return
        self.conn.execute(
            "UPDATE players SET display_name=?, updated_at=? WHERE uuid=?",
            (display_name, now, uuid))
        self.conn.commit()

    def set_number(self, uuid, number, now=None):
        now = _now(now)
        prow = self.conn.execute(
            "SELECT jersey_history FROM players WHERE uuid=?", (uuid,)).fetchone()
        if prow is None:
            return
        history = _jloads(prow["jersey_history"], [])
        history, current = self._bump_jersey_history(history, number, now)
        role = role_for_number(current)[0]
        self.conn.execute(
            "UPDATE players SET current_number=?, jersey_history=?, role=?, "
            "updated_at=? WHERE uuid=?",
            (current, _jdumps(history), role, now, uuid))
        self.conn.commit()

    def merge(self, uuid_keep, uuid_drop, now=None):
        """Fold uuid_drop into uuid_keep: reassign observations, combine the two
        signatures (drift-bounded), then delete the dropped player. For fixing
        bootstrap mistakes."""
        now = _now(now)
        if uuid_keep == uuid_drop:
            return
        keep = self.conn.execute(
            "SELECT * FROM players WHERE uuid=?", (uuid_keep,)).fetchone()
        drop = self.conn.execute(
            "SELECT * FROM players WHERE uuid=?", (uuid_drop,)).fetchone()
        if keep is None or drop is None:
            return

        ks = self.get_signature(uuid_keep) or {}
        ds = self.get_signature(uuid_drop) or {}

        # combine height (weighted by stored windows, capped)
        k_h = _finite(ks.get("height_resid_mean"))
        k_n = int(ks.get("height_n") or 0)
        d_h = _finite(ds.get("height_resid_mean"))
        d_n = int(ds.get("height_n") or 0)
        if k_h is None:
            new_h, new_hn = d_h, d_n
        elif d_h is None:
            new_h, new_hn = k_h, k_n
        else:
            tot = k_n + d_n
            new_h = (k_h * k_n + d_h * d_n) / tot if tot > 0 else k_h
            new_hn = min(MAX_N, tot)
        new_hs = _finite(ks.get("height_resid_std"))
        if new_hs is None:
            new_hs = _finite(ds.get("height_resid_std"))

        # combine fingerprint
        new_fp, new_fpn = self._blend_fp(
            _clean_vec(ks.get("fingerprint")), int(ks.get("fingerprint_n") or 0),
            _clean_vec(ds.get("fingerprint")), int(ds.get("fingerprint_n") or 0))

        # combine number votes
        votes = dict(ks.get("number_votes") or {})
        for d, w in (ds.get("number_votes") or {}).items():
            votes[str(d)] = float(votes.get(str(d), 0.0)) + float(w)

        self._ensure_signature_row(uuid_keep, now)
        self.conn.execute(
            "UPDATE player_signatures SET height_resid_mean=?, height_resid_std=?, "
            "height_n=?, fingerprint=?, fingerprint_n=?, number_votes=?, "
            "updated_at=? WHERE uuid=?",
            (new_h, new_hs, new_hn,
             _jdumps(new_fp) if new_fp is not None else None, new_fpn,
             _jdumps(votes), now, uuid_keep))

        # reassign observations + fold bookkeeping
        self.conn.execute(
            "UPDATE match_observations SET uuid=? WHERE uuid=?",
            (uuid_keep, uuid_drop))
        kh = _jloads(keep["jersey_history"], [])
        for e in _jloads(drop["jersey_history"], []):
            kh, _ = self._bump_jersey_history(
                kh, e.get("number"), e.get("last_seen") or now)
        n_frames = int(keep["n_frames"] or 0) + int(drop["n_frames"] or 0)
        n_clips = int(keep["n_clips"] or 0) + int(drop["n_clips"] or 0)
        n_matches = int(keep["n_matches"] or 0) + int(drop["n_matches"] or 0)
        current = keep["current_number"]
        if current is None:
            current = drop["current_number"]
        role = role_for_number(current)[0] or keep["role"]
        self.conn.execute(
            "UPDATE players SET current_number=?, jersey_history=?, role=?, "
            "n_frames=?, n_clips=?, n_matches=?, updated_at=? WHERE uuid=?",
            (current, _jdumps(kh), role, n_frames, n_clips, n_matches, now,
             uuid_keep))

        self.conn.execute("DELETE FROM player_signatures WHERE uuid=?", (uuid_drop,))
        self.conn.execute("DELETE FROM players WHERE uuid=?", (uuid_drop,))
        self.conn.commit()


# ------------------------------------------------------------------ selftest --
def _mk_profile(pid, number=None, votes=None, h_mean=None, h_spread=4,
                zone=None, fp=None, n_frames=120, clips=("A",)):
    """Build a finalized PlayerProfile from simple values (selftest helper)."""
    p = PlayerProfile(pid=pid)
    if votes:
        for d, w in votes.items():
            p.vote_number(d, w)
    elif number is not None:
        p.vote_number(str(number), 3.0)
    if h_mean is not None:
        # fold a few samples around h_mean to populate the Welford accumulator
        import statistics
        xs = [h_mean - h_spread, h_mean, h_mean + h_spread]
        n = len(xs); mean = sum(xs) / n
        M2 = sum((x - mean) ** 2 for x in xs)
        p.fold_height(n, mean, M2)
    if zone is not None:
        p.fold_zone(3, zone[0], 0.001, 3, zone[1], 0.001)
    if fp is not None:
        p.fold_fingerprint(fp, 3)
    p.n_frames = n_frames
    for c in clips:
        p.note_clip(c)
    p.finalize(min_number_weight=2.0)
    return p


def _selftest():
    import tempfile, os
    print("[registry-selftest] persistent cross-match player registry")

    db = Path(tempfile.gettempdir()) / "rt2_registry_selftest.sqlite"
    try:
        os.unlink(db)
    except OSError:
        pass

    T = "2026-06-04T10:00:00"
    reg = PlayerRegistry(db)

    # --- create "me" (idempotent) ---
    me = reg.set_me("Pou", number=7, now=T)
    me2 = reg.set_me("Pou", number=7, now=T)
    assert me == me2, "set_me must be idempotent (one me)"
    assert sum(1 for p in reg.list_players() if p["is_me"]) == 1
    mp = reg.get_player(me)
    assert mp["is_me"] == 1 and mp["is_teammate"] == 1 and mp["current_number"] == 7
    print("[registry-selftest] set_me idempotent; one me  OK")

    # --- ingest a match: #7 (me, central) + #11 (wing, wide) + 1 unnumbered ---
    p7 = _mk_profile("t7", number=7, h_mean=2.0, zone=(0.50, 0.60),
                     fp=[0.5, 0.1, 0.0], n_frames=300, clips=("A", "B"))
    p11 = _mk_profile("t11", number=11, h_mean=-8.0, zone=(0.10, 0.30),
                      fp=[0.1, 0.5, 0.0], n_frames=240, clips=("A",))
    pun = _mk_profile("tun", number=None, h_mean=40.0, zone=(0.50, 0.55),
                      fp=[0.0, 0.0, 0.5], n_frames=80, clips=("A",))
    gp = GameProfiles(game="match_alpha", team="Blue&Gold",
                      players=[p7, p11, pun])

    s1 = reg.ingest_game_profiles(gp, match_id="match_alpha", video="match_alpha.mp4",
                                  pipeline_version="v2", now=T)
    assert s1["n_ingested"] == 3, s1
    # #7 must have linked to the existing "me" (number match), not created new
    u7 = reg.find_by_number(7)["uuid"]
    assert u7 == me, "ingesting #7 should link to the existing me by number"
    u11 = reg.find_by_number(11)["uuid"]
    assert u11 is not None and u11 != me
    print(f"[registry-selftest] match ingest: {s1['n_ingested']} players, "
          f"#7->me, new #11  OK")

    # snapshot frame counts to prove re-ingest doesn't double-count
    f7_before = reg.get_player(u7)["n_frames"]
    f11_before = reg.get_player(u11)["n_frames"]
    n_players_before = len(reg.list_players())

    # --- re-ingest the SAME match: idempotent (no double-count, no new players) ---
    s1b = reg.ingest_game_profiles(gp, match_id="match_alpha", video="match_alpha.mp4",
                                   pipeline_version="v2", now=T)
    assert len(reg.list_players()) == n_players_before, "re-ingest created players"
    assert reg.get_player(u7)["n_frames"] == f7_before, "re-ingest doubled frames"
    assert reg.get_player(u11)["n_frames"] == f11_before, "re-ingest doubled frames"
    assert s1b["reingest"] is True
    print("[registry-selftest] re-ingest SAME match is idempotent (counts stable)  OK")

    # --- SECOND match: same numbers must link to the SAME uuids (persistence) ---
    T2 = "2026-06-11T10:00:00"
    q7 = _mk_profile("u7", number=7, h_mean=3.0, zone=(0.52, 0.58),
                     fp=[0.5, 0.1, 0.0], n_frames=280, clips=("A",))
    q11 = _mk_profile("u11", number=11, h_mean=-7.0, zone=(0.12, 0.28),
                      fp=[0.1, 0.5, 0.0], n_frames=200, clips=("A",))
    gp2 = GameProfiles(game="match_beta", team="Blue&Gold", players=[q7, q11])
    s2 = reg.ingest_game_profiles(gp2, match_id="match_beta", video="match_beta.mp4",
                                  now=T2)
    assert reg.find_by_number(7)["uuid"] == u7, "cross-match #7 identity broke"
    assert reg.find_by_number(11)["uuid"] == u11, "cross-match #11 identity broke"
    assert s2["n_linked"] == 2 and s2["n_new"] == 0, s2
    # drift-bounded update moved height a little but did not blow up
    sig11 = reg.get_signature(u11)
    assert -12.0 < sig11["height_resid_mean"] < -4.0, sig11["height_resid_mean"]
    assert reg.get_player(u7)["n_matches"] == 2, "match count should grow"
    print("[registry-selftest] 2nd match links same numbers to same uuids; "
          "drift-bounded update OK")

    # --- jersey_history grows when a player's number changes ---
    T3 = "2026-06-18T10:00:00"
    r11 = _mk_profile("v11", number=23, h_mean=-7.0, zone=(0.12, 0.30),
                      fp=[0.1, 0.5, 0.0], n_frames=150)
    # height links it to u11 (number 23 is new, so it can't link by number)
    gp3 = GameProfiles(game="match_gamma", players=[r11])
    reg.ingest_game_profiles(gp3, match_id="match_gamma", now=T3)
    hist = reg.get_player(u11)["jersey_history"]
    nums = sorted(e["number"] for e in hist)
    assert 11 in nums and 23 in nums, f"jersey history should hold 11 and 23: {nums}"
    assert reg.get_player(u11)["current_number"] == 23
    print(f"[registry-selftest] jersey_history grew on number change: {nums}  OK")

    # --- rule_out_for(#7) returns the OTHER teammates, not me ---
    ro = reg.rule_out_for(7)
    ro_uuids = {d["uuid"] for d in ro}
    assert me not in ro_uuids, "rule_out_for(#7) must NOT include me (#7)"
    assert u11 in ro_uuids, "rule_out_for(#7) should include #11/23"
    print(f"[registry-selftest] rule_out_for(#7) -> {len(ro)} others, excludes self  OK")

    # export the seed file in the canonical shape
    seed = Path(tempfile.gettempdir()) / "rt2_registry_ruleout.json"
    reg.export_ruleout_seed(7, seed, now=T3)
    seed_obj = json.loads(seed.read_text())
    assert seed_obj["my_number"] == 7
    assert seed_obj["my_role"] == "openside flanker"
    assert "rule_out" in seed_obj and isinstance(seed_obj["rule_out"], list)
    print("[registry-selftest] export_ruleout_seed canonical shape  OK")

    # --- merge() folds two players (the unnumbered one into #11) ---
    # find the unnumbered player explicitly
    unnum = [p for p in reg.list_players()
             if p["current_number"] is None and not p["is_me"]]
    assert unnum, "expected an unnumbered player to exist"
    drop_uuid = unnum[0]["uuid"]
    keep_frames = reg.get_player(u11)["n_frames"]
    drop_frames = reg.get_player(drop_uuid)["n_frames"]
    n_before_merge = len(reg.list_players())
    reg.merge(u11, drop_uuid, now=T3)
    assert reg.get_player(drop_uuid) is None, "dropped player should be gone"
    assert len(reg.list_players()) == n_before_merge - 1
    assert reg.get_player(u11)["n_frames"] == keep_frames + drop_frames, \
        "merge should fold frame counts"
    # observations got reassigned
    obs = reg.conn.execute(
        "SELECT COUNT(*) FROM match_observations WHERE uuid=?", (drop_uuid,)
    ).fetchone()[0]
    assert obs == 0, "dropped player's observations should be reassigned"
    print("[registry-selftest] merge folds player (signatures+obs+counts)  OK")

    # --- round-trip: reopen the DB file, confirm players persist ---
    reg.close()
    reg2 = PlayerRegistry(db)
    assert reg2.find_by_number(7)["uuid"] == me, "persistence: #7 lost on reopen"
    assert reg2.find_by_number(23)["uuid"] == u11, "persistence: #23 lost on reopen"
    assert reg2.get_player(me)["display_name"] == "Pou"
    print("[registry-selftest] reopen DB -> players persist  OK")
    reg2.close()

    for f in (db, seed):
        try:
            os.unlink(f)
        except OSError:
            pass

    print("[registry-selftest] PASS")


if __name__ == "__main__":
    _selftest()
