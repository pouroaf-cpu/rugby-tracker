"""profile_game.py - the OVERNIGHT BATCH PROFILER (unattended, NOT Qt).

"The more it watches the more it learns." Point it at one clip, several clips, or
the whole input folder and leave it running. It does NOT need you there:

  for each clip:
    1. load the team-filtered detections (prerender's .detections.parquet)
    2. LINK them into tracklets (greedy IoU/centre association across frames)
    3. for each tracklet, fold evidence into an aggregate:
         - JERSEY NUMBER  (OCR on the bigger, mid-field crops where the number
           is actually legible - the user's key insight)
         - HEIGHT RESIDUAL vs the team perspective model (the one trait that
           individuates same-kit teammates at this resolution)
         - play ZONE, SPEED, colour FINGERPRINT
  then ACROSS all clips:
    4. cluster tracklets into candidate players (number-anchored where a number
       was read; otherwise by height residual), -> rt2.profiles.PlayerProfile
    5. derive ROLE from number (11=wing, 6/7=flanker, ...), cross-check vs zone
    6. write output/game.profiles.json + (optionally) update the persistent
       cross-match registry, so every clip watched tightens the IDs.

"ID ME first": pass --me <number> (and optionally --player <name>). The matching
profile is reported first and a rule-out seed is written naming every OTHER
identified teammate - so the live tracker can exclude them and gain confidence on
you. If your number isn't auto-read yet (cold start), the profiler still nails the
OTHERS to rule out.

Usage (PowerShell):
  python v2\\apps\\profile_game.py --all --me 7 --player pou
  python v2\\apps\\profile_game.py --video input\\chunk_028.mp4 --passes 2
  python v2\\apps\\profile_game.py --selftest          # headless, no video
"""
from __future__ import annotations

import argparse
import sys
import pathlib
import time

# Make the rt2 shared lib importable (v2/ is this file's grandparent).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rt2.paths import ProjectPaths
from rt2 import bbox, profiles
from rt2.profiles import PlayerProfile, GameProfiles, write_ruleout_seed

# -----------------------------------------------------------------------------
# Tunables (overridable on the CLI)
IOU_GATE = 0.20            # min IoU to associate a detection with an open tracklet
DIST_FRAC = 0.04           # fallback: centre distance < DIST_FRAC*diag also matches
MAX_GAP = 8                # frames a tracklet may go unmatched before it closes
MIN_TRACKLET = 8           # drop tracklets shorter than this many frames (noise)
SAMPLE_STRIDE = 4          # decode every Nth frame of a tracklet for heavy work
MAX_SAMPLES = 24           # cap heavy (fingerprint/OCR) samples per tracklet
OCR_MIN_H = 46             # only OCR crops at least this tall (px) - legibility gate
HEIGHT_LINK_STD = 1.5      # cluster split when height-resid gap > this * resid_std


# =============================================================================
# 1) Tracklet linking  (PURE - no cv2 - so it is headless-testable)
# =============================================================================
class _Open:
    __slots__ = ("frames", "boxes", "last_frame", "last_box")

    def __init__(self, frame, box):
        self.frames = [frame]
        self.boxes = [box]
        self.last_frame = frame
        self.last_box = box

    def extend(self, frame, box):
        self.frames.append(frame)
        self.boxes.append(box)
        self.last_frame = frame
        self.last_box = box


def link_tracklets(dets_by_frame, diag, iou_gate=IOU_GATE, dist_frac=DIST_FRAC,
                   max_gap=MAX_GAP, min_len=MIN_TRACKLET):
    """Greedy frame-to-frame association of detection boxes into tracklets.

    dets_by_frame : dict frame -> list of boxes (x1,y1,x2,y2[,conf]); only the
                    first 4 coords are used.
    diag          : frame diagonal in px (scales the distance fallback gate).
    Returns a list of tracklets, each {"frames":[...], "boxes":[(x1,y1,x2,y2),...]}.
    Tracklets shorter than min_len frames are dropped.
    """
    open_tracks: list[_Open] = []
    closed: list[_Open] = []
    dist_gate = dist_frac * diag

    for fr in sorted(dets_by_frame.keys()):
        boxes = [tuple(b[:4]) for b in dets_by_frame[fr]]
        # close stale tracks
        still_open = []
        for t in open_tracks:
            if fr - t.last_frame > max_gap:
                closed.append(t)
            else:
                still_open.append(t)
        open_tracks = still_open

        # score every (track, box) pair, then greedily take the best disjoint set
        pairs = []
        for ti, t in enumerate(open_tracks):
            for bi, b in enumerate(boxes):
                iou = bbox.iou(t.last_box, b)
                d2 = bbox.dist2(t.last_box, b)
                if iou >= iou_gate or d2 <= dist_gate * dist_gate:
                    # higher score = better; iou dominates, distance breaks ties
                    score = iou + 1.0 / (1.0 + d2 / (dist_gate * dist_gate + 1.0))
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)
        used_t, used_b = set(), set()
        for _score, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            open_tracks[ti].extend(fr, boxes[bi])
            used_t.add(ti)
            used_b.add(bi)
        # unmatched boxes -> new tracks
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                open_tracks.append(_Open(fr, b))

    closed.extend(open_tracks)
    out = []
    for t in closed:
        if len(t.frames) >= min_len:
            out.append({"frames": t.frames, "boxes": t.boxes})
    return out


# =============================================================================
# 2) Height-residual clustering  (PURE - headless-testable)
# =============================================================================
def cluster_by_height(resid_means, resid_std, gap_std=HEIGHT_LINK_STD):
    """1-D agglomerative split of tracklets by their mean height residual.

    resid_means : list of per-tracklet mean height residuals (px). None entries
                  are placed in their own 'unknown-height' bucket (-1 label).
    resid_std   : the team height model's residual std (the natural scale).
    Returns a list of integer cluster labels parallel to resid_means (-1 = the
    height-unknown bucket; 0..K-1 = real clusters ordered by ascending residual).
    """
    scale = max(float(resid_std or 0.0), 1.0)
    gap = gap_std * scale
    idx_known = [i for i, r in enumerate(resid_means) if r is not None]
    labels = [-1] * len(resid_means)
    if not idx_known:
        return labels
    idx_known.sort(key=lambda i: resid_means[i])
    cur = 0
    labels[idx_known[0]] = 0
    for prev, i in zip(idx_known, idx_known[1:]):
        if resid_means[i] - resid_means[prev] > gap:
            cur += 1
        labels[i] = cur
    return labels


# =============================================================================
# 3) Per-tracklet heavy aggregation  (needs cv2 / VideoReader - real runs only)
# =============================================================================
def _aggregate_tracklet(reader, tk, height_model, frame_w, frame_h,
                        do_ocr, ocr_min_h, stride, max_samples):
    """Decode sampled frames of a tracklet and return an aggregate dict:
    {n_frames, h:(n,mean,M2), zx/zy:(n,mean,M2), sp:(n,mean,M2),
     fp:(vec or None, n), number_votes:{digits:weight}}."""
    import numpy as np
    from rt2 import features
    try:
        from rt2 import ocr
    except Exception:
        ocr = None

    frames, boxes = tk["frames"], tk["boxes"]
    n = len(frames)

    # --- light stats over ALL boxes (no decode): height resid, zone, speed ---
    def _wf():
        return [0, 0.0, 0.0]   # n, mean, M2

    def _fold(acc, x):
        acc[0] += 1
        d = x - acc[1]
        acc[1] += d / acc[0]
        acc[2] += d * (x - acc[1])

    h_acc, zx_acc, zy_acc, sp_acc = _wf(), _wf(), _wf(), _wf()
    prev_c = prev_f = None
    for fr, b in zip(frames, boxes):
        cx = 0.5 * (b[0] + b[2]); cy = 0.5 * (b[1] + b[3])
        bh = b[3] - b[1]
        if height_model is not None:
            try:
                resid = bh - height_model.predict(cy)
            except Exception:
                resid = bh
        else:
            resid = bh
        _fold(h_acc, float(resid))
        if frame_w > 0 and frame_h > 0:
            _fold(zx_acc, cx / frame_w)
            _fold(zy_acc, cy / frame_h)
        if prev_c is not None and fr > prev_f:
            d = ((cx - prev_c[0]) ** 2 + (cy - prev_c[1]) ** 2) ** 0.5
            _fold(sp_acc, d / (fr - prev_f))
        prev_c, prev_f = (cx, cy), fr

    # --- heavy stats on SAMPLED frames: fingerprint mean + OCR number votes ---
    samp = list(range(0, n, max(1, stride)))[:max_samples]
    fp_sum = None; fp_n = 0
    votes: dict[str, float] = {}
    for si in samp:
        fr, b = frames[si], boxes[si]
        img = reader.frame(fr)
        if img is None:
            continue
        try:
            vec = features.feature_vector(img, b)
            if vec is not None:
                if fp_sum is None:
                    fp_sum = [0.0] * len(vec)
                for k, v in enumerate(vec):
                    fp_sum[k] += float(v)
                fp_n += 1
        except Exception:
            pass
        if do_ocr and ocr is not None and (b[3] - b[1]) >= ocr_min_h:
            try:
                from rt2 import regions
                crop = regions.torso_patch(img, b)
                res = ocr.read_number(crop)
                if res:
                    digits, conf = res
                    votes[digits] = votes.get(digits, 0.0) + float(conf)
            except Exception:
                pass

    fp = [s / fp_n for s in fp_sum] if (fp_sum and fp_n) else None
    return {
        "n_frames": n, "h": h_acc, "zx": zx_acc, "zy": zy_acc, "sp": sp_acc,
        "fp": (fp, fp_n), "number_votes": votes,
    }


# =============================================================================
# 4) Orchestration
# =============================================================================
def _resolve_videos(pp, args):
    vids = []
    if args.all:
        vids = pp.videos()
    for v in (args.video or []):
        p = pathlib.Path(v)
        if not p.is_absolute():
            p = pp.root / v
        vids.append(p)
    # de-dupe, keep order
    seen, out = set(), []
    for v in vids:
        key = str(v.resolve()).lower()
        if key not in seen and v.exists():
            seen.add(key); out.append(v)
    return out


def _load_height_model(pp, video, override):
    """Return the tracked team's HeightModel (or None) from the resolved
    calibration, plus the team name."""
    from rt2.calibration import MatchCalibration
    cpath = pp.resolve_calibration(video, override)
    if not cpath:
        return None, ""
    try:
        cal = MatchCalibration.load(cpath)
    except Exception:
        return None, ""
    tracked = cal.tracked_teams() or cal.teams
    for t in tracked:
        if getattr(t, "height", None) is not None:
            return t.height, t.name
    return None, (tracked[0].name if tracked else "")


def run(args):
    from rt2.video import VideoReader
    from rt2.detections import read_detections, kept_by_frame
    try:
        from rt2 import ocr
        ocr_avail = ocr.available()
    except Exception:
        ocr_avail = False

    pp = ProjectPaths().ensure()
    videos = _resolve_videos(pp, args)
    if not videos:
        print("[profile] no videos found. Use --video <path> or --all "
              "(input/ folder). Aborting.")
        return 2

    do_ocr = ocr_avail and not args.no_ocr
    print(f"[profile] {len(videos)} clip(s); passes={args.passes}; "
          f"OCR={'ON' if do_ocr else 'OFF (tesseract not installed)'}")
    if not do_ocr and not args.no_ocr:
        print("[profile]   -> jersey numbers cannot be read without the "
              "tesseract binary; profiling on height/zone only this run.")
        print("[profile]   -> install once with: "
              "winget install UB-Mannheim.TesseractOCR")

    # accumulate tracklet aggregates across all clips & passes
    agg_list = []     # list of (aggregate_dict, clip_stem)
    frame_w = frame_h = 0
    fps = 29.97
    team_name = ""

    for video in videos:
        dpath = pp.detections(video)
        if not dpath.exists():
            print(f"[profile] SKIP {video.name}: no detections "
                  f"({dpath.name}). Run prerender on it first.")
            continue
        try:
            df = read_detections(dpath)
        except Exception as e:
            print(f"[profile] SKIP {video.name}: cannot read detections ({e})")
            continue
        kbf = kept_by_frame(df)
        if not kbf:
            print(f"[profile] SKIP {video.name}: 0 kept detections.")
            continue

        hm, tname = _load_height_model(pp, video, args.calibration)
        if tname:
            team_name = tname
        try:
            reader = VideoReader(video)
        except Exception as e:
            print(f"[profile] SKIP {video.name}: cannot open video ({e})")
            continue
        frame_w = frame_w or reader.width
        frame_h = frame_h or reader.height
        fps = reader.fps or fps
        diag = (reader.width ** 2 + reader.height ** 2) ** 0.5

        if args.max_frames:
            kbf = {f: v for f, v in kbf.items() if f <= args.max_frames}

        tracks = link_tracklets(kbf, diag, args.iou, args.dist_frac,
                                args.max_gap, args.min_tracklet)
        print(f"[profile] {video.name}: {len(kbf)} det-frames -> "
              f"{len(tracks)} tracklets (team='{tname}', "
              f"height_model={'yes' if hm else 'no'})")

        for p in range(args.passes):
            # vary the sampling offset each pass so re-watching reads NEW frames
            stride = max(1, args.stride)
            offset = p % stride
            t0 = time.time()
            for ti, tk in enumerate(tracks):
                tk_off = {"frames": tk["frames"][offset:],
                          "boxes": tk["boxes"][offset:]}
                if len(tk_off["frames"]) < max(2, args.min_tracklet // 2):
                    continue
                ag = _aggregate_tracklet(
                    reader, tk_off, hm, reader.width, reader.height,
                    do_ocr, args.ocr_min_h, stride, args.max_samples)
                agg_list.append((ag, video.stem))
                if args.verbose and (ti + 1) % 25 == 0:
                    print(f"    pass {p+1}/{args.passes}: "
                          f"{ti+1}/{len(tracks)} tracklets "
                          f"({time.time()-t0:.0f}s)")
            print(f"[profile]   pass {p+1}/{args.passes} done "
                  f"({time.time()-t0:.0f}s)")
        reader.release()

    if not agg_list:
        print("[profile] no tracklets aggregated; nothing to profile.")
        return 3

    gp = _build_profiles(agg_list, team_name, fps, frame_w, frame_h,
                         len(videos), args)
    out = args.out or (pp.output / "game.profiles.json")
    gp.save(out)
    print(f"\n[profile] wrote {len(gp.players)} player profile(s) -> {out}")
    for line in gp.summary_lines():
        print(line)

    _report_me(gp, pp, args)
    _maybe_ingest_registry(gp, pp, videos, args)
    return 0


def _tracklet_number(votes, max_number, per_tracklet_min):
    """Best whitelisted number READ ON A SINGLE TRACKLET (the per-tracklet anchor),
    or None. votes: {digits: weight}. Returns (int_number, weight) or (None, w)."""
    v = {d: w for d, w in votes.items()
         if d.isdigit() and 1 <= int(d) <= max_number}
    if not v:
        return None, 0.0
    d, w = max(v.items(), key=lambda kv: kv[1])
    return (int(d), w) if w >= per_tracklet_min else (None, w)


def _build_profiles(agg_list, team_name, fps, frame_w, frame_h, n_clips, args):
    """Cluster tracklet aggregates into PlayerProfiles, NUMBER-ANCHORED.

    Strategy (the fix for the 'one blob' collapse):
      1. each tracklet gets its own best whitelisted number (if its OCR votes are
         confident enough) -> seeds one identity PER DISTINCT NUMBER;
      2. unnumbered tracklets ATTACH to the numbered identity nearest in height
         residual (within a tolerance) - so a player's silent tracklets join the
         frames where their number WAS read;
      3. whatever's left (no number, no height match) is height-clustered into an
         honest 'unidentified' pool - we don't fake-individuate it.
    """
    import numpy as np
    profiles.MAX_JERSEY = args.max_number          # apply the closed-set constraint

    resid = [ag["h"][1] if ag["h"][0] > 0 else None for ag, _ in agg_list]
    known = np.array([r for r in resid if r is not None], dtype=float)
    resid_std = float(np.std(known)) if len(known) >= 2 else 1.0
    resid_std = resid_std or 1.0

    # 1) per-tracklet number -> seed numbered clusters
    clusters: dict[tuple, list[int]] = {}
    tk_num = []
    for i, (ag, _clip) in enumerate(agg_list):
        n, _w = _tracklet_number(ag["number_votes"], args.max_number,
                                 args.per_tracklet_min)
        tk_num.append(n)
        if n is not None:
            clusters.setdefault(("num", n), []).append(i)

    # --- OCR legibility DIAGNOSTIC (free: uses the votes already collected) ----
    from collections import Counter as _Counter
    valid_reads = 0
    per_tk_best = _Counter()
    total_weight = _Counter()
    for ag, _clip in agg_list:
        vv = {d: w for d, w in ag["number_votes"].items()
              if d.isdigit() and 1 <= int(d) <= args.max_number}
        if vv:
            valid_reads += 1
            best = max(vv.items(), key=lambda kv: kv[1])[0]
            per_tk_best[int(best)] += 1
            for d, w in vv.items():
                total_weight[int(d)] += w
    seeds = sum(1 for n in tk_num if n is not None)
    print(f"[profile] OCR diag: {valid_reads}/{len(agg_list)} tracklets had "
          f">=1 valid read; {seeds} crossed per-tracklet-min="
          f"{args.per_tracklet_min}")
    print(f"[profile] OCR diag: per-tracklet BEST number (count): "
          f"{dict(per_tk_best.most_common(12))}")
    print(f"[profile] OCR diag: total weight per number: "
          f"{ {k: round(v, 1) for k, v in total_weight.most_common(12)} }")
    try:
        import json as _json
        from rt2.paths import ProjectPaths as _PP
        dump = [{"clip": c, "frames": ag["n_frames"],
                 "resid": (ag["h"][1] if ag["h"][0] > 0 else None),
                 "votes": ag["number_votes"]} for ag, c in agg_list]
        (_PP().output / "_tracklet_reads.json").write_text(
            _json.dumps(dump, indent=1), encoding="utf-8")
    except Exception:
        pass

    def _cluster_resid(idxs):
        rs = [resid[i] for i in idxs if resid[i] is not None]
        return float(np.mean(rs)) if rs else None
    num_resid = {k: _cluster_resid(v) for k, v in clusters.items()}

    # 2) attach unnumbered tracklets to the nearest numbered cluster by height
    attach_tol = max(args.height_gap_std * resid_std, 8.0)
    leftover = []
    for i, (ag, _clip) in enumerate(agg_list):
        if tk_num[i] is not None:
            continue
        ri = resid[i]
        best, bestd = None, None
        if ri is not None:
            for k, mr in num_resid.items():
                if mr is None:
                    continue
                d = abs(ri - mr)
                if d <= attach_tol and (bestd is None or d < bestd):
                    bestd, best = d, k
        if best is not None:
            clusters[best].append(i)
        else:
            leftover.append(i)

    # 3) height-cluster the leftovers into an 'unidentified' pool
    lo_labels = cluster_by_height([resid[i] for i in leftover], resid_std,
                                  args.height_gap_std)
    for lab, i in zip(lo_labels, leftover):
        clusters.setdefault(("p", lab), []).append(i)

    # 4) build a PlayerProfile per cluster (numbered first, then the pool)
    players = []
    pn = 0
    for key in sorted(clusters.keys(), key=lambda k: (k[0] != "num", k[1])):
        pn += 1
        prof = PlayerProfile(pid=f"p{pn:02d}")
        for i in clusters[key]:
            ag, clip = agg_list[i]
            prof.fold_height(*ag["h"])
            prof.fold_zone(*ag["zx"], *ag["zy"])
            prof.fold_speed(*ag["sp"])
            fp, fpn = ag["fp"]
            if fp is not None:
                prof.fold_fingerprint(fp, fpn)
            for digits, w in ag["number_votes"].items():
                prof.vote_number(digits, w)        # whitelisted inside vote_number
            prof.n_frames += ag["n_frames"]
            prof.n_tracklets += 1
            prof.note_clip(clip)
        prof.finalize(min_number_weight=args.min_number_weight)
        players.append(prof)

    # if two clusters resolved to the SAME confident number, merge them
    players = _merge_same_number(players, args.min_number_weight)
    return GameProfiles(game=team_name or "match", fps=fps,
                        frame_w=frame_w, frame_h=frame_h, team=team_name,
                        n_clips=n_clips, players=players)


def _merge_same_number(players, min_w):
    by_num: dict[int, PlayerProfile] = {}
    out = []
    for p in players:
        if p.number is None:
            out.append(p); continue
        if p.number in by_num:
            keep = by_num[p.number]
            keep.fold_height(p.h_n, p.h_mean, p.h_M2)
            keep.fold_zone(p.zx_n, p.zx_mean, p.zx_M2,
                           p.zy_n, p.zy_mean, p.zy_M2)
            keep.fold_speed(p.sp_n, p.sp_mean, p.sp_M2)
            if p.fingerprint is not None:
                keep.fold_fingerprint(p.fingerprint, p.fp_n)
            for d, w in p.number_votes.items():
                keep.vote_number(d, w)
            keep.n_frames += p.n_frames
            keep.n_tracklets += p.n_tracklets
            for c in p.clips:
                keep.note_clip(c)
            keep.finalize(min_number_weight=min_w)
        else:
            by_num[p.number] = p
            out.append(p)
    return out


def _report_me(gp, pp, args):
    if args.me is None:
        return
    me = gp.by_number(args.me)
    print("\n[profile] === ID ME FIRST ===")
    if me is not None:
        role = me.role or "role unknown"
        print(f"[profile] YOU = #{args.me} ({role}); "
              f"seen {me.n_frames} frames over {len(me.clips)} clip(s); "
              f"evidence {me.evidence_score():.2f}")
    else:
        r = profiles.role_for_number(args.me)[0] or "role unknown"
        print(f"[profile] #{args.me} ({r}) not auto-identified yet (cold start / "
              f"number not read). Track yourself once in track.py to seed it; "
              f"the rule-out list below still narrows the field.")
    seeds = gp.rule_out_for(args.me)
    print(f"[profile] rule out {len(seeds)} other identified teammate(s):")
    for s in seeds:
        print(f"    #{s['number']:>2} {s['role'] or '?':<20} "
              f"ev={s['evidence']:.2f}")
    seed_path = pp.output / (f"player_{args.player}.ruleout.json"
                             if args.player else "ruleout.json")
    write_ruleout_seed(seed_path, args.me, gp)
    print(f"[profile] rule-out seed -> {seed_path}")


def _maybe_ingest_registry(gp, pp, videos, args):
    """Fold this run into the persistent cross-match registry, if available.
    The registry module is built separately; we import it lazily so the
    profiler works with or without it."""
    if args.no_registry:
        return
    try:
        from rt2.registry import PlayerRegistry
    except Exception:
        print("[profile] (registry module not present yet - skipped persistent "
              "update; game.profiles.json was still written.)")
        return
    try:
        match_id = args.match_id or (videos[0].stem if videos else None)
        with PlayerRegistry() as reg:
            if args.me is not None and args.player:
                try:
                    reg.set_me(args.player, number=args.me)
                except Exception:
                    pass
            summary = reg.ingest_game_profiles(
                gp, match_id=match_id,
                video=(str(videos[0]) if videos else ""),
                pipeline_version="profile_game/1")
        print(f"[profile] registry updated: {summary}")
    except Exception as e:
        print(f"[profile] registry update failed (non-fatal): {e}")


# =============================================================================
# 5) Self-test (headless - no video, no torch, no display)
# =============================================================================
def _selftest():
    print("[profile-selftest] tracklet linker + height clustering + profiling")

    # --- linker: two players moving linearly across 30 frames -> 2 tracklets ---
    diag = (1920 ** 2 + 1080 ** 2) ** 0.5
    dets = {}
    for fr in range(1, 31):
        a = (100 + fr * 5, 400, 130 + fr * 5, 480)      # player A drifts right
        b = (900 - fr * 4, 300, 925 - fr * 4, 360)      # player B drifts left
        dets[fr] = [a, b]
    tracks = link_tracklets(dets, diag)
    assert len(tracks) == 2, f"expected 2 tracklets, got {len(tracks)}"
    assert all(len(t["frames"]) == 30 for t in tracks), "tracklets should be full-length"
    print(f"[profile-selftest] linker: 2 players -> {len(tracks)} clean tracklets  OK")

    # a brief gap inside the max_gap is bridged, not split
    dets2 = {fr: [(100 + fr * 5, 400, 130 + fr * 5, 480)] for fr in range(1, 21)}
    del dets2[10]; del dets2[11]            # 2-frame gap (< MAX_GAP)
    t2 = link_tracklets(dets2, diag)
    assert len(t2) == 1, f"a short gap must not split a tracklet (got {len(t2)})"
    print("[profile-selftest] linker bridges a short gap  OK")

    # --- clustering: two height bands -> two clusters --------------------------
    resid_means = [-20.0, -18.0, -21.0, +25.0, +24.0, +27.0]
    labels = cluster_by_height(resid_means, resid_std=4.0, gap_std=1.5)
    assert len(set(labels)) == 2, f"expected 2 height clusters, got {set(labels)}"
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    # a None residual goes to the -1 unknown bucket
    labels2 = cluster_by_height([-20.0, None, +25.0], resid_std=4.0)
    assert labels2[1] == -1
    print("[profile-selftest] height clustering splits two bands  OK")

    # --- build profiles from synthetic aggregates (no video) -------------------
    def wf(vals):
        n = 0; mean = 0.0; M2 = 0.0
        for x in vals:
            n += 1; d = x - mean; mean += d / n; M2 += d * (x - mean)
        return [n, mean, M2]

    short = (180 ** 2 + 90 ** 2) ** 0.5  # unused; keep imports honest
    agg_list = [
        ({"n_frames": 120, "h": wf([-20, -19, -21]), "zx": wf([0.08, 0.10]),
          "zy": wf([0.30, 0.28]), "sp": wf([3.0, 3.2]), "fp": ([0.1] * 10, 3),
          "number_votes": {"11": 1.4, "11": 1.4}}, "clipA"),
        ({"n_frames": 110, "h": wf([-20, -22]), "zx": wf([0.07]), "zy": wf([0.31]),
          "sp": wf([3.1]), "fp": ([0.1] * 10, 2), "number_votes": {"11": 0.8}},
         "clipB"),
        ({"n_frames": 140, "h": wf([+26, +25, +27]), "zx": wf([0.5, 0.52]),
          "zy": wf([0.6, 0.58]), "sp": wf([2.0]), "fp": ([0.2] * 10, 3),
          "number_votes": {"7": 3.0}}, "clipA"),
    ]

    class _A:
        me = 7; player = None; min_number_weight = 2.0; height_gap_std = 1.5
        max_number = 23; per_tracklet_min = 1.5
    gp = _build_profiles(agg_list, "Otorohanga", 29.97, 1920, 1080, 2, _A())
    nums = {p.number for p in gp.players if p.number is not None}
    assert 11 in nums and 7 in nums, f"expected #11 and #7, got {nums}"
    p11 = gp.by_number(11); p7 = gp.by_number(7)
    assert p11.role == "left wing" and p7.role == "openside flanker"
    # #11 wing was seen wide+deep -> zone cross-check should hold
    assert p11.role_zone_ok is True
    # the two #11 tracklets (clipA+clipB) merged into one player across clips
    assert len(p11.clips) == 2, f"#11 should span 2 clips, got {p11.clips}"
    ro = gp.rule_out_for(7)
    assert [r["number"] for r in ro] == [11], "rule-out for #7 should name #11"
    print(f"[profile-selftest] profiles: #11 {p11.role} (2 clips), "
          f"#7 {p7.role}; rule_out(#7)->#11  OK")

    # NUMBER-ANCHORED split: two players at the SAME height but DIFFERENT numbers
    # must stay SEPARATE (height clustering alone would have merged them).
    agg2 = [
        ({"n_frames": 100, "h": wf([0, 0, 0]), "zx": wf([0.5]), "zy": wf([0.5]),
          "sp": wf([2.0]), "fp": (None, 0), "number_votes": {"1": 3.0}}, "c"),
        ({"n_frames": 100, "h": wf([0, 0, 0]), "zx": wf([0.5]), "zy": wf([0.5]),
          "sp": wf([2.0]), "fp": (None, 0), "number_votes": {"3": 3.0}}, "c"),
    ]
    gp2 = _build_profiles(agg2, "T", 30, 1920, 1080, 1, _A())
    assert {p.number for p in gp2.players if p.number is not None} == {1, 3}, \
        "same-height different-number tracklets must NOT merge"
    print("[profile-selftest] number-anchored: same-height #1 vs #3 stay split  OK")

    # WHITELIST: an impossible number (73) is dropped, the valid one wins
    agg3 = [
        ({"n_frames": 100, "h": wf([0]), "zx": wf([0.5]), "zy": wf([0.5]),
          "sp": wf([2.0]), "fp": (None, 0),
          "number_votes": {"73": 5.0, "3": 3.0}}, "c"),
    ]
    gp3 = _build_profiles(agg3, "T", 30, 1920, 1080, 1, _A())
    assert gp3.players[0].number == 3, "impossible #73 must be dropped by whitelist"
    print("[profile-selftest] whitelist drops impossible #73 -> names #3  OK")

    # round-trip
    import tempfile, os
    tmp = pathlib.Path(tempfile.gettempdir()) / "rt2_game_profiles_selftest.json"
    gp.save(tmp)
    gp2 = GameProfiles.load(tmp)
    assert gp2.by_number(11).role == "left wing"
    try:
        os.unlink(tmp)
    except OSError:
        pass
    print("[profile-selftest] game.profiles.json round-trip  OK")
    print("[profile-selftest] PASS")


def main():
    ap = argparse.ArgumentParser(description="Overnight batch player profiler")
    ap.add_argument("--video", action="append",
                    help="a clip to profile (repeatable); path rel to repo root OK")
    ap.add_argument("--all", action="store_true",
                    help="profile every video in the input/ folder")
    ap.add_argument("--calibration", help="explicit calibration json override")
    ap.add_argument("--me", type=int, default=None,
                    help="YOUR jersey number - reported first + rule-out seed")
    ap.add_argument("--player", help="player name (namespaces the rule-out seed)")
    ap.add_argument("--out", help="output profiles json (default output/game.profiles.json)")
    ap.add_argument("--match-id", help="registry match id (default first clip stem)")
    ap.add_argument("--passes", type=int, default=1,
                    help="re-watch passes; each reads different sampled frames")
    ap.add_argument("--stride", type=int, default=SAMPLE_STRIDE,
                    help="decode every Nth tracklet frame for fingerprint/OCR")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--ocr-min-h", type=int, default=OCR_MIN_H,
                    help="only OCR crops at least this tall (px)")
    ap.add_argument("--no-ocr", action="store_true", help="disable jersey OCR")
    ap.add_argument("--no-registry", action="store_true",
                    help="don't update the persistent cross-match registry")
    ap.add_argument("--iou", type=float, default=IOU_GATE)
    ap.add_argument("--dist-frac", type=float, default=DIST_FRAC)
    ap.add_argument("--max-gap", type=int, default=MAX_GAP)
    ap.add_argument("--min-tracklet", type=int, default=MIN_TRACKLET)
    ap.add_argument("--height-gap-std", type=float, default=HEIGHT_LINK_STD)
    ap.add_argument("--min-number-weight", type=float, default=2.0,
                    help="OCR vote weight needed to NAME a number (per cluster)")
    ap.add_argument("--max-number", type=int, default=23,
                    help="largest valid jersey number (closed-set constraint; "
                         "reads above this are dropped as misreads)")
    ap.add_argument("--per-tracklet-min", type=float, default=1.5,
                    help="OCR vote weight on a SINGLE tracklet to seed a numbered "
                         "identity (number-anchored clustering)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="only process frames <= this (quick test runs)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    sys.exit(run(args))


if __name__ == "__main__":
    main()
