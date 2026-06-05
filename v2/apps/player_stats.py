"""player_stats.py - distance / speed run by each tracked player, in real METRES.

Uses the pitch homography (rt2.homography) + per-frame camera-motion compensation
(rt2.cmc) to convert every tracked foot position (from mark_all's objects.csv)
into pitch metres, then sums frame-to-frame movement per object id:
  * distance_m   - metres covered
  * top_speed_ms - fastest plausible step (m/s)
  * avg_speed_ms - mean over moving frames
  * minutes      - time the id was on screen

HONEST CAVEATS (printed in the output too):
  1. PANNING camera: a single calibration is only valid at its anchor frame, so we
     ride CMC forward from there. CMC drifts over long spans -> stats are ACCURATE
     OVER A PASSAGE (default a window from the calibration frame), UNRELIABLE over
     the whole game. Use --start/--n to pick a clean passage; --all is approximate.
  2. ID FRAGMENTATION: mark_all assigns many ids to one real player across a game,
     so this is distance PER TRACKLET, not per named player. Stitch ids first
     (your loose-marks / id-merge) for true per-player totals.
  3. A per-step SPEED CAP rejects teleports (bad homography / id swaps) so glitches
     don't inflate distance.

Usage (PowerShell):
  python v2\\apps\\player_stats.py --video input\\<match>.mp4              # window from the calib frame
  python v2\\apps\\player_stats.py --video input\\<match>.mp4 --start 1500 --n 900
  python v2\\apps\\player_stats.py --video input\\<match>.mp4 --selftest   # headless math test
"""
from __future__ import annotations

import argparse
import csv
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

MAX_SPEED_MS = 12.0          # cap a plausible human step (sprint ~10-11 m/s + slack)
DEFAULT_WINDOW = 900         # frames analysed by default (~30s) from the calib frame
MIN_TRACKLET_FRAMES = 15     # ignore blink tracklets


def track_distances(positions_by_obj, fps, max_speed=MAX_SPEED_MS):
    """PURE + testable. positions_by_obj: {obj_id -> list of (frame, X_m, Y_m)}.
    Returns {obj_id -> dict(distance_m, top_speed_ms, avg_speed_ms, n_frames,
    minutes)}. Steps faster than max_speed are dropped (glitch/teleport)."""
    out = {}
    fps = float(fps) or 30.0
    for oid, pts in positions_by_obj.items():
        pts = sorted(pts, key=lambda p: p[0])
        dist = 0.0
        speeds = []
        for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
            gap = f1 - f0
            if gap <= 0:
                continue
            step = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            spd = step / (gap / fps)            # m/s
            if spd > max_speed:                 # teleport / bad H -> skip
                continue
            dist += step
            if step > 0.05:                     # count only real movement
                speeds.append(spd)
        n = len(pts)
        out[oid] = {
            "distance_m": round(dist, 1),
            "top_speed_ms": round(max(speeds), 2) if speeds else 0.0,
            "avg_speed_ms": round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
            "n_frames": n,
            "minutes": round(n / fps / 60.0, 2),
        }
    return out


def run(args):
    import cv2
    from rt2.paths import ProjectPaths
    from rt2.homography import PitchHomography
    from rt2.cmc import CameraTracker, LiveHomography

    pp = ProjectPaths()
    video = pathlib.Path(args.video)
    if not video.is_absolute():
        video = pp.root / args.video
    stem = video.stem

    obj_path = pathlib.Path(args.objects) if args.objects else (
        pp.output / f"{stem}.objects.csv")
    if not obj_path.exists():
        print(f"[stats] no objects file at {obj_path} - run mark_all first.")
        return 2
    pitch_path = pathlib.Path(args.pitch) if args.pitch else None
    if pitch_path is None:
        for cand in (pp.output / f"{stem}.pitch.json", pp.output / "game.pitch.json"):
            if cand.exists():
                pitch_path = cand
                break
    ph = PitchHomography.load(pitch_path) if pitch_path else None
    if ph is None or not ph.ok:
        print("[stats] no usable pitch calibration (.pitch.json). Calibrate the "
              "pitch first (Calibrate pitch in the tracker / calibrate_pitch.py).")
        return 2

    anchor = ph.anchor_frame if ph.anchor_frame is not None else 1
    start = args.start if args.start is not None else anchor
    if start != anchor:
        print(f"[stats] NOTE: pitch anchor is frame {anchor}; CMC must ride forward "
              f"from there, so analysing from {anchor} regardless of --start.")
        start = anchor
    n = args.n if args.all is False else None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[stats] cannot open {video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if n is None else min(total, start + n)

    # objects boxes per frame in [start, end)
    print(f"[stats] loading {obj_path.name} for frames {start}..{end}...")
    boxes_by_frame = {}
    with open(obj_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                fr = int(float(r["frame"]))
            except (KeyError, ValueError):
                continue
            if fr < start or fr >= end:
                continue
            try:
                oid = int(float(r["obj_id"]))
                box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            except (KeyError, ValueError):
                continue
            boxes_by_frame.setdefault(fr, []).append((oid, box, r.get("team", "")))

    live = LiveHomography(ph, CameraTracker())
    positions = {}          # obj_id -> [(frame, X, Y)]
    teams = {}              # obj_id -> dominant team label (last seen)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start - 1))
    import time
    t0 = time.time()
    for fr in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        ig = [b for _, b, _ in boxes_by_frame.get(fr, [])]   # mask players from CMC
        if fr == anchor:
            live.tracker.anchor(img)
        else:
            live.tracker.update(img, ignore_boxes=ig)
        if not live.healthy:
            continue
        for oid, box, team in boxes_by_frame.get(fr, []):
            xy = live.foot_point(box)
            if xy is None:
                continue
            positions.setdefault(oid, []).append((fr, xy[0], xy[1]))
            if team:
                teams[oid] = team
        if (fr - start) % 200 == 0:
            done = fr - start + 1
            print(f"[stats] {done}/{end-start} frames "
                  f"({done/max(time.time()-t0,1e-9):.1f} fps)", flush=True)
    cap.release()

    stats = track_distances(positions, fps)
    stats = {o: s for o, s in stats.items() if s["n_frames"] >= MIN_TRACKLET_FRAMES}

    out_path = pp.output / f"{stem}.player_stats.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "team", "distance_m", "top_speed_ms",
                    "avg_speed_ms", "minutes", "n_frames"])
        for oid, s in sorted(stats.items(), key=lambda kv: -kv[1]["distance_m"]):
            w.writerow([oid, teams.get(oid, ""), s["distance_m"], s["top_speed_ms"],
                        s["avg_speed_ms"], s["minutes"], s["n_frames"]])

    print(f"\n[stats] window frames {start}..{end} (~{(end-start)/fps:.0f}s) | "
          f"{len(stats)} tracklets >= {MIN_TRACKLET_FRAMES} frames -> {out_path}")
    print("[stats] CAVEATS: per-TRACKLET (ids fragment, not per player); accurate "
          "over this passage only (CMC drifts long-term); teleports capped at "
          f"{MAX_SPEED_MS} m/s.")
    top = sorted(stats.items(), key=lambda kv: -kv[1]["distance_m"])[:12]
    print("\n  obj   team        dist(m)  top(m/s)  avg   mins")
    for oid, s in top:
        print(f"  {oid:>5}  {teams.get(oid,'')[:9]:<9}  {s['distance_m']:>7.1f}  "
              f"{s['top_speed_ms']:>6.2f}  {s['avg_speed_ms']:>4.1f}  {s['minutes']:>4.1f}")
    return 0


def _selftest():
    print("[stats-selftest] distance / speed math")
    fps = 30.0
    # one object walks +1m/frame in X for 10 frames at 30fps = 30 m/s -> CAPPED out;
    # use a realistic 0.2 m/frame = 6 m/s
    pos = {1: [(i, 0.2 * i, 0.0) for i in range(1, 31)],     # straight 6 m/s
           2: [(i, 0.0, 0.0) for i in range(1, 31)],          # stationary
           3: [(1, 0.0, 0.0), (2, 50.0, 0.0), (3, 50.2, 0.0)]}  # a teleport then small
    s = track_distances(pos, fps)
    # obj1: 29 steps * 0.2m = 5.8m, speed 0.2*30=6 m/s
    assert abs(s[1]["distance_m"] - 5.8) < 0.1, s[1]
    assert abs(s[1]["top_speed_ms"] - 6.0) < 0.1, s[1]
    assert s[2]["distance_m"] == 0.0 and s[2]["top_speed_ms"] == 0.0
    # obj3: the 50m teleport (1500 m/s) is dropped; only the 0.2m step counts
    assert abs(s[3]["distance_m"] - 0.2) < 0.01, s[3]
    print(f"[stats-selftest] runner 6m/s -> {s[1]['distance_m']}m top "
          f"{s[1]['top_speed_ms']}m/s; stationary 0m; teleport capped  OK")
    print("[stats-selftest] PASS")


def main():
    ap = argparse.ArgumentParser(description="Per-player distance/speed in metres")
    ap.add_argument("--video", help="match video")
    ap.add_argument("--objects", help="objects.csv override")
    ap.add_argument("--pitch", help=".pitch.json override")
    ap.add_argument("--start", type=int, default=None,
                    help="first frame (defaults to the pitch anchor frame)")
    ap.add_argument("--n", type=int, default=DEFAULT_WINDOW,
                    help=f"frames to analyse from the start (default {DEFAULT_WINDOW})")
    ap.add_argument("--all", action="store_true",
                    help="analyse to the end (APPROXIMATE - CMC drifts over a full game)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.video:
        ap.error("--video is required")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
