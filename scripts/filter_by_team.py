"""
filter_by_team.py - Keep only the blue & gold team's tracks (multi-signal scorer).

No model training. Each TRACK is classified by what its player wears, sampled
over the whole track lifetime and combined from several weak colour signals.

SIGNALS (per frame, in a region of the player bbox):
  white  - WHITE SHORTS, lower body            (strong: staff & maroon team lack it)
  navy   - NAVY TOP, torso                      (weak: sideline staff also wear navy)
  gold   - GOLD numbers/trim, torso             (weak: collides with skin/turf)
  maroon - MAROON TOP, torso  -> VETO           (the opposition)

HOW IT DECIDES:
  For each signal we compute the fraction of a track's frames where the signal
  exceeds its per-frame pixel threshold (--*-pix). If that fraction reaches the
  signal's presence ratio (--min-ratio) the signal "qualifies" and adds its
  WEIGHT to the track score. A track is kept when:
      median maroon < --maroon-max         (not the maroon team)
      AND total score >= --keep-score
      AND frames sampled >= --min-frames

  Default weights (white 1.0, navy 0.5, gold 0.5) with --keep-score 1.0 mean:
      white shorts alone           -> keep            (1.0)
      navy + gold (seen from behind)-> keep            (1.0)
      navy alone (sideline staff)  -> reject           (0.5)
      gold alone (skin/turf noise) -> reject           (0.5)

Outputs (next to --output):
  - filtered video showing ONLY blue & gold players
  - <source>_bluegold.csv : same schema as tracks.csv, blue&gold tracks only

Usage:
  python scripts\\filter_by_team.py --video input\\chunk_028.mp4 ^
      --csv output\\chunk_028_tracks.csv --output output\\chunk_028_bluegold.mp4
  # tune, e.g.:  --white-pix 0.05 --navy-pix 0.02 --gold-pix 0.03 --keep-score 1.0
  # add --report-only to print the score table without writing video/csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# --- HSV gates (OpenCV HSV: H 0-179, S/V 0-255), calibrated from footage ---
WHITE_LO = np.array([0, 0, 140]);     WHITE_HI = np.array([179, 50, 255])
NAVY_LO = np.array([95, 40, 15]);     NAVY_HI = np.array([125, 255, 120])
GOLD_LO = np.array([16, 80, 90]);     GOLD_HI = np.array([34, 255, 255])
MAROON1_LO = np.array([0, 70, 40]);   MAROON1_HI = np.array([12, 255, 210])
MAROON2_LO = np.array([165, 70, 40]); MAROON2_HI = np.array([179, 255, 210])


def region_fracs(frame, x1, y1, x2, y2):
    """(maroon_torso, white_shorts, navy_torso, gold_torso) for a bbox, or None."""
    w, h = x2 - x1, y2 - y1
    if h < 22 or w < 7:
        return None
    cx1, cx2 = x1 + int(0.20 * w), x1 + int(0.80 * w)
    torso = frame[y1 + int(0.18 * h):y1 + int(0.48 * h), cx1:cx2]
    shorts = frame[y1 + int(0.50 * h):y1 + int(0.78 * h), cx1:cx2]
    if torso.size == 0 or shorts.size == 0:
        return None
    th = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    sh = cv2.cvtColor(shorts, cv2.COLOR_BGR2HSV)
    nt = torso.shape[0] * torso.shape[1]
    ns = shorts.shape[0] * shorts.shape[1]
    maroon = (cv2.inRange(th, MAROON1_LO, MAROON1_HI).sum()
              + cv2.inRange(th, MAROON2_LO, MAROON2_HI).sum()) / 255 / nt
    navy = cv2.inRange(th, NAVY_LO, NAVY_HI).sum() / 255 / nt
    gold = cv2.inRange(th, GOLD_LO, GOLD_HI).sum() / 255 / nt
    white = cv2.inRange(sh, WHITE_LO, WHITE_HI).sum() / 255 / ns
    return maroon, white, navy, gold


def main():
    ap = argparse.ArgumentParser(description="Keep only the blue & gold team's tracks (multi-signal)")
    ap.add_argument("--video", required=True, help="source clip (clean)")
    ap.add_argument("--csv", required=True, help="tracks CSV from track.py")
    ap.add_argument("--output", required=True, help="output video path")
    # per-frame pixel thresholds (fraction of the region's pixels)
    ap.add_argument("--white-pix", type=float, default=0.04)
    ap.add_argument("--navy-pix", type=float, default=0.05)
    ap.add_argument("--gold-pix", type=float, default=0.03)
    # a signal "qualifies" if it exceeds its pixel threshold in this fraction of frames
    ap.add_argument("--min-ratio", type=float, default=0.20)
    # signal weights and overall keep threshold
    ap.add_argument("--w-white", type=float, default=1.0)
    ap.add_argument("--w-navy", type=float, default=0.5)
    ap.add_argument("--w-gold", type=float, default=0.5)
    ap.add_argument("--keep-score", type=float, default=1.0)
    # maroon veto + minimum track length
    ap.add_argument("--maroon-max", type=float, default=0.12)
    ap.add_argument("--min-frames", type=int, default=12)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    by_frame = defaultdict(list)
    for r in csv.DictReader(open(csv_path, newline="")):
        by_frame[int(r["frame"])].append((
            int(r["track_id"]),
            float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
            float(r["confidence"]),
        ))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stats = defaultdict(lambda: {"maroon": [], "white": [], "navy": [], "gold": []})
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        for (tid, x1, y1, x2, y2, _c) in by_frame.get(frame_idx, []):
            fr = region_fracs(frame, int(x1), int(y1), int(x2), int(y2))
            if fr is None:
                continue
            maroon, white, navy, gold = fr
            s = stats[tid]
            s["maroon"].append(maroon); s["white"].append(white)
            s["navy"].append(navy); s["gold"].append(gold)
    cap.release()

    # --- Score each track ---
    blue_gold_ids = set()
    report = []
    for tid, s in stats.items():
        nf = len(s["white"])
        maroon_med = float(np.median(s["maroon"])) if nf else 1.0
        wr = float((np.array(s["white"]) > args.white_pix).mean()) if nf else 0.0
        nr = float((np.array(s["navy"]) > args.navy_pix).mean()) if nf else 0.0
        gr = float((np.array(s["gold"]) > args.gold_pix).mean()) if nf else 0.0

        score = (args.w_white * (wr >= args.min_ratio)
                 + args.w_navy * (nr >= args.min_ratio)
                 + args.w_gold * (gr >= args.min_ratio))
        is_bg = (nf >= args.min_frames
                 and maroon_med < args.maroon_max
                 and score >= args.keep_score)
        if is_bg:
            blue_gold_ids.add(tid)
        report.append((tid, nf, wr, nr, gr, maroon_med, score, is_bg))

    report.sort(key=lambda r: (-r[7], -r[6], -r[1]))
    print(f"{'ID':>4} {'frames':>6} {'white%':>7} {'navy%':>6} {'gold%':>6} "
          f"{'maroon%':>8} {'score':>6}  team")
    for tid, nf, wr, nr, gr, mm, sc, bg in report:
        print(f"{tid:>4} {nf:>6} {wr*100:>6.0f}% {nr*100:>5.0f}% {gr*100:>5.0f}% "
              f"{mm*100:>7.1f} {sc:>6.1f}  {'BLUE&GOLD' if bg else '-'}")
    print("-" * 64)
    print(f"weights: white={args.w_white} navy={args.w_navy} gold={args.w_gold} "
          f"| keep-score={args.keep_score} | min-ratio={args.min_ratio}")
    print(f"pix thresholds: white>{args.white_pix} navy>{args.navy_pix} gold>{args.gold_pix} "
          f"| maroon veto>{args.maroon_max}")
    print(f"tracks total      : {len(report)}")
    print(f"tracks blue&gold  : {len(blue_gold_ids)}")
    print(f"blue&gold IDs     : {sorted(blue_gold_ids)}")

    if args.report_only:
        return

    out_csv = out_path.parent / f"{video_path.stem}_bluegold.csv"
    with open(out_csv, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"])
        for fr in sorted(by_frame):
            for (tid, x1, y1, x2, y2, conf) in by_frame[fr]:
                if tid in blue_gold_ids:
                    w_.writerow([fr, tid, f"{x1:.2f}", f"{y1:.2f}",
                                 f"{x2:.2f}", f"{y2:.2f}", f"{conf:.4f}"])

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    color = (0, 215, 255)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        for (tid, x1, y1, x2, y2, conf) in by_frame.get(frame_idx, []):
            if tid not in blue_gold_ids:
                continue
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, color, 2)
            label = f"ID {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
            cv2.putText(frame, label, (p1[0] + 2, p1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"filtered video -> {out_path}")
    print(f"filtered csv   -> {out_csv}")


if __name__ == "__main__":
    main()
