"""
export.py - Phase 6 EXPORT pipeline (batch CLI, not Qt).

Takes the single-target track (Phase 5 output, tracks.csv) and produces three
deliverables for one rugby player followed through wide Veo footage:

  OUTPUT 1  output/<stem>.annotated.mp4
      Full-length video. On frames where the target is present, draw a thick
      distinct bbox + ID label + an optional fading trail of the last N centres.

  OUTPUT 2  output/<stem>.clips.mp4
      A REEL of ONLY the frames where the target is on-screen. Contiguous
      present-segments (a gap of more than a few frames splits a segment) are
      padded by --pad seconds each side (clamped to video bounds) and
      concatenated, target boxed. This is the "2-3 min reel of touches".

  OUTPUT 3  stats
      Printed to console AND written to output/<stem>.stats.json:
      total frames, tracked frames, coverage %, gaps, longest gap (frames+s),
      clip-segment count, reel duration (s), and a per-source breakdown.

Frame numbers are 1-based throughout (matching VideoReader.frame and the CSV).

Target marker styles (--marker), all scaled by --marker-size (default 1.0):
  arrow  downward chevron hovering just above the head (default)
  ring   broadcast-style circle on the ground at the player's feet (a
         flattened ellipse centred on the bottom of the box)
  dot    a minimal small filled circle just above the head
  box    a thick bounding box + "TARGET <id>" label
  both   box + arrow
  adaptive  confidence-adaptive marker mirroring the live tracker: a tight green
         body ellipse at high confidence (>=0.7), a larger amber ellipse that
         grows as confidence falls (0.4-0.7), and a growing red dashed search
         circle at low confidence (<0.4). Driven by the per-frame "confidence"
         column in tracks.csv.

When --player <name> is given, the export targets that player's namespaced
track and writes namespaced outputs (<stem>.<player>.annotated.mp4,
<stem>.<player>.clips.mp4, <stem>.<player>.stats.json). Without --player the
behaviour is unchanged (the default un-namespaced track and outputs).

Usage:
  python v2\\apps\\export.py --video input\\chunk_028.mp4
  python v2\\apps\\export.py --video input\\chunk_028.mp4 --player frank
  python v2\\apps\\export.py --video input\\chunk_028.mp4 --trail 25 --pad 0.5
  python v2\\apps\\export.py --video input\\chunk_028.mp4 --marker ring --marker-size 1.4
  python v2\\apps\\export.py --video input\\chunk_028.mp4 --marker dot --marker-size 0.7
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from collections import Counter

import cv2
import numpy as np

from rt2.paths import ProjectPaths
from rt2.video import VideoReader
from rt2 import bbox
from rt2.tracks_io import read_tracks


# A gap of more than this many frames between consecutive present frames splits
# a contiguous on-screen segment in two (small dropouts are bridged).
SEGMENT_GAP_FRAMES = 5

TARGET_COLOR = (0, 215, 255)   # BGR - bright amber, distinct from track.py's per-ID colours
TRAIL_COLOR = (0, 215, 255)    # trail fades via alpha against this colour
BOX_THICK = 3


def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def draw_arrow(frame, box, marker_size=1.0):
    """Draw a downward chevron hovering just above the target's head, so a
    reviewer (or an AI) can follow the player at a glance. Sized to the box but
    floored for visibility on small/distant players; bright fill + dark outline.
    `marker_size` scales the chevron width/height and the gap above the head."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    bw, bh = x2 - x1, y2 - y1
    aw = int(max(20, 0.7 * bw) * marker_size)   # arrow width
    ah = int(max(24, 0.9 * bw) * marker_size)   # arrow height
    gap = int(max(8, 0.10 * bh) * marker_size)  # gap above the head
    aw = max(2, aw); ah = max(2, ah)
    tip_y = int(y1 - gap)                 # bottom point (aimed at the head)
    top_y = tip_y - ah
    if top_y < 2:                         # clamp so it stays on-screen near the top
        shift = 2 - top_y
        tip_y += shift; top_y += shift
    h, w = frame.shape[:2]
    cx = max(aw // 2 + 1, min(cx, w - aw // 2 - 1))
    pts = np.array([(cx, tip_y), (cx - aw // 2, top_y), (cx + aw // 2, top_y)], np.int32)
    cv2.fillConvexPoly(frame, pts, TARGET_COLOR, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2, lineType=cv2.LINE_AA)


def draw_ground_ring(frame, box, marker_size=1.0):
    """Draw a broadcast-style circle on the ground at the player's feet: a
    flattened ellipse centred on the bottom-centre of the box (cx, y2). Width
    scales with the box width and `marker_size`; height is ~0.38 of the width so
    it reads as a ring lying on the pitch. A dark thin outline is drawn first,
    then a bright TARGET_COLOR outline on top."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    bw = x2 - x1
    rw = int(max(8, 1.1 * bw * marker_size) / 2)   # horizontal radius
    rh = int(max(3, 0.38 * (rw * 2)) / 2)          # vertical radius (flattened)
    rw = max(2, rw); rh = max(1, rh)
    axes = (rw, rh)
    # dark backing outline for contrast, then bright ring on top
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (0, 0, 0), 4, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, TARGET_COLOR, 2, lineType=cv2.LINE_AA)


def draw_dot(frame, box, marker_size=1.0):
    """Draw a minimal marker: a small filled TARGET_COLOR circle just above the
    head, with a thin dark edge. Radius scales with `marker_size` (min 4)."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    bh = y2 - y1
    r = max(4, int(6 * marker_size))
    gap = int(max(8, 0.10 * bh) * marker_size)
    cy = int(y1 - gap - r)
    h, w = frame.shape[:2]
    cy = max(r + 1, cy)                   # clamp so it stays on-screen near the top
    cx = max(r + 1, min(cx, w - r - 1))
    cv2.circle(frame, (cx, cy), r, TARGET_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (0, 0, 0), 2, lineType=cv2.LINE_AA)


def draw_adaptive(frame, box, confidence, marker_size=1.0):
    """Confidence-adaptive marker mirroring the live tracker's overlay. The
    geometry reacts to the per-frame tracking `confidence` (0..1):

      conf >= 0.7  TIGHT body ellipse around the box centre (axes ~0.55*w,
                   ~0.50*h), GREEN - locked on.
      0.4 - 0.7    LARGER ellipse, axes lerp from the tight high-conf size at
                   0.7 up to ~1.3x the box at 0.4, AMBER - drifting.
      < 0.4        DASHED CIRCLE (search region) centred on the box, radius
                   growing from ~0.8*box_diag at 0.4 up to ~2.5*box_diag near 0,
                   RED - searching.

    All geometry is scaled by `marker_size`. A thin dark backing outline is drawn
    first for contrast, then the coloured outline on top (cv2.LINE_AA)."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    bw = x2 - x1
    bh = y2 - y1
    diag = (bw ** 2 + bh ** 2) ** 0.5
    conf = max(0.0, min(1.0, confidence))

    if conf >= 0.7:
        # tight body ellipse, green
        rw = max(2, int(0.55 * bw * marker_size))
        rh = max(2, int(0.50 * bh * marker_size))
        colour = (0, 200, 0)
        cv2.ellipse(frame, (cx, cy), (rw, rh), 0, 0, 360, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (rw, rh), 0, 0, 360, colour, 2, lineType=cv2.LINE_AA)
        return

    if conf >= 0.4:
        # larger ellipse, axes lerp tight(@0.7) -> ~1.3x box(@0.4), amber
        t = (0.7 - conf) / 0.3                 # 0 at conf 0.7, 1 at conf 0.4
        t = max(0.0, min(1.0, t))
        fw = 0.55 + t * (1.30 - 0.55)          # half-width fraction of bw
        fh = 0.50 + t * (1.30 - 0.50)          # half-height fraction of bh
        rw = max(2, int(fw * bw * marker_size))
        rh = max(2, int(fh * bh * marker_size))
        colour = (0, 170, 255)
        cv2.ellipse(frame, (cx, cy), (rw, rh), 0, 0, 360, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (rw, rh), 0, 0, 360, colour, 2, lineType=cv2.LINE_AA)
        return

    # conf < 0.4: dashed search circle, red, growing as conf -> 0
    # radius ~0.8*diag at conf 0.4 up to ~2.5*diag near conf 0
    t = (0.4 - conf) / 0.4                      # 0 at conf 0.4, 1 at conf 0
    t = max(0.0, min(1.0, t))
    radius = max(2, int((0.8 + t * (2.5 - 0.8)) * diag * marker_size))
    colour = (0, 0, 255)
    # dashed: alternating drawn/skipped arc segments
    seg = 18          # degrees drawn
    gap = 14          # degrees skipped
    a = 0
    while a < 360:
        end = min(360, a + seg)
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, a, end, (0, 0, 0), 4,
                    lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, a, end, colour, 2,
                    lineType=cv2.LINE_AA)
        a += seg + gap


def draw_target(frame, box, target_id, trail, marker="arrow", marker_size=1.0,
                confidence=1.0):
    """Draw the fading trail (oldest->newest), then the chosen target marker."""
    # --- trail: list of (cx, cy), oldest first. Fade alpha old->new. ---
    n = len(trail)
    for i, (cx, cy) in enumerate(trail):
        alpha = (i + 1) / n if n else 1.0          # 0..1, newest brightest
        radius = max(2, int(2 + 4 * alpha))
        overlay = frame.copy()
        cv2.circle(overlay, (int(cx), int(cy)), radius, TRAIL_COLOR, -1,
                   lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6 * alpha, frame, 1 - 0.6 * alpha, 0, frame)

    x1, y1, x2, y2 = box
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))

    if marker == "adaptive":
        draw_adaptive(frame, box, confidence, marker_size)
        return
    if marker == "ring":
        draw_ground_ring(frame, box, marker_size)
        return
    if marker == "dot":
        draw_dot(frame, box, marker_size)
        return

    if marker in ("box", "both"):
        cv2.rectangle(frame, p1, p2, TARGET_COLOR, BOX_THICK, lineType=cv2.LINE_AA)
        label = f"TARGET {target_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ly = max(th + 6, p1[1])
        cv2.rectangle(frame, (p1[0], ly - th - 6), (p1[0] + tw + 6, ly), TARGET_COLOR,
                      -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, label, (p1[0] + 3, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2, lineType=cv2.LINE_AA)

    if marker in ("arrow", "both"):
        draw_arrow(frame, box, marker_size)


def build_segments(present_frames, n_frames, pad_frames):
    """present_frames: sorted list of 1-based frames where target is on-screen.

    Returns (raw_segments, padded_segments) where each is a list of (start, end)
    inclusive 1-based frame ranges. raw = contiguous present runs (bridging gaps
    up to SEGMENT_GAP_FRAMES); padded = raw expanded by pad_frames and clamped,
    then merged where padding causes overlap.
    """
    if not present_frames:
        return [], []

    raw = []
    s = prev = present_frames[0]
    for f in present_frames[1:]:
        if f - prev > SEGMENT_GAP_FRAMES:
            raw.append((s, prev))
            s = f
        prev = f
    raw.append((s, prev))

    # pad + clamp
    padded = [(max(1, a - pad_frames), min(n_frames, b + pad_frames))
              for (a, b) in raw]

    # merge overlapping / touching padded ranges
    merged = []
    for a, b in padded:
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return raw, merged


def main():
    ap = argparse.ArgumentParser(description="Phase 6 export: annotated video + touch reel + stats")
    ap.add_argument("--video", default=None,
                    help="source video (default: first ProjectPaths().videos())")
    ap.add_argument("--player", default=None,
                    help="target a specific player's namespaced track + outputs "
                         "(e.g. --player frank uses <stem>.frank.* ); "
                         "default None = the un-namespaced track/outputs")
    ap.add_argument("--trail", type=int, default=25,
                    help="number of recent target centres to draw as a fading trail (0 = off)")
    ap.add_argument("--pad", type=float, default=0.5,
                    help="padding in seconds around each on-screen segment in the clip reel")
    ap.add_argument("--marker", choices=["arrow", "box", "both", "ring", "dot", "adaptive"],
                    default="arrow",
                    help="target marker style: 'arrow' chevron above the head (default), "
                         "'ring' ground circle at the feet, 'dot' small circle above the head, "
                         "'box' bounding box + label, 'both' box + arrow, "
                         "'adaptive' confidence-adaptive marker (green tight ellipse high-conf, "
                         "amber growing ellipse mid, red dashed search circle low-conf)")
    ap.add_argument("--marker-size", type=float, default=1.0,
                    help="scale factor for the marker geometry (arrow/ring/dot); "
                         "<1 smaller, >1 larger (default 1.0)")
    ap.add_argument("--max-frame", type=int, default=None,
                    help=argparse.SUPPRESS)  # smoke-test cap; render only up to this 1-based frame
    args = ap.parse_args()

    pp = ProjectPaths()

    if args.video:
        video = pathlib.Path(args.video).resolve()
    else:
        vids = pp.videos()
        if not vids:
            raise SystemExit(f"[export] no videos found in {pp.input}")
        video = vids[0]
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    tracks_path = pp.tracks(video, args.player)
    if not tracks_path.exists():
        if args.player:
            raise SystemExit(
                f"[export] tracks CSV not found: {tracks_path}\n"
                f"          no track for player '{args.player}' on this clip - "
                f"run the tracker with --player {args.player} first.")
        raise SystemExit(f"[export] tracks CSV not found: {tracks_path}\n"
                         f"          run the Phase 5 re-anchor pipeline first.")

    annotated_path = pp.annotated(video, args.player)
    clips_path = pp.clips(video, args.player)
    stats_path = (pp.output / f"{video.stem}.{args.player}.stats.json"
                  if args.player else pp.output / f"{video.stem}.stats.json")
    pp.output.mkdir(parents=True, exist_ok=True)

    # --- Load target track, key boxes by frame ---
    rows = read_tracks(tracks_path)
    box_by_frame = {}        # frame -> (x1,y1,x2,y2)
    source_by_frame = {}     # frame -> source
    conf_by_frame = {}       # frame -> per-frame confidence (0..1, default 1.0)
    target_ids = set()
    for r in rows:
        f = r["frame"]
        box_by_frame[f] = (r["x1"], r["y1"], r["x2"], r["y2"])
        source_by_frame[f] = r["source"]
        conf_by_frame[f] = r.get("confidence", 1.0)
        target_ids.add(r["target_id"])
    target_id = sorted(target_ids)[0] if target_ids else 0

    vr = VideoReader(video)
    fps = vr.fps or 29.97
    n_frames = vr.n_frames
    width, height = vr.width, vr.height

    last_frame = n_frames if args.max_frame is None else min(n_frames, args.max_frame)

    present_frames = sorted(f for f in box_by_frame if 1 <= f <= last_frame)
    pad_frames = int(round(args.pad * fps))
    raw_segments, segments = build_segments(present_frames, last_frame, pad_frames)

    print(f"[export] source    : {video}")
    print(f"[export] tracks     : {tracks_path}")
    print(f"[export] video      : {width}x{height} @ {fps:.2f} fps, {n_frames} frames"
          + (f" (capped at {last_frame})" if last_frame != n_frames else ""))
    print(f"[export] target id  : {target_id}")
    print(f"[export] present    : {len(present_frames)} frames")
    print(f"[export] trail      : {args.trail}   pad : {args.pad}s ({pad_frames} frames)")
    marker_line = f"[export] marker     : {args.marker}   size : {args.marker_size}"
    if args.marker == "adaptive":
        present_confs = [conf_by_frame.get(f, 1.0) for f in present_frames]
        if present_confs:
            marker_line += (f"   conf range : {min(present_confs):.3f}-{max(present_confs):.3f}"
                            f" (>=0.7 green / 0.4-0.7 amber / <0.4 red search)")
        else:
            marker_line += "   conf range : n/a (>=0.7 green / 0.4-0.7 amber / <0.4 red search)"
    print(marker_line)
    print(f"[export] segments   : {len(raw_segments)} raw -> {len(segments)} padded/merged")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # =========================================================================
    # OUTPUT 1 - annotated.mp4 (full length, the long pass)
    # =========================================================================
    ann_writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
    if not ann_writer.isOpened():
        vr.release()
        raise RuntimeError(f"Could not open VideoWriter for: {annotated_path}")

    trail = []   # list of recent target centres (cx, cy), oldest first
    t_start = time.time()
    drawn = 0
    try:
        for fidx in range(1, last_frame + 1):
            frame = vr.frame(fidx)
            if frame is None:
                break

            box = box_by_frame.get(fidx)
            if box is not None:
                trail.append(bbox.center(box))
                if args.trail > 0:
                    while len(trail) > args.trail:
                        trail.pop(0)
                else:
                    trail = trail[-1:]   # keep nothing to draw when trail off
                draw_trail = trail if args.trail > 0 else []
                draw_target(frame, box, target_id, draw_trail, args.marker, args.marker_size,
                            confidence=conf_by_frame.get(fidx, 1.0))
                drawn += 1
            else:
                # target absent this frame: let the trail decay so it doesn't
                # linger forever across long gaps
                if trail:
                    trail.pop(0)

            ann_writer.write(frame)

            if fidx % 100 == 0 or fidx == last_frame:
                elapsed = time.time() - t_start
                rate = fidx / elapsed if elapsed > 0 else 0.0
                remaining = (last_frame - fidx) / rate if rate > 0 else 0.0
                print(f"[export] annotated {fidx}/{last_frame}  "
                      f"({rate:.1f} FPS, ETA {fmt_eta(remaining)})")
    finally:
        ann_writer.release()

    print(f"[export] annotated  -> {annotated_path}  ({drawn} boxed frames)")

    # =========================================================================
    # OUTPUT 2 - clips.mp4 (reel of on-screen segments only)
    # =========================================================================
    clip_writer = cv2.VideoWriter(str(clips_path), fourcc, fps, (width, height))
    if not clip_writer.isOpened():
        vr.release()
        raise RuntimeError(f"Could not open VideoWriter for: {clips_path}")

    reel_frames = 0
    try:
        for si, (a, b) in enumerate(segments, 1):
            seg_trail = []
            for fidx in range(a, b + 1):
                frame = vr.frame(fidx)
                if frame is None:
                    break
                box = box_by_frame.get(fidx)
                if box is not None:
                    seg_trail.append(bbox.center(box))
                    if args.trail > 0:
                        while len(seg_trail) > args.trail:
                            seg_trail.pop(0)
                    draw_trail = seg_trail if args.trail > 0 else []
                    draw_target(frame, box, target_id, draw_trail, args.marker, args.marker_size,
                                confidence=conf_by_frame.get(fidx, 1.0))
                else:
                    if seg_trail:
                        seg_trail.pop(0)
                clip_writer.write(frame)
                reel_frames += 1
            print(f"[export] reel seg {si}/{len(segments)}  frames {a}-{b}  "
                  f"({reel_frames} written)")
    finally:
        clip_writer.release()
        vr.release()

    reel_duration = reel_frames / fps if fps else 0.0
    print(f"[export] clips      -> {clips_path}  "
          f"({reel_frames} frames, {reel_duration:.1f}s)")

    # =========================================================================
    # OUTPUT 3 - stats (console + json)
    # =========================================================================
    tracked = len(present_frames)
    coverage = (tracked / last_frame * 100.0) if last_frame else 0.0

    # gaps = runs of absent frames strictly inside [first_present, last_present]
    gaps = []
    if present_frames:
        for prev, nxt in zip(present_frames, present_frames[1:]):
            if nxt - prev > 1:
                gaps.append(nxt - prev - 1)
    n_gaps = len(gaps)
    longest_gap_frames = max(gaps) if gaps else 0
    longest_gap_seconds = longest_gap_frames / fps if fps else 0.0

    source_breakdown = dict(Counter(
        source_by_frame[f] for f in present_frames
    ))

    stats = {
        "video": str(video),
        "fps": round(fps, 4),
        "target_id": target_id,
        "total_frames": last_frame,
        "tracked_frames": tracked,
        "coverage_pct": round(coverage, 2),
        "num_gaps": n_gaps,
        "longest_gap_frames": longest_gap_frames,
        "longest_gap_seconds": round(longest_gap_seconds, 2),
        "num_clip_segments": len(segments),
        "clip_reel_frames": reel_frames,
        "clip_reel_duration_s": round(reel_duration, 2),
        "source_breakdown": source_breakdown,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("-" * 60)
    print(f"[stats] total frames     : {stats['total_frames']}")
    print(f"[stats] tracked frames   : {stats['tracked_frames']}")
    print(f"[stats] coverage         : {stats['coverage_pct']:.2f}%")
    print(f"[stats] gaps             : {stats['num_gaps']}")
    print(f"[stats] longest gap      : {stats['longest_gap_frames']} frames "
          f"({stats['longest_gap_seconds']:.2f}s)")
    print(f"[stats] clip segments    : {stats['num_clip_segments']}")
    print(f"[stats] reel duration    : {stats['clip_reel_duration_s']:.2f}s")
    print(f"[stats] source breakdown : {stats['source_breakdown']}")
    print(f"[stats] json             -> {stats_path}")
    print("-" * 60)
    print(f"[export] DONE")


if __name__ == "__main__":
    main()
