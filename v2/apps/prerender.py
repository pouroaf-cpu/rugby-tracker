"""prerender.py - Phase 3 PRE-RENDER pipeline (batch CLI, NOT Qt).

Runs YOLOv11m person detection across a FULL Veo match video and scores every
detection box against the video's MatchCalibration (learned team colour
signatures + a perspective height model). Every detection is recorded -- kept
ones (passing the colour + height filters) AND rejected ones (kept=False) --
so nothing is lost downstream.

Output: output/<stem>.detections.parquet
  columns: frame, x1, y1, x2, y2, conf, team_score, kept

Usage:
  python v2\\apps\\prerender.py --video input\\chunk_028.mp4
  python v2\\apps\\prerender.py --video input\\chunk_028.mp4 --conf 0.3 \\
         --colour-thresh 0.05 --height-tol 3.0
"""
from __future__ import annotations

import argparse
import sys
import pathlib
import time

import cv2
import numpy as np

# Make the rt2 shared lib importable (v2/ is this file's grandparent).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rt2.paths import ProjectPaths
from rt2.video import VideoReader
from rt2 import bbox, regions, field, features  # noqa: F401  (bbox/regions parity)
from rt2.calibration import MatchCalibration
from rt2.detections import write_detections

# Project root = three levels up from this file: v2/apps/prerender.py -> root.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
YOLO_WEIGHTS = ROOT / "models" / "yolo11m.pt"

PERSON_CLASS = 0          # COCO 'person'
DEVICE = "0"              # GPU index 0 (RTX 3060)

# Team assignment uses perceptual fingerprints (rt2.features): each detection is
# matched to the nearest team fingerprint by L1 distance. Learned HSV signatures
# don't separate teams at this resolution (skin dominates) -- fingerprints do.


def height_ok(box, teams, tol: float) -> bool:
    """True if ANY tracked team's height model finds the box plausible.

    A team with no height model is treated as OK (does not veto).
    """
    x1, y1, x2, y2 = box
    center_y = 0.5 * (y1 + y2)
    box_h = y2 - y1
    for t in teams:
        hm = getattr(t, "height", None)
        if hm is None:
            return True
        try:
            if hm.plausible(center_y, box_h, tol=tol):
                return True
        except Exception:
            return True
    return False


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def run(video_path: pathlib.Path, conf: float, max_dist: float,
        height_tol: float, max_frames: int | None = None,
        use_field: bool = True, feet_pad: int = 10, margin: float = 0.03,
        calibration=None, imgsz: int = 640) -> pathlib.Path:
    pp = ProjectPaths()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")

    calib_path = pp.resolve_calibration(video_path, calibration)
    if calib_path is None:
        raise FileNotFoundError(
            "No calibration for this clip or game. Run the calibration tool "
            "first (it also saves a shared game.calibration.json).")
    calib = MatchCalibration.load(calib_path)
    tracked = [t for t in calib.teams if t.track]
    others = [t for t in calib.teams if not t.track]
    if not tracked:
        print("[prerender] WARNING: calibration has no tracked teams; "
              "nothing will be kept.")

    out_path = pp.detections(video_path)

    # VideoReader gives us authoritative geometry + 1-based frame indexing.
    vr = VideoReader(video_path)
    total_frames = vr.n_frames
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    print(f"[prerender] source      : {video_path}")
    print(f"[prerender] calibration : {calib_path}")
    print(f"[prerender] output      : {out_path}")
    print(f"[prerender] video       : {vr.width}x{vr.height} @ {vr.fps:.2f} fps, "
          f"{vr.n_frames} frames")
    have_fp = [t.name for t in calib.teams if t.fingerprint]
    print(f"[prerender] track       : {[t.name for t in tracked]}")
    print(f"[prerender] vs other    : {[t.name for t in others] or '(none - single-team mode)'}")
    print(f"[prerender] fingerprints: {have_fp or '(none - calibration has no fingerprints!)'}")
    print(f"[prerender] params      : conf={conf} margin={margin} max_dist={max_dist} "
          f"height_tol={height_tol} field_mask={use_field}"
          + (f" (LIMITED to first {max_frames} frames)" if max_frames else ""))

    # Import here so --help stays instant (matches scripts/track.py).
    from ultralytics import YOLO
    model = YOLO(str(YOLO_WEIGHTS))

    # Sequential decode (cv2) -- same frames VideoReader.frame() would return,
    # but fast for a full pass. We close VideoReader's cap to avoid two readers.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        vr.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    rows = []
    kept_count = 0
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1                      # 1-based, matches VideoReader
            if max_frames is not None and frame_idx > max_frames:
                frame_idx -= 1
                break

            results = model.predict(
                frame, classes=[PERSON_CLASS], conf=conf, imgsz=imgsz,
                device=DEVICE, half=True, verbose=False,
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                # field mask once per frame (only when needed); follows the pan
                fmask = field.field_mask(frame) if use_field else None
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    box = (float(x1), float(y1), float(x2), float(y2))
                    fv = features.feature_vector(frame, box)
                    bt, bo = features.classify(fv, calib.teams)   # L1 distances
                    h_ok = height_ok(box, tracked, height_tol)
                    field_ok = (fmask is None) or field.feet_in_field(fmask, box, feet_pad)
                    if bt == float("inf"):
                        team_score, colour_ok = 0.0, False        # no tracked fingerprint
                    elif bo == float("inf"):
                        team_score = -bt                          # single-team: distance gate
                        colour_ok = bt <= max_dist
                    else:
                        team_score = bo - bt                      # >0 => closer to your team
                        colour_ok = team_score >= margin
                    kept = bool(colour_ok and h_ok and field_ok)
                    if kept:
                        kept_count += 1
                    rows.append({
                        "frame": frame_idx,
                        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                        "conf": float(c),
                        "team_score": float(team_score),
                        "kept": kept,
                    })

            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                fps_now = frame_idx / elapsed if elapsed > 0 else 0.0
                remaining = total_frames - frame_idx
                eta = remaining / fps_now if fps_now > 0 else 0.0
                pct = 100.0 * frame_idx / total_frames if total_frames else 0.0
                print(f"[prerender] frame {frame_idx}/{total_frames} ({pct:4.1f}%)  "
                      f"{fps_now:5.1f} FPS  dets={len(rows)} kept={kept_count}  "
                      f"elapsed={_fmt_eta(elapsed)} ETA={_fmt_eta(eta)}")
    finally:
        cap.release()
        vr.release()

    write_detections(out_path, rows)

    elapsed = time.time() - t_start
    fps_avg = frame_idx / elapsed if elapsed > 0 else 0.0
    print("-" * 64)
    print(f"[prerender] DONE  frames={frame_idx}  detections={len(rows)}  "
          f"kept={kept_count}  rejected={len(rows) - kept_count}")
    print(f"[prerender] wall={_fmt_eta(elapsed)} ({elapsed:.1f}s)  "
          f"avg={fps_avg:.1f} FPS")
    print(f"[prerender] detections -> {out_path}")
    return out_path


def main(argv=None):
    pp = ProjectPaths()
    vids = pp.videos()
    default_video = str(vids[0]) if vids else None

    ap = argparse.ArgumentParser(
        description="Phase 3 pre-render: YOLO detect + team/height filter -> "
                    "detections.parquet")
    ap.add_argument("--video", default=default_video,
                    help="path to source video "
                         "(default: first ProjectPaths().videos())")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="YOLO detection confidence floor (default 0.3)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO inference image size. The Veo shot is 1920 wide and "
                         "players are tiny; raise to 1280/1536 to detect far/packed "
                         "players the default 640 misses (slower). (default 640)")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="min fingerprint-distance margin (opposition - your team) "
                         "to keep, in 2-team mode (default 0.03)")
    ap.add_argument("--max-dist", type=float, default=0.6,
                    help="single-team mode: max fingerprint distance to keep (default 0.6)")
    ap.add_argument("--height-tol", type=float, default=3.0,
                    help="height-model tolerance in resid_std units (default 3.0)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only the first N frames (smoke test only)")
    ap.add_argument("--field-mask", action=argparse.BooleanOptionalAction, default=True,
                    help="reject detections whose feet are off the pitch "
                         "(--no-field-mask to disable; default on)")
    ap.add_argument("--feet-pad", type=int, default=10,
                    help="tolerance (px) around the feet point for the field test")
    ap.add_argument("--calibration", default=None,
                    help="explicit calibration JSON; otherwise uses this clip's, "
                         "then the shared game.calibration.json")
    args = ap.parse_args(argv)

    if not args.video:
        ap.error("no --video given and no videos found in input/")

    run(pathlib.Path(args.video).resolve(),
        conf=args.conf,
        max_dist=args.max_dist,
        height_tol=args.height_tol,
        max_frames=args.max_frames,
        use_field=args.field_mask,
        feet_pad=args.feet_pad,
        margin=args.margin,
        calibration=args.calibration,
        imgsz=args.imgsz)


if __name__ == "__main__":
    main()
