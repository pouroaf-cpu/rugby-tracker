"""mark_all.py - OVERNIGHT "mark every object" batch (NOT Qt).

Run it before bed; in the morning every player is already detected + tracked with
a persistent-ish ID and a team label, so grabbing YOURSELF (and ruling others
out) is much easier. It does NOT mask everyone (SAM is far too slow per object) -
masks stay for your target (live SAM2 / SAM3 re-track on a window).

Pipeline (reuses the proven v1 stack): YOLOv11m person detection at HIGH imgsz
(so far/packed players are found) -> BoT-SORT + OSNet ReID multi-object tracking
-> per-box TEAM label from the match calibration fingerprints. Output:

  output/<stem>.objects.csv   columns: frame, obj_id, x1, y1, x2, y2, team, conf
    team : your tracked team's name | "opp" | "unsure" | "" (no calibration)

HONEST LIMIT: at this resolution IDs FRAGMENT at occlusions/crossovers (expect
many IDs over a full game). This still does the grunt work - every player boxed
every frame, segments pre-linked, teams labelled - leaving only a handful of
re-anchors for the morning (loose-mark + SAM3 re-track).

Usage (PowerShell):
  python v2\\apps\\mark_all.py --video input\\chunk_028.mp4            # whole clip
  python v2\\apps\\mark_all.py --video input\\chunk_028.mp4 --imgsz 1280 --max-frames 200
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import pathlib

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rt2.paths import ProjectPaths
from rt2 import features, regions, numocr
from rt2.calibration import MatchCalibration

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
YOLO_WEIGHTS = ROOT / "models" / "yolo11m.pt"
REID_WEIGHTS = ROOT / "models" / "osnet_x1_0_msmt17.pt"
PERSON_CLASS = 0
DEVICE = "0"                 # GPU index (BoxMOT wants an index, NOT 'cuda')
TEAM_MARGIN_MIN = 0.02       # min fingerprint-distance gap to call a team confidently
OCR_EVERY = 8                # OCR each track at most every Nth frame (throttle)
OCR_MIN_H = 46               # only OCR boxes at least this tall (legibility)
MAX_OCR_PER_TRACK = 6        # STOP OCRing a track after this many reads (a few is
                             # plenty to vote a number; without this cap a full game
                             # is ~33h of OCR - with it, minutes)


def _load_teams(pp, video, override):
    """Tracked + opposition TeamProfiles with fingerprints, or [] if none."""
    cpath = pp.resolve_calibration(video, override)
    if not cpath:
        return []
    try:
        cal = MatchCalibration.load(cpath)
    except Exception:
        return []
    return [t for t in cal.teams if getattr(t, "fingerprint", None)]


def _team_label(frame, box, teams):
    """'<tracked-name>' | 'opp' | 'unsure' | '' for a player box."""
    if not teams:
        return ""
    try:
        vec = features.feature_vector(frame, box)
    except Exception:
        return "unsure"
    scored = sorted(((features.distance(vec, t.fingerprint), t) for t in teams),
                    key=lambda kv: kv[0])
    best_d, best_t = scored[0]
    if len(scored) >= 2 and (scored[1][0] - best_d) < TEAM_MARGIN_MIN:
        return "unsure"
    return best_t.name if getattr(best_t, "track", False) else "opp"


def _fmt_eta(s):
    s = int(max(0, s)); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def run(video_path, conf, imgsz, max_frames, calibration, use_ocr=False):
    pp = ProjectPaths().ensure()
    video_path = pathlib.Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(YOLO_WEIGHTS)
    if not REID_WEIGHTS.exists():
        raise FileNotFoundError(REID_WEIGHTS)

    teams = _load_teams(pp, video_path, calibration)
    print(f"[mark_all] calibration teams with fingerprints: "
          f"{[t.name for t in teams] or 'NONE (team labels disabled)'}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    from ultralytics import YOLO
    from boxmot.reid import ReID
    from boxmot.trackers import BotSort

    model = YOLO(str(YOLO_WEIGHTS))
    reid = ReID(weights=REID_WEIGHTS, device=DEVICE, half=True)
    tracker = BotSort(reid_model=reid.model, with_reid=True,
                      frame_rate=int(round(fps)))

    do_ocr = bool(use_ocr) and numocr.available()
    if use_ocr and not do_ocr:
        print("[mark_all] --ocr requested but no OCR backend available; skipping")
    elif do_ocr:
        print(f"[mark_all] OCR ON ({numocr.backend()}): reading jersey numbers per "
              "track; fills the 'number' column so the morning shortlist auto-IDs")

    out_path = pp.output / f"{video_path.stem}.objects.csv"
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["frame", "obj_id", "x1", "y1", "x2", "y2", "team", "conf", "number"])

    print(f"[mark_all] {video_path.name}: {total} frames @ {fps:.1f}fps, "
          f"imgsz={imgsz} -> {out_path.name}")
    ids = set(); n_rows = 0; t0 = time.time(); frame_idx = 0
    last_ocr = {}                       # obj_id -> last frame we OCR'd (throttle)
    ocr_count = {}                      # obj_id -> how many times we've OCR'd it
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break
            frame_idx += 1
            res = model.predict(frame, classes=[PERSON_CLASS], conf=conf,
                                imgsz=imgsz, device=DEVICE, half=True,
                                verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
                cl = res.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, cf, cl]).astype(np.float32)
            else:
                dets = np.empty((0, 6), np.float32)
            tracks = tracker.update(dets, frame)   # x1,y1,x2,y2,id,conf,cls,det_ind
            if tracks is not None and len(tracks):
                for tr in tracks:
                    x1, y1, x2, y2 = (float(v) for v in tr[:4])
                    tid = int(tr[4]); tconf = float(tr[5])
                    team = _team_label(frame, (x1, y1, x2, y2), teams)
                    number = ""
                    if (do_ocr and (y2 - y1) >= OCR_MIN_H
                            and ocr_count.get(tid, 0) < MAX_OCR_PER_TRACK
                            and frame_idx - last_ocr.get(tid, -OCR_EVERY) >= OCR_EVERY):
                        last_ocr[tid] = frame_idx
                        ocr_count[tid] = ocr_count.get(tid, 0) + 1
                        try:
                            rd = numocr.read_number(
                                regions.torso_patch(frame, (x1, y1, x2, y2)))
                            if rd:
                                number = rd[0]
                        except Exception:
                            pass
                    w.writerow([frame_idx, tid, f"{x1:.1f}", f"{y1:.1f}",
                                f"{x2:.1f}", f"{y2:.1f}", team, f"{tconf:.3f}",
                                number])
                    ids.add(tid); n_rows += 1
            if frame_idx % 100 == 0 or frame_idx == total:
                el = time.time() - t0
                fps_now = frame_idx / max(el, 1e-9)
                eta = (total - frame_idx) / max(fps_now, 1e-9)
                print(f"[mark_all] {frame_idx}/{total} "
                      f"({fps_now:.1f} fps, ETA {_fmt_eta(eta)}) "
                      f"ids={len(ids)} rows={n_rows}", flush=True)
    finally:
        f.close(); cap.release()
    el = time.time() - t0
    print(f"[mark_all] DONE {frame_idx} frames in {_fmt_eta(el)}; "
          f"{len(ids)} unique ids, {n_rows} rows -> {out_path}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Overnight mark-every-object batch")
    ap.add_argument("--video", required=True)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO detection confidence floor (default 0.25, lower than "
                         "prerender to catch more far players)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="YOLO inference size (default 1280 - finds far/packed "
                         "players the 640 default misses)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="process only the first N frames (quick test)")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--ocr", action="store_true",
                    help="OCR each track's jersey number into the 'number' column "
                         "(slower; lets the morning shortlist auto-identify by #). "
                         "Needs the tesseract binary.")
    args = ap.parse_args(argv)
    run(args.video, args.conf, args.imgsz, args.max_frames or None,
        args.calibration, use_ocr=args.ocr)


if __name__ == "__main__":
    main()
