"""
track.py - YOLOv11m + BoT-SORT (OSNet ReID) player tracker.

Detects people (class 0) with YOLOv11m and tracks them with BoxMOT's BoT-SORT,
which uses ByteTrack-style association + camera-motion compensation + OSNet ReID
appearance matching for identity persistence through occlusion/scrums.

Outputs:
  - annotated video (bbox + track ID per player)
  - <videoname>_tracks.csv : frame, track_id, x1, y1, x2, y2, confidence

Usage:
  python scripts\\track.py --video input\\test.mp4 --output output\\test_tracked.mp4
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

# Project root = parent of this script's folder
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
YOLO_WEIGHTS = MODELS / "yolo11m.pt"
REID_WEIGHTS = MODELS / "osnet_x1_0_msmt17.pt"

PERSON_CLASS = 0          # COCO 'person'
DET_CONF = 0.3            # detection confidence floor
DEVICE = "0"             # GPU index 0 (RTX 3060). BoxMOT wants an index, not 'cuda'.


def color_for_id(track_id: int):
    """Deterministic BGR color per track ID."""
    rng = (track_id * 47) % 256, (track_id * 97) % 256, (track_id * 151) % 256
    return int(rng[0]), int(rng[1]), int(rng[2])


def main():
    ap = argparse.ArgumentParser(description="YOLOv11m + BoT-SORT/OSNet player tracker")
    ap.add_argument("--video", required=True, help="path to source video")
    ap.add_argument("--output", required=True, help="path to annotated output video")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
    if not REID_WEIGHTS.exists():
        raise FileNotFoundError(f"ReID weights not found: {REID_WEIGHTS}")

    # CSV sits next to the output video, named after the SOURCE video.
    csv_path = out_path.parent / f"{video_path.stem}_tracks.csv"

    # --- Open video ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    print(f"[track] source : {video_path}")
    print(f"[track] output : {out_path}")
    print(f"[track] csv    : {csv_path}")
    print(f"[track] video  : {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # --- Models (imported here so --help is instant) ---
    from ultralytics import YOLO
    from boxmot.reid import ReID
    from boxmot.trackers import BotSort

    model = YOLO(str(YOLO_WEIGHTS))

    reid = ReID(weights=REID_WEIGHTS, device=DEVICE, half=True)
    tracker = BotSort(
        reid_model=reid.model,      # OSNet backend exposing .get_features()
        with_reid=True,
        frame_rate=int(round(fps)),
    )

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"])

    frame_idx = 0
    unique_ids = set()
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # --- Detect people only ---
            results = model.predict(
                frame, classes=[PERSON_CLASS], conf=DET_CONF,
                device=DEVICE, half=True, verbose=False,
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
                cls = results.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, conf, cls]).astype(np.float32)  # N x 6
            else:
                dets = np.empty((0, 6), dtype=np.float32)

            # --- Track ---
            tracks = tracker.update(dets, frame)  # cols: x1,y1,x2,y2,id,conf,cls,det_ind

            if tracks is not None and len(tracks) > 0:
                for row in np.asarray(tracks):
                    x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                    tid = int(row[4])
                    tconf = float(row[5])
                    unique_ids.add(tid)

                    csv_writer.writerow([
                        frame_idx, tid,
                        f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                        f"{tconf:.4f}",
                    ])

                    # draw
                    c = color_for_id(tid)
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(frame, p1, p2, c, 2)
                    label = f"ID {tid}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (p1[0], p1[1] - th - 6),
                                  (p1[0] + tw + 4, p1[1]), c, -1)
                    cv2.putText(frame, label, (p1[0] + 2, p1[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.write(frame)

            # --- Progress every 100 frames ---
            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                fps_now = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"[track] frame {frame_idx}/{total_frames}  "
                      f"({fps_now:.1f} FPS, {len(unique_ids)} IDs so far)")
    finally:
        cap.release()
        writer.release()
        csv_file.close()

    elapsed = time.time() - t_start
    fps_avg = frame_idx / elapsed if elapsed > 0 else 0.0
    print("-" * 60)
    print(f"[track] DONE  frames={frame_idx}  unique_ids={len(unique_ids)}  "
          f"time={elapsed:.1f}s  avg={fps_avg:.1f} FPS")
    print(f"[track] video -> {out_path}")
    print(f"[track] csv   -> {csv_path}")


if __name__ == "__main__":
    main()
