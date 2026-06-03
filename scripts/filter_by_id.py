"""
filter_by_id.py - Re-render a video keeping ONLY one track ID's bbox.

Manual re-anchor workflow: scrub the annotated video, find the track ID you
want, then run this to produce a video showing only that player's box (pulled
from the tracks CSV).

  --video   the video to render onto. Pass the ORIGINAL clean source clip for a
            truly clean single-player result (recommended). Passing the annotated
            video will highlight the target on top of the existing boxes.
  --csv     the *_tracks.csv produced by track.py
  --id      the single track ID to keep
  --output  output video path

Single-ID by design: if your ID changes after a scrum/sub, run this again with
the new ID and stitch the segments together afterwards.

Usage:
  python scripts\\filter_by_id.py --video input\\test.mp4 --csv output\\test_tracks.csv --id 7 --output output\\player7.mp4
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(description="Keep only one track ID's bbox in a re-rendered video")
    ap.add_argument("--video", required=True, help="video to render onto (original source recommended)")
    ap.add_argument("--csv", required=True, help="tracks CSV from track.py")
    ap.add_argument("--id", type=int, required=True, help="track ID to keep")
    ap.add_argument("--output", required=True, help="output video path")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    target = args.id

    # --- Load only the target ID's boxes, keyed by frame number ---
    boxes_by_frame = defaultdict(list)  # frame -> list of (x1,y1,x2,y2,conf)
    rows_kept = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r["track_id"]) != target:
                continue
            frame = int(r["frame"])
            boxes_by_frame[frame].append((
                float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                float(r["confidence"]),
            ))
            rows_kept += 1

    if rows_kept == 0:
        raise SystemExit(f"[filter] track_id {target} not found in {csv_path.name}. "
                         f"Nothing to render.")

    frames_present = sorted(boxes_by_frame.keys())
    print(f"[filter] target ID : {target}")
    print(f"[filter] rows kept : {rows_kept} across {len(frames_present)} frames "
          f"(first={frames_present[0]}, last={frames_present[-1]})")

    # --- Open video / writer ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    color = (0, 255, 0)  # bright green for the chosen player
    frame_idx = 0
    drawn = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1  # track.py numbers the first frame as 1

            for (x1, y1, x2, y2, conf) in boxes_by_frame.get(frame_idx, []):
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, color, 3)
                label = f"ID {target} ({conf:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (p1[0], p1[1] - th - 6),
                              (p1[0] + tw + 4, p1[1]), color, -1)
                cv2.putText(frame, label, (p1[0] + 2, p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                drawn += 1

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"[filter] frames written : {frame_idx}")
    print(f"[filter] boxes drawn    : {drawn}")
    print(f"[filter] output -> {out_path}")


if __name__ == "__main__":
    main()
