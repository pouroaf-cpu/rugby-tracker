"""sam3_segment.py - run ONE SAM 3 segment re-track in a SEPARATE PROCESS.

The GUI (track.py) launches this with QProcess so the heavy (~2s/frame) SAM 3
video propagation runs OUTSIDE the GUI process (responsive UI, no torch in the
GUI). It seeds from one loose ROI box on the window's first frame, propagates,
and writes {frame_number: [x1,y1,x2,y2]} JSON for the GUI to merge into the
target track.

Progress is printed to STDERR as `PROGRESS i/n` lines so the GUI can show a bar.

Usage:
  python v2\\apps\\sam3_segment.py --video input\\chunk_028.mp4 \\
      --start 1500 --n 60 --box 1070 520 1118 615 --out output\\seg.json
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rt2.sam3track import Sam3Segment, available

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = ROOT / "models" / "sam3.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(description="SAM3 segment re-track (one window)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, required=True,
                    help="first window frame (1-based, matches the tracks CSV)")
    ap.add_argument("--n", type=int, required=True, help="window length in frames")
    ap.add_argument("--box", type=float, nargs=4, required=True,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="seed ROI box on the FIRST window frame")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=512,
                    help="SAM3 input size; 512 + crop is ~3x faster than 1024 with "
                         "the player filling the cropped window (default 512)")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args(argv)

    if not available(args.model):
        print(f"ERROR: SAM3 model not available at {args.model} "
              f"(need the gated models/sam3.pt)", file=sys.stderr, flush=True)
        return 2

    def on_progress(i, n):
        print(f"PROGRESS {i}/{n}", file=sys.stderr, flush=True)

    seg = Sam3Segment(args.model, device=args.device, imgsz=args.imgsz)
    boxes = seg.track_segment(args.video, args.start, args.n, args.box,
                              on_progress=on_progress)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON keys must be strings; the GUI casts back to int frame numbers
    out.write_text(json.dumps({str(k): v for k, v in boxes.items()}),
                   encoding="utf-8")
    print(f"DONE {len(boxes)} frames -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
