# Rugby Tracker

Player detection + tracking for rugby footage on an RTX 3060 (12GB).

## Stack
- **Detector:** YOLOv11m (`yolo11m.pt`), person class only (class 0)
- **Tracker:** ByteTrack via [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- **ReID backbone:** OSNet (`osnet_x1_0_msmt17.pt`) for identity re-association
- **Framework:** PyTorch (CUDA 12.1 build) + Ultralytics + Supervision
- **Python:** 3.11 (venv)
- **GPU:** NVIDIA GeForce RTX 3060, CUDA 12.1

## Setup (already done by build, kept for reference)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics boxmot supervision opencv-python
```

## Re-activate the venv each session
```powershell
cd C:\Users\PFrew\Projects\rugby-tracker
.\venv\Scripts\Activate.ps1
```

## Run: track all players
```powershell
python scripts\track.py --video input\test.mp4 --output output\test_tracked.mp4
```
Writes the annotated video plus `output\<videoname>_tracks.csv`
(`frame, track_id, x1, y1, x2, y2, confidence`).

## Run: filter to a single track ID (manual re-anchor workflow)
```powershell
python scripts\filter_by_id.py --video output\test_tracked.mp4 --csv output\test_tracks.csv --id 7 --output output\player7.mp4
```
Scrub the annotated video, find the ID you want, then re-render keeping only that ID.

## Re-anchoring note
Track IDs can change after scrums, subs, or heavy occlusion. **This is expected, not a bug.**
When your ID changes, run `filter_by_id.py` again with the new ID and stitch the segments
together afterwards.

## Folders
- `input/` — source clips (gitignored)
- `output/` — annotated videos + track CSVs (gitignored)
- `models/` — downloaded weights (gitignored)
- `scripts/` — `track.py`, `filter_by_id.py`
