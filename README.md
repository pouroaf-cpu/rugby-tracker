# Rugby Tracker

Player detection + tracking for rugby footage on an RTX 3060 (12GB).

## Studio — two modes
Launch the tabbed studio to choose a mode:
```powershell
python v2\apps\studio.py
```
- **Game Analyse** — opens the full match player-tracker (`v2/apps/track.py`) in its own window.
- **Training Analyse** — load two clips (two angles of yourself, or you vs a pro
  reference), sync them (audio auto-sync + manual offset), draw a BlazePose
  skeleton on each, play them side by side, and export the annotated side-by-side
  video to feed into another AI for body-positioning feedback. Built for bodyweight
  work (squats, lunges, press-ups, sit-ups). Standalone: `python v2\apps\training.py`.

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
# Training Analyse (pose overlay):
#   BlazePose (CPU, MediaPipe Tasks) + GPU engine (SynthPose/ViTPose via transformers)
pip install mediapipe scipy
pip install "transformers==4.50.3" accelerate   # pinned for torch 2.5 compatibility
```

### Training Analyse pose engines
Two selectable engines (dropdown, top-right of the Training tab):
- **SynthPose (GPU, feet)** — default when CUDA is present. ViTPose fine-tuned for
  biomechanics (52 keypoints incl. heel/toe), runs on the GPU via `transformers`;
  uses YOLO (`yolo11m.pt`) for the person box. Model `stanfordmimi/synthpose-vitpose-base-hf`
  downloads to the HF cache on first use. Fast (~real-time on an RTX 3060).
- **BlazePose (CPU, feet)** — MediaPipe heavy model; no GPU, slower first pass.

### Training Analyse pose model (one-time download, gitignored)
The BlazePose **heavy** model (33 landmarks incl. feet — most accurate for ankle/leg
form) is not bundled. Download it once into `models/`:
```powershell
Invoke-WebRequest "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" -OutFile "models\pose_landmarker_heavy.task"
```
Audio auto-sync uses `ffmpeg` (must be on PATH).

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
