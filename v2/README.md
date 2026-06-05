# rugby-tracker v2

Track a specified player through Veo match footage → clean annotated MP4 + a
clip reel of their touches, with minimal post-game attention cost.

Built parallel to v1 (the `scripts/` tools); v2 reuses v1's proven colour-filter
and re-anchor logic under a PySide6 app shell.

## Locked spec decisions
- **GUI:** PySide6 / Qt full app.
- **Tracking (Phase 4):** CSRT ROI + snap-to-nearest-detection (needs
  `opencv-contrib-python`, already installed).
- **SAM 2 fallback:** deferred to post-ship.
- **First build:** lean vertical slice — one video, end-to-end to a clip reel.

## Layout
```
v2/
  rt2/                     shared library (phase-agnostic, headless-testable)
    paths.py               per-video artifact path conventions
    bbox.py                bbox math (iou, centre, nearest, clamp)
    video.py               VideoReader (seek + decode cache)
    calibration.py         ColorSignature + HeightModel + MatchCalibration
    detections.py          detections.parquet IO
    tracks_io.py           tracks.csv IO (frame,target_id,xyxy,source)
  apps/                    (built per phase)
    calibrate.py           Phase 2 - team calibration tool
    prerender.py           Phase 3 - YOLO cache + colour/height filter
    track.py               Phase 4 - interactive CSRT ROI tracker
    export.py              Phase 6 - annotated video + clip reel + stats
  selftest_lib.py          headless smoke test for rt2
```

## Per-video artifacts (in ../output/)
```
<stem>.calibration.json     learned team colour signatures + height model
<stem>.detections.parquet   cached, filtered person detections
<stem>.tracks.csv           final target track (frame,target_id,xyxy,source)
<stem>.annotated.mp4        target bbox + optional trail
<stem>.clips.mp4            reel of frames where the target is on screen
```

## Key modelling notes
- **Colours are learned, not hardcoded.** Calibration pools HSV pixels from the
  sample boxes (grass dropped) and stores a percentile band per signature.
- **Height filter is perspective-aware.** Player pixel-height is modelled as a
  linear function of box-centre y (higher in frame = further = shorter), so the
  filter doesn't reject distant real players.

## Build status
- [x] Env prep (opencv-contrib + PySide6 + pyarrow)
- [x] Scaffold + shared lib (rt2) — self-tested
- [x] Phase 2 calibration tool (`apps/calibrate.py`) — compiles; needs display to test
- [x] Phase 3 pre-render pipeline (`apps/prerender.py`) — validated on real footage (32 FPS)
- [x] Phase 4 interactive CSRT ROI tracker (`apps/track.py`) — compiles; needs display to test
- [x] Phase 5 re-anchor tools wired in (in `apps/track.py`)
- [x] Phase 6 export (`apps/export.py`) — validated on real footage (annotated + clip reel + stats)

## Apps
```powershell
.\venv\Scripts\Activate.ps1
python v2\apps\calibrate.py  --video input\chunk_028.mp4   # Phase 2 (GUI): drop sample boxes per team, learn profiles
python v2\apps\prerender.py  --video input\chunk_028.mp4   # Phase 3 (batch): YOLO + colour/height/field filter -> detections.parquet
python v2\apps\track.py      --video input\chunk_028.mp4   # Phase 4/5 (GUI): ROI track + re-anchor -> tracks.csv
python v2\apps\export.py     --video input\chunk_028.mp4 --marker arrow   # Phase 6: annotated.mp4 + clips.mp4 + stats
```

## Full-game, multi-player workflow
- **Calibrate once** — `calibrate.py` saves both the per-clip calibration AND a shared
  `output/game.calibration.json`. Every other clip reuses it automatically
  (`prerender`/`track` resolve: explicit `--calibration` > per-clip > shared game).
- **Multiple clips** — in the tracker, **File → Open clip (Ctrl+O)** switches to another
  clip; the shared calibration + the persistent player profile carry over.
- **Multiple players** — `--player <name>` on `track.py`/`export.py` gives each person their
  own appearance profile (`player_<name>.json`) and their own per-clip
  `tracks` / `annotated` / `clips` / `stats`. Omit it for the default single player.
- **Per-game, per-player example:**
  ```powershell
  python v2\apps\prerender.py --video input\clip01.mp4            # uses shared game calibration
  python v2\apps\track.py     --video input\clip01.mp4 --player me  # follow yourself; Ctrl+O to next clip
  python v2\apps\export.py    --video input\clip01.mp4 --player me --marker ring
  ```

## Tracker (track.py) — key features
- **ROI tracking** auto-enables when you place an ROI; **Play records** each frame
  (`learned Nf` climbs). **Save button** + `s`. **Track: ON/OFF** button.
- **Online learning** of your motion + appearance; **recovery** re-grabs you within a
  larger search box after a crossover. **`L`** resets the session model, **`p`** saves /
  **`K`** clears the persistent profile.
- **View filters**: `h` hide other teams, `g` hide off-field bystanders (target always shown).
- **Markers (export)**: `--marker arrow|ring|dot|box|both` + `--marker-size`.
- Zoom (wheel), pan (Space/middle-drag), `o` arrow marker, `<`/`>` speed.
- **Controls panel (left dock)** — every action has a labelled button (playback, tracking,
  learning, re-anchor, view, file), so the tool is fully mouse-usable; checkable buttons
  stay in sync with the hotkeys.
- **Player selector** (top of the panel) — switch which player you're tracking; pick an
  existing `player_<name>` profile or create a new one. Switching relaunches on the same
  clip with `--player <name>` (its own profile/track/output). Lets you pick an easy-to-track
  player and let the model build up knowledge of them.

## Confidence rating & rule-out (raising tracking confidence)
The philosophy: accumulate many weak, resolution-robust signals into a per-frame
confidence, and hard-exclude anyone confidently NOT the target.
- **Height-consistency gating** — learns the target's pixel-height *residual* vs the
  team's perspective height model (a stable per-player trait, persisted across clips);
  once ≥15 samples it rules out candidates of clearly different height. Inert without a
  calibration/height model or until enough samples.
- **Team-constraint** — confident-opposition detections are never eligible (already in).
- **Per-frame confidence (0–1)** — blends source (detection/held/manual), height match,
  appearance, and margin-over-next-best. Surfaced as the target box colour (green/amber/red),
  a `conf 0.NN` status readout, and a panel label. **Auto-pause on low conf** (`a`, panel
  toggle) stops playback when unsure instead of drifting.
- **Roadmap (not yet built — needs steer):** physical motion gating (reject impossible m/s),
  track exclusivity (claim teammates' own tracks), learn-from-corrections ("not me" tags),
  timeline filmstrip + low-confidence review queue.

## Tried and rejected: pose / limb tracking
YOLO11-pose was tested for limb tracking / motion prediction. On this wide Veo footage it
**only detected close/large players (≈6/frame) and missed all small/mid-field players**, with
mediocre keypoints (~10/17 @ 0.59 conf) even on the close ones — the same resolution wall as
appearance ReID. Not integrated. Revisit only with zoomed/follow-cam footage (`yolo11m-pose.pt`
is downloaded in `models/`).

## Pre-render filter stack
A detection is `kept` only if it passes all three (`kept = colour AND height AND field`):
1. **Colour (team fingerprints)** — each detection is matched to the nearest team
   *fingerprint* and kept only if it's closest to your tracked team by `--margin`
   (2-team mode) — see below.
2. **Height** — plausible vs the perspective height model (rejects crowd / distant noise by size).
3. **Field-of-play** (`rt2/field.py`, `--no-field-mask` to disable) — feet on the pitch; a per-frame greedy pitch hull that follows the camera pan. Light secondary guard for off-green crowd; never drops on-pitch players. Goalposts/signs aren't people so YOLO never proposes them.

## Team fingerprints (the colour model)
Learned HSV *signatures* can't separate teams at Veo resolution — the dominant
colour in the torso/shorts regions is **skin**, common to both teams (proven:
learned classifier got blue&gold 20%/0% right). Instead, `rt2/features.py`
describes each box by a **vector of skin-immune perceptual colour fractions**
(white/black shorts, navy/red/gold shirt, ...). Each team's fingerprint is the
mean vector over its calibration samples; a detection is assigned to the
**nearest fingerprint** (skin cancels in the comparison).

**Calibrate BOTH teams** (the opposition profile is what makes the comparison
work — the GUI now learns every team with samples, the `track` flag just picks
which to follow). On the test match, 30 samples each → Otorohanga vs Hautapu
classified at **100% / 87%**, and the full pre-render keeps **51%** (your team's
share) vs the old learner's 93%-kept mush.

## Run the lib self-test
```powershell
.\venv\Scripts\Activate.ps1
python v2\selftest_lib.py
```
