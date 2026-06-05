"""
track.py - Interactive single-player CSRT ROI tracker + re-anchor toolkit (Phases 4 & 5).

MULTI-PLAYER MODE (--player <name>)
  Pass --player <name> so several people can each track THEIR OWN player on the
  same footage without clobbering each other's work. The name namespaces three
  things to that player: the persistent appearance PROFILE
  (output/player_<name>.json instead of player_profile.json), the TRACK CSV and
  the per-clip OUTPUTS (file named <stem>.<name>.tracks.csv / .annotated / clips).
  The name carries across clips via File -> Open clip. Without --player the
  behaviour is IDENTICAL to before (single shared player_profile.json + plain
  <stem> track/output names).

A PySide6 desktop app for tracking ONE rugby player through wide Veo match footage.
You place a Region-Of-Interest (ROI) box over your player; a cv2.TrackerCSRT carries
that ROI forward as the match plays. Inside the ROI the pre-rendered "kept" detections
are drawn; the candidate that best matches the ONLINE TARGET MODEL (centre + predicted
motion + accumulated appearance) becomes the ACTIVE TARGET and is appended to the
consolidated target track. Phase 5 re-anchor tools (merge / delete / nuke / bulk-ID
dialog / undo) let you repair the resulting track. Nothing is written until you press 's'.

ONLINE TARGET MODEL (TargetModel)
  As you track, the app LEARNS the followed player so it can pick the right candidate
  inside the ROI instead of blindly taking the box nearest the ROI centre:
    * MOTION  - a short history of confirmed target centres; predicts where the player
                will be on the next frame from average recent velocity.
    * APPEARANCE - a running mean L2-normalised H-S colour histogram of confirmed target
                crops (grass hue masked out). New candidates are scored by cosine
                similarity to this mean.
  Each in-ROI candidate is scored by a weighted blend of nearness-to-ROI-centre,
  nearness-to-predicted-centre and appearance similarity; the highest-scoring candidate
  wins. When the model is still empty the behaviour is identical to the old
  nearest-centre selection. The model UPDATES at the point a target box is CONFIRMED for
  each frame (snapped detection OR the CSRT/ROI fallback box). The status bar shows
  `learned <Nf>` (appearance crops folded in / motion samples held).
  Press 'L' (shift+l) to RESET the learned model; it also resets when you PLACE a fresh
  ROI (a small drag-correct of the existing ROI keeps the model).

TEAM-CONSTRAINED SELECTION (never lock onto the opposition)
  When a calibration is loaded, the ROI snap AND recovery candidate pools are
  restricted to ELIGIBLE detections = all EXCEPT those CONFIDENTLY classified as a
  NON-tracked (opposition) team (eligible_target_indices()). Tracked-team AND
  'unsure' detections always stay eligible, so the real target is never dropped
  when the classifier is uncertain; only boxes the classifier is SURE belong to
  the other team are excluded. Among the eligible candidates a confidently-
  tracked-team box is PREFERRED, falling back to 'unsure' only if there is no
  tracked-team candidate. With no calibration everything is eligible (old
  behaviour). This is why the box can no longer jump to the opposition / far ref.

HEIGHT-CONSISTENCY gating (rule out wrong-height players)
  When a calibration with a TRACKED-team HEIGHT model is loaded, the followed
  player's HEIGHT RESIDUAL (box_height - tracked_team.height.predict(centre_y))
  is accumulated as the target is tracked (running count/mean/std via Welford,
  folded in by _learn_target). Once >= 15 residual samples are seen, the snap
  AND recovery candidate pools EXCLUDE any candidate whose residual deviates
  from the learned mean by more than HEIGHT_TOL = max(18px, 2.5*resid_std) -
  i.e. players who are clearly the wrong height for that position on the pitch.
  Candidates within tolerance, or with no measurable height, are kept; the
  current active target and the armed detection are NEVER excluded; gating is
  conservative (only with >=15 samples). The residual trait PERSISTS in the
  player profile (version 2), so it carries across clips. With no calibration /
  no tracked-team height model / <15 samples the feature is INERT (no gating).

CONFIDENCE rating (per tracking frame, 0..1)
  After the target box for a frame is chosen, a CONFIDENCE in [0,1] is computed
  as a weighted blend of: (a) SOURCE base - snapped same-team detection 0.7,
  held/csrt 0.35, manual 0.9; (b) HEIGHT closeness of the chosen box residual
  to the learned mean (1 within ~0.5*std, decaying to 0 by HEIGHT_TOL; neutral
  0.5 until the residual model is ready); (c) APPEARANCE similarity of the chosen
  crop to the learned player (neutral 0.5 if none); (d) MARGIN - how dominant the
  chosen candidate is over the next-best eligible candidate. It is stored per
  frame (conf_by_frame) so scrubbing back shows the rating the frame was tracked
  at. SURFACED three ways: the TARGET box is coloured by confidence (green >=0.7,
  amber 0.4-0.7, red <0.4); the status bar shows "conf 0.NN"; the Controls panel
  shows a "conf: 0.NN" label. AUTO-PAUSE on low confidence ('a' / panel "Auto-
  pause on low conf", default OFF): when ON and playing, a frame with confidence
  < 0.4 pauses playback ("paused: low confidence - check the target").

JERSEY-NUMBER OCR soft signal (toggle 'j'; DEFAULT OFF; optional, low-risk)
  A BONUS confidence cue for when the player's BACK faces the camera. When ON
  AND the tesseract binary is installed (rt2.ocr.available()), the target's
  upper-back/torso region (top ~18-55% of the box, central 60%) is OCR'd every
  ~10th tracked frame, but only when the target box is >= ~80px tall. Confident
  1-2 digit reads accumulate in a Counter for the TARGET; the most-frequent one
  is exposed as self.target_number and shown as "target #<n>" (with the last
  read) in the status bar / Controls panel. SOFT weighting only: in the per-
  frame confidence, a fresh read MATCHING target_number nudges confidence UP a
  little (+0.1), a confident DIFFERENT 1-2 digit number nudges it DOWN a little
  (-0.08); both re-clamped to [0,1]. The whole feature is a strict NO-OP when
  the toggle is OFF or OCR is unavailable - it never touches confidence or what
  is tracked/saved then. The "Jersey OCR" panel button / 'j' no-op with a status
  note when the binary is missing (install: winget install
  UB-Mannheim.TesseractOCR). See rt2/ocr.py - it is gracefully inert (returns
  None, never raises) without the binary.

CONFIDENCE-ADAPTIVE MARKER (toggle 'b'; DEFAULT ON)
  A marker drawn ON the target box whose SHAPE encodes how sure the tracker is
  (the arrow marker 'o' is an alternative; both may be on). The pure
  adaptive_marker_spec(box, conf) gives the geometry (headlessly testable):
    * conf >= 0.7      -> a TIGHT body ELLIPSE (semi-axes ~0.55*w x 0.50*h) -
                          reads as "locked on".
    * 0.4 <= conf <0.7 -> a LARGER ellipse, axes GROWING (lerp up to ~1.3x the
                          box) as confidence drops.
    * conf < 0.4       -> a dashed SEARCH CIRCLE whose radius GROWS as confidence
                          falls (~0.8*box-diag at 0.4 up to ~2.5*box-diag near 0).
  Coloured by confidence (green >=0.7, amber 0.4-0.7, red <0.4) over a thin dark
  backing for contrast; the circle case is DASHED ("searching"). The drawn size
  is EMA-smoothed frame-to-frame to avoid flicker. A scrubbed frame with no
  rating shows the tight ellipse.

CONFIDENCE-ADAPTIVE SPEED (toggle 'e'; DEFAULT OFF)
  When ON during play, the per-tick interval is driven by the latest tracking
  confidence instead of the manual speed: conf<0.4 -> 0.25x (crawl so you can
  check the target), 0.4-0.7 -> 0.5x, >=0.7 -> 1.75x (run ahead through
  confident stretches), all relative to the base fps interval. When OFF the
  existing manual '<'/'>' speed applies unchanged; the two never fight.

PERSISTED CONFIDENCE (per-frame)
  On save ('s') each written row carries its per-frame confidence under the
  "confidence" column (default 1.0 for any frame with no rating), so the rating
  the frame was tracked at survives in the track CSV.

HOLD-don't-guess vs RECOVERY on loss (toggle 'v'; DEFAULT = HOLD)
  When a tracking step finds NO good target THIS frame - the CSRT lost the player
  (`_advance_csrt_to` returned None) OR there is no ELIGIBLE (same-team / unsure)
  detection inside the ROI - the behaviour depends on self.recovery_on:
    * recovery OFF (the DEFAULT, "HOLD"): keep the ROI exactly where it is and
      record the held box (SOURCE_CSRT) WITHOUT moving to any far detection. The
      big search-box recovery does NOT run and the magenta search box is NOT drawn.
      This is the safe default - it holds rather than guessing onto a stranger.
    * recovery ON: try to RE-ACQUIRE over a WIDER search box. The search box is
      centred on the model's predicted centre (or the ROI centre) and sized
      SEARCH_MULT x the ROI's (w, h), clamped to the frame. ELIGIBLE detections
      (team-constrained - opposition excluded) whose centre lands inside it are
      each scored `recov = W_RPOS*pos_term + W_RAPP*app_term`. Only if the model
      HAS data and the best `recov >= RECOVER_MIN` do we re-acquire: the ROI snaps
      onto that detection (padded ~15%), the CSRT re-inits there, the box is
      recorded as a DETECTION ("recovered"). While recovery is ON the search box is
      drawn as a faint DASHED MAGENTA rectangle (correct under zoom/pan).

PERSISTENT PLAYER PROFILE (full-game carry-over)
  The learned APPEARANCE running-mean (vector + crop count) is persisted to a single
  project-level file, ProjectPaths().output / "player_profile.json", so the player's
  look ACCUMULATES across clips. MOTION is NOT saved (velocity is session-only). On
  startup the profile is loaded if present ("loaded player profile: N crops"). It is
  saved on every track save ('s') and on the dedicated 'p' key; new crops keep folding
  into the running mean across sessions. Press 'K' to CLEAR the persistent profile
  (deletes player_profile.json and resets appearance) - distinct from 'L', which only
  resets the in-session model and leaves the file intact.

MULTI-CLIP GAME (Open clip)
  A full game is usually split into several clips. File -> "Open clip..." (Ctrl+O)
  loads another clip without re-doing setup: it offers to save the current track,
  then RELAUNCHES the tracker on the chosen video and closes this window. Two
  things carry across clips automatically: the CALIBRATION (a clip with no
  calibration of its own falls back to the shared output/game.calibration.json,
  so team labels stay consistent) and the persistent player_profile.json (the
  learned appearance), so each new clip opens already knowing the player's look.

TEAM LABELS (calibration, optional)
  If a MatchCalibration is resolved for the video (clip's own calibration, else the
  shared game.calibration.json, else --calibration override), each
  detection is classified to the nearest CALIBRATED TEAM by its perceptual fingerprint
  (rt2.features). The detection list shows the team name beside each "#i conf=..", and
  the plain detection boxes are coloured by team (tracked team = teal, opposition =
  orange, "unsure" = grey). This is a BEST-GUESS, per-box appearance match; it is
  imperfect (a single box can be mislabelled) and is purely a visual aid - it does not
  affect tracking or what is saved. With no calibration / no fingerprints the feature is
  silently inert (list/boxes keep their previous look).

DISPLAY FILTERS (declutter only - never change what is tracked/saved)
  Two optional VIEW filters hide distracting detections from the list + the boxes
  drawn on the video. They are PURELY cosmetic - they never touch the target track,
  the candidate pool or what is written on 's':
    * HIDE OTHER TEAMS ('h' or View menu): when a calibration is loaded, hide boxes
      classified to a NON-tracked team with CONFIDENT team labels (e.g. Hautapu).
      "unsure" boxes are always kept so the target is never hidden by mistake. With
      no calibration this hides nothing (no team info to act on).
    * HIDE OFF-FIELD / BYSTANDERS ('g' or View menu): compute the per-frame
      rt2.field.field_mask and hide boxes whose feet are off the pitch (crowd /
      sideline). The field mask is CACHED per frame (only computed when this filter
      is on). The ACTIVE TARGET det and the ARMED detection are ALWAYS kept visible,
      even when a filter would otherwise hide them. The status bar shows
      "view: team-only" / "view: on-field" when each filter is active.

MANUAL CURSOR-FOLLOW MODE (toggle 'm' or the bottom-bar "Manual" button)
  A fully hand-driven mode for when the automatic tracker keeps losing the player.
  When MANUAL is ON there is NO CSRT / NO snap / NO recovery: the ROI simply
  FOLLOWS THE MOUSE CURSOR - move the mouse to point at yourself, no dragging
  needed (the ROI keeps its current width/height and recentres on the cursor on
  every mouse-move). Every time you ADVANCE to a frame (Play or a single forward
  step) the CURRENT ROI box is recorded as the target for that frame
  (SOURCE_MANUAL) and folded into the learned model. The status bar shows
  "** MANUAL **". Saves normally (same target track + 's').

  Toggling 'm' is STUTTER-SAFE: it only flips the flag - it never stops/restarts
  the play timer, re-seeks or re-shows the frame, so it can be toggled mid-play
  without hitching (the per-frame advance loop just branches on the flag). On
  toggle ON the ROI snaps to the last cursor position; on toggle OFF the CSRT is
  re-anchored on the current ROI so AUTO tracking resumes from wherever the cursor
  left the player ("auto-follow resumed"). Left-drag may still nudge the ROI.

ONLINE IDENTITY CLASSIFIER ("me vs not-me"; rt2.identity)
  A self-contained online logistic-regression model that learns YOUR player from
  the user's CONFIRMATIONS and CORRECTIONS, so confidence tightens over rewatches.
  It is wired to existing per-detection signals via _identity_features(frame,box),
  a FIXED-LENGTH vector: [team-fingerprint distance to the tracked team, to the
  opposition, their signed margin, |height-residual deviation| vs the target's
  learned mean, normalised centre x and y, motion-consistency (dist to predicted
  centre / frame diag), box height / frame height, appearance similarity to the
  learned player]; missing signals fall back to neutral 0 so the dimension is
  constant.
    * LEARNS automatically: every CONFIRMED target frame (snap / manual / held /
      recovered) adds the target as a POSITIVE and the OTHER eligible in-frame
      detections as WEAK negatives (weight ~0.3).
    * LEARNS from CORRECTIONS (the key feature): "Rule out (not me)" - press 'n'
      or the RE-ANCHOR panel button while a detection is ARMED - adds it as a
      HARD NEGATIVE (weight ~2.0) AND records its track id in a session ruled_out
      set so that track is EXCLUDED from eligibility going forward (like the team
      constraint). Status: "ruled out id N (not me)".
    * USED once ready() (>= 8 positives AND >= 8 negatives): prob(target) is folded
      into the per-frame CONFIDENCE blend (weight CONF_W_ID) and used as a
      candidate-selection TIEBREAKER (prefer higher-prob eligible candidates). It
      NEVER hard-excludes - only the explicit rule-out + team + height do that.
      Shown as "id NN%" in the status bar / Controls panel when ready.
    * PERSISTED in a sibling file next to the appearance profile
      (player_<name>.id.json / player_profile.id.json), loaded on startup, saved
      on 's' (do_save) and 'p'. Inert (prob 0.5, no effect) until ready.

DATA FLOW
  ProjectPaths().detections(video) -> read_detections -> kept_by_frame  ... candidate pool
  ROI (CSRT) + nearest kept detection  ----------------------------------> target track
  write_tracks(ProjectPaths().tracks(video), rows)  (columns: frame,target_id,x1,y1,x2,y2,source)

CONTROLS
  --- playback / navigation ---
  space ................ play / pause   (HOLD space + left-drag = PAN when zoomed)
  Left / Right ......... step -1 / +1 frame   (when zoomed & holding space: pan)
  , / . ................ step -1s / +1s
  [ / ] ................ step -10s / +10s
  scrubber (bottom) .... seek to any frame; play/pause button at far left
  ◀ / ▶ buttons ........ bottom bar: step -1 / +1 frame (same as Left/Right; pauses)
  Del fwd button ....... bottom bar: delete the armed/target track forward from this
                         frame (same as 'd')
  Clean jumps button ... bottom bar: smart-clean teleport/outlier frames (same as 'c')

  --- zoom / pan (the wide Veo shot - players are tiny) ---
  mouse wheel .......... zoom in / out, anchored under the cursor (~1x fit .. 12x)
  middle-drag .......... pan the zoomed view
  space + left-drag .... pan the zoomed view (keeps left-button ROI semantics)
  arrow keys (zoomed) .. while holding space, nudge the pan
  0 ................... reset view: fit-to-window (zoom = fit, pan = 0)

  --- target id ---
  1-9 ................. set armed / active target ID to that digit
  t ................... begin typing a target ID > 9 (type digits, then Enter)
  Enter ............... commit typed target ID  (or, if an ID is armed by click and
                        digits were typed into the merge buffer, MERGE armed->typed)
  L (shift+l) ......... reset the ONLINE TARGET MODEL (forget learned motion+appearance,
                        SESSION only - keeps the persistent player_profile.json)
  p ................... save the PERSISTENT player appearance profile now
  K (shift+k) ......... clear the PERSISTENT player profile (delete player_profile.json
                        + reset appearance); distinct from 'L'

  --- ROI tracking (Phase 4) ---
  r ................... toggle ROI tracking on / off
  R (shift+r) ......... reset the ROI to the current target box
  m ................... manual cursor-follow (move mouse to point at yourself;
                        records each frame); m again = resume auto-follow.
                        (also the bottom-bar "Manual" button): no CSRT/snap/
                        recovery; the ROI follows the cursor and each forward
                        step records it (SOURCE_MANUAL)
  v ................... toggle RECOVERY on loss (Track menu too). DEFAULT is OFF:
                        on loss the ROI is HELD in place rather than guessing onto
                        a far detection. ON enables the team-constrained search-box
                        recovery + the magenta search box.
  + / = ............... grow the ROI ~10% (square, centred, re-anchors CSRT)
  - / _ ............... shrink the ROI ~10% (square, centred, re-anchors CSRT)
                        (when the ARROW marker is ON, +/- resize the arrow)
  click empty area .... place a fresh ~220x220 ROI centred on the click (tighter
                        default so it is less likely to span two players)
  drag ROI body ....... move the ROI
  drag a corner ....... resize the ROI (8 handles)
  click a detection ... arm that detection's track id (for merge / delete)

  --- playback speed / marker ---
  < / > ............... slower / faster playback (0.25 .. 4x; affects play only)
  e ................... toggle CONFIDENCE-ADAPTIVE SPEED (during play, slow down
                        on uncertain frames, run ahead on confident ones; ignores
                        the manual speed while on)
  o ................... toggle ARROW marker (amber chevron above the target's head)
  b ................... toggle CONFIDENCE-ADAPTIVE MARKER (default ON): a tight
                        body ellipse when confident, growing to a big dashed
                        search circle when not; coloured green/amber/red by conf

  --- display filters (View menu; cosmetic only) ---
  h ................... toggle HIDE OTHER TEAMS (confidently non-tracked teams)
  g ................... toggle HIDE OFF-FIELD / bystanders (feet off the pitch)
                        (active target + armed detection always stay visible)

  --- re-anchor tools (Phase 5) ---
  i ................... bulk ID manager (Qt dialog: checkbox list, sort radios, bulk buttons)
  d ................... delete forward: armed id from this frame to the next >30f gap
  D (shift+d) ......... nuke: delete ALL rows for the armed id
  n ................... RULE OUT (not me): mark the ARMED detection as a HARD
                        NEGATIVE for the online identity classifier AND exclude
                        its track id from selection for the rest of the session.
                        This is the key CORRECTION - it makes the "me vs not-me"
                        confidence tighten over rewatches.
  c ................... SMART-CLEAN ("clean jumps"): remove teleport/outlier frames
                        from the target track over the last ~20s up to the current
                        frame. For each scoped point the centre is compared to a
                        MEDIAN-smoothed trajectory; frames deviating by more than
                        max(100px, 3 x median box larger-side) are dropped as ONE
                        undoable op (a single 'u' restores them all). With <5 rows
                        in scope the whole track is cleaned instead.
  u ................... undo last change (merge / delete / nuke / clean-jumps /
                        whole bulk dialog)

  --- misc ---
  a ................... toggle AUTO-PAUSE on low confidence (pauses play when a
                        frame's confidence < 0.4; default OFF)
  j ................... toggle JERSEY-NUMBER OCR soft signal (default OFF; a
                        BONUS confidence cue for when the back faces the camera).
                        No-ops with a status note if the tesseract binary is not
                        installed (winget install UB-Mannheim.TesseractOCR).
  f ................... toggle SAM 2 (body-TIGHT silhouette hugging the player +
                        dim the grass inside the box, replacing the ellipse; also
                        tightens the tracked box to the mask; default OFF,
                        ~130ms/frame on GPU; uses the installed ultralytics SAM2)
  s ................... save target track CSV (NO autosave)
  q ................... quit
  Esc ................. clear armed id / typed-input buffer

CONTROLS PANEL + PLAYER SELECTOR (fully mouse-usable)
  A LEFT dock "Controls" mirrors EVERY hotkey action as a labelled button so the
  app can be driven entirely with the mouse - the keys all still work too. The
  panel is a scrollable column of grouped sections:
    PLAYER   - a combo box to switch which player is tracked (scans
               output/player_*.json + the current/default player + "New
               player..."); changing it relaunches the tracker on the SAME video
               for the chosen player (offering to save the current track first).
    PLAYBACK - Play/Pause, -1s/-1f/+1f/+1s, -10s/+10s, Speed -/+ (with a speed
               label), Conf-adaptive speed (checkable), Fit zoom.
    TRACKING - Track / Manual / Recovery-on-loss / Arrow marker / Adaptive
               marker (checkable), ROI -/+, Reset ROI->target.
    LEARNING - Save profile, Reset model, Clear profile, a "learned: Nf" label,
               a live "conf: 0.NN" label, an "Auto-pause on low conf" toggle, a
               "Jersey OCR" toggle (default OFF; soft jersey-number cue) + a
               "target #<n>" label.
    RE-ANCHOR- Delete forward, Nuke track, Clean jumps, Undo, Bulk IDs...
    VIEW     - Hide other teams / Hide off-field (checkable).
    TACTICAL - Player circles (checkable; Z) + Clear circles, Set offside @
               player + Clear offside. True-metre ground rings + an offside line
               drawn via the live pitch homography.
    FILE     - Save track, Open clip...
  Each button is tooltipped with its hotkey. The CHECKABLE buttons (Track, Manual,
  Recovery, Arrow, Adaptive marker, Conf-adaptive speed, Hide teams, Hide
  off-field, Auto-pause on low conf, Jersey OCR) reflect
  window state and STAY IN SYNC when the equivalent hotkey is pressed (_sync_buttons(), called after the
  toggles + once per frame, using blockSignals to avoid feedback loops). The
  bottom bar is kept as-is; the left panel is the full mouse superset.

FRAME NUMBERS ARE 1-BASED THROUGHOUT.

Usage:
  python v2\\apps\\track.py --video input\\chunk_028.mp4
  python v2\\apps\\track.py --video input\\chunk_028.mp4 --player alice   # multi-player
  python v2\\apps\\track.py --selftest      # headless logic test (no GUI / no display)
"""
from __future__ import annotations

import argparse
import csv
import sys
import pathlib
from collections import defaultdict, deque, Counter

import numpy as np
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rt2.paths import ProjectPaths, FPS_ASSUMED
from rt2.video import VideoReader
from rt2 import bbox
from rt2 import features
from rt2 import field
from rt2 import ocr
from rt2 import regions    # SAFE: pure numpy crops (no torch / no pyarrow)
from rt2 import identity
from rt2 import sam2track  # SAFE to import: torch is lazy-loaded only on first use
from rt2 import registry    # SAFE to import: sqlite/json only (no torch, no pyarrow)
from rt2 import candidates  # SAFE: pure logic (stdlib only); "who could be ME" shortlist
from rt2.homography import PitchHomography, PITCH  # SAFE: numpy/cv2 only (no torch)
from rt2.cmc import LiveHomography, CameraTracker   # SAFE: numpy/cv2 only (no torch)
from rt2.detections import read_detections, kept_by_frame, all_by_frame
from rt2.tracks_io import (
    write_tracks, read_tracks,
    SOURCE_DETECTION, SOURCE_CSRT, SOURCE_MANUAL, SOURCE_SAM2,
)

GAP = 30                 # frames; a forward segment ends at the first gap larger than this
DEFAULT_ROI = 220        # default ROI side length (px) in original-frame coords (tighter box)
HANDLE = 14              # corner/edge handle hit radius (original-frame px)

RESIZE_STEP = 1.10       # +/- hotkey ROI/spotlight grow factor (~10%); 1/x to shrink
MIN_SIDE = 16            # smallest allowed square side (frame px)
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]   # playback speed multipliers
DEFAULT_SPEED = 1.0

# ---- CONFIDENCE-ADAPTIVE SPEED (Feature 3) ----
# When conf-adaptive speed is ON, the per-tick playback interval is driven by
# the latest tracking confidence instead of the manual speed: slow right down
# to inspect when the tracker is unsure, speed up when it is confident. The
# multipliers are relative to the base fps interval.
CONF_SPEED_LOW = 0.25    # conf < 0.4  -> crawl so the user can check the target
CONF_SPEED_MID = 0.5     # 0.4 <= conf < 0.7
CONF_SPEED_HI = 1.75     # conf >= 0.7 -> run ahead through confident stretches
CONF_SPEED_HI_THRESH = 0.70
CONF_SPEED_LOW_THRESH = 0.40
SPOTLIGHT_DEFAULT = 320  # default spotlight square side (frame px)
SPOTLIGHT_TARGET_MULT = 1.6   # spotlight side relative to target box larger side
SPOTLIGHT_EMA = 0.4      # smoothing factor for the spotlight centre (higher = snappier)

# ---- CONFIDENCE-ADAPTIVE MARKER tuning (Feature 1) ----
# A confidence-driven marker drawn ON the target box: a TIGHT body ELLIPSE when
# confident, growing to a big dashed SEARCH CIRCLE as confidence falls. The
# geometry is computed by the pure adaptive_marker_spec() so --selftest can
# exercise it headlessly. Confidence band cut points:
ADAPT_CONF_HI = 0.70     # >= this: tight body ellipse (green)
ADAPT_CONF_LO = 0.40     # >= this (and < HI): larger amber ellipse; below: search circle (red)
ADAPT_ELL_AX_HI = 0.55   # tight ellipse semi-axis x as a fraction of box width (at conf>=HI)
ADAPT_ELL_AY_HI = 0.50   # tight ellipse semi-axis y as a fraction of box height
ADAPT_ELL_GROW = 1.30    # ellipse axes grow up to ~this multiple of the box at conf==LO
ADAPT_CIRC_R_LO = 0.80   # search-circle radius as a fraction of box diagonal at conf==LO
ADAPT_CIRC_R_MIN = 2.50  # search-circle radius (x box diagonal) as conf -> 0 (grows as conf falls)
ADAPT_MARKER_EMA = 0.4   # smoothing factor on the drawn marker size (anti-flicker)

# ---- ONLINE TARGET MODEL tuning ----
TM_MOTION_LEN = 8        # motion deque length (confirmed centres kept)
TM_HIST_BINS = (12, 4)   # H-S histogram bins (Hue x Saturation)
TM_GRASS_H_LO = 33       # grass hue range (OpenCV H in 0..179) masked from histograms
TM_GRASS_H_HI = 90
TM_W_CENTER = 0.4        # weight: nearness of candidate to ROI centre
TM_W_PRED = 0.4          # weight: nearness of candidate to predicted centre
TM_W_APP = 0.2           # weight: appearance similarity to learned mean

# ---- ONLINE IDENTITY CLASSIFIER ("me vs not-me") tuning ----
# A self-contained logistic-regression model (rt2.identity) that learns the
# target from the user's CONFIRMATIONS (positives) and CORRECTIONS (hard
# negatives = "rule out this player"). Once ready() its prob(target) in [0,1]
# is folded into the per-frame CONFIDENCE and used as a candidate-selection
# tiebreaker; it NEVER hard-excludes (only the explicit rule-out + team +
# height do that). See _identity_features() for the FIXED-LENGTH feature vector.
IDENTITY_DIM = 9         # MUST equal the length identity_features() returns
ID_WEAK_NEG = 0.30       # weight: WEAK negative for an other-in-frame detection
ID_HARD_NEG = 2.00       # weight: HARD negative for an explicit "Rule out (not me)"
CONF_W_ID = 0.20         # how much the identity prob pulls the per-frame confidence
ID_TIEBREAK = 0.15       # max selection-score bonus from identity prob (tiebreaker)

# ---- RECOVERY (larger search box) tuning ----
SEARCH_MULT = 2.5        # search box side = SEARCH_MULT x the ROI's (w, h)
RECOVER_MIN = 0.45       # min blended recovery score to RE-ACQUIRE a candidate
FOCUS_MULT = 3.0         # FOCUS box side = FOCUS_MULT x the ROI's (w, h). ONLY
                         # detections inside it are ID-traced/selectable (declutter
                         # + anti-jump: far players can't be picked or crossed onto)
CSRT_SCALE = 0.5         # run CSRT on a frame DOWNSCALED by this (its coord frame
                         # just scales). 1080p CSRT is ~79ms/frame -> ~48ms at 0.5,
                         # so AUTO playback stops being CSRT-bound. 1.0 = full res.
W_RPOS = 0.6             # recovery weight: nearness to predicted/ROI centre
W_RAPP = 0.4             # recovery weight: appearance similarity to learned player

# ---- SMART-CLEAN ("clean jumps") tuning ----
MIN_JUMP_PX = 100.0      # min deviation (px) from the smoothed trajectory to call a jump
JUMP_K = 3.0             # deviation threshold also scales with K x median box larger-side
CLEAN_WINDOW_S = 20.0    # default scope: clean the last N seconds up to the current frame
CLEAN_MEDIAN_W = 7       # median-smoothing half-window (neighbours i-W..i+W)

# ---- PERSISTENT player profile ----
PROFILE_NAME = "player_profile.json"   # project-level appearance carry-over file

# ---- TEAM LABEL (calibration) tuning ----
TEAM_MARGIN_MIN = 0.05   # min L1-distance gap between best & 2nd-best team to be "confident"

# ---- HEIGHT-CONSISTENCY gating (rule out wrong-height candidates) ----
# A box's height RESIDUAL = box_height - tracked_team.height.predict(centre_y).
# The TargetModel accumulates the TARGET's residual (Welford). Once we have
# >= HEIGHT_MIN_SAMPLES residual samples, candidates whose residual deviates
# from the learned mean by more than HEIGHT_TOL are EXCLUDED from the snap /
# recovery pools (but never the active target or the armed detection). Inert
# (no-op) when there is no calibration / no tracked-team height model.
HEIGHT_MIN_SAMPLES = 15  # min residual samples before height gating activates
HEIGHT_MIN_PX = 18.0     # absolute floor for the height tolerance band (px)
HEIGHT_TOL_K = 2.5       # tolerance = max(HEIGHT_MIN_PX, HEIGHT_TOL_K * resid_std)

# ---- per-frame CONFIDENCE rating (0..1) ----
CONF_BASE_DETECTION = 0.70   # base when the frame snapped a same-team detection
CONF_BASE_HELD = 0.35        # base when the box was HELD / CSRT fallback
CONF_BASE_MANUAL = 0.90      # base when the box was placed manually
MANUAL_CONF_FLOOR = 0.85     # in manual mode the HUMAN is the tracker -> always
                             # read confident (green) immediately, never red/amber

# JUMP GATE ("wait before jumping"): a real player can't teleport. If the snapped
# box is an implausibly large step from the last confirmed target (e.g. it latched
# onto a different player metres away), HOLD for a few frames instead of jumping;
# only accept the new location if it PERSISTS (a genuine fast move / re-acquire).
JUMP_MAX_BODYHEIGHTS = 0.9   # max plausible target travel per frame, in box-heights
JUMP_CONFIRM = 3             # frames a far jump must persist before it is accepted
JUMP_GAP_RESET = 8           # after this many frames with no lock, gate stops (re-acquire freely)


def jump_too_far(prev_centre, prev_frame, box, frame,
                 max_bodyheights=JUMP_MAX_BODYHEIGHTS, gap_reset=JUMP_GAP_RESET):
    """True if `box` is an implausibly large jump from the last confirmed target
    centre for the elapsed frame gap (a teleport / mis-association onto another
    player). The budget scales with the box height (a proxy for the player's
    size/distance) and the frame gap. Pure logic so --selftest can exercise it."""
    if prev_centre is None or prev_frame is None:
        return False
    gap = frame - prev_frame
    if gap <= 0 or gap > gap_reset:
        return False
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    bh = box[3] - box[1]
    dist = ((cx - prev_centre[0]) ** 2 + (cy - prev_centre[1]) ** 2) ** 0.5
    budget = max_bodyheights * max(float(bh), 20.0) * gap
    return dist > budget
CONF_W_BASE = 0.40           # blend weight: source base
CONF_W_HEIGHT = 0.20         # blend weight: height-residual closeness term
CONF_W_APP = 0.20            # blend weight: appearance similarity term
CONF_W_MARGIN = 0.20         # blend weight: dominance vs the next-best candidate
AUTO_PAUSE_THRESH = 0.40     # auto-pause-on-low fires below this confidence
RUCK_LOST_FRAMES = 5         # consecutive sub-threshold frames -> "lost in the ruck"

# ---- JERSEY-NUMBER OCR soft signal (BONUS cue; DEFAULT OFF, hotkey 'j') ----
# An OPTIONAL, low-weight confidence cue for when the player's back faces the
# camera. OCR the target's upper-back/torso region; accumulate digit reads for
# the TARGET in a Counter and expose the most-frequent confident one as
# self.target_number. In the per-frame confidence: a fresh read MATCHING the
# learned number nudges confidence UP a little; a confident read of a DIFFERENT
# number nudges it DOWN a little. The weighting is SMALL and the whole feature
# is a strict NO-OP when the toggle is OFF or the tesseract binary is absent
# (rt2.ocr.available() is False) - it never touches tracking/confidence then.
OCR_MIN_BOX_H = 80           # only OCR when the target box is at least this tall (px)
OCR_EVERY_N = 10             # throttle: only OCR every Nth tracked frame
OCR_CONF_FLOOR = 0.50        # min OCR read confidence (0..1) to count a read
OCR_BOOST = 0.10             # confidence nudge UP when a read matches target_number
OCR_PENALTY = 0.08           # confidence nudge DOWN on a confident DIFFERENT number
OCR_REGION_TOP = 0.18        # upper-back region: top fraction of the box
OCR_REGION_BOT = 0.55        # upper-back region: bottom fraction of the box
OCR_REGION_CENTRAL = 0.60    # central horizontal fraction of the box kept

# ---- MY-NUMBER live jersey exclusion (Feature 1; INERT unless OCR available &
# self.my_number set). The user configures THEIR jersey number; in-ROI candidate
# crops are OCR'd (throttled) and confident reads are tallied per track id. A
# track whose top voted number is confidently a DIFFERENT number than mine is
# DROPPED from eligibility (like a rule-out), so the target can't cross onto a
# player whose number is confidently not mine; a track reading MY number gets a
# selection preference + a small confidence bump. The whole feature is a strict
# NO-OP when ocr.available() is False OR self.my_number is None.
MAX_JERSEY = 23              # plausible rugby shirt numbers are 1..23 (reject others)
MYNUM_OCR_EVERY_N = 5        # throttle: OCR candidate crops every Nth tracked frame
MYNUM_OCR_MAX_CANDS = 3      # at most this many in-ROI candidates OCR'd per pass
MYNUM_OCR_MIN_H = 46         # only OCR candidate boxes at least this tall (px)
MYNUM_OCR_CONF_FLOOR = 0.50  # min OCR read confidence (0..1) to count a vote
MYNUM_DECIDE_WEIGHT = 1.5    # accumulated weight a top number needs to be "decided"
MYNUM_PREF_BOOST = 0.30      # selection-score boost for a candidate reading MY number
MYNUM_CONF_BUMP = 0.12       # confidence bump for a target box reading MY number


def jersey_decided_number(counter, min_weight=MYNUM_DECIDE_WEIGHT):
    """PURE. Given a Counter of accumulated jersey-number votes (digit-string ->
    weight) for ONE track, return the top-voted number STRING if its weight is at
    least `min_weight` (a confident decision), else None. Ties / not-enough-weight
    -> None. Headlessly testable by --selftest."""
    if not counter:
        return None
    num, wt = counter.most_common(1)[0]
    return num if wt >= min_weight else None


def confidently_someone_else(counter, my_number, min_weight=MYNUM_DECIDE_WEIGHT):
    """PURE. True when this track is CONFIDENTLY a DIFFERENT player than me: its
    votes have decided on a number (>= min_weight accumulated) AND that number is
    not my_number. False when my_number is None, the votes are undecided, or the
    decided number IS mine. This is the exact predicate the eligibility filter
    uses to drop wrong-numbered tracks (mirrors the ruled_out filter)."""
    if my_number is None:
        return False
    decided = jersey_decided_number(counter, min_weight)
    if decided is None:
        return False
    try:
        return int(decided) != int(my_number)
    except (TypeError, ValueError):
        return False


# Manual-exit SNAP distance budget (Feature 2): the nearest eligible detection is
# only snapped to when its centre is within this many ROI WIDTHS of the ROI centre.
MANUAL_SNAP_ROI_WIDTHS = 1.5
MANUAL_SNAP_PAD = 0.15       # pad the snapped ROI by this fraction around the box


def manual_snap_choice(roi, boxes, max_roi_widths=MANUAL_SNAP_ROI_WIDTHS):
    """PURE. On manual-mode EXIT, choose the box to lock onto from `boxes` (the
    eligible candidate pool, xyxy). Picks the one whose centre is NEAREST the ROI
    centre, but ONLY if that centre is within `max_roi_widths` x the ROI width of
    the ROI centre. Returns the chosen box, or None if nothing is near enough /
    no candidates / no ROI. Headlessly testable by --selftest."""
    if roi is None or not boxes:
        return None
    rx1, ry1, rx2, ry2 = roi
    cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
    roi_w = max(1.0, float(rx2 - rx1))
    budget = max_roi_widths * roi_w
    best, best_d = None, None
    for b in boxes:
        bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        d = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best, best_d = b, d
    if best is None or best_d > budget:
        return None
    return best


# ===========================================================================
# TargetModel - the ONLINE model of the followed player (motion + appearance).
#
# Learns the player you track so candidate selection inside the ROI prefers the
# box that looks/moves like the target rather than just the one nearest the ROI
# centre. Pure logic, Qt-free, exercised headlessly by --selftest.
#
#   MOTION     : deque of confirmed (frame, cx, cy); predict_centre() extrapolates
#                from the average per-frame velocity over the recent samples.
#   APPEARANCE : incremental MEAN of L2-normalised H-S colour histograms of
#                confirmed target crops (grass hue masked); appearance_sim(crop)
#                is the cosine similarity in [0, 1] (0 if empty / invalid).
# ===========================================================================
class TargetModel:
    """Online motion + appearance model of the single followed player."""

    def __init__(self):
        self.motion = deque(maxlen=TM_MOTION_LEN)   # (frame, cx, cy)
        self._mean = None       # running mean histogram vector (np.float32, L2-normed input)
        self.count = 0          # number of crops folded into the appearance mean
        # HEIGHT-CONSISTENCY: running Welford stats over the TARGET's height
        # RESIDUAL (box_height - tracked_team.height.predict(centre_y)). Used to
        # gate out wrong-height candidates once enough samples are seen. These
        # persist in the profile so a player's height trait carries across clips.
        self.resid_count = 0
        self.resid_mean = 0.0
        self.resid_M2 = 0.0     # sum of squares of differences from the mean
        # MY JERSEY NUMBER (Feature 1): the user's own shirt number, persisted in
        # the profile so it carries across clips. None = not configured (the live
        # number-exclusion feature is then inert). Lives here purely so it rides
        # the same save_profile/load_profile JSON; it is NOT part of appearance.
        self.my_number = None

    # ---- lifecycle ----
    def reset(self):
        self.motion.clear()
        self._mean = None
        self.count = 0
        self.resid_count = 0
        self.resid_mean = 0.0
        self.resid_M2 = 0.0
        # NOTE: my_number is a deliberate, user-set config trait - reset()
        # (forget motion+appearance) must NOT wipe it.

    def learned(self):
        """(appearance_count, motion_samples) for status display."""
        return (self.count, len(self.motion))

    def has_data(self):
        return self.count > 0 or len(self.motion) >= 2

    # ---- appearance descriptor ----
    @staticmethod
    def _crop(img, box):
        """Integer-clamped BGR crop for `box`, or None if invalid/empty."""
        if img is None or box is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(round(x1), w - 1)))
        y1 = int(max(0, min(round(y1), h - 1)))
        x2 = int(max(0, min(round(x2), w)))
        y2 = int(max(0, min(round(y2), h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    @staticmethod
    def _hist(crop):
        """L2-normalised flattened H-S histogram of a BGR crop (grass masked).
        Returns a float32 vector, or None if the crop is unusable."""
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # mask out obvious grass hue so the descriptor keys on the player
        hue = hsv[:, :, 0]
        mask = ((hue < TM_GRASS_H_LO) | (hue > TM_GRASS_H_HI)).astype(np.uint8) * 255
        if int(mask.sum()) == 0:
            mask = None          # crop is all grass -> use everything rather than nothing
        hist = cv2.calcHist([hsv], [0, 1], mask, list(TM_HIST_BINS), [0, 180, 0, 256])
        v = hist.flatten().astype(np.float32)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return None
        return v / n

    def appearance_sim(self, crop):
        """Cosine similarity in [0,1] of `crop` to the learned mean; 0 if empty."""
        if self.count == 0 or self._mean is None:
            return 0.0
        v = self._hist(crop)
        if v is None:
            return 0.0
        mn = float(np.linalg.norm(self._mean))
        if mn <= 1e-12:
            return 0.0
        sim = float(np.dot(v, self._mean) / mn)   # v already unit-norm
        return max(0.0, min(1.0, sim))

    # ---- height residual (Welford running count/mean/variance) ----
    def update_height(self, resid):
        """Fold one TARGET height RESIDUAL into the running Welford stats.

        Silently ignores a None / non-finite residual (e.g. no calibration or
        no tracked-team height model on this frame), so the caller can always
        call this without first checking whether the model is ready."""
        if resid is None:
            return
        try:
            r = float(resid)
        except (TypeError, ValueError):
            return
        if not np.isfinite(r):
            return
        self.resid_count += 1
        delta = r - self.resid_mean
        self.resid_mean += delta / self.resid_count
        self.resid_M2 += delta * (r - self.resid_mean)

    def resid_std(self):
        """Population std of the learned height residual (0 with <2 samples)."""
        if self.resid_count < 2:
            return 0.0
        return float((self.resid_M2 / self.resid_count) ** 0.5)

    def height_ready(self):
        """True once enough residual samples are accumulated to gate on height."""
        return self.resid_count >= HEIGHT_MIN_SAMPLES

    def height_tol(self):
        """Tolerance band (px) for height gating: max(HEIGHT_MIN_PX, K*std)."""
        return max(HEIGHT_MIN_PX, HEIGHT_TOL_K * self.resid_std())

    # ---- motion prediction ----
    def predict_centre(self, next_frame):
        """Predicted (cx, cy) for `next_frame` from recent velocity, or None if
        fewer than 2 samples."""
        if len(self.motion) < 2:
            return None
        pts = list(self.motion)
        f0, x0, y0 = pts[0]
        f1, x1, y1 = pts[-1]
        df = f1 - f0
        if df <= 0:
            return (x1, y1)
        vx = (x1 - x0) / df
        vy = (y1 - y0) / df
        steps = next_frame - f1
        return (x1 + vx * steps, y1 + vy * steps)

    # ---- update (call when a target box is CONFIRMED for a frame) ----
    def update(self, frame, box, img):
        """Fold a confirmed target box for `frame` into motion + appearance.
        Silently skips invalid boxes / empty crops."""
        if box is None:
            return
        x1, y1, x2, y2 = box
        if x2 - x1 < 1 or y2 - y1 < 1:
            return
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.motion.append((int(frame), float(cx), float(cy)))
        v = self._hist(self._crop(img, box))
        if v is None:
            return
        if self._mean is None:
            self._mean = v.copy()
            self.count = 1
        else:
            # incremental running mean of the unit-norm histograms
            self._mean = (self._mean * self.count + v) / (self.count + 1)
            self.count += 1

    # ---- PERSISTENT appearance profile (carry-over across clips) ----
    # Only the APPEARANCE running-mean vector + its crop count persist; MOTION is
    # session-only (velocity from one clip is meaningless in the next). Stored as
    # plain JSON so it is human-inspectable and tiny.
    def save_profile(self, path):
        """Write the appearance mean vector + count to JSON. No-op (still writes an
        empty profile) when nothing has been learned, so the file always reflects
        the current state."""
        import json
        if self.count == 0 and self.my_number is None:
            return          # don't clobber an existing profile with an empty one
            #               (but DO persist a my_number set before any learning)
        path = pathlib.Path(path)
        data = {
            "version": 2,
            "count": int(self.count),
            "bins": list(TM_HIST_BINS),
            "mean": ([] if self._mean is None else self._mean.astype(float).tolist()),
            # HEIGHT-CONSISTENCY: the player's height trait carries across clips.
            "resid_count": int(self.resid_count),
            "resid_mean": float(self.resid_mean),
            "resid_M2": float(self.resid_M2),
            # MY JERSEY NUMBER (carries across clips). null when not configured.
            "my_number": (None if self.my_number is None else int(self.my_number)),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def load_profile(self, path):
        """Load an appearance mean vector + count from JSON into this model.
        Returns the crop count loaded (0 if missing / unusable / shape mismatch).
        Does not touch motion."""
        import json
        path = pathlib.Path(path)
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # MY JERSEY NUMBER: load FIRST so it carries over even when there is
            # no appearance data yet (a number set before any learning). Tolerate
            # old profiles (key absent) and malformed values.
            try:
                mn = data.get("my_number", None)
                self.my_number = None if mn is None else int(mn)
            except (TypeError, ValueError):
                self.my_number = None
            mean = data.get("mean") or []
            count = int(data.get("count", 0))
            expected = int(np.prod(TM_HIST_BINS))
            if count <= 0 or len(mean) != expected:
                return 0
            self._mean = np.asarray(mean, dtype=np.float32)
            self.count = count
            # HEIGHT-CONSISTENCY (added in version 2): tolerate OLD profiles that
            # predate it (the keys are simply absent -> start the residual model
            # fresh). Guard against malformed values.
            try:
                rc = int(data.get("resid_count", 0))
                rm = float(data.get("resid_mean", 0.0))
                rM2 = float(data.get("resid_M2", 0.0))
                if rc > 0 and np.isfinite(rm) and np.isfinite(rM2) and rM2 >= 0.0:
                    self.resid_count = rc
                    self.resid_mean = rm
                    self.resid_M2 = rM2
            except (TypeError, ValueError):
                pass
            return count
        except Exception:
            return 0


# ===========================================================================
# SAM 3 re-track window selection (Feature: loose-mark live, SAM3 re-tracks
# behind). Given the CURRENT frame, an optional MARKED start frame, the fps and
# a cap, return the (start, n) window to re-track. Marked start wins; otherwise
# default to the last ~3 seconds up to the current frame. The window length is
# capped at `max_frames` (SAM3 is ~2s/frame). Pure logic (Qt-free) so --selftest
# can exercise it headlessly.
# ===========================================================================
SAM3_DEFAULT_BACK_S = 3.0     # default look-back when no start is marked
SAM3_MAX_WINDOW_S = 12.0      # warn/cap: 12s window is ~6 min at ~2s/frame


def sam3_window(frame, start_mark, fps, max_frames):
    """-> (start, n): the 1-based first frame and frame count to re-track.

    `start_mark` is the user's marked IN-point (or None for the last ~3s).
    Both `start` and the window are clamped to >=1 frame and capped at
    `max_frames` (keeping the END = `frame` fixed, so the window shrinks from
    its start when capped)."""
    frame = int(frame)
    fps = float(fps) if fps else FPS_ASSUMED
    max_frames = max(1, int(max_frames))
    if start_mark is not None and 1 <= int(start_mark) <= frame:
        start = int(start_mark)
    else:
        start = max(1, frame - int(round(SAM3_DEFAULT_BACK_S * fps)))
    n = frame - start + 1
    if n > max_frames:                 # cap: keep END fixed, move start forward
        start = frame - max_frames + 1
        n = max_frames
    if n < 1:
        n = 1
        start = frame
    return start, n


# ===========================================================================
# LIVE PITCH-HOMOGRAPHY (M1) sequential-advance decision.
#
# Per-frame camera-motion compensation composes the anchor->current transform
# ONE frame at a time, so the live homography is only valid when we reach a
# frame sequentially from the last one we advanced to (or land exactly on the
# pitch anchor frame). This pure decision picks what to do at frame `f` given
# the pitch anchor frame and the last frame CMC was advanced to. Qt-free so
# --selftest exercises the (delicate) jump handling headlessly.
# ===========================================================================
def live_cmc_action(f, anchor_frame, last_frame):
    """-> 'anchor' | 'update' | 'stale' for displaying frame `f`.

    'anchor'  : f IS the pitch anchor frame -> (re)anchor the CMC here (trusted).
    'update'  : f is exactly one frame after the last advanced frame -> fold in
                this frame's motion (trusted).
    'stale'   : any jump / non-sequential seek / not-yet-anchored -> the
                homography cannot be trusted at this frame.
    """
    if anchor_frame is not None and f == anchor_frame:
        return "anchor"
    if last_frame is not None and f == last_frame + 1:
        return "update"
    return "stale"


# ===========================================================================
# TargetTrack - the consolidated single-player track + reversible op log.
#
# Ported from the v1 TrackStore (scripts/reanchor_gui.py): records keyed by a
# stable rid, forward_segment scoping to a >GAP-frame gap, merge_forward /
# delete_forward / nuke / bulk_delete / undo. Adapted to the v2 tracks schema
# (target_id + source) and to per-frame upsert from the live ROI tracker.
#
# Pure logic, no GUI -- exercised by --selftest.
# ===========================================================================
class TargetTrack:
    """In-memory target rows with stable row ids and a single-undo op log."""

    def __init__(self, target_id: int = 1):
        self.records = {}          # rid -> dict(frame,tid,x1,y1,x2,y2,source)
        self._next_rid = 0
        self.undo_stack = []       # list of undo records (one per user op)
        self.merges = 0
        self.deletes = 0           # counts tracks/segments deleted
        self.target = int(target_id)
        self._reindex()

    # ---- load / save (v2 schema) ----
    def load_rows(self, rows):
        """Populate from read_tracks() output (list of dicts)."""
        for r in rows:
            rid = self._next_rid; self._next_rid += 1
            self.records[rid] = {
                "frame": int(r["frame"]), "tid": int(r["target_id"]),
                "x1": float(r["x1"]), "y1": float(r["y1"]),
                "x2": float(r["x2"]), "y2": float(r["y2"]),
                "source": r.get("source", SOURCE_DETECTION),
            }
        self._reindex()

    def to_rows(self):
        """Rows in tracks_io / write_tracks shape."""
        return [
            {"frame": d["frame"], "target_id": d["tid"],
             "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
             "source": d["source"]}
            for d in self.records.values()
        ]

    def save(self, path):
        path = write_tracks(path, self.to_rows())
        return path

    # ---- indices ----
    def _reindex(self):
        self.by_frame = defaultdict(list)        # frame -> [rid]
        self.by_id = defaultdict(list)           # tid -> [(frame, rid)]
        for rid, d in self.records.items():
            self.by_frame[d["frame"]].append(rid)
            self.by_id[d["tid"]].append((d["frame"], rid))
        for tid in self.by_id:
            self.by_id[tid].sort()

    def unique_ids(self):
        return sorted({d["tid"] for d in self.records.values()})

    def id_stats(self, tid):
        frames = [fr for fr, _ in self.by_id.get(tid, [])]
        if not frames:
            return {"count": 0, "first": 0, "last": 0, "dur": 0.0}
        return {"count": len(frames), "first": min(frames), "last": max(frames),
                "dur": (max(frames) - min(frames) + 1) / FPS_ASSUMED}

    def target_box_at(self, frame):
        """Box of the target id at `frame`, or None."""
        for rid in self.by_frame.get(frame, []):
            d = self.records[rid]
            if d["tid"] == self.target:
                return (d["x1"], d["y1"], d["x2"], d["y2"])
        return None

    # ---- live ROI tracker upsert (Phase 4) ----
    def set_target_box(self, frame, box, source):
        """Upsert the target id's box for `frame`. Returns the rid.

        Recorded as ONE undoable op so 'u' steps back through tracking frames.
        """
        x1, y1, x2, y2 = box
        existing = None
        for rid in self.by_frame.get(frame, []):
            if self.records[rid]["tid"] == self.target:
                existing = rid
                break
        if existing is not None:
            old = dict(self.records[existing])
            self.records[existing].update(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "source": source})
            self.undo_stack.append({"kind": "upsert", "rid": existing, "old": old})
            self._reindex()
            return existing
        rid = self._next_rid; self._next_rid += 1
        self.records[rid] = {"frame": int(frame), "tid": self.target,
                             "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "source": source}
        self.undo_stack.append({"kind": "insert", "rid": rid})
        self._reindex()
        return rid

    def set_target_boxes(self, items, source):
        """Upsert MANY (frame, box) pairs for the target id as ONE undoable op.

        Used by the SAM3 background re-track merge so a single 'u' reverts the
        whole merge (mirrors remove_frames / bulk_delete which record one undo
        record for a batch). Only the listed frames are touched; frames outside
        the set are never modified. Returns the number of frames written.
        """
        inserts = []        # rids newly created (undo -> pop)
        upserts = []        # (rid, old-dict) overwritten (undo -> restore)
        for frame, box in items:
            x1, y1, x2, y2 = box
            existing = None
            for rid in self.by_frame.get(int(frame), []):
                if self.records[rid]["tid"] == self.target:
                    existing = rid
                    break
            if existing is not None:
                upserts.append((existing, dict(self.records[existing])))
                self.records[existing].update(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "source": source})
            else:
                rid = self._next_rid; self._next_rid += 1
                self.records[rid] = {"frame": int(frame), "tid": self.target,
                                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                     "source": source}
                inserts.append(rid)
        n = len(inserts) + len(upserts)
        if n:
            self.undo_stack.append({"kind": "batch_upsert",
                                    "inserts": inserts, "upserts": upserts})
            self._reindex()
        return n

    # ---- segment scoping (shared by merge-forward and delete-forward) ----
    def forward_segment(self, tid, start_frame, gap=GAP):
        """rids of `tid` from start_frame forward, stopping before the first
        absence gap larger than `gap` frames."""
        appics = [(fr, rid) for fr, rid in self.by_id.get(tid, []) if fr >= start_frame]
        if not appics:
            return []
        seg = [appics[0][1]]
        prev = appics[0][0]
        for fr, rid in appics[1:]:
            if fr - prev > gap:
                break
            seg.append(rid); prev = fr
        return seg

    # ---- mutations (each pushes one undo record) ----
    def merge_forward(self, src_id, dst_id, start_frame):
        seg = self.forward_segment(src_id, start_frame)
        if not seg:
            return 0
        changes = [(rid, self.records[rid]["tid"]) for rid in seg]
        for rid in seg:
            self.records[rid]["tid"] = dst_id
        self.undo_stack.append({"kind": "relabel", "data": changes, "counter": "merges"})
        self.merges += 1
        self._reindex()
        return len(seg)

    def _remove_rids(self, rids, counter, n_tracks):
        removed = [(rid, dict(self.records[rid])) for rid in rids if rid in self.records]
        for rid, _ in removed:
            del self.records[rid]
        self.undo_stack.append({"kind": "remove", "data": removed,
                                "counter": counter, "n": n_tracks})
        if counter == "deletes":
            self.deletes += n_tracks
        self._reindex()
        return len(removed)

    def delete_forward(self, tid, start_frame):
        seg = self.forward_segment(tid, start_frame)
        return self._remove_rids(seg, "deletes", 1) if seg else 0

    def nuke(self, tid):
        rids = [rid for _, rid in self.by_id.get(tid, [])]
        return self._remove_rids(rids, "deletes", 1) if rids else 0

    def bulk_delete(self, ids):
        ids = set(ids)
        rids = [rid for rid, d in self.records.items() if d["tid"] in ids]
        return self._remove_rids(rids, "deletes", len(ids)) if rids else 0

    def remove_frames(self, frames, tid=None):
        """Remove the `tid` rows at the given `frames` as ONE undoable op.

        Used by smart-clean ("clean jumps") to drop a set of outlier/teleport
        frames from the target track so a single 'u' restores them all. `tid`
        defaults to the current target id. Returns the number of rows removed.
        """
        if tid is None:
            tid = self.target
        want = set(int(f) for f in frames)
        rids = [rid for rid, d in self.records.items()
                if d["tid"] == tid and d["frame"] in want]
        return self._remove_rids(rids, "deletes", 0) if rids else 0

    def undo(self):
        if not self.undo_stack:
            return False
        rec = self.undo_stack.pop()
        kind = rec["kind"]
        if kind == "relabel":
            for rid, old in rec["data"]:
                if rid in self.records:
                    self.records[rid]["tid"] = old
            if rec.get("counter") == "merges":
                self.merges = max(0, self.merges - 1)
        elif kind == "remove":
            for rid, data in rec["data"]:
                self.records[rid] = data
            if rec.get("counter") == "deletes":
                self.deletes = max(0, self.deletes - rec.get("n", 1))
        elif kind == "insert":
            self.records.pop(rec["rid"], None)
        elif kind == "upsert":
            if rec["rid"] in self.records:
                self.records[rec["rid"]] = rec["old"]
        elif kind == "batch_upsert":
            for rid in rec["inserts"]:
                self.records.pop(rid, None)
            for rid, old in rec["upserts"]:
                if rid in self.records:
                    self.records[rid] = old
        self._reindex()
        return True


# ===========================================================================
# ROI <-> CSRT helpers
# ===========================================================================
def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def xywh_to_xyxy(b):
    x, y, w, h = b
    return (float(x), float(y), float(x + w), float(y + h))


def clamp_box(b, w, h):
    """Clamp an xyxy box to [0,w]x[0,h] keeping x1<x2, y1<y2."""
    x1, y1, x2, y2 = b
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(1.0, min(x2, w))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(1.0, min(y2, h))
    if x2 - x1 < 4:
        x2 = min(w, x1 + 4)
    if y2 - y1 < 4:
        y2 = min(h, y1 + 4)
    return (x1, y1, x2, y2)


def recovery_score(box, centre, search_box, tmodel, img):
    """Blended RECOVERY score for a single candidate `box` (xyxy).

    recov = W_RPOS*pos_term + W_RAPP*app_term
      pos_term = clamp(1 - dist(box_centre, centre) / search_diag, 0, 1)
      app_term = tmodel.appearance_sim(crop(box))  (0 when the model is empty / no img)

    Pure logic (Qt-free) so --selftest can exercise candidate selection headlessly.
    """
    bx, by = bbox.center(box)
    ccx, ccy = centre
    sx1, sy1, sx2, sy2 = search_box
    diag = ((sx2 - sx1) ** 2 + (sy2 - sy1) ** 2) ** 0.5
    if diag <= 1e-6:
        diag = 1.0
    d = ((bx - ccx) ** 2 + (by - ccy) ** 2) ** 0.5
    pos_term = max(0.0, min(1.0, 1.0 - d / diag))
    if img is not None and tmodel is not None and tmodel.count > 0:
        app_term = tmodel.appearance_sim(TargetModel._crop(img, box))
    else:
        app_term = 0.0
    return W_RPOS * pos_term + W_RAPP * app_term


def eligible_target_indices(labels):
    """Indices ELIGIBLE to be the followed target, given team_labels_for_frame
    output (a list of (team_name, is_tracked, confident) aligned with the
    detections at a frame).

    TEAM-CONSTRAINED SELECTION: never lock onto the opposition. Eligible = all
    detections EXCEPT those CONFIDENTLY classified as a NON-tracked team
    (is_tracked is False and confident is True). Tracked-team AND 'unsure'
    detections stay eligible so we never drop the real target when the
    classifier is uncertain. When there is no team info (team_name is None for
    every entry, i.e. calibration off / inert) everything is eligible.

    Returns a set of integer indices. Pure logic (Qt-free) so --selftest can
    exercise the filter headlessly.
    """
    out = set()
    for i, lab in enumerate(labels):
        name, is_tracked, confident = lab
        if name is not None and confident and not is_tracked:
            continue        # confident opposition -> never eligible
        out.add(i)
    return out


def height_gate_indices(indices, residuals, resid_mean, tol, protect=None):
    """HEIGHT-CONSISTENCY gating over a candidate index pool.

    `indices`     : iterable of candidate detection indices.
    `residuals`   : dict index -> height residual (or None when not measurable).
    `resid_mean`  : the TARGET's learned mean residual.
    `tol`         : tolerance band (px); a candidate is excluded when its
                    residual deviates from `resid_mean` by more than `tol`.
    `protect`     : optional iterable of indices NEVER excluded (the active
                    target + the armed detection), regardless of residual.

    Returns the kept set: candidates within tolerance, OR with no measurable
    height (residual is None), OR protected. Pure logic (Qt-free) so --selftest
    can exercise it headlessly. The CALLER decides whether gating is active
    (>= HEIGHT_MIN_SAMPLES + a height model present); when inert it must just
    pass every index through.
    """
    protect = set(protect or ())
    keep = set()
    for i in indices:
        if i in protect:
            keep.add(i)
            continue
        r = residuals.get(i)
        if r is None:
            keep.add(i)                 # no measurable height -> keep
            continue
        if abs(r - resid_mean) <= tol:
            keep.add(i)
    return keep


def identity_features(team_d_tracked, team_d_other, height_resid_dev,
                      norm_cx, norm_cy, motion_consistency,
                      box_h_frac, appearance_sim):
    """Build the FIXED-LENGTH (IDENTITY_DIM) "me vs not-me" feature vector from
    EXISTING per-detection signals. Pure logic (Qt-free) so --selftest can build
    a constant-dim vector for a synthetic detection headlessly.

    All inputs are plain floats; any None / non-finite value is replaced by the
    NEUTRAL 0.0 so the dimension is ALWAYS constant regardless of which signals
    are available this frame (no calibration, empty model, etc.):

      0 team_d_tracked     L1 fingerprint distance to the TRACKED team (0 if none)
      1 team_d_other       L1 fingerprint distance to the OPPOSITION (0 if none)
      2 team_margin        (d_other - d_tracked): >0 when this box looks tracked-team
      3 height_resid_dev   |box height residual - target's learned resid_mean| (px)
      4 norm_cx            box centre x / frame width   (0..1)
      5 norm_cy            box centre y / frame height  (0..1)
      6 motion_consistency dist(box centre, predicted centre)/frame_diag (0 if none)
      7 box_h_frac         box height / frame height    (0..1)
      8 appearance_sim     similarity of the crop to the learned player (0..1)

    NOTE the feature LIST documents 9 names but two are derived from the same two
    team distances; the returned vector length is held to IDENTITY_DIM exactly.
    """
    def f(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        return v if v == v and v not in (float("inf"), float("-inf")) else 0.0

    dt = f(team_d_tracked)
    do = f(team_d_other)
    vec = [
        dt,
        do,
        do - dt,                       # team margin (signed)
        f(height_resid_dev),
        f(norm_cx),
        f(norm_cy),
        f(motion_consistency),
        f(box_h_frac),
        f(appearance_sim),
    ]
    # hold the dimension constant at IDENTITY_DIM (pad / truncate defensively)
    if len(vec) < IDENTITY_DIM:
        vec = vec + [0.0] * (IDENTITY_DIM - len(vec))
    elif len(vec) > IDENTITY_DIM:
        vec = vec[:IDENTITY_DIM]
    return vec


def frame_confidence(source, height_term=None, app_term=None, margin_term=None,
                     id_term=None):
    """Blend a per-frame tracking CONFIDENCE in [0,1].

    Terms (each already in [0,1], or None when not available -> neutral 0.5):
      * SOURCE base: snapped same-team detection -> CONF_BASE_DETECTION (0.7),
        held / csrt -> CONF_BASE_HELD (0.35), manual -> CONF_BASE_MANUAL (0.9).
      * HEIGHT term : closeness of the chosen box residual to the learned mean
        (1 within ~0.5*std, decaying to 0 by HEIGHT_TOL) when the residual model
        is ready, else neutral 0.5.
      * APPEARANCE  : appearance similarity of the chosen crop to the learned
        player, else neutral 0.5.
      * MARGIN      : how dominant the chosen candidate is over the next-best
        eligible candidate (bigger score gap -> higher), else neutral 0.5.
      * IDENTITY    : the ONLINE "me vs not-me" classifier's prob(target) in
        [0,1] (rt2.identity), passed only once the model is ready(); else None
        (the whole identity term is then absent and the blend is UNCHANGED).

    The base four terms blend by CONF_W_* (summing to 1). When `id_term` is given
    that blend is then pulled toward the identity prob by CONF_W_ID, so a frame
    the classifier is sure is the target reads MORE confident and a frame it is
    sure is someone else reads LESS confident. Clamped to [0,1]. Pure logic so
    --selftest can exercise it headlessly.
    """
    base = {
        SOURCE_DETECTION: CONF_BASE_DETECTION,
        SOURCE_CSRT: CONF_BASE_HELD,
        SOURCE_MANUAL: CONF_BASE_MANUAL,
    }.get(source, CONF_BASE_HELD)
    h = 0.5 if height_term is None else max(0.0, min(1.0, height_term))
    a = 0.5 if app_term is None else max(0.0, min(1.0, app_term))
    m = 0.5 if margin_term is None else max(0.0, min(1.0, margin_term))
    conf = (CONF_W_BASE * base + CONF_W_HEIGHT * h
            + CONF_W_APP * a + CONF_W_MARGIN * m)
    # IDENTITY: fold in the classifier prob when the model is ready (id_term set).
    if id_term is not None:
        idt = max(0.0, min(1.0, id_term))
        conf = (1.0 - CONF_W_ID) * conf + CONF_W_ID * idt
    # MANUAL is hand-driven by the user, so trust it: floor it high (and let the
    # identity term never drag a user-placed box down into amber/red).
    if source == SOURCE_MANUAL:
        conf = max(conf, MANUAL_CONF_FLOOR)
    return max(0.0, min(1.0, conf))


def adaptive_marker_spec(box, conf):
    """CONFIDENCE-ADAPTIVE MARKER geometry for an xyxy target `box` at `conf`.

    Returns the shape to DRAW (all values in FRAME px), so the GUI just maps the
    points and strokes the outline. Pure logic (Qt-free) so --selftest can
    exercise it headlessly.

    Three confidence tiers (cut points ADAPT_CONF_HI / ADAPT_CONF_LO):
      * conf >= ADAPT_CONF_HI (0.70): a TIGHT body ELLIPSE centred on the box,
          semi-axes (ax, ay) ~ (0.55*w, 0.50*h)  ->  read as "locked on".
          -> ("ellipse", cx, cy, ax, ay)
      * ADAPT_CONF_LO <= conf < ADAPT_CONF_HI (0.40..0.70): a LARGER ellipse;
          the axes grow as confidence drops, lerping from the tight axes at HI up
          to ~ADAPT_ELL_GROW x the box at LO.
          -> ("ellipse", cx, cy, ax, ay)
      * conf < ADAPT_CONF_LO (0.40): a SEARCH CIRCLE whose radius GROWS as
          confidence falls - from ~ADAPT_CIRC_R_LO*diag at LO up to
          ~ADAPT_CIRC_R_MIN*diag as conf -> 0.
          -> ("circle", cx, cy, r)

    `conf` is clamped to [0,1]; a None conf is treated as fully confident (1.0)
    so a scrubbed frame with no rating shows the tight ellipse.
    """
    x1, y1, x2, y2 = box
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    diag = (w * w + h * h) ** 0.5
    c = 1.0 if conf is None else max(0.0, min(1.0, float(conf)))

    if c >= ADAPT_CONF_HI:
        return ("ellipse", cx, cy, ADAPT_ELL_AX_HI * w, ADAPT_ELL_AY_HI * h)

    if c >= ADAPT_CONF_LO:
        # lerp 0 (at HI -> tight) .. 1 (at LO -> grown); axes scale tight..GROW.
        span = max(1e-6, ADAPT_CONF_HI - ADAPT_CONF_LO)
        t = (ADAPT_CONF_HI - c) / span                 # 0 at HI, 1 at LO
        ax = (ADAPT_ELL_AX_HI + t * (ADAPT_ELL_GROW - ADAPT_ELL_AX_HI)) * w
        ay = (ADAPT_ELL_AY_HI + t * (ADAPT_ELL_GROW - ADAPT_ELL_AY_HI)) * h
        return ("ellipse", cx, cy, ax, ay)

    # conf < LO: search circle, radius grows as conf falls toward 0
    span = max(1e-6, ADAPT_CONF_LO)
    t = (ADAPT_CONF_LO - c) / span                     # 0 at LO, 1 at conf==0
    r = (ADAPT_CIRC_R_LO + t * (ADAPT_CIRC_R_MIN - ADAPT_CIRC_R_LO)) * diag
    return ("circle", cx, cy, r)


# TACTICAL (M2): true-metre player CIRCLES + OFFSIDE line. CIRCLE_R_M is the
# ground radius of a player's foot ring in pitch metres (~1 m reads cleanly).
CIRCLE_R_M = 1.0


def pitch_circle_points(cx_m, cy_m, r_m, n=28):
    """A ring of `n` PITCH (X, Y) points (metres) of radius `r_m` around the foot
    point (cx_m, cy_m). The GUI maps each point through the live homography
    (pitch -> image) so the ring renders as a perspective-correct, foreshortened
    ground ellipse at the player's feet. Pure geometry (Qt-free, no homography)
    so --selftest can exercise it headlessly."""
    pts = []
    for k in range(int(n)):
        a = 2.0 * np.pi * k / float(n)
        pts.append((cx_m + r_m * np.cos(a), cy_m + r_m * np.sin(a)))
    return pts


def conf_speed_mult(conf):
    """CONFIDENCE-ADAPTIVE SPEED multiplier (relative to the base fps interval).

    conf < CONF_SPEED_LOW_THRESH (0.40)  -> CONF_SPEED_LOW  (0.25x, crawl)
    0.40 <= conf < CONF_SPEED_HI_THRESH (0.70) -> CONF_SPEED_MID (0.5x)
    conf >= 0.70                         -> CONF_SPEED_HI   (1.75x)

    A None conf (no rating yet) defaults to full confidence (HI) so playback is
    never accidentally throttled before the first frame is tracked. Pure mapping
    (Qt-free) so --selftest can exercise it headlessly.
    """
    if conf is None:
        return CONF_SPEED_HI
    c = float(conf)
    if c >= CONF_SPEED_HI_THRESH:
        return CONF_SPEED_HI
    if c >= CONF_SPEED_LOW_THRESH:
        return CONF_SPEED_MID
    return CONF_SPEED_LOW


def find_jump_frames(rows, median_w=CLEAN_MEDIAN_W,
                     min_jump_px=MIN_JUMP_PX, jump_k=JUMP_K):
    """Identify teleport/outlier frames in a target track segment.

    `rows` is a list of (frame, (cx, cy), box) sorted by frame (the scoped
    target track). For each point i we compute the MEDIAN centre over the
    neighbour window i-W..i+W (W=median_w) -> a robust smoothed trajectory.
    deviation_i = euclidean distance(centre_i, smoothed_i). A point is an
    OUTLIER when deviation_i > threshold, where

        threshold = max(min_jump_px, jump_k * median(box larger-side))

    Returns the sorted list of outlier FRAME numbers. Pure logic (Qt-free) so
    --selftest can exercise it headlessly.
    """
    n = len(rows)
    if n == 0:
        return []
    xs = [r[1][0] for r in rows]
    ys = [r[1][1] for r in rows]
    sides = []
    for _, _, box in rows:
        x1, y1, x2, y2 = box
        sides.append(max(abs(x2 - x1), abs(y2 - y1)))
    med_side = float(np.median(sides)) if sides else 0.0
    threshold = max(float(min_jump_px), float(jump_k) * med_side)
    outliers = []
    for i in range(n):
        lo = max(0, i - median_w)
        hi = min(n, i + median_w + 1)
        sx = float(np.median(xs[lo:hi]))
        sy = float(np.median(ys[lo:hi]))
        dev = ((xs[i] - sx) ** 2 + (ys[i] - sy) ** 2) ** 0.5
        if dev > threshold:
            outliers.append(int(rows[i][0]))
    return sorted(outliers)


def aggregate_objects_rows(rows, fw, fh):
    """PURE: fold mark_all `objects.csv` rows into the per-frame overlay map +
    per-obj `candidates.TrackAgg` aggregates. Headlessly testable.

    `rows` is an iterable of dicts (a csv.DictReader, or hand-built) with keys
    frame, obj_id, x1, y1, x2, y2, team, conf. `fw`/`fh` are the frame size used
    to normalise the box centre into zone coords (zx 0=left..1=right touch;
    zy 0=top/far..1=bottom/near). Bad / short / garbled rows are skipped.

    Returns (objects_by_frame, aggs) where:
      objects_by_frame: dict[int frame] -> list[(obj_id, (x1,y1,x2,y2), team, conf)]
      aggs: dict[int obj_id] -> candidates.TrackAgg
    The optional `number` column (mark_all --ocr) feeds number_votes; without it
    number_votes stays {}. height_resid_mean is left None (not derivable here)."""
    objects_by_frame = {}
    aggs = {}
    fw = float(fw) if fw else 1.0
    fh = float(fh) if fh else 1.0
    for row in rows:
        try:
            frame = int(float(row["frame"]))
            oid = int(float(row["obj_id"]))
            x1 = float(row["x1"]); y1 = float(row["y1"])
            x2 = float(row["x2"]); y2 = float(row["y2"])
        except (KeyError, TypeError, ValueError):
            continue                                  # skip garbled row
        team = (row.get("team") or "").strip()
        try:
            conf = float(row.get("conf") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        box = (x1, y1, x2, y2)
        objects_by_frame.setdefault(frame, []).append((oid, box, team, conf))

        agg = aggs.get(oid)
        if agg is None:
            agg = candidates.TrackAgg(obj_id=oid)
            aggs[oid] = agg
        agg.n_frames += 1
        if team:
            agg.team_counts[team] = agg.team_counts.get(team, 0) + 1
        # jersey NUMBER (mark_all --ocr fills this) -> votes feed the shortlist, so
        # a track reading YOUR number is promoted to me-likely / others ruled out.
        num = (row.get("number") or "").strip()
        if num.isdigit() and 1 <= len(num) <= 2:
            agg.number_votes[num] = agg.number_votes.get(num, 0.0) + 1.0
        # running-mean box-centre, normalised to [0,1] zone coords
        zx = ((x1 + x2) * 0.5) / fw
        zy = ((y1 + y2) * 0.5) / fh
        n = agg.n_frames
        agg.zx_mean = zx if agg.zx_mean is None else agg.zx_mean + (zx - agg.zx_mean) / n
        agg.zy_mean = zy if agg.zy_mean is None else agg.zy_mean + (zy - agg.zy_mean) / n
    return objects_by_frame, aggs


def make_csrt():
    import cv2
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()   # pragma: no cover


# ===========================================================================
# ViewTransform - pure (Qt-free) screen<->ORIGINAL-frame coordinate mapping
# with letterboxing + zoom + pan.  This is THE coordinate-correctness core:
# the GUI's widget_to_frame / frame_to_widget delegate straight to it, so all
# click / ROI / handle / detection math goes through one exact, testable place.
# Exercised headlessly by selftest() (no QApplication, no display).
#
# Model:
#   fit  = min(ww/W, wh/H)            # aspect-preserving fit scale
#   s    = fit * zoom                 # total px-per-frame-pixel scale
#   the frame is drawn at scale `s`, its top-left at widget (ox, oy):
#       widget = (ox + fx*s, oy + fy*s)
#       frame  = ((wx-ox)/s, (wy-oy)/s)
#   pan is expressed as a frame-pixel offset (panx, pany): the frame point that
#   sits at the widget centre when pan == 0 is the frame centre; pan shifts it.
# ===========================================================================
ZOOM_MIN = 1.0           # 1x == fit-to-window
ZOOM_MAX = 12.0          # clamp hard zoom-in (players are tiny in the wide shot)


class ViewTransform:
    """Letterbox + zoom + pan mapping between widget px and ORIGINAL frame px.

    All values are floats. Round-trip exact: frame_to_widget(widget_to_frame(p)) == p.
    """

    def __init__(self, fw, fh):
        self.fw = float(fw)
        self.fh = float(fh)
        self.ww = 1.0
        self.wh = 1.0
        self.zoom = 1.0          # multiplier over the fit scale (1.0 == fit)
        self.panx = 0.0          # pan offset in FRAME pixels (added to centre)
        self.pany = 0.0

    # ---- geometry ----
    def set_widget_size(self, ww, wh):
        self.ww = float(max(1, ww))
        self.wh = float(max(1, wh))
        self.clamp()

    def fit_scale(self):
        if self.fw <= 0 or self.fh <= 0:
            return 1.0
        return min(self.ww / self.fw, self.wh / self.fh)

    def scale(self):
        """Total widget-px per frame-px."""
        return self.fit_scale() * self.zoom

    def draw_rect(self):
        """(ox, oy, dw, dh, s, s): where/at-what-scale the frame is painted.

        The frame point (cx + panx, cy + pany) - where (cx, cy) is the frame
        centre - is anchored to the widget centre.
        """
        s = self.scale()
        dw = self.fw * s
        dh = self.fh * s
        cx, cy = self.fw / 2.0 + self.panx, self.fh / 2.0 + self.pany
        # widget centre should map to frame point (cx, cy):
        #   ww/2 = ox + cx*s  ->  ox = ww/2 - cx*s
        ox = self.ww / 2.0 - cx * s
        oy = self.wh / 2.0 - cy * s
        return (ox, oy, dw, dh, s, s)

    # ---- mapping (exact round-trip) ----
    def widget_to_frame(self, wx, wy):
        ox, oy, dw, dh, sx, sy = self.draw_rect()
        if sx == 0 or sy == 0:
            return (0.0, 0.0)
        return ((wx - ox) / sx, (wy - oy) / sy)

    def frame_to_widget(self, fx, fy):
        ox, oy, dw, dh, sx, sy = self.draw_rect()
        return (ox + fx * sx, oy + fy * sy)

    def in_frame(self, fx, fy):
        return 0.0 <= fx <= self.fw and 0.0 <= fy <= self.fh

    # ---- pan clamping: never scroll the frame entirely off-screen ----
    def clamp(self):
        # zoom clamp
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom))
        s = self.scale()
        if s <= 0:
            self.panx = self.pany = 0.0
            return
        dw = self.fw * s
        dh = self.fh * s
        # If the drawn frame is wider/taller than the widget, allow panning up to
        # the point where an edge meets the widget edge; otherwise (frame fits)
        # keep it centred (pan == 0 on that axis).
        # ox = ww/2 - (fw/2 + panx)*s; we want 0 <= visible. Solve pan bounds so
        # that 0 <= -ox... -> express directly in terms of panx.
        # Visible-left frame edge at widget x: ox. Constrain ox in [ww-dw, 0]
        # when dw>=ww, else ox == (ww-dw)/2 (centred).
        if dw >= self.ww:
            # ox = ww/2 - (fw/2 + panx)*s  in  [ww - dw, 0]
            # -> panx in [ (ww/2)/s - fw/2 - (dw-ww)/s ... ]
            max_off = (dw - self.ww) / 2.0 / s     # max |panx| in frame px
            self.panx = max(-max_off, min(max_off, self.panx))
        else:
            self.panx = 0.0
        if dh >= self.wh:
            max_off = (dh - self.wh) / 2.0 / s
            self.pany = max(-max_off, min(max_off, self.pany))
        else:
            self.pany = 0.0

    # ---- zoom anchored under a widget point ----
    def zoom_at(self, wx, wy, factor):
        """Multiply zoom by `factor`, keeping the frame point currently under
        (wx, wy) under the cursor afterwards."""
        fx, fy = self.widget_to_frame(wx, wy)
        old = self.zoom
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        if self.zoom == old:
            return
        # After changing scale, choose pan so (fx, fy) maps back to (wx, wy):
        #   wx = ox + fx*s  and  ox = ww/2 - (fw/2 + panx)*s
        #   -> panx = (ww/2 - wx)/s + fx - fw/2
        s = self.scale()
        self.panx = (self.ww / 2.0 - wx) / s + fx - self.fw / 2.0
        self.pany = (self.wh / 2.0 - wy) / s + fy - self.fh / 2.0
        self.clamp()

    def pan_by_widget(self, dwx, dwy):
        """Pan by a widget-pixel delta (e.g. a drag)."""
        s = self.scale()
        if s == 0:
            return
        # moving the view content right by dwx widget px == decreasing panx
        self.panx -= dwx / s
        self.pany -= dwy / s
        self.clamp()

    def reset(self):
        self.zoom = 1.0
        self.panx = 0.0
        self.pany = 0.0
        self.clamp()


# ===========================================================================
# Headless self-test (no GUI, no display)
# ===========================================================================
def selftest():
    print("[selftest] building synthetic target track")
    t = TargetTrack(target_id=1)

    # synth: two ids. id 5 is a long contiguous run with one big internal gap;
    # id 2 is short. Use them to exercise scoping + the re-anchor ops.
    rows = []
    for fr in range(1, 41):                       # id 5: frames 1..40 contiguous
        rows.append({"frame": fr, "target_id": 5, "x1": fr, "y1": fr,
                     "x2": fr + 50, "y2": fr + 80, "source": SOURCE_DETECTION})
    for fr in range(120, 131):                     # id 5: frames 120..130 (gap=80 > GAP)
        rows.append({"frame": fr, "target_id": 5, "x1": fr, "y1": fr,
                     "x2": fr + 50, "y2": fr + 80, "source": SOURCE_DETECTION})
    for fr in range(1, 16):                        # id 2: short
        rows.append({"frame": fr, "target_id": 2, "x1": 500, "y1": 500,
                     "x2": 560, "y2": 600, "source": SOURCE_CSRT})
    t.load_rows(rows)

    ids = t.unique_ids()
    print(f"[selftest] {len(t.records)} rows, ids {ids}")
    assert ids == [2, 5]

    # forward_segment stops at the >GAP gap (must NOT cross frame 40 -> 120)
    seg = t.forward_segment(5, 1)
    seg_frames = sorted(t.records[r]["frame"] for r in seg)
    assert seg_frames == list(range(1, 41)), seg_frames
    assert all(seg_frames[i + 1] - seg_frames[i] <= GAP for i in range(len(seg_frames) - 1))
    print(f"[selftest] forward_segment(5, 1) -> {len(seg)} rows, "
          f"frames {seg_frames[0]}..{seg_frames[-1]} (stops before gap)  OK")

    # starting mid-gap picks up only the second cluster
    seg2 = t.forward_segment(5, 100)
    f2 = sorted(t.records[r]["frame"] for r in seg2)
    assert f2 == list(range(120, 131)), f2
    print(f"[selftest] forward_segment(5, 100) -> frames {f2[0]}..{f2[-1]}  OK")

    # merge_forward(5 -> 1) then undo restores exactly
    before = {rid: d["tid"] for rid, d in t.records.items()}
    n = t.merge_forward(5, 1, 1)
    assert n == len(seg)
    assert all(t.records[r]["tid"] == 1 for r in seg), "merge didn't relabel"
    assert t.merges == 1
    t.undo()
    after = {rid: d["tid"] for rid, d in t.records.items()}
    assert before == after, "undo(merge) did not restore tids"
    assert t.merges == 0
    print(f"[selftest] merge_forward(5->1) relabelled {n}; undo restored  OK")

    # delete_forward then undo restores row set
    total_before = len(t.records)
    n = t.delete_forward(5, 1)
    assert len(t.records) == total_before - n
    assert t.deletes == 1
    t.undo()
    assert len(t.records) == total_before, "undo(delete) did not restore rows"
    assert t.deletes == 0
    print(f"[selftest] delete_forward removed {n}; undo restored {total_before}  OK")

    # nuke then undo
    cnt = t.id_stats(5)["count"]
    n = t.nuke(5)
    assert n == cnt and 5 not in t.unique_ids(), "nuke incomplete"
    t.undo()
    assert 5 in t.unique_ids(), "undo(nuke) did not restore id"
    print(f"[selftest] nuke(5) removed {n}; undo restored  OK")

    # bulk_delete (single undo) then undo
    total_before = len(t.records)
    victims = [2, 5]
    n = t.bulk_delete(victims)
    assert all(v not in t.unique_ids() for v in victims), "bulk_delete incomplete"
    assert len(t.undo_stack) >= 1
    t.undo()
    assert len(t.records) == total_before, "undo(bulk) did not restore"
    assert all(v in t.unique_ids() for v in victims), "undo(bulk) lost ids"
    print(f"[selftest] bulk_delete({victims}) removed {n} rows as ONE undo; restored  OK")

    # live ROI upsert: insert then overwrite the same frame, each undoable
    t2 = TargetTrack(target_id=1)
    t2.set_target_box(200, (10, 10, 60, 90), SOURCE_CSRT)
    assert t2.target_box_at(200) == (10, 10, 60, 90)
    t2.set_target_box(200, (11, 11, 61, 91), SOURCE_DETECTION)   # overwrite
    assert t2.target_box_at(200) == (11, 11, 61, 91)
    assert len([r for r in t2.records.values() if r["frame"] == 200]) == 1, "upsert duplicated"
    t2.undo()                                            # back to first box
    assert t2.target_box_at(200) == (10, 10, 60, 90), "undo(upsert) failed"
    t2.undo()                                            # back to nothing
    assert t2.target_box_at(200) is None, "undo(insert) failed"
    print("[selftest] live ROI upsert/insert + undo  OK")

    # SAM3 batch merge: many (frame, box) pairs upserted as ONE undoable op
    # (a single 'u' reverts the whole merge). Mixes new inserts with overwrites.
    t3 = TargetTrack(target_id=1)
    t3.set_target_box(300, (1, 1, 2, 2), SOURCE_CSRT)          # pre-existing frame
    base = len(t3.undo_stack)
    items = [(300, (10, 10, 60, 90)), (301, (11, 11, 61, 91)),
             (302, (12, 12, 62, 92))]
    n = t3.set_target_boxes(items, SOURCE_SAM2)
    assert n == 3, n
    assert len(t3.undo_stack) == base + 1, "batch merge must be ONE undo record"
    assert t3.target_box_at(300) == (10, 10, 60, 90)          # overwrote
    assert t3.target_box_at(302) == (12, 12, 62, 92)          # inserted
    t3.undo()                                                  # ONE undo reverts all
    assert t3.target_box_at(300) == (1, 1, 2, 2), "undo(batch) didn't restore overwrite"
    assert t3.target_box_at(301) is None, "undo(batch) didn't drop insert"
    assert t3.target_box_at(302) is None, "undo(batch) didn't drop insert"
    print("[selftest] SAM3 batch merge (set_target_boxes) + single undo  OK")

    # sam3_window: marked-start wins; default = last ~3s; cap at max_frames.
    fps = 30.0
    s, n = sam3_window(1000, 880, fps, max_frames=12 * 30)     # marked start
    assert (s, n) == (880, 121), (s, n)
    s, n = sam3_window(1000, None, fps, max_frames=12 * 30)    # default last 3s
    assert (s, n) == (1000 - 90, 91), (s, n)
    s, n = sam3_window(1000, 1, fps, max_frames=12 * 30)       # huge -> capped
    assert n == 360 and s == 1000 - 360 + 1, (s, n)
    s, n = sam3_window(1000, 1200, fps, max_frames=360)        # bad start (>frame) -> default
    assert (s, n) == (1000 - 90, 91), (s, n)
    s, n = sam3_window(5, None, fps, max_frames=360)           # near start clamps to 1
    assert s == 1 and n == 5, (s, n)
    print("[selftest] sam3_window: marked/default/cap/clamp  OK")

    # ROI<->CSRT round-trip
    b = (100.0, 200.0, 300.0, 500.0)
    assert xywh_to_xyxy(xyxy_to_xywh(b)) == b, "xyxy<->xywh round-trip"
    print("[selftest] ROI<->CSRT (xyxy<->xywh) round-trip  OK")

    # ---------------------------------------------------------------
    # ViewTransform: screen<->ORIGINAL-frame round-trip under zoom+pan.
    # This runs WITHOUT any display / QApplication - pure float math.
    # frame_to_widget(widget_to_frame(p)) must equal p to < 1px at every
    # zoom level and pan offset (the make-or-break coordinate guarantee).
    # ---------------------------------------------------------------
    vt = ViewTransform(1920, 1080)
    # a few realistic widget sizes (incl. non-matching aspect -> letterbox)
    worst = 0.0
    checks = 0
    for (ww, wh) in [(1280, 820), (1920, 1080), (800, 800), (640, 360), (1000, 1300)]:
        vt.set_widget_size(ww, wh)
        for zoom in [1.0, 1.5, 3.0, 7.0, 12.0]:
            vt.reset()
            # zoom anchored under a couple of cursor points (also sets pan)
            vt.zoom_at(ww * 0.30, wh * 0.40, zoom)
            for (panx, pany) in [(0, 0), (300, -200), (-700, 400), (5000, -5000)]:
                vt.panx, vt.pany = panx, pany
                vt.clamp()
                for (wx, wy) in [(0, 0), (ww * 0.5, wh * 0.5),
                                 (ww, wh), (ww * 0.13, wh * 0.77),
                                 (ww * 0.91, wh * 0.05)]:
                    fx, fy = vt.widget_to_frame(wx, wy)
                    bx, by = vt.frame_to_widget(fx, fy)
                    err = max(abs(bx - wx), abs(by - wy))
                    worst = max(worst, err)
                    checks += 1
                    assert err < 1e-6, (
                        f"round-trip widget->frame->widget off by {err} "
                        f"at ww={ww},wh={wh},zoom={zoom},pan=({panx},{pany}),"
                        f"w=({wx},{wy})")
                    # also frame->widget->frame
                    fbx, fby = vt.widget_to_frame(bx, by)
                    ferr = max(abs(fbx - fx), abs(fby - fy))
                    assert ferr < 1e-6, f"frame round-trip off by {ferr}"
    # clamp keeps the frame partly on screen: at any zoom>1 the widget centre
    # must still land inside the frame.
    vt.set_widget_size(1280, 820)
    vt.reset(); vt.zoom_at(0, 0, 12.0)
    vt.panx, vt.pany = 1e9, 1e9; vt.clamp()
    cfx, cfy = vt.widget_to_frame(1280 / 2, 820 / 2)
    assert vt.in_frame(cfx, cfy), f"pan clamp let view leave frame: {(cfx, cfy)}"
    print(f"[selftest] ViewTransform zoom+pan round-trip: {checks} checks, "
          f"worst err {worst:.2e}px (<1px), pan-clamp OK")

    # ---------------------------------------------------------------
    # TargetModel: motion prediction + appearance similarity on synthetic
    # crops. Pure numpy/cv2, no GUI / display.
    # ---------------------------------------------------------------
    tm = TargetModel()
    assert tm.predict_centre(10) is None, "empty model must not predict"
    assert tm.has_data() is False
    assert tm.appearance_sim(np.zeros((8, 8, 3), np.uint8)) == 0.0

    # synthetic frame: a RED player box on a GREEN (grass) background
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (0, 200, 0)                       # BGR green background
    red_box = (40, 40, 80, 120)
    img[40:120, 40:80] = (0, 0, 220)              # BGR red "player"
    blue_box = (120, 40, 160, 120)
    img[40:120, 120:160] = (220, 0, 0)            # BGR blue "other player"

    # feed two confirmed centres so motion can predict the third
    tm.update(1, red_box, img)
    tm.update(2, (red_box[0] + 10, red_box[1], red_box[2] + 10, red_box[3]), img)
    assert tm.has_data() is True
    cnt, ms = tm.learned()
    assert cnt == 2 and ms == 2, (cnt, ms)
    pred = tm.predict_centre(3)
    assert pred is not None
    # moved +10px/frame in x -> next centre ~ +10 from last
    last_cx = (red_box[0] + 10 + red_box[2] + 10) / 2.0
    assert abs(pred[0] - (last_cx + 10)) < 1e-6, pred
    assert abs(pred[1] - (red_box[1] + red_box[3]) / 2.0) < 1e-6, pred

    # appearance: the RED crop must match the learned (red) mean better than BLUE
    sim_red = tm.appearance_sim(TargetModel._crop(img, red_box))
    sim_blue = tm.appearance_sim(TargetModel._crop(img, blue_box))
    assert 0.0 <= sim_blue <= sim_red <= 1.0, (sim_red, sim_blue)
    assert sim_red > sim_blue, (sim_red, sim_blue)
    # invalid crop -> 0 similarity, no crash
    assert tm.appearance_sim(None) == 0.0
    assert tm.appearance_sim(np.zeros((0, 0, 3), np.uint8)) == 0.0

    tm.reset()
    assert tm.predict_centre(3) is None and tm.learned() == (0, 0)
    print(f"[selftest] TargetModel motion+appearance: pred {pred[0]:.1f},{pred[1]:.1f}  "
          f"sim red {sim_red:.3f} > blue {sim_blue:.3f}; reset OK")

    # ---------------------------------------------------------------
    # RECOVERY scoring: on a synthetic candidate set the nearer / better-
    # matching box must win, and the blend must be position-dominant.
    # ---------------------------------------------------------------
    centre = (100.0, 100.0)
    search_box = (0.0, 0.0, 200.0, 200.0)        # diag ~= 283
    near = (90.0, 90.0, 110.0, 110.0)            # centre (100,100) - bang on
    far = (170.0, 170.0, 190.0, 190.0)           # centre (180,180) - far corner
    cands = [far, near]
    # no appearance model -> pure position; the NEAR box must score highest
    scores = [recovery_score(b, centre, search_box, None, None) for b in cands]
    assert scores[1] > scores[0], scores         # near > far
    best = max(cands, key=lambda b: recovery_score(b, centre, search_box, None, None))
    assert best == near, best
    # exact pos-only value for the on-centre box: dist 0 -> pos_term 1 -> W_RPOS
    assert abs(scores[1] - W_RPOS) < 1e-6, scores
    # a box dead-centre with NO appearance signal must fail the RECOVER_MIN gate
    # only via appearance? No: pos alone here is W_RPOS(0.6) >= RECOVER_MIN(0.45).
    # Verify the gate maths is the documented blend, not silently inverted:
    assert W_RPOS + W_RAPP == 1.0
    # appearance tilts ties: build a model that likes RED, give two equidistant
    # boxes (one red, one blue) -> the red one must win on the app term.
    tm2 = TargetModel()
    img2 = np.zeros((220, 260, 3), np.uint8)
    img2[:, :] = (0, 200, 0)
    red2 = (40, 40, 80, 120); img2[40:120, 40:80] = (0, 0, 220)
    blue2 = (140, 40, 180, 120); img2[40:120, 140:180] = (220, 0, 0)
    tm2.update(1, red2, img2); tm2.update(2, red2, img2)
    c2 = (110.0, 80.0)                            # equidistant-ish midpoint
    sb2 = (0.0, 0.0, 260.0, 220.0)
    s_red = recovery_score(red2, c2, sb2, tm2, img2)
    s_blue = recovery_score(blue2, c2, sb2, tm2, img2)
    assert s_red > s_blue, (s_red, s_blue)        # appearance breaks the tie toward red
    print(f"[selftest] recovery scoring: near {scores[1]:.3f} > far {scores[0]:.3f}; "
          f"red {s_red:.3f} > blue {s_blue:.3f}  OK")

    # ---------------------------------------------------------------
    # PERSISTENT profile: save_profile / load_profile round-trips the
    # appearance mean vector + count (motion is NOT persisted).
    # ---------------------------------------------------------------
    import tempfile, os
    tm_src = TargetModel()
    tm_src.update(1, red2, img2)
    tm_src.update(2, (red2[0] + 8, red2[1], red2[2] + 8, red2[3]), img2)
    assert tm_src.count == 2 and tm_src._mean is not None
    tmpd = tempfile.mkdtemp(prefix="rt2_profile_")
    pfile = os.path.join(tmpd, PROFILE_NAME)
    tm_src.save_profile(pfile)
    assert os.path.exists(pfile)
    tm_dst = TargetModel()
    n_loaded = tm_dst.load_profile(pfile)
    assert n_loaded == 2, n_loaded
    assert tm_dst.count == tm_src.count
    assert tm_dst._mean is not None
    assert np.allclose(tm_dst._mean, tm_src._mean, atol=1e-6), "profile mean mismatch"
    assert len(tm_dst.motion) == 0, "motion must NOT be persisted"
    # an empty model round-trips to a no-op load (count 0)
    empty = TargetModel()
    epath = os.path.join(tmpd, "empty.json")
    empty.save_profile(epath)
    assert TargetModel().load_profile(epath) == 0
    # missing file -> 0, no crash
    assert TargetModel().load_profile(os.path.join(tmpd, "nope.json")) == 0

    # HEIGHT residual persists in the profile (version 2) and round-trips.
    tm_h = TargetModel()
    tm_h.update(1, red2, img2)                # need >=1 crop or save is a no-op
    for r in (10.0, 12.0, 8.0, 11.0, 9.0):
        tm_h.update_height(r)
    assert tm_h.resid_count == 5
    hpath = os.path.join(tmpd, "height.json")
    tm_h.save_profile(hpath)
    import json as _json
    saved = _json.loads(open(hpath, encoding="utf-8").read())
    assert saved.get("version") == 2, saved.get("version")
    tm_h2 = TargetModel()
    tm_h2.load_profile(hpath)
    assert tm_h2.resid_count == tm_h.resid_count
    assert abs(tm_h2.resid_mean - tm_h.resid_mean) < 1e-9
    assert abs(tm_h2.resid_std() - tm_h.resid_std()) < 1e-9, \
        (tm_h2.resid_std(), tm_h.resid_std())
    # OLD profile (version 1, NO height keys) is tolerated -> residual starts fresh
    old_path = os.path.join(tmpd, "old_v1.json")
    open(old_path, "w", encoding="utf-8").write(_json.dumps({
        "version": 1, "count": tm_src.count, "bins": list(TM_HIST_BINS),
        "mean": tm_src._mean.astype(float).tolist()}))
    tm_old = TargetModel()
    assert tm_old.load_profile(old_path) == tm_src.count
    assert tm_old.resid_count == 0, "old file must not invent residual data"
    for f in (hpath, old_path):
        try:
            os.remove(f)
        except OSError:
            pass
    # cleanup
    for f in (pfile, epath):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmpd)
    except OSError:
        pass
    print(f"[selftest] persistent profile save/load round-trip "
          f"({n_loaded} crops, motion not persisted)  OK")

    # ---------------------------------------------------------------
    # DISPLAY FILTERS: visible_det_indices logic (cosmetic declutter).
    # The GUI method depends on Qt, so exercise the SAME logic headlessly
    # against synthetic team labels + a stub field mask. Mirrors the rules
    # in MainWindow.visible_det_indices: keep target + armed always; hide
    # confidently non-tracked teams; hide feet-off-field detections.
    # ---------------------------------------------------------------
    def _visible_indices(dets, labels, mask, hide_teams, hide_off,
                         keep_active=None, armed_box=None):
        visible = set(range(len(dets)))
        if not dets:
            return visible
        keep = set()
        if keep_active is not None:
            keep.add(keep_active)
        if armed_box is not None:
            for i, d in enumerate(dets):
                if tuple(d[:4]) == tuple(armed_box):
                    keep.add(i); break
        if hide_teams:
            for i in list(visible):
                if i in keep:
                    continue
                name, is_tracked, confident = labels[i]
                if name is not None and confident and not is_tracked:
                    visible.discard(i)
        if hide_off and mask is not None:
            for i in list(visible):
                if i in keep:
                    continue
                if not field.feet_in_field(mask, dets[i][:4]):
                    visible.discard(i)
        visible |= keep
        return visible

    # dets: 0 = tracked team on-field, 1 = OTHER team on-field (confident),
    #       2 = unsure on-field, 3 = tracked team OFF-field (crowd)
    fdets = [(10, 10, 30, 90, 0.9),     # tracked, on field
             (40, 10, 60, 90, 0.9),     # other team, confident, on field
             (70, 10, 90, 90, 0.9),     # unsure, on field
             (10, 110, 30, 150, 0.9)]   # tracked, but feet off field
    flabels = [("Mine", True, True), ("Hautapu", False, True),
               ("Mine", True, False), ("Mine", True, True)]
    # stub field mask: pitch only in the top 100 rows
    fmask = np.zeros((200, 120), np.uint8)
    fmask[0:100, :] = 255
    assert field.feet_in_field(fmask, fdets[0][:4])
    assert not field.feet_in_field(fmask, fdets[3][:4])

    # no filters -> everything visible
    assert _visible_indices(fdets, flabels, fmask, False, False) == {0, 1, 2, 3}
    # hide other teams -> drop the confident OTHER team (#1), keep unsure (#2)
    assert _visible_indices(fdets, flabels, fmask, True, False) == {0, 2, 3}
    # hide off-field -> drop #3 (feet off pitch)
    assert _visible_indices(fdets, flabels, fmask, False, True) == {0, 1, 2}
    # both -> drop #1 and #3
    assert _visible_indices(fdets, flabels, fmask, True, True) == {0, 2}
    # target (#1) always kept even though it's a confident other team
    assert 1 in _visible_indices(fdets, flabels, fmask, True, True, keep_active=1)
    # armed box (#3 off-field) always kept too
    assert 3 in _visible_indices(fdets, flabels, fmask, True, True,
                                 armed_box=fdets[3][:4])
    # no team info (calibration off) -> hide_other_teams hides nothing
    nolabels = [(None, None, False) for _ in fdets]
    assert _visible_indices(fdets, nolabels, fmask, True, False) == {0, 1, 2, 3}
    print("[selftest] display filters visible_det_indices "
          "(team/off-field + always-keep target/armed)  OK")

    # ---------------------------------------------------------------
    # SMART-CLEAN ("clean jumps"): find_jump_frames + remove_frames undo.
    # Build a smooth synthetic target track moving +5px/frame, inject a few
    # teleport spike frames, and assert the spikes are flagged/removed while
    # clean frames are kept, and that ONE undo restores everything.
    # ---------------------------------------------------------------
    tc = TargetTrack(target_id=1)
    spike_frames = {25, 26, 60}            # injected teleports
    crows = []
    for fr in range(1, 101):
        cx = 100.0 + 5.0 * fr              # smooth diagonal drift
        cy = 200.0 + 3.0 * fr
        if fr in spike_frames:
            cx += 600.0                    # huge jump (ROI snapped to ref)
            cy -= 500.0
        crows.append({"frame": fr, "target_id": 1,
                      "x1": cx - 20, "y1": cy - 40, "x2": cx + 20, "y2": cy + 40,
                      "source": SOURCE_CSRT})
    tc.load_rows(crows)

    scoped = []
    for fr, rid in tc.by_id.get(1, []):
        d = tc.records[rid]
        box = (d["x1"], d["y1"], d["x2"], d["y2"])
        scoped.append((fr, bbox.center(box), box))
    scoped.sort(key=lambda r: r[0])
    found = find_jump_frames(scoped)
    assert set(found) == spike_frames, (found, spike_frames)
    # the threshold (>=MIN_JUMP_PX) must NOT flag any smooth frame
    assert all(f in spike_frames for f in found), found

    before = len(tc.records)
    n = tc.remove_frames(found, tid=1)
    assert n == len(spike_frames), (n, spike_frames)
    assert len(tc.records) == before - n
    # the spike frames are gone, clean frames remain
    remaining = {tc.records[r]["frame"] for _, r in tc.by_id.get(1, [])}
    assert remaining.isdisjoint(spike_frames), remaining & spike_frames
    assert 1 in remaining and 100 in remaining and 50 in remaining
    # ONE undo restores all removed frames
    tc.undo()
    assert len(tc.records) == before, "undo(remove_frames) did not restore"
    restored = {tc.records[r]["frame"] for _, r in tc.by_id.get(1, [])}
    assert spike_frames.issubset(restored), "undo did not restore spike frames"
    # no jumps in a perfectly smooth track -> empty
    smooth = TargetTrack(target_id=1)
    smooth.load_rows([
        {"frame": fr, "target_id": 1,
         "x1": 5.0 * fr, "y1": 3.0 * fr, "x2": 5.0 * fr + 40, "y2": 3.0 * fr + 80,
         "source": SOURCE_CSRT} for fr in range(1, 51)])
    srows = []
    for fr, rid in smooth.by_id.get(1, []):
        d = smooth.records[rid]
        box = (d["x1"], d["y1"], d["x2"], d["y2"])
        srows.append((fr, bbox.center(box), box))
    srows.sort(key=lambda r: r[0])
    assert find_jump_frames(srows) == [], "false positive on smooth track"
    print(f"[selftest] clean_jumps: flagged+removed {sorted(spike_frames)} as ONE "
          f"undo, kept clean frames, restored on undo  OK")

    # ---------------------------------------------------------------
    # TEAM-CONSTRAINED SELECTION: eligible_target_indices excludes a
    # CONFIDENT opposition (non-tracked) detection but keeps tracked-team
    # AND 'unsure' detections, and keeps everything when calibration is off.
    # ---------------------------------------------------------------
    # 0=tracked confident, 1=opposition confident, 2=unsure, 3=opposition UNSURE
    elig_labels = [("Mine", True, True), ("Hautapu", False, True),
                   ("Mine", True, False), ("Hautapu", False, False)]
    elig = eligible_target_indices(elig_labels)
    assert 1 not in elig, "confident opposition must be excluded"
    assert 0 in elig and 2 in elig, "tracked + unsure must stay eligible"
    assert 3 in elig, "UNSURE opposition must stay eligible (classifier unsure)"
    # no calibration / inert -> everything eligible
    elig_none = eligible_target_indices([(None, None, False) for _ in range(4)])
    assert elig_none == {0, 1, 2, 3}, elig_none
    # all-confident-opposition -> nothing eligible (would force a HOLD)
    assert eligible_target_indices([("Opp", False, True)]) == set()
    print("[selftest] team-constrained eligible_target_indices "
          "(drop confident opposition, keep tracked/unsure)  OK")

    # ---------------------------------------------------------------
    # MANUAL FOLLOW MODE recording: the manual step must write the CURRENT
    # ROI box for the frame with SOURCE_MANUAL (mirrors _record_manual,
    # which the GUI calls on every advance while manual_mode is ON).
    # ---------------------------------------------------------------
    tman = TargetTrack(target_id=7)
    manual_roi = (300.0, 150.0, 360.0, 270.0)
    # what _record_manual does for a frame, headlessly:
    tman.set_target_box(42, manual_roi, SOURCE_MANUAL)
    assert tman.target_box_at(42) == manual_roi, "manual ROI not recorded"
    rec = next(d for d in tman.records.values() if d["frame"] == 42)
    assert rec["source"] == SOURCE_MANUAL, rec["source"]
    assert rec["tid"] == 7, "manual record must use the target id"
    # a second advance to a new frame records the (possibly moved) ROI again
    moved_roi = (320.0, 160.0, 380.0, 280.0)
    tman.set_target_box(43, moved_roi, SOURCE_MANUAL)
    assert tman.target_box_at(43) == moved_roi
    print("[selftest] manual-mode ROI recording (SOURCE_MANUAL per frame)  OK")

    # ---------------------------------------------------------------
    # HEIGHT-CONSISTENCY: Welford residual accumulation matches numpy,
    # gating activates only at >=HEIGHT_MIN_SAMPLES, excludes a clearly-
    # wrong-height candidate, keeps in-tolerance ones, and never drops the
    # protected (active target / armed) index.
    # ---------------------------------------------------------------
    tmh = TargetModel()
    samples = [50.0, 52.0, 48.0, 51.0, 49.0, 53.0, 47.0, 50.5,
               49.5, 51.5, 48.5, 52.5, 47.5, 50.0, 50.0]   # 15 samples
    assert not tmh.height_ready()
    for i, r in enumerate(samples, 1):
        tmh.update_height(r)
        assert tmh.resid_count == i
    # Welford mean/std match numpy population stats
    assert abs(tmh.resid_mean - float(np.mean(samples))) < 1e-9, tmh.resid_mean
    assert abs(tmh.resid_std() - float(np.std(samples))) < 1e-9, tmh.resid_std()
    assert tmh.height_ready(), "gating must activate at 15 samples"
    # a None / non-finite residual is a no-op (does not corrupt the count)
    before_c = tmh.resid_count
    tmh.update_height(None); tmh.update_height(float("nan")); tmh.update_height(float("inf"))
    assert tmh.resid_count == before_c, "non-finite residuals must be ignored"

    mean = tmh.resid_mean
    tol = tmh.height_tol()
    assert tol >= HEIGHT_MIN_PX
    # candidate residuals: 0 in-tolerance, 1 way off (wrong height), 2 unmeasured
    resids = {0: mean + 2.0, 1: mean + 500.0, 2: None}
    kept = height_gate_indices([0, 1, 2], resids, mean, tol)
    assert 0 in kept, "in-tolerance candidate kept"
    assert 1 not in kept, "wrong-height candidate excluded"
    assert 2 in kept, "no-measurable-height candidate kept"
    # the wrong-height candidate is NEVER excluded when protected (active/armed)
    kept_p = height_gate_indices([0, 1, 2], resids, mean, tol, protect={1})
    assert 1 in kept_p, "protected (active/armed) candidate must never be excluded"
    # with <15 samples the CALLER keeps gating inert; verify the readiness gate
    tmh_few = TargetModel()
    for r in samples[:14]:
        tmh_few.update_height(r)
    assert not tmh_few.height_ready(), "must NOT gate with <15 samples"
    print("[selftest] height-consistency Welford + gating (exclude wrong height, "
          "keep in-tol/unmeasured, never protected)  OK")

    # ---------------------------------------------------------------
    # CONFIDENCE rating: a dominant, correct, same-team detection scores
    # HIGHER than an ambiguous one, and a held/csrt fallback scores LOWER
    # than a clean snap. Source bases, terms + clamping behave.
    # ---------------------------------------------------------------
    # dominant correct pick: detection source, perfect height/app/margin
    conf_good = frame_confidence(SOURCE_DETECTION, height_term=1.0,
                                 app_term=1.0, margin_term=1.0)
    # ambiguous pick: detection source but poor height/app and no dominance
    conf_amb = frame_confidence(SOURCE_DETECTION, height_term=0.0,
                                app_term=0.0, margin_term=0.0)
    assert conf_good > conf_amb, (conf_good, conf_amb)
    assert conf_good >= 0.85, conf_good         # detection base 0.7 + maxed terms
    # a held/csrt frame with the SAME ancillary terms is lower than a snap
    conf_snap = frame_confidence(SOURCE_DETECTION, height_term=0.5,
                                 app_term=0.5, margin_term=0.5)
    conf_held = frame_confidence(SOURCE_CSRT, height_term=0.5,
                                 app_term=0.5, margin_term=0.5)
    assert conf_snap > conf_held, (conf_snap, conf_held)
    # manual is the most trusted source base
    conf_man = frame_confidence(SOURCE_MANUAL, height_term=0.5,
                                app_term=0.5, margin_term=0.5)
    assert conf_man > conf_snap > conf_held, (conf_man, conf_snap, conf_held)
    # None terms => neutral 0.5 (not a crash, not 0)
    conf_neutral = frame_confidence(SOURCE_DETECTION)
    assert 0.0 <= conf_neutral <= 1.0
    # a poor held frame (bad ancillary terms) trips the auto-pause threshold,
    # while a clean dominant snap stays above it
    conf_held_bad = frame_confidence(SOURCE_CSRT, height_term=0.0,
                                     app_term=0.0, margin_term=0.0)
    assert conf_held_bad < AUTO_PAUSE_THRESH, conf_held_bad
    assert conf_good >= AUTO_PAUSE_THRESH, conf_good
    print(f"[selftest] confidence: good {conf_good:.2f} > ambiguous {conf_amb:.2f}; "
          f"snap {conf_snap:.2f} > held {conf_held:.2f}; manual {conf_man:.2f}  OK")

    # ---------------------------------------------------------------
    # CONFIDENCE-ADAPTIVE MARKER: adaptive_marker_spec geometry tiers.
    # High conf -> a TIGHT body ellipse; mid conf -> a LARGER ellipse;
    # low conf -> a SEARCH circle whose radius GROWS as conf falls.
    # Pure float math, no GUI / display.
    # ---------------------------------------------------------------
    mbox = (100.0, 200.0, 160.0, 320.0)        # w=60, h=120
    mw, mh = 60.0, 120.0
    mdiag = (mw * mw + mh * mh) ** 0.5
    mcx, mcy = 130.0, 260.0

    # conf 0.9 -> a TIGHT body ellipse (ax ~ 0.55*w, ay ~ 0.50*h), centred.
    s_hi = adaptive_marker_spec(mbox, 0.9)
    assert s_hi[0] == "ellipse", s_hi
    _, hcx, hcy, hax, hay = s_hi
    assert abs(hcx - mcx) < 1e-9 and abs(hcy - mcy) < 1e-9, s_hi
    assert abs(hax - ADAPT_ELL_AX_HI * mw) < 1e-9, s_hi
    assert abs(hay - ADAPT_ELL_AY_HI * mh) < 1e-9, s_hi
    # at exactly the HI cut point it is still the tight ellipse
    assert adaptive_marker_spec(mbox, ADAPT_CONF_HI)[3:] == s_hi[3:]
    # a None conf (scrubbed/unrated frame) -> treated as confident -> tight ellipse
    assert adaptive_marker_spec(mbox, None) == s_hi

    # conf 0.55 -> a LARGER ellipse (both axes strictly bigger than the tight one)
    s_mid = adaptive_marker_spec(mbox, 0.55)
    assert s_mid[0] == "ellipse", s_mid
    _, _, _, max_, may = s_mid
    assert max_ > hax and may > hay, (s_mid, s_hi)
    # ... and an ellipse at 0.45 (closer to LO) is bigger still than at 0.65
    s_065 = adaptive_marker_spec(mbox, 0.65)
    s_045 = adaptive_marker_spec(mbox, 0.45)
    assert s_045[3] > s_065[3] and s_045[4] > s_065[4], (s_045, s_065)

    # conf 0.2 and 0.05 -> a SEARCH CIRCLE whose radius GROWS as conf falls
    s_c20 = adaptive_marker_spec(mbox, 0.2)
    s_c05 = adaptive_marker_spec(mbox, 0.05)
    assert s_c20[0] == "circle" and s_c05[0] == "circle", (s_c20, s_c05)
    _, ccx, ccy, r20 = s_c20
    assert abs(ccx - mcx) < 1e-9 and abs(ccy - mcy) < 1e-9, s_c20
    r05 = s_c05[3]
    assert r05 > r20, (r05, r20)               # lower conf -> bigger search circle
    # the circle radius is in the documented diag-scaled band
    assert r20 >= ADAPT_CIRC_R_LO * mdiag - 1e-9, (r20, mdiag)
    assert r05 <= ADAPT_CIRC_R_MIN * mdiag + 1e-9, (r05, mdiag)
    # monotonic across the whole low band: 0.0 >= 0.2 >= just-below-LO
    r00 = adaptive_marker_spec(mbox, 0.0)[3]
    r_lo = adaptive_marker_spec(mbox, ADAPT_CONF_LO - 1e-6)[3]
    assert r00 >= r05 >= r20 >= r_lo - 1e-6, (r00, r05, r20, r_lo)
    # KIND FLIP at the 0.4 boundary: just above LO is an ellipse, just below is a
    # circle. The drawn-size EMA must reset across this flip (the GUI keys the
    # reset on the spec kind changing) - assert the kind genuinely changes so the
    # GUI's _adapt_kind comparison has something to act on.
    just_above = adaptive_marker_spec(mbox, ADAPT_CONF_LO + 1e-6)[0]
    just_below = adaptive_marker_spec(mbox, ADAPT_CONF_LO - 1e-6)[0]
    assert just_above == "ellipse" and just_below == "circle", \
        (just_above, just_below)
    assert just_above != just_below, "kind must flip across the LO boundary"
    print(f"[selftest] adaptive marker: tight ellipse @0.9 ax {hax:.1f}; "
          f"larger ellipse @0.55 ax {max_:.1f}; growing circle r@0.2 {r20:.1f} "
          f"< r@0.05 {r05:.1f}; kind flips ellipse->circle at LO  OK")

    # ---------------------------------------------------------------
    # CONFIDENCE-ADAPTIVE SPEED: conf -> per-tick speed multiplier mapping.
    # conf<0.4 -> 0.25x (slow to inspect), 0.4-0.7 -> 0.5x, >=0.7 -> 1.75x.
    # Pure mapping, no GUI.
    # ---------------------------------------------------------------
    assert conf_speed_mult(0.2) == CONF_SPEED_LOW, conf_speed_mult(0.2)
    assert conf_speed_mult(0.39) == CONF_SPEED_LOW
    assert conf_speed_mult(0.4) == CONF_SPEED_MID
    assert conf_speed_mult(0.55) == CONF_SPEED_MID
    assert conf_speed_mult(0.69) == CONF_SPEED_MID
    assert conf_speed_mult(0.7) == CONF_SPEED_HI
    assert conf_speed_mult(0.95) == CONF_SPEED_HI
    # a None conf (no rating yet) defaults to full speed (HI), never crashes
    assert conf_speed_mult(None) == CONF_SPEED_HI
    # lower confidence => slower (smaller multiplier) => longer interval
    assert CONF_SPEED_LOW < CONF_SPEED_MID < CONF_SPEED_HI, \
        (CONF_SPEED_LOW, CONF_SPEED_MID, CONF_SPEED_HI)
    print(f"[selftest] conf-adaptive speed: <0.4 -> {CONF_SPEED_LOW}x, "
          f"0.4-0.7 -> {CONF_SPEED_MID}x, >=0.7 -> {CONF_SPEED_HI}x  OK")

    # ---------------------------------------------------------------
    # JERSEY-NUMBER OCR soft signal: the module must be IMPORTABLE and
    # GRACEFULLY INERT without the tesseract binary - available() returns a
    # bool (never raises) and read_number() returns None when unavailable. The
    # full GUI soft-signal wiring (default-off, throttled boost/penalty) is
    # exercised at runtime; here we only confirm the inert contract that keeps
    # tracking unaffected when the binary is absent / the toggle is off.
    avail = ocr.available()
    assert isinstance(avail, bool)
    if not avail:
        assert ocr.read_number(None) is None
        import numpy as _np
        assert ocr.read_number(_np.zeros((40, 30, 3), dtype=_np.uint8)) is None
        print("[selftest] jersey OCR inert (no tesseract binary) -> "
              "available() False, read_number None  OK")
    else:
        # binary present: a confident 1-2 digit read is (str, 0..1) or None
        r = ocr.read_number(None)
        assert r is None
        print("[selftest] jersey OCR available (tesseract binary present)  OK")

    # ---------------------------------------------------------------
    # tracks_io CONFIDENCE column round-trip (rt2.tracks_io, imported
    # READ-ONLY): write_tracks persists a per-row "confidence" and
    # read_tracks reads it back; a row with no confidence defaults to 1.0
    # on write, and an OLD CSV with NO confidence column reads back as 1.0.
    # This is the on-disk contract do_save() relies on (Fix A keeps the
    # per-frame ratings alive long enough to reach here).
    # ---------------------------------------------------------------
    import tempfile as _tempfile, os as _os
    iod = _tempfile.mkdtemp(prefix="rt2_tracks_io_")
    iopath = _os.path.join(iod, "io.tracks.csv")
    io_rows = [
        {"frame": 1, "target_id": 3, "x1": 10.0, "y1": 20.0, "x2": 40.0,
         "y2": 90.0, "source": SOURCE_DETECTION, "confidence": 0.812},
        {"frame": 2, "target_id": 3, "x1": 11.0, "y1": 21.0, "x2": 41.0,
         "y2": 91.0, "source": SOURCE_CSRT, "confidence": 0.350},
        # a row with NO confidence key -> write default 1.0
        {"frame": 3, "target_id": 3, "x1": 12.0, "y1": 22.0, "x2": 42.0,
         "y2": 92.0, "source": SOURCE_MANUAL},
    ]
    write_tracks(iopath, io_rows)
    back = {r["frame"]: r for r in read_tracks(iopath)}
    assert abs(back[1]["confidence"] - 0.812) < 1e-3, back[1]
    assert abs(back[2]["confidence"] - 0.350) < 1e-3, back[2]
    assert abs(back[3]["confidence"] - 1.0) < 1e-9, \
        ("missing confidence must default to 1.0", back[3])
    assert all(0.0 <= back[f]["confidence"] <= 1.0 for f in back)
    assert back[2]["source"] == SOURCE_CSRT and back[3]["source"] == SOURCE_MANUAL
    # an OLD CSV with NO confidence column at all reads back as 1.0 per row.
    oldpath = _os.path.join(iod, "old.tracks.csv")
    with open(oldpath, "w", newline="", encoding="utf-8") as _f:
        _f.write("frame,target_id,x1,y1,x2,y2,source\n")
        _f.write("1,5,10.00,20.00,40.00,90.00,detection\n")
        _f.write("2,5,11.00,21.00,41.00,91.00,csrt\n")
    old_back = read_tracks(oldpath)
    assert len(old_back) == 2, old_back
    assert all(abs(r["confidence"] - 1.0) < 1e-9 for r in old_back), \
        ("old file with no confidence column must default to 1.0", old_back)
    for _p in (iopath, oldpath):
        try:
            _os.remove(_p)
        except OSError:
            pass
    try:
        _os.rmdir(iod)
    except OSError:
        pass
    print("[selftest] tracks_io confidence-column round-trip "
          "(persisted, default 1.0, old-file back-compat)  OK")

    # ---------------------------------------------------------------
    # recovery_score gate: a dead-centre box with NO appearance signal scores
    # exactly W_RPOS (pos-only) and that clears RECOVER_MIN, while a box at the
    # search-box edge with no appearance falls below it (would NOT re-acquire).
    # Confirms the documented blend is not silently inverted.
    # ---------------------------------------------------------------
    rc_centre = (100.0, 100.0)
    rc_search = (0.0, 0.0, 200.0, 200.0)
    rc_on = recovery_score((90.0, 90.0, 110.0, 110.0), rc_centre, rc_search, None, None)
    assert abs(rc_on - W_RPOS) < 1e-6, rc_on
    assert rc_on >= RECOVER_MIN, (rc_on, RECOVER_MIN)
    rc_edge = recovery_score((180.0, 180.0, 200.0, 200.0), rc_centre, rc_search,
                             None, None)
    assert rc_edge < RECOVER_MIN, (rc_edge, RECOVER_MIN)
    print(f"[selftest] recovery_score gate: centre {rc_on:.2f} >= "
          f"RECOVER_MIN {RECOVER_MIN} > edge {rc_edge:.2f}  OK")

    # JUMP GATE: a small step is allowed; a teleport onto a far player is rejected;
    # a long gap (lost lock) disables the gate so re-acquire is free.
    prev_c, prev_f = (500.0, 400.0), 100        # last good centre at frame 100
    near = (488, 360, 512, 440)                 # ~same place next frame (24x80 box)
    far = (760, 360, 784, 440)                  # ~260px jump = many body-heights
    assert not jump_too_far(prev_c, prev_f, near, 101), "a normal step must pass"
    assert jump_too_far(prev_c, prev_f, far, 101), "a teleport must be rejected"
    assert not jump_too_far(prev_c, prev_f, far, 101 + JUMP_GAP_RESET + 1), \
        "after a long gap the gate must allow free re-acquire"
    assert not jump_too_far(None, None, far, 101), "no history -> no gating"
    print("[selftest] jump gate: step ok, teleport rejected, gap re-acquire free  OK")

    # ---------------------------------------------------------------
    # ONLINE IDENTITY ("me vs not-me") wiring:
    #   (a) the rt2.identity module's own __main__ selftest passes;
    #   (b) identity_features() builds a CONSTANT-DIM (IDENTITY_DIM) vector for a
    #       synthetic detection - even with EVERY signal missing (all None);
    #   (c) the ruled_out exclusion drops a detection that maps to a ruled-out
    #       track id from the eligible set (mirrors _eligible_indices' filter).
    # ---------------------------------------------------------------
    identity._selftest()
    # (b) constant-dim feature vector, all-present and all-missing
    fv_full = identity_features(0.3, 0.8, 12.0, 0.5, 0.6, 0.1, 0.2, 0.7)
    fv_none = identity_features(None, None, None, None, None, None, None, None)
    assert len(fv_full) == IDENTITY_DIM, len(fv_full)
    assert len(fv_none) == IDENTITY_DIM, len(fv_none)
    assert all(isinstance(x, float) and x == x for x in fv_full + fv_none)
    assert fv_none == [0.0] * IDENTITY_DIM, fv_none
    # a ready model built from those features yields a usable prob in [0,1]
    idm = identity.IdentityModel(dim=IDENTITY_DIM)
    for _ in range(10):
        idm.add_positive(fv_full)
        idm.add_negative(fv_none, weight=ID_HARD_NEG)
    assert idm.ready()
    assert 0.0 <= idm.prob(fv_full) <= 1.0
    print(f"[selftest] identity_features constant dim {IDENTITY_DIM}; "
          f"ready model prob(full)={idm.prob(fv_full):.2f}  OK")

    # (c) ruled_out exclusion: replicate _eligible_indices' exact filter over a
    # tiny synthetic frame. dets 0,1,2 all team-eligible; det 1 maps to tid 7.
    syn_dets = [(0, 0, 10, 30, 0.9), (50, 0, 60, 30, 0.9), (100, 0, 110, 30, 0.9)]
    det_tid = {0: 1, 1: 7, 2: 1}            # _det_track_hint() result per index
    base_eligible = eligible_target_indices(
        [(None, None, False)] * len(syn_dets))   # no calibration -> all eligible
    assert base_eligible == {0, 1, 2}
    ruled_out = {7}
    excluded = {i for i in base_eligible if det_tid[i] not in ruled_out}
    assert excluded == {0, 2}, excluded       # det 1 (tid 7) removed
    assert 1 not in excluded
    print("[selftest] ruled_out excludes the ruled-out track id from eligibility "
          f"({sorted(base_eligible)} -> {sorted(excluded)})  OK")

    # ---------------------------------------------------------------
    # DETECTION -> PLAYER NAMING wiring (FIX 3). Two pure checks:
    #   (a) the in-session map track_id -> (uuid, display_name) drives the list /
    #       box label suffix EXACTLY as the GUI builds it, keyed on the STABLE
    #       track id (so the name survives the per-frame index rebuild);
    #   (b) naming a detection actually writes/links a row in the SAME registry
    #       DB the profiler feeds (output/registry.sqlite via PlayerRegistry),
    #       and that row persists across a reopen.
    # ---------------------------------------------------------------
    # (a) label suffix logic mirrored from _refresh_det_list / paintEvent
    det_names = {7: ("uuid-abc", "Pou"), 11: ("uuid-def", "Hautapu")}

    def _name_suffix(tid):
        if tid is not None and tid in det_names:
            return f'  "{det_names[tid][1]}"'
        return ""
    assert _name_suffix(7) == '  "Pou"', _name_suffix(7)
    assert _name_suffix(11) == '  "Hautapu"'
    assert _name_suffix(3) == ""          # unnamed track -> no suffix
    assert _name_suffix(None) == ""
    print("[selftest] detection naming: track_id->name label suffix  OK")

    # (b) naming round-trips through the registry DB (the 'link to the database')
    import tempfile as _tf, os as _os
    _db = pathlib.Path(_tf.gettempdir()) / "rt2_track_naming_selftest.sqlite"
    try:
        _os.unlink(_db)
    except OSError:
        pass
    with registry.PlayerRegistry(_db) as _reg:
        _uuid_new = _reg.add_player(display_name="Pou", is_teammate=True, number=7)
        assert _reg.get_player(_uuid_new)["display_name"] == "Pou"
        _me = _reg.set_me("Me", number=10)
        assert _reg.get_player(_me)["is_me"] == 1
    with registry.PlayerRegistry(_db) as _reg2:           # reopen -> persists
        names = {p["display_name"] for p in _reg2.list_players()}
        assert "Pou" in names and "Me" in names, names
    try:
        _os.unlink(_db)
    except OSError:
        pass
    print("[selftest] detection naming: writes/links + persists in registry DB  OK")

    # ---------------------------------------------------------------
    # MY-NUMBER live jersey exclusion (Feature 1) - PURE decision logic:
    #   confidently_someone_else(counter, my_number) is the exact predicate the
    #   eligibility filter uses to drop a wrong-numbered track. Check the four
    #   regimes: my_number None -> inert; decided + same -> keep; decided +
    #   different -> exclude; undecided (under weight) -> keep.
    # ---------------------------------------------------------------
    assert confidently_someone_else(Counter({"3": 2.0}), None) is False  # inert
    assert confidently_someone_else(Counter({"1": 2.0}), 1) is False     # same #
    assert confidently_someone_else(Counter({"7": 2.0}), 1) is True      # diff #
    assert confidently_someone_else(Counter({"7": 0.9}), 1) is False     # under wt
    assert confidently_someone_else(Counter(), 1) is False               # no votes
    # decided number helper: needs >= MYNUM_DECIDE_WEIGHT accumulated weight
    assert jersey_decided_number(Counter({"3": 1.6})) == "3"
    assert jersey_decided_number(Counter({"3": 1.0})) is None
    # the PROFILE round-trips my_number across save/load (carry across clips)
    import tempfile as _tf2
    _pf = pathlib.Path(_tf2.gettempdir()) / "rt2_mynum_selftest.json"
    try:
        _pf.unlink()
    except OSError:
        pass
    _tm = TargetModel()
    _tm.my_number = 3                 # number set BEFORE any appearance learning
    _tm.save_profile(_pf)             # must still persist (count==0 but my_number set)
    _tm2 = TargetModel()
    _tm2.load_profile(_pf)
    assert _tm2.my_number == 3, _tm2.my_number
    try:
        _pf.unlink()
    except OSError:
        pass
    print("[selftest] my-number: confidently_someone_else decision + profile "
          "persist  OK")

    # ---------------------------------------------------------------
    # MANUAL-EXIT SNAP-TO-NEAREST (Feature 2) - PURE choice logic:
    #   manual_snap_choice(roi, boxes) snaps to the nearest box ONLY if its centre
    #   is within MANUAL_SNAP_ROI_WIDTHS x the ROI width; else None.
    # ---------------------------------------------------------------
    _roi = (100, 100, 200, 200)       # 100px-wide ROI centred at (150,150)
    _near = (140, 140, 180, 220)      # centre (160,180) ~36px away -> within budget
    _far = (700, 100, 760, 220)       # centre way outside 1.5*100=150px budget
    assert manual_snap_choice(_roi, [_far, _near]) == _near       # picks the near one
    assert manual_snap_choice(_roi, [_far]) is None               # nothing near enough
    assert manual_snap_choice(_roi, []) is None                   # no candidates
    assert manual_snap_choice(None, [_near]) is None              # no ROI
    print("[selftest] manual-exit snap: nearest-within-budget choice  OK")

    # ---------------------------------------------------------------
    # OBJECTS-OVERLAY (mark_all): aggregate_objects_rows folds per-frame rows
    # into the overlay map + per-obj TrackAggs, and candidates.shortlist groups
    # them me-likely / possible / ruled-out. PURE (no GUI, no file needed).
    # ---------------------------------------------------------------
    _fw, _fh = 1000.0, 1000.0
    _rows = []
    # obj 1: opposition (every frame) -> ruled out
    for fr in range(1, 21):
        _rows.append({"frame": fr, "obj_id": 1, "x1": 600, "y1": 500,
                      "x2": 660, "y2": 600, "team": "opp", "conf": 0.9})
    # obj 2: my team, central -> possible (no number to confirm)
    for fr in range(1, 21):
        _rows.append({"frame": fr, "obj_id": 2, "x1": 480, "y1": 560,
                      "x2": 540, "y2": 660, "team": "blue&gold", "conf": 0.8})
    # obj 3: a short track (< min_frames) -> skipped by shortlist
    for fr in range(1, 4):
        _rows.append({"frame": fr, "obj_id": 3, "x1": 100, "y1": 100,
                      "x2": 160, "y2": 200, "team": "blue&gold", "conf": 0.7})
    # one garbled row must be skipped without crashing
    _rows.append({"frame": "x", "obj_id": 9, "x1": "?", "y1": 1,
                  "x2": 2, "y2": 3, "team": "opp", "conf": "?"})

    _obf, _aggs = aggregate_objects_rows(_rows, _fw, _fh)
    assert {o[0] for o in _obf[1]} == {1, 2, 3}, _obf[1]       # objs 1,2,3 at frame 1
    assert {o[0] for o in _obf[10]} == {1, 2}, _obf[10]        # obj 3 gone by frame 10
    assert _aggs[1].n_frames == 20 and _aggs[1].dominant_team() == "opp"
    assert _aggs[2].team_counts.get("blue&gold") == 20
    # central track: zx_mean ~ 0.51, zy_mean ~ 0.61
    assert 0.45 <= _aggs[2].zx_mean <= 0.55, _aggs[2].zx_mean
    assert 9 not in _aggs                                       # garbled skipped

    _me = candidates.MeProfile(number=None, team="blue&gold")   # role unknown
    _verdicts, _summ = candidates.shortlist(list(_aggs.values()), _me, min_frames=8)
    assert _summ["total"] == 2, _summ                          # obj 3 (short) skipped
    assert _summ["ruled_out"] >= 1                             # opposition ruled out
    by_id = {v.obj_id: v for v in _verdicts}
    assert by_id[1].status == "ruled-out"
    assert by_id[2].status in ("possible", "me-likely")
    print(f"[selftest] objects-overlay: aggregate + shortlist {_summ}  OK")

    # --- TACTICAL (M2): pitch_circle_points ring geometry --------------------
    # n points, all exactly r_m from the centre, closing back near the start.
    _cx, _cy, _r, _n = 60.0, 35.0, 1.0, 28
    _ring = pitch_circle_points(_cx, _cy, _r, n=_n)
    assert len(_ring) == _n, len(_ring)
    for _px, _py in _ring:
        _d = ((_px - _cx) ** 2 + (_py - _cy) ** 2) ** 0.5
        assert abs(_d - _r) < 1e-9, _d                         # all on the circle
    # a larger radius keeps the invariant; degenerate r=0 collapses to the centre
    _ring2 = pitch_circle_points(10.0, 5.0, 2.5, n=12)
    assert len(_ring2) == 12
    assert all(abs(((x - 10.0) ** 2 + (y - 5.0) ** 2) ** 0.5 - 2.5) < 1e-9
               for x, y in _ring2)
    assert all(p == (3.0, 4.0) for p in pitch_circle_points(3.0, 4.0, 0.0, n=6))
    print(f"[selftest] tactical: pitch_circle_points {_n} pts on r={_r}m  OK")

    # --- live pitch homography: sequential-advance decision (M1) -------------
    # anchor at the clicked frame; then ONLY exact +1 steps update; everything
    # else (jumps, backward, repeats, gaps, pre-anchor) is STALE.
    assert live_cmc_action(50, 50, None) == "anchor"        # land on anchor
    assert live_cmc_action(51, 50, 50) == "update"          # +1 from last
    assert live_cmc_action(52, 50, 51) == "update"          # next +1
    assert live_cmc_action(60, 50, 51) == "stale"           # forward jump
    assert live_cmc_action(51, 50, 51) == "stale"           # repeat (no move)
    assert live_cmc_action(50, 50, 80) == "anchor"          # re-land on anchor wins
    assert live_cmc_action(49, 50, 51) == "stale"           # backward
    assert live_cmc_action(10, None, None) == "stale"       # no calibration anchor
    assert live_cmc_action(10, 50, None) == "stale"         # not yet anchored
    # a clean play-through from the anchor stays trusted frame after frame
    last = None
    for fr in range(50, 56):
        act = live_cmc_action(fr, 50, last)
        assert act in ("anchor", "update"), (fr, act)
        last = fr
    print("[selftest] live homography: sequential-advance / jump=stale  OK")

    print("[selftest] ALL PASSED")


# ===========================================================================
# GUI  (PySide6).  Only imported when actually launching.
# ===========================================================================
def run_gui(video_path, store, det_by_frame, tracks_path, calib=None, player=None,
            kept_only=False, my_number=None, sam2_model=None):
    import numpy as np
    from PySide6 import QtCore, QtGui, QtWidgets

    Qt = QtCore.Qt

    # -------------------------------------------------------------------
    # Bulk-ID "Manage IDs" dialog (Qt port of the v1 Tkinter dialog).
    # checkbox list + sort radios + bulk buttons + guardrails.
    # -------------------------------------------------------------------
    class ManageIdsDialog(QtWidgets.QDialog):
        def __init__(self, store, parent=None):
            super().__init__(parent)
            self.store = store
            self.setWindowTitle("Manage IDs")
            self.resize(560, 680)
            self.setModal(True)

            self.ids = store.unique_ids()
            self.stats = {t: store.id_stats(t) for t in self.ids}
            self.checks = {}                  # tid -> QCheckBox
            self._sort = "id"

            lay = QtWidgets.QVBoxLayout(self)

            # --- sort radios ---
            top = QtWidgets.QHBoxLayout()
            top.addWidget(QtWidgets.QLabel("Sort:"))
            self.rb_group = QtWidgets.QButtonGroup(self)
            for label, val in [("By ID", "id"), ("By frame count", "count"),
                               ("By first appearance", "first")]:
                rb = QtWidgets.QRadioButton(label)
                rb.setChecked(val == "id")
                rb.toggled.connect(lambda checked, v=val: self._set_sort(v) if checked else None)
                self.rb_group.addButton(rb)
                top.addWidget(rb)
            top.addStretch(1)
            lay.addLayout(top)

            # --- bulk action buttons ---
            bar = QtWidgets.QHBoxLayout()
            for label, fn in [("Tick all", self._tick_all),
                              ("Untick all", self._untick_all),
                              ("Untick short (<30)", self._untick_short),
                              ("Untick all except target", self._untick_except_target)]:
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(fn)
                bar.addWidget(b)
            lay.addLayout(bar)

            # --- header ---
            hdr = QtWidgets.QLabel(
                f"{'keep':<6}{'ID':>5}{'frames':>9}{'   span':>16}{'   dur(s)':>10}")
            hdr.setFont(QtGui.QFont("Consolas", 9, QtGui.QFont.Bold))
            lay.addWidget(hdr)

            # --- scrollable checkbox list ---
            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.rows_host = QtWidgets.QWidget()
            self.rows_lay = QtWidgets.QVBoxLayout(self.rows_host)
            self.rows_lay.setAlignment(Qt.AlignTop)
            self.scroll.setWidget(self.rows_host)
            lay.addWidget(self.scroll, 1)

            # all ticked == keep by default (we delete the UNticked)
            for t in self.ids:
                cb = QtWidgets.QCheckBox()
                cb.setChecked(True)
                self.checks[t] = cb
            self._rebuild_rows()

            # --- apply / cancel ---
            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel)
            btns.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._on_apply)
            btns.rejected.connect(self.reject)
            lay.addWidget(btns)

            self.applied = False
            self.removed_ids = []
            self.removed_rows = 0

        def _sorted_ids(self):
            if self._sort == "count":
                return sorted(self.ids, key=lambda t: -self.stats[t]["count"])
            if self._sort == "first":
                return sorted(self.ids, key=lambda t: self.stats[t]["first"])
            return sorted(self.ids)

        def _set_sort(self, val):
            self._sort = val
            self._rebuild_rows()

        def _rebuild_rows(self):
            while self.rows_lay.count():
                item = self.rows_lay.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
            for t in self._sorted_ids():
                s = self.stats[t]
                is_target = (t == self.store.target)
                star = "* " if is_target else "  "
                txt = (f"{star}{t:>4}{s['count']:>9}"
                       f"  {s['first']:>5}->{s['last']:<5}{s['dur']:>8.1f}")
                row = QtWidgets.QWidget()
                rl = QtWidgets.QHBoxLayout(row)
                rl.setContentsMargins(2, 0, 2, 0)
                rl.addWidget(self.checks[t])
                lbl = QtWidgets.QLabel(txt)
                fnt = QtGui.QFont("Consolas", 9)
                fnt.setBold(is_target)
                lbl.setFont(fnt)
                if is_target:
                    lbl.setStyleSheet("color:#b8860b;")
                rl.addWidget(lbl)
                rl.addStretch(1)
                self.rows_lay.addWidget(row)

        def _tick_all(self):
            for cb in self.checks.values():
                cb.setChecked(True)

        def _untick_all(self):
            for cb in self.checks.values():
                cb.setChecked(False)

        def _untick_short(self):
            for t, cb in self.checks.items():
                if self.stats[t]["count"] < 30:
                    cb.setChecked(False)

        def _untick_except_target(self):
            for t, cb in self.checks.items():
                cb.setChecked(t == self.store.target)

        def _on_apply(self):
            unticked = [t for t in self.ids if not self.checks[t].isChecked()]
            if not unticked:
                self.applied = False
                self.accept()
                return
            # guardrail: deleting the active target
            if self.store.target in unticked:
                if QtWidgets.QMessageBox.question(
                        self, "Confirm", "Delete your active target ID?") \
                        != QtWidgets.QMessageBox.Yes:
                    return
            # guardrail: > 50% of rows
            total = len(self.store.records)
            doomed = sum(self.stats[t]["count"] for t in unticked)
            if total and doomed / total > 0.5:
                pct = round(100 * doomed / total)
                if QtWidgets.QMessageBox.question(
                        self, "Confirm",
                        f"This will remove {pct}% of all tracking data. Continue?") \
                        != QtWidgets.QMessageBox.Yes:
                    return
            self.applied = True
            self.removed_ids = unticked
            self.removed_rows = doomed
            self.accept()

    # -------------------------------------------------------------------
    # "Assign to player" dialog: name a detection / give it a player profile.
    # The user EITHER picks an existing registry player OR types a new name
    # (+ optional jersey number, + "is teammate" / "this is me"). result_value()
    # returns a small dict the caller turns into a registry write. This dialog is
    # PURE UI -- it touches no DB, so a registry failure can never crash it.
    # -------------------------------------------------------------------
    class _AssignPlayerDialog(QtWidgets.QDialog):
        def __init__(self, parent, existing, current_tid=None):
            super().__init__(parent)
            self.setWindowTitle("Assign to player")
            self.setModal(True)
            self.resize(360, 300)
            self._existing = list(existing or [])
            self._result = None

            lay = QtWidgets.QVBoxLayout(self)
            if current_tid is not None:
                lay.addWidget(QtWidgets.QLabel(
                    f"Assign track id {current_tid} to a player:"))

            # --- EXISTING player (radio + combo) ---
            self.rb_existing = QtWidgets.QRadioButton("Existing player")
            lay.addWidget(self.rb_existing)
            self.combo = QtWidgets.QComboBox()
            for p in self._existing:
                name = p.get("display_name") or "(unnamed)"
                num = p.get("current_number")
                tag = " (me)" if p.get("is_me") else ""
                label = f"{name}{tag}" + (f"  #{num}" if num is not None else "")
                self.combo.addItem(label, p.get("uuid"))
            self.combo.setEnabled(False)
            lay.addWidget(self.combo)

            # --- NEW player (radio + name + number + flags) ---
            self.rb_new = QtWidgets.QRadioButton("New player")
            lay.addWidget(self.rb_new)
            form = QtWidgets.QFormLayout()
            self.name_edit = QtWidgets.QLineEdit()
            self.name_edit.setPlaceholderText("e.g. Pou")
            form.addRow("Name:", self.name_edit)
            self.num_edit = QtWidgets.QLineEdit()
            self.num_edit.setPlaceholderText("optional, e.g. 7")
            self.num_edit.setValidator(QtGui.QIntValidator(0, 99, self))
            form.addRow("Jersey #:", self.num_edit)
            lay.addLayout(form)
            self.cb_teammate = QtWidgets.QCheckBox("Is teammate")
            self.cb_teammate.setChecked(True)
            lay.addWidget(self.cb_teammate)
            self.cb_me = QtWidgets.QCheckBox("This is me")
            lay.addWidget(self.cb_me)

            # default selection: existing if any exist, else new
            if self._existing:
                self.rb_existing.setChecked(True)
            else:
                self.rb_new.setChecked(True)
            self._sync_enabled()
            self.rb_existing.toggled.connect(self._sync_enabled)
            self.rb_new.toggled.connect(self._sync_enabled)
            # "this is me" implies teammate (registry treats me as is_teammate=1)
            self.cb_me.toggled.connect(
                lambda on: (self.cb_teammate.setChecked(True) if on else None))

            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            btns.accepted.connect(self._on_ok)
            btns.rejected.connect(self.reject)
            lay.addWidget(btns)

        def _sync_enabled(self, *_):
            ex = self.rb_existing.isChecked()
            self.combo.setEnabled(ex and bool(self._existing))
            for w in (self.name_edit, self.num_edit, self.cb_teammate, self.cb_me):
                w.setEnabled(not ex)

        def _on_ok(self):
            if self.rb_existing.isChecked():
                if self.combo.count() == 0:
                    return
                self._result = {
                    "kind": "existing",
                    "uuid": self.combo.currentData(),
                    "name": self.combo.currentText(),
                }
                self.accept()
                return
            # new player: a non-empty name is required
            name = self.name_edit.text().strip()
            if not name:
                QtWidgets.QMessageBox.warning(
                    self, "Assign to player", "Enter a name for the new player.")
                return
            num_txt = self.num_edit.text().strip()
            number = int(num_txt) if num_txt else None
            self._result = {
                "kind": "new",
                "name": name,
                "number": number,
                "is_teammate": self.cb_teammate.isChecked(),
                "is_me": self.cb_me.isChecked(),
            }
            self.accept()

        def result_value(self):
            return self._result

    # -------------------------------------------------------------------
    # Video widget: draws the frame + ROI overlay, maps mouse<->frame px.
    # -------------------------------------------------------------------
    class VideoView(QtWidgets.QWidget):
        clicked = QtCore.Signal(float, float)            # frame-px click (no drag)

        def __init__(self, app):
            super().__init__()
            self.app = app
            self.setMinimumSize(640, 360)
            self.setMouseTracking(True)
            self._qimg = None                            # current QImage (RGB)
            # zoom + pan transform (letterbox + zoom + pan); single source of
            # truth for all screen<->frame mapping.
            self.vt = ViewTransform(app.fw, app.fh)
            # drag state
            self._mode = None                            # None|'move'|'resize'|'maybe-click'|'pan'
            self._handle = None                          # which corner/edge
            self._press = None                           # (fx, fy) press point in frame coords
            self._roi0 = None                            # ROI at press time
            self._moved = False
            self._pan_anchor = None                      # (wx, wy) last pan widget pos
            self._space_down = False                     # Space held -> left-drag pans

        # ---- coordinate mapping (THE #1 correctness risk) -------------
        # All mapping flows through self.vt (ViewTransform), which folds the
        # letterbox fit-scale, the zoom multiplier and the pan offset into one
        # exact, round-trip-tested transform. widget_to_frame / frame_to_widget
        # therefore stay pixel-accurate at ANY zoom / pan, and every click,
        # ROI move, handle grab and detection overlay that uses them does too.
        def _sync_vt(self):
            self.vt.set_widget_size(self.width(), self.height())

        def _draw_rect(self):
            """Letterboxed+zoomed+panned dest rect (widget px) the frame is
            painted into. Returns (ox, oy, dw, dh, sx, sy)."""
            self._sync_vt()
            return self.vt.draw_rect()

        def widget_to_frame(self, wx, wy):
            """Map a widget point to ORIGINAL-frame px. Returns None if the
            point falls outside the frame (so clicks in the letterbox/zoom
            margins are ignored, exactly as before)."""
            self._sync_vt()
            fx, fy = self.vt.widget_to_frame(wx, wy)
            if not self.vt.in_frame(fx, fy):
                return None
            return (fx, fy)

        def frame_to_widget(self, fx, fy):
            self._sync_vt()
            return self.vt.frame_to_widget(fx, fy)

        def frame_box_to_widget(self, b):
            x1, y1 = self.frame_to_widget(b[0], b[1])
            x2, y2 = self.frame_to_widget(b[2], b[3])
            return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

        def set_image(self, qimg):
            self._qimg = qimg
            self.update()

        # ---- keep the frame fixed in the middle on resize ----------
        def resizeEvent(self, ev):
            """Keep the picture centred + stable when the window/dock layout
            changes size. We re-sync the transform to the new widget size, which
            re-clamps the pan: when the frame fits (the default zoom==1 fit case)
            clamp() forces pan to 0 on both axes, so the frame stays centred in
            the viewport instead of drifting. Wheel-zoom / pan still work because
            they only change zoom/pan, not this re-centring of a fitted view."""
            self._sync_vt()
            super().resizeEvent(ev)

        # ---- ROI handle hit-testing (in frame coords) -----------------
        def _handles(self, roi):
            x1, y1, x2, y2 = roi
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return {
                "nw": (x1, y1), "n": (cx, y1), "ne": (x2, y1),
                "w": (x1, cy), "e": (x2, cy),
                "sw": (x1, y2), "s": (cx, y2), "se": (x2, y2),
            }

        def _hit_handle(self, roi, fx, fy):
            for name, (hx, hy) in self._handles(roi).items():
                if abs(fx - hx) <= HANDLE and abs(fy - hy) <= HANDLE:
                    return name
            return None

        # ---- zoom (wheel, anchored under cursor) --------------------
        def wheelEvent(self, ev):
            self._sync_vt()
            dy = ev.angleDelta().y()
            if dy == 0:
                return
            factor = 1.0015 ** dy        # smooth, multiplicative; +120 ~= 1.20x
            self.vt.zoom_at(ev.position().x(), ev.position().y(), factor)
            self.app._refresh()
            ev.accept()

        # ---- mouse --------------------------------------------------
        def _start_pan(self, ev):
            self._mode = "pan"
            self._pan_anchor = (ev.position().x(), ev.position().y())
            self.setCursor(Qt.ClosedHandCursor)

        def mousePressEvent(self, ev):
            # PAN: middle button anywhere, OR Space + left button. This is kept
            # strictly separate from the left-button ROI / arm semantics.
            if ev.button() == Qt.MiddleButton or \
                    (ev.button() == Qt.LeftButton and self._space_down):
                self._start_pan(ev)
                return
            if ev.button() != Qt.LeftButton:
                return
            pt = self.widget_to_frame(ev.position().x(), ev.position().y())
            if pt is None:
                return
            fx, fy = pt
            self._press = (fx, fy)
            self._moved = False
            roi = self.app.roi
            if roi is not None:
                h = self._hit_handle(roi, fx, fy)
                if h is not None:
                    self._mode = "resize"; self._handle = h; self._roi0 = roi
                    return
                if bbox.contains(roi, fx, fy):
                    self._mode = "move"; self._roi0 = roi
                    return
            self._mode = "maybe-click"

        def mouseMoveEvent(self, ev):
            if self._mode == "pan" and self._pan_anchor is not None:
                wx, wy = ev.position().x(), ev.position().y()
                self.vt.pan_by_widget(wx - self._pan_anchor[0],
                                      wy - self._pan_anchor[1])
                self._pan_anchor = (wx, wy)
                self.app._refresh()
                return
            pt = self.widget_to_frame(ev.position().x(), ev.position().y())
            # MANUAL CURSOR-FOLLOW: while manual mode is ON and we are NOT
            # dragging/panning, a plain mouse-move (no button) recentres the ROI
            # on the cursor so the user just points at themselves. mouseTracking
            # is enabled so this fires with no button held.
            if pt is not None:
                self.app._cursor_fp = pt
                if (self.app.manual_mode and self._mode is None):
                    self.app._roi_follow_cursor(pt)
            if pt is None or self._press is None:
                return
            fx, fy = pt
            dx, dy = fx - self._press[0], fy - self._press[1]
            if abs(dx) > 2 or abs(dy) > 2:
                self._moved = True
            if self._mode == "move" and self._roi0 is not None:
                x1, y1, x2, y2 = self._roi0
                self.app.set_roi(clamp_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                                           self.app.fw, self.app.fh))
            elif self._mode == "resize" and self._roi0 is not None:
                self.app.set_roi(self._resized(self._roi0, self._handle, dx, dy))

        def mouseReleaseEvent(self, ev):
            if self._mode == "pan":
                if ev.button() in (Qt.MiddleButton, Qt.LeftButton):
                    self._mode = None; self._pan_anchor = None
                    self.unsetCursor()
                return
            if ev.button() != Qt.LeftButton:
                return
            if self._mode == "maybe-click" and not self._moved and self._press is not None:
                self.clicked.emit(self._press[0], self._press[1])
            elif self._mode in ("move", "resize") and self._moved:
                # A manual drag/resize re-anchors the tracker to the corrected ROI,
                # so it follows YOU from here instead of snapping back to whoever
                # it had locked. (CSRT is re-initialised on the new box.)
                self.app._reset_csrt()
                self.app.status_msg = "ROI moved - tracker re-anchored"
                self.app._refresh()
            self._mode = None; self._handle = None; self._press = None; self._roi0 = None

        def _resized(self, roi, handle, dx, dy):
            x1, y1, x2, y2 = roi
            if "n" in handle:
                y1 += dy
            if "s" in handle:
                y2 += dy
            if "w" in handle:
                x1 += dx
            if "e" in handle:
                x2 += dx
            return clamp_box((x1, y1, x2, y2), self.app.fw, self.app.fh)

        # ---- paint --------------------------------------------------
        def paintEvent(self, ev):
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), QtGui.QColor(20, 20, 20))
            if self._qimg is None:
                return
            ox, oy, dw, dh, sx, sy = self._draw_rect()
            target = QtCore.QRectF(ox, oy, dw, dh)
            p.drawImage(target, self._qimg)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)

            frame = self.app.frame
            # --- consolidated target box at this frame (gold) ---
            tb = self.app.store.target_box_at(frame)
            if tb is not None:
                r = self.frame_box_to_widget(tb)
                # CONFIDENCE colouring: green (>=0.7), amber (0.4-0.7), red (<0.4);
                # gold when this frame has no confidence rating (e.g. scrubbed).
                conf = self.app.conf_by_frame.get(frame)
                if conf is None:
                    tcol = QtGui.QColor(255, 215, 0)         # gold (no rating)
                    tlabel = f"TARGET {self.app.store.target}"
                else:
                    if conf >= 0.7:
                        tcol = QtGui.QColor(0, 220, 0)       # green
                    elif conf >= 0.4:
                        tcol = QtGui.QColor(255, 190, 0)     # amber
                    else:
                        tcol = QtGui.QColor(255, 60, 60)     # red
                    tlabel = f"TARGET {self.app.store.target}  conf {conf:.2f}"
                # MASK-FIRST: prefer a body-TIGHT SAM2 silhouette over the big
                # square box (the square is "too large"). When a mask is available
                # this frame, draw the silhouette + a small label and SUPPRESS the
                # square entirely; only fall back to the square + ellipse marker
                # when there is no mask (SAM2 off / unavailable / implausible).
                sil = self.app.target_silhouette(frame, tb)
                if sil:
                    self._draw_silhouette(p, sil, tb, frame)
                    self._label(p, r, tlabel, tcol)
                else:
                    p.setPen(QtGui.QPen(tcol, 3))
                    p.drawRect(r)
                    self._label(p, r, tlabel, tcol)
                    if self.app.adaptive_marker_on:
                        self._draw_adaptive_marker(p, tb, frame)

            roi = self.app.roi
            # --- OBJECTS OVERLAY: when an objects.csv is loaded it becomes the
            # visible player set. Draw each tracked obj at this frame as #<obj_id>
            # + team, coloured by candidate status (me-likely=bright green,
            # possible=teal, ruled-out=hidden unless the toggle is on). This
            # REPLACES the raw parquet det boxes below (which would double up). The
            # gold TARGET box + ROI + armed box are still drawn (above / below).
            if self.app.has_objects_overlay():
                for oid, box, team, conf, status in self.app.visible_objects_at(frame):
                    r = self.frame_box_to_widget(box)
                    ruled = oid in self.app.ruled_out      # user said NOT me -> RED
                    inside = roi is not None and bbox.contains(roi, *bbox.center(box))
                    if ruled:
                        col = QtGui.QColor(255, 40, 40)        # ruled out: RED
                        th = 2
                    elif inside:
                        col = QtGui.QColor(0, 200, 255)        # inside ROI: cyan
                        th = 2
                    else:
                        col = self.app._obj_status_color(status, QtGui)
                        th = 3 if status == "me-likely" else 2
                    p.setPen(QtGui.QPen(col, th))
                    p.drawRect(r)
                    team_txt = f" {team}" if team else ""
                    name_suffix = ' NOT ME' if ruled else ""
                    if oid in self.app.det_names:
                        name_suffix = f' "{self.app.det_names[oid][1]}"'
                    self._label(p, r, f"#{oid}{team_txt}{name_suffix}", col)
                # armed box + ROI + recovery + spotlight still draw below
                if self.app.armed_box is not None:
                    r = self.frame_box_to_widget(self.app.armed_box)
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 60, 60), 3))
                    p.drawRect(r)
                if roi is not None:
                    r = self.frame_box_to_widget(roi)
                    col = QtGui.QColor(255, 255, 255) if self.app.roi_on \
                        else QtGui.QColor(160, 160, 160)
                    pen = QtGui.QPen(col, 2)
                    if not self.app.roi_on:
                        pen.setStyle(Qt.DashLine)
                    p.setPen(pen)
                    p.drawRect(r)
                    p.setBrush(col)
                    hs = 5
                    for (hx, hy) in self._handles(roi).values():
                        wx, wy = self.frame_to_widget(hx, hy)
                        p.drawRect(QtCore.QRectF(wx - hs, wy - hs, 2 * hs, 2 * hs))
                    p.setBrush(Qt.NoBrush)
                if self.app.roi_on and self.app.search_box is not None:
                    rs = self.frame_box_to_widget(self.app.search_box)
                    pen = QtGui.QPen(QtGui.QColor(200, 0, 200), 1)
                    pen.setStyle(Qt.DashLine)
                    p.setPen(pen); p.setBrush(Qt.NoBrush)
                    p.drawRect(rs)
                self._draw_focus(p)
                if self.app.spotlight_on:
                    self._draw_spotlight(p)
                self._draw_minimap(p)
                # TACTICAL (M2): player circles + offside line, in pitch metres
                self._draw_circles(p)
                self._draw_offside(p)
                p.end()
                return

            # --- kept detections; those inside the ROI are highlighted ---
            # Plain (not active, not in-ROI) boxes are coloured by CALIBRATED TEAM
            # when a calibration is loaded (teal=tracked team, orange=opposition,
            # grey=unsure); the active/ROI colours keep priority.
            dets = self.app.dets_at(frame)
            active_idx = self.app.active_target_idx()
            team_labels = self.app.team_labels_for_frame(frame)
            visible = self.app.visible_det_indices(frame)
            for i, d in enumerate(dets):
                if i not in visible:
                    continue
                box = d[:4]
                inside = roi is not None and bbox.contains(roi, *bbox.center(box))
                r = self.frame_box_to_widget(box)
                name, is_tracked, confident = (team_labels[i] if i < len(team_labels)
                                               else (None, None, False))
                suffix = ""
                if i == active_idx:
                    col = QtGui.QColor(0, 255, 0)      # ACTIVE target det: green
                    th = 3
                elif inside:
                    col = QtGui.QColor(0, 200, 255)    # inside ROI: cyan
                    th = 2
                elif name is not None and confident:
                    # plain box, coloured by team
                    col = (QtGui.QColor(0, 200, 200) if is_tracked    # tracked: teal
                           else QtGui.QColor(255, 150, 40))           # opposition: orange
                    th = 2
                    suffix = f" {name[:1].upper()}"
                elif name is not None:
                    col = QtGui.QColor(150, 150, 150)  # unsure: grey
                    th = 1
                else:
                    col = QtGui.QColor(150, 150, 150)  # no calibration: grey (as before)
                    th = 1
                p.setPen(QtGui.QPen(col, th))
                p.drawRect(r)
                # ASSIGNED PLAYER NAME drawn on the box (from the track_id->player
                # map): comes from the persistent per-session naming so it follows
                # the detection's stable track id, not the per-frame index.
                name_suffix = ""
                tid = self.app._det_track_hint(box)
                if tid is not None and tid in self.app.det_names:
                    name_suffix = f' "{self.app.det_names[tid][1]}"'
                self._label(p, r, f"#{i}{suffix}{name_suffix}", col)

            # --- armed detection-id box (red), from a click ---
            if self.app.armed_box is not None:
                r = self.frame_box_to_widget(self.app.armed_box)
                p.setPen(QtGui.QPen(QtGui.QColor(255, 60, 60), 3))
                p.drawRect(r)

            # --- the ROI itself (white, with corner handles) ---
            if roi is not None:
                r = self.frame_box_to_widget(roi)
                col = QtGui.QColor(255, 255, 255) if self.app.roi_on \
                    else QtGui.QColor(160, 160, 160)
                pen = QtGui.QPen(col, 2)
                if not self.app.roi_on:
                    pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.drawRect(r)
                p.setBrush(col)
                hs = 5
                for (hx, hy) in self._handles(roi).values():
                    wx, wy = self.frame_to_widget(hx, hy)
                    p.drawRect(QtCore.QRectF(wx - hs, wy - hs, 2 * hs, 2 * hs))
                p.setBrush(Qt.NoBrush)

            # --- RECOVERY search box (faint dashed magenta) while tracking ON ---
            # The wider region recovery scans when CSRT is lost / nothing is in the
            # ROI. Mapped via frame_to_widget so it stays correct under zoom / pan.
            if self.app.roi_on and self.app.search_box is not None:
                rs = self.frame_box_to_widget(self.app.search_box)
                pen = QtGui.QPen(QtGui.QColor(200, 0, 200), 1)   # dim magenta
                pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(rs)

            self._draw_focus(p)
            # --- spotlight follow overlay (drawn last, on top of everything) ---
            if self.app.spotlight_on:
                self._draw_spotlight(p)
            self._draw_minimap(p)
            # TACTICAL (M2): player circles + offside line, in pitch metres
            self._draw_circles(p)
            self._draw_offside(p)
            p.end()

        def _draw_minimap(self, p):
            """Top-down pitch minimap (M1): the field outline + key lines, every
            visible player (overlay status colour), and the gold tracked target,
            using the LIVE (CMC-compensated) homography to map feet -> pitch metres.
            Drawn only when a calibration is loaded, the user has it toggled ON, and
            the live mapping is HEALTHY (not stale). Fully inert otherwise."""
            if not getattr(self.app, "minimap_on", False):
                return
            lh = self.app.live_h
            if not (lh and lh.healthy and not self.app._live_stale):
                return
            mw = 240; mh = 240 * PITCH.width / PITCH.length; pad = 12
            rect = QtCore.QRectF(self.width() - mw - pad, pad, mw, mh)
            p.fillRect(rect, QtGui.QColor(0, 40, 0, 180))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 160), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)
            sx, sy = mw / PITCH.length, mh / PITCH.width

            def to_map(X, Y):    # pitch metres -> minimap px (flip Y so top touch up)
                return rect.left() + X * sx, rect.top() + (PITCH.width - Y) * sy

            # try-lines + halfway
            for X in (PITCH.in_goal, PITCH.in_goal + PITCH.play_length,
                      PITCH.length / 2):
                x0, y0 = to_map(X, 0); x1, y1 = to_map(X, PITCH.width)
                p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

            # players from the objects overlay (if present); foot -> pitch metres
            try:
                objs = self.app.visible_objects_at(self.app.frame)
            except Exception:
                objs = []
            for tup in objs:
                box = tup[1]
                xy = lh.foot_point(box)
                if xy is None:
                    continue
                mx, my = to_map(*xy)
                if hasattr(self.app, "_obj_status_color"):
                    col = self.app._obj_status_color(tup[4], QtGui)
                else:
                    col = QtGui.QColor(0, 200, 255)
                p.setBrush(col); p.setPen(Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(mx, my), 3, 3)

            # the tracked target (gold), drawn on top
            tb = self.app.store.target_box_at(self.app.frame)
            if tb is not None:
                xy = lh.foot_point(tb)
                if xy is not None:
                    mx, my = to_map(*xy)
                    p.setBrush(QtGui.QColor(255, 215, 0))
                    p.setPen(QtGui.QPen(Qt.black, 1))
                    p.drawEllipse(QtCore.QPointF(mx, my), 5, 5)
            p.setBrush(Qt.NoBrush)

        # ---- TACTICAL (M2): player circles + offside line ----------------
        def _circle_polygon(self, cx_m, cy_m, r_m):
            """Build a widget-px QPolygonF for a true-metre ground ring centred on
            the pitch point (cx_m, cy_m): map each ring point pitch->image via the
            live homography, then image->widget via frame_to_widget. Returns None
            if too few points map (any homography call may return None). The result
            is a foreshortened ellipse automatically (perspective-correct)."""
            lh = self.app.live_h
            poly = QtGui.QPolygonF()
            for X, Y in pitch_circle_points(cx_m, cy_m, r_m):
                img = lh.pitch_to_image(X, Y)
                if img is None:
                    continue
                wx, wy = self.frame_to_widget(img[0], img[1])
                poly.append(QtCore.QPointF(wx, wy))
            return poly if poly.count() >= 8 else None

        def _draw_flat_ellipse(self, p, box, col):
            """FALLBACK ground ring when there is no live pitch: a dim, DASHED flat
            pixel ellipse at the feet (bottom-centre of the box, axes ~ box width)
            so the tool still shows something. Visually de-emphasised vs the true
            metre ring."""
            r = self.frame_box_to_widget(box)
            cx = r.center().x()
            cy = r.bottom()
            ax = max(6.0, r.width() * 0.55)
            ay = max(3.0, r.width() * 0.22)
            dim = QtGui.QColor(col); dim.setAlpha(150)
            pen = QtGui.QPen(dim, 2); pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QtCore.QPointF(cx, cy), ax, ay)

        def _draw_circles(self, p):
            """Draw a ground ring under every SELECTED overlay player + the TARGET
            (always). True-metre perspective ring via the live homography; a dim
            dashed pixel ellipse as a fallback when the pitch is missing/stale.
            Inert unless circles_on. Cheap: only runs when the toggle is on."""
            app = self.app
            if not app.circles_on:
                return
            lh = app.live_h
            live = bool(lh and lh.healthy and not app._live_stale)
            frame = app.frame
            p.setBrush(Qt.NoBrush)

            def ring(box, col):
                if live:
                    xy = lh.foot_point(box)
                    if xy is not None:
                        poly = self._circle_polygon(xy[0], xy[1], CIRCLE_R_M)
                        if poly is not None:
                            p.setPen(QtGui.QPen(col, 2))
                            p.drawPolygon(poly)
                            return
                # no live pitch (or this box didn't map): dim flat fallback
                self._draw_flat_ellipse(p, box, col)

            # selected overlay players, coloured by candidate status/team
            if app.has_objects_overlay() and app.selected_objs:
                for oid, box, _team, _conf, status in app.visible_objects_at(frame):
                    if oid in app.selected_objs:
                        ring(box, app._obj_status_color(status, QtGui))

            # the TARGET is ALWAYS ringed (gold), on top
            tb = app.store.target_box_at(frame)
            if tb is not None:
                ring(tb, QtGui.QColor(255, 215, 0))

            # de-emphasised status hint when we fell back to pixel ellipses
            if not live:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 235, 120, 180), 1))
                f = p.font(); f.setPointSize(9); f.setBold(False); p.setFont(f)
                p.drawText(QtCore.QPointF(10.0, self.height() - 10.0),
                           "circles approximate - no live pitch")

        def _draw_offside(self, p):
            """Draw the OFFSIDE line: a constant pitch-X line across the field,
            sampled Y=0..PITCH.width and mapped pitch->image->widget via the live
            homography. Drawn ONLY with a live (non-stale) pitch. Bold cyan, labelled
            'offside'. Inert when offside_x is None."""
            app = self.app
            if app.offside_x is None:
                return
            lh = app.live_h
            if not (lh and lh.healthy and not app._live_stale):
                return
            X = app.offside_x
            poly = QtGui.QPolygonF()
            n = 12
            for k in range(n + 1):
                Y = PITCH.width * k / float(n)
                img = lh.pitch_to_image(X, Y)
                if img is None:
                    continue
                wx, wy = self.frame_to_widget(img[0], img[1])
                poly.append(QtCore.QPointF(wx, wy))
            if poly.count() < 2:
                return
            p.setPen(QtGui.QPen(QtGui.QColor(0, 230, 230), 3))   # bold cyan
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(poly)
            self._label(p, QtCore.QRectF(poly.first(), poly.first()),
                        "offside", QtGui.QColor(0, 230, 230))

        def _draw_adaptive_marker(self, p, tb, frame):
            """CONFIDENCE-ADAPTIVE MARKER: draw the shape from adaptive_marker_spec
            for the target box `tb` at the current confidence. The spec is in
            FRAME coords; points are mapped via frame_to_widget so it stays
            correct under zoom / pan. The ellipse cases are an OUTLINE; the circle
            case is a DASHED circle (reads as "searching"). Coloured by confidence
            (green >=0.7, amber 0.4-0.7, red <0.4) over a thin dark backing. The
            drawn size is EMA-smoothed frame-to-frame to damp flicker."""
            # confidence for this frame (default to last value, then fully
            # confident if nothing tracked yet so a scrubbed frame shows tight).
            # in manual mode the human is the tracker -> default to confident so
            # the marker is a tight green ellipse immediately, not a red search ring
            _dflt = 1.0 if self.app.manual_mode else self.app.last_confidence
            conf = self.app.conf_by_frame.get(frame, _dflt)
            spec = adaptive_marker_spec(tb, conf)
            kind, cx, cy, a = spec[0], spec[1], spec[2], spec[3]
            b = spec[4] if kind == "ellipse" else a   # circle: equal radii
            # EMA-smooth the size (a, b) to avoid flicker as conf jitters. The
            # ellipse and circle geometries are NOT on the same scale, so blending
            # across the ellipse<->circle "kind" change at the 0.4 boundary would
            # draw a wrong-sized transitional marker. RESET the EMA when the kind
            # differs from the previous frame so each shape starts clean.
            prev = self.app._adapt_size
            if self.app._adapt_kind != kind:
                prev = None
            self.app._adapt_kind = kind
            if prev is None:
                sa, sb = a, b
            else:
                e = ADAPT_MARKER_EMA
                sa = e * a + (1.0 - e) * prev[0]
                sb = e * b + (1.0 - e) * prev[1]
            self.app._adapt_size = (sa, sb)

            # colour by confidence (mirrors the target-box colouring)
            c = 1.0 if conf is None else conf
            if c >= ADAPT_CONF_HI:
                col = QtGui.QColor(0, 220, 0)         # green
            elif c >= ADAPT_CONF_LO:
                col = QtGui.QColor(255, 190, 0)       # amber
            else:
                col = QtGui.QColor(255, 60, 60)       # red

            # map centre + axis endpoints to widget px (correct under zoom/pan)
            wcx, wcy = self.frame_to_widget(cx, cy)
            wex, _ = self.frame_to_widget(cx + sa, cy)   # +x axis endpoint
            _, wey = self.frame_to_widget(cx, cy + sb)   # +y axis endpoint
            rw = abs(wex - wcx)
            rh = abs(wey - wcy)
            rect = QtCore.QRectF(wcx - rw, wcy - rh, 2 * rw, 2 * rh)

            p.setBrush(Qt.NoBrush)
            dashed = (kind == "circle")
            # thin dark backing for contrast, then the coloured marker on top
            back = QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 4)
            front = QtGui.QPen(col, 2)
            if dashed:
                back.setStyle(Qt.DashLine)
                front.setStyle(Qt.DashLine)
            p.setPen(back)
            p.drawEllipse(rect)
            p.setPen(front)
            p.drawEllipse(rect)

        def _draw_silhouette(self, p, polys, tb, frame):
            """SAM2 SILHOUETTE marker: draw a body-TIGHT outline following the
            player's segmentation mask, and DIM the area inside the target box but
            OUTSIDE the mask (suppress the green grass). polys are lists of (x, y)
            in FRAME coords; mapped via frame_to_widget so it stays correct under
            zoom / pan. Confidence-coloured (green/amber/red) like the box."""
            _dflt = 1.0 if self.app.manual_mode else self.app.last_confidence
            conf = self.app.conf_by_frame.get(frame, _dflt)
            c = 1.0 if conf is None else conf
            if c >= ADAPT_CONF_HI:
                col = QtGui.QColor(0, 220, 0)
            elif c >= ADAPT_CONF_LO:
                col = QtGui.QColor(255, 190, 0)
            else:
                col = QtGui.QColor(255, 60, 60)
            # union of the mask polygons, in widget coords
            sil = QtGui.QPainterPath()
            for pts in polys:
                if len(pts) < 3:
                    continue
                poly = QtGui.QPolygonF(
                    [QtCore.QPointF(*self.frame_to_widget(x, y)) for x, y in pts])
                sub = QtGui.QPainterPath()
                sub.addPolygon(poly)
                sub.closeSubpath()
                sil = sil.united(sub)
            if sil.isEmpty():
                return
            # DIM the green: (target box) minus silhouette, translucent dark fill
            rect = self.frame_box_to_widget(tb)
            box_path = QtGui.QPainterPath()
            box_path.addRect(rect)
            p.fillPath(box_path.subtracted(sil), QtGui.QColor(0, 0, 0, 120))
            # tight outline: dark backing then the coloured silhouette on top
            p.setBrush(Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 4))
            p.drawPath(sil)
            p.setPen(QtGui.QPen(col, 2))
            p.drawPath(sil)

        def _draw_spotlight(self, p):
            """Draw a downward chevron hovering just above the target's head (the
            live equivalent of export.py's arrow). The anchor is computed in FRAME
            coords and mapped via frame_to_widget so it stays correct under zoom /
            pan. Size scales with the marker-size state (+/- adjusts it)."""
            c = self.app._spotlight_centre()        # smoothed (cx, head_top) frame px
            if c is None:
                return
            cx, top_y = c
            side = self.app.spotlight_side
            aw = max(16.0, 0.28 * side)             # arrow width  (frame px)
            ah = max(20.0, 0.34 * side)             # arrow height
            gap = max(6.0, 0.06 * side)             # gap above the head
            tip = self.frame_to_widget(cx, top_y - gap)
            bl = self.frame_to_widget(cx - aw / 2.0, top_y - gap - ah)
            br = self.frame_to_widget(cx + aw / 2.0, top_y - gap - ah)
            poly = QtGui.QPolygonF([QtCore.QPointF(*tip),
                                    QtCore.QPointF(*bl),
                                    QtCore.QPointF(*br)])
            p.setBrush(QtGui.QColor(0, 215, 255))   # amber, matches export.py
            p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
            p.drawPolygon(poly)

        def _draw_focus(self, p):
            """Draw the FOCUS box (FOCUS_MULT x ROI) - the ONLY region being
            ID-traced (detections outside it are ignored + hidden)."""
            fb = self.app.focus_box()
            if fb is None:
                return
            r = self.frame_box_to_widget(fb)
            pen = QtGui.QPen(QtGui.QColor(255, 235, 120), 2)     # soft yellow
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)

        def _label(self, p, rect, text, col):
            p.setPen(QtGui.QPen(col, 1))
            f = p.font(); f.setPointSize(9); f.setBold(True); p.setFont(f)
            p.drawText(QtCore.QPointF(rect.left(), max(10.0, rect.top() - 3)), text)

    # -------------------------------------------------------------------
    # Main window
    # -------------------------------------------------------------------
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            # MULTI-PLAYER: the name (or None) is threaded from --player; it
            # namespaces this player's profile / track / outputs.
            self.player = player
            # --kept-only flag this session was launched with. Threaded through the
            # open_clip / player-switch RELAUNCH so the detection-pool mode is
            # preserved across clips/players instead of silently resetting to ALL.
            self.kept_only = kept_only
            self.video_path = video_path      # source video (for player-selector relaunch)
            clip = pathlib.Path(video_path).name
            if self.player:
                self.setWindowTitle(f"track.py - {clip} - player:{self.player}")
            else:
                self.setWindowTitle(f"track.py - {clip}")

            self.reader = VideoReader(video_path)
            self.fw = self.reader.width or 1920
            self.fh = self.reader.height or 1080
            self.fps = self.reader.fps or FPS_ASSUMED
            self.n_frames = self.reader.n_frames

            self.store = store
            self.det_by_frame = det_by_frame
            self.tracks_path = tracks_path

            # match calibration (team fingerprints) + per-frame team-label cache.
            # calib may be None (feature inert) or have <2 fingerprinted teams.
            self.calib = calib
            self._team_fp_teams = []          # teams that HAVE a fingerprint
            if calib is not None:
                self._team_fp_teams = [t for t in calib.teams
                                       if getattr(t, "fingerprint", None)]
            self._team_cache = {}             # frame -> list aligned with dets_at(frame)

            # OBJECTS OVERLAY (overnight mark_all output/<stem>.objects.csv): EVERY
            # player, tracked, per frame, with a stable obj_id + team. When present
            # for this clip it becomes the visible player set and is auto-classified
            # ("who could be ME?") via rt2.candidates so the user picks from a short
            # list instead of the whole field. ALL state stays empty / the feature
            # is fully INERT when the file is absent (existing behaviour unchanged).
            self.objects_by_frame = {}        # frame -> [(obj_id,(x1,y1,x2,y2),team,conf)]
            self._obj_aggs = {}               # obj_id -> candidates.TrackAgg
            self._obj_verdicts = {}           # obj_id -> candidates.Verdict
            self._obj_summary = {"me_likely": 0, "possible": 0,
                                 "ruled_out": 0, "total": 0}
            self.show_ruled_out = False       # CANDIDATES toggle: draw ruled-out tracks?
            self._load_objects_overlay()

            # PITCH HOMOGRAPHY + per-frame CMC (M1): a calibration clicked on ONE
            # anchor frame (output/<stem>.pitch.json, else the shared
            # game.pitch.json) is slid along the pan/zoom by camera-motion
            # compensation so a top-down minimap + metre-accurate foot points are
            # valid on the CURRENT frame. ALL of this is fully INERT (self.live_h
            # is None) when no calibration is present; nothing below ever crashes.
            self.live_h = None                # LiveHomography or None (inert)
            self._pitch_anchor = None         # frame the pitch points were clicked on
            self._live_last_frame = None      # last frame CMC was advanced TO (sequential)
            self._live_stale = False          # True when the homography can't be trusted
            self.minimap_on = False           # drawn only when a calibration loads
            self._load_pitch_calibration()

            # TACTICAL OVERLAY (M2): a "character selector" that draws true-metre
            # ground CIRCLES under chosen players + an OFFSIDE line, all in pitch
            # metres via the live homography (so they stay correct under zoom/pan
            # AND the camera pan). circles_on gates the whole feature (DEFAULT OFF);
            # while ON a left-click on an overlay player TOGGLES it in selected_objs
            # instead of placing an ROI. The TARGET is always circled. offside_x is
            # a constant pitch X (metres) for the offside line, or None. Fully inert
            # until circles_on / a selection / offside_x is set.
            self.circles_on = False           # draw player circles? (hotkey z)
            self.selected_objs = set()        # overlay obj_ids with a circle
            self.offside_x = None             # offside line pitch X (m) or None

            # DISPLAY FILTERS (cosmetic declutter only; never affect tracking/saving)
            # Opposition boxes are HIDDEN BY DEFAULT: the opposition is still
            # detected + tracked (so it can rule players out) but not DRAWN, so the
            # screen isn't cluttered with the other team. Toggle back on with 'h'.
            self.hide_other_teams = True      # hide confidently non-tracked-team boxes
            self.hide_off_field = False       # hide boxes whose feet are off the pitch
            self.focus_on = True              # restrict ID tracing to the FOCUS box
            self._field_mask_cache = {}       # frame -> field_mask (only when hide_off_field)

            # state
            self.frame = 1
            self.playing = False
            self.armed = None                 # armed detection-pool track id (for merge/del)
            self.armed_box = None
            self.roi = None                   # xyxy ROI in frame coords
            self.roi_on = False               # ROI tracking active?
            # HOLD-don't-guess: when recovery is OFF (DEFAULT) the tracker HOLDS
            # the ROI in place on loss instead of running the big search-box
            # recovery (which can grab the opposition / a far ref). Toggle 'v'.
            self.recovery_on = False
            # MANUAL FOLLOW MODE: when ON there is NO CSRT / snap / recovery -
            # the ROI simply FOLLOWS THE MOUSE CURSOR and each forward step
            # records wherever it sits. Toggle 'm'.
            self.manual_mode = False
            # last cursor position in FRAME px (set by VideoView mouse-move so
            # manual cursor-follow can snap the ROI onto it). None until the mouse
            # has moved over the frame.
            self._cursor_fp = None
            self.search_box = None            # last RECOVERY search box (xyxy) for drawing
            self.csrt = None                  # cv2 tracker
            self.csrt_frame = None            # frame the csrt was last advanced TO
            self.input_buf = ""               # typed-digit buffer
            self.input_mode = None            # None | 'target' | 'merge'
            self.status_msg = ""

            # playback speed (multiplier over the base frame interval)
            self.speed = DEFAULT_SPEED

            # spotlight follow mode
            self.spotlight_on = False
            self.spotlight_side = float(SPOTLIGHT_DEFAULT)   # square side, frame px
            self._spot_cx = None              # smoothed spotlight centre (frame px)
            self._spot_cy = None

            # CONFIDENCE-ADAPTIVE MARKER (Feature 1): a tight body ellipse when
            # confident, growing to a big dashed SEARCH circle when not. Default
            # ON (the arrow marker is the alternative). _adapt_size is an EMA of
            # the drawn axes/radius to damp frame-to-frame flicker.
            self.adaptive_marker_on = True
            self._adapt_size = None           # smoothed (a, b) ellipse axes / (r, r) circle
            self._adapt_kind = None           # last marker kind ("ellipse"/"circle"); EMA
                                              # resets on a kind flip to avoid blending shapes

            # CONFIDENCE-ADAPTIVE SPEED (Feature 3): when ON during play the
            # per-tick interval is driven by the latest confidence rather than the
            # manual speed (slow when unsure, fast when confident). Default OFF.
            self.conf_speed = False

            # frame cache: frame -> {"roi": xyxy, "target": xyxy} (positions, not images)
            self.frame_cache = {}

            # per-frame CONFIDENCE rating (0..1): last value + a per-frame dict so
            # scrubbing back shows the confidence the frame was tracked at.
            self.last_confidence = None
            self.conf_by_frame = {}
            # AUTO-PAUSE on low confidence: when ON and playing, a frame whose
            # confidence < AUTO_PAUSE_THRESH pauses playback. Default OFF.
            self.autopause_low = False
            # LOST-IN-THE-RUCK prompt (Feature 3): when auto-pause is ON and the
            # target is lost (an auto-pause fires, OR RUCK_LOST_FRAMES consecutive
            # very-low-confidence frames accrue while playing), prompt ONCE asking
            # whether to switch to manual mode. _ruck_low_run counts the current
            # run of sub-threshold frames; _ruck_prompted latches so we only ask
            # once per loss (reset when confidence recovers >= AUTO_PAUSE_THRESH).
            self._ruck_low_run = 0
            self._ruck_prompted = False

            # JERSEY-NUMBER OCR soft signal (BONUS cue; DEFAULT OFF). When ON and
            # rt2.ocr.available(), the target's upper-back is OCR'd every Nth
            # tracked frame; confident 1-2 digit reads accumulate in a Counter and
            # the most-frequent one is exposed as self.target_number. It is a SOFT
            # confidence cue only (small boost/penalty) and a strict NO-OP when OFF
            # or unavailable. Toggle 'j' / the "Jersey OCR" panel button.
            self.ocr_on = False
            self.target_number = None         # most-frequent confident read (str) or None
            self._ocr_counter = Counter()     # digit-string -> count of confident reads
            self._ocr_last = None             # last (digits, conf) read this session
            self._ocr_frame_tick = 0          # counts tracked frames for throttling

            # MY-NUMBER live jersey exclusion (Feature 1). self.my_number (int or
            # None) is the user's own shirt number; it is set from the profile /
            # CLI / UI just below. _track_numbers tallies confident jersey reads
            # PER track id (keyed on _det_track_hint), weighted by OCR confidence.
            # A track confidently reading a DIFFERENT number is dropped from
            # eligibility; one reading MY number is preferred. INERT (zero cost /
            # no behaviour change) whenever ocr.available() is False OR
            # self.my_number is None. _mynum_frame_tick throttles the per-frame
            # candidate OCR so it never tanks the framerate.
            self.my_number = None             # set after the profile load below
            self._track_numbers = defaultdict(Counter)  # track_id -> Counter(num->weight)
            self._mynum_frame_tick = 0        # counts tracked frames for throttling

            # SAM 2 box-refinement (DEFAULT OFF; hotkey 'f' / panel button). When ON
            # and ultralytics is available, the per-frame target box is tightened to
            # SAM2's segmentation mask (a cleaner box -> better marker + height).
            # self.sam2 is the lazy Sam2Box (created on first enable so torch is
            # imported only AFTER pyarrow, avoiding the import-order segfault).
            self.sam2_on = False
            self.sam2 = None
            self.sam2_model = sam2_model or sam2track.DEFAULT_MODEL
            # cache of the target's SAM2 silhouette polygons per (frame, box) so
            # paintEvent doesn't re-segment (130ms) on every repaint of a frame.
            self._sil_cache = {}

            # SAM 3 BACKGROUND RE-TRACK (loose-mark live, SAM3 re-tracks behind):
            # the user marks a loose IN-point ('w'), scrubs to the end and triggers
            # a re-track ('W'). A SEPARATE PROCESS (apps/sam3_segment.py, launched
            # via QProcess so torch never loads in this GUI process) propagates a
            # high-quality, ruck-holding box across the window; the clean result is
            # merged into the target track as ONE undoable op. _sam3_start is the
            # marked IN-point (None = default last ~3s); _sam3_proc the running
            # QProcess; _sam3_out the temp JSON path it is writing.
            self._sam3_start = None
            self._sam3_proc = None
            self._sam3_out = None
            self._sam3_window = None      # (start, n) of the run in flight
            self._sam3_stderr_buf = ""

            # JUMP GATE state: last CONFIRMED target centre/frame + how many frames
            # a big jump has persisted (so we HOLD before accepting a teleport).
            self._last_good_centre = None
            self._last_good_frame = None
            self._jump_pending = 0

            # ONLINE TARGET MODEL: learns the followed player (motion + appearance)
            self.tmodel = TargetModel()

            # PERSISTENT player appearance profile (carries across clips). Loaded
            # at startup if present; saved on 's' and on the dedicated 'p' key.
            # In MULTI-PLAYER mode each player has their own profile file
            # (output/player_<name>.json); single-player uses player_profile.json.
            self.profile_path = ProjectPaths().player_profile(self.player)
            n = self.tmodel.load_profile(self.profile_path)
            if n:
                print(f"[track] loaded player profile: {n} crops "
                      f"({self.profile_path})")
            # MY JERSEY NUMBER: an explicit --my-number on the CLI WINS over the
            # value persisted in the profile (the user is re-configuring it this
            # launch); otherwise inherit whatever the profile carried over. Mirror
            # it onto the TargetModel so save_profile persists it. Clamp to 1..23.
            if my_number is not None and 1 <= int(my_number) <= MAX_JERSEY:
                self.my_number = int(my_number)
            else:
                self.my_number = self.tmodel.my_number
            self.tmodel.my_number = self.my_number
            if self.my_number is not None:
                print(f"[track] my jersey number = {self.my_number}")

            # OBJECTS OVERLAY shortlist: now that my_number is resolved, classify
            # every overlay track ("who could be ME?"). Recomputed on my_number
            # changes (set_my_number) + the CANDIDATES "Re-shortlist" button.
            # No-op when no objects.csv was loaded.
            self._recompute_shortlist()

            # ONLINE IDENTITY CLASSIFIER ("me vs not-me"): learns the target from
            # the user's CONFIRMATIONS (positives) and CORRECTIONS (hard negatives
            # = "rule out this player"). Persisted in a SIBLING file next to the
            # appearance profile (player_<name>.id.json), loaded on startup and
            # saved on 's' / 'p'. ruled_out holds track ids the user has marked as
            # NOT-me this session so they are excluded from eligibility (like the
            # team constraint).
            self.idmodel = identity.IdentityModel(dim=IDENTITY_DIM)
            self.id_path = self.profile_path.with_suffix(".id.json")
            if self.idmodel.load(self.id_path):
                ip, ineg = self.idmodel.counts()
                print(f"[track] loaded identity model: {ip} pos / {ineg} neg "
                      f"({self.id_path})")
            self.ruled_out = set()            # track ids the user said are NOT me
            self.ruleout_mode = False         # ON: clicking a player rules it out

            # DETECTION -> PLAYER NAMING (links to the registry DB).
            # In-session map track_id -> (registry_uuid, display_name). The
            # detection list is rebuilt every frame, so the name must come from
            # this PERSISTENT-per-session map (keyed on the stable track id), not
            # from the per-frame detection index. The registry row itself lives
            # in output/registry.sqlite so the name is reusable across clips and
            # is the SAME DB the overnight profiler feeds.
            self.det_names = {}               # track_id -> (uuid, display_name)

            self._build_ui()
            self._show_frame()

        # ---- OBJECTS OVERLAY (overnight mark_all "mark everyone") ----
        def _load_objects_overlay(self):
            """Parse output/<stem>.objects.csv (if present) into the per-frame
            overlay map + per-obj candidates.TrackAgg aggregates. ROBUST: a
            missing/short/garbled file never crashes -- on any failure the overlay
            stays empty and the existing parquet-detection display is used."""
            try:
                stem = pathlib.Path(self.video_path).stem
                path = ProjectPaths().output / f"{stem}.objects.csv"
                if not path.exists():
                    return
                with open(path, newline="", encoding="utf-8", errors="replace") as f:
                    rows = list(csv.DictReader(f))
                obf, aggs = aggregate_objects_rows(rows, self.fw, self.fh)
                self.objects_by_frame = obf
                self._obj_aggs = aggs
                print(f"[track] loaded objects overlay: {path} "
                      f"({len(obf)} frames, {len(aggs)} obj ids)")
            except Exception as e:
                self.objects_by_frame = {}
                self._obj_aggs = {}
                print(f"[track] WARNING could not read objects overlay: {e}")

        def _pitch_file(self):
            """Resolve the pitch calibration path: per-clip output/<stem>.pitch.json
            if it exists, else the shared output/game.pitch.json."""
            out = ProjectPaths().output
            stem = pathlib.Path(self.video_path).stem
            per_clip = out / f"{stem}.pitch.json"
            return per_clip if per_clip.exists() else out / "game.pitch.json"

        def _load_pitch_calibration(self):
            """Load the pitch homography (per-clip, else shared game) and wrap it in
            a LiveHomography + CameraTracker for per-frame CMC. TOLERANT: any
            missing/garbled file leaves self.live_h = None (the minimap + live
            mapping stay fully inert). Sets the minimap default ON when a
            calibration is present."""
            self.live_h = None
            self._pitch_anchor = None
            self._live_last_frame = None
            self._live_stale = False
            try:
                path = self._pitch_file()
                pitch_h = PitchHomography.load(path)
                if pitch_h is not None and pitch_h.ok:
                    self.live_h = LiveHomography(pitch_h, CameraTracker())
                    self._pitch_anchor = pitch_h.anchor_frame
                    self.minimap_on = True
                    print(f"[track] loaded pitch calibration: {path} "
                          f"(anchor frame {self._pitch_anchor})")
                else:
                    print(f"[track] no usable pitch calibration at {path}")
            except Exception as e:                       # noqa: BLE001 - never crash
                self.live_h = None
                print(f"[track] WARNING could not load pitch calibration: {e}")

        def _live_player_boxes(self):
            """Player boxes to MASK OUT of the camera-motion estimate this frame
            (moving players bias it). Prefer the objects overlay; else the kept
            detections. Returns a list of (x1,y1,x2,y2)."""
            try:
                if self.has_objects_overlay():
                    return [box for (_oid, box, *_rest)
                            in self.objects_at(self.frame)]
                return [d[:4] for d in self.dets_at(self.frame)]
            except Exception:                            # noqa: BLE001
                return []

        def _update_live_homography(self):
            """Advance the per-frame CMC to the CURRENT displayed frame. SEQUENTIAL
            ONLY: it composes the anchor->current transform one frame at a time, so
            it only updates when stepping forward by exactly one frame from the last
            updated frame (or anchoring at the pitch's anchor frame). Any jump /
            non-sequential move marks the homography STALE (can't be trusted here)
            until play resumes from the anchor frame (or a re-anchor). Fully inert
            when no calibration is loaded. ~30ms ORB cost; runs at most once/frame."""
            if self.live_h is None:
                return
            f = self.frame
            img = self._frame_img_for_appearance()       # current BGR (already decoded)
            if img is None:
                self._live_stale = True
                return
            action = live_cmc_action(f, self._pitch_anchor, self._live_last_frame)
            if action == "anchor":
                # anchor exactly where the pitch points were clicked -> trusted
                self.live_h.tracker.anchor(img)
                self._live_last_frame = f
                self._live_stale = False
            elif action == "update":
                # one sequential step forward -> fold this frame's motion in
                self.live_h.tracker.update(
                    img, ignore_boxes=self._live_player_boxes())
                self._live_last_frame = f
                self._live_stale = False
            else:
                # a JUMP / non-sequential seek / not yet anchored: do NOT update;
                # mark stale so overlays know the mapping can't be trusted here.
                self._live_stale = True

        def _live_health_text(self):
            """Status string for the pitch calibration: None when inert, else a
            'pitch: live (N inl)' / 'pitch: STALE ...' message."""
            if self.live_h is None:
                return None
            if self._live_stale or not self.live_h.healthy:
                return ("pitch: STALE - play forward from calib frame "
                        f"{self._pitch_anchor} or re-calibrate")
            return f"pitch: live ({self.live_h.tracker.last_inliers} inl)"

        def has_objects_overlay(self):
            """True when an objects.csv overlay is loaded (the visible player set
            is the overlay). False -> fully inert, existing behaviour unchanged."""
            return bool(self.objects_by_frame)

        def _me_profile(self):
            """Build a candidates.MeProfile from LIVE state: my jersey number, the
            tracked team name (from calibration) and the learned height residual
            mean (if the target model exposes one, else None)."""
            team = None
            try:
                if self.calib is not None:
                    tt = self.calib.tracked_teams()
                    if tt:
                        team = tt[0].name
            except Exception:
                team = None
            # the target's learned residual mean, only once enough samples make it
            # meaningful (height_ready); else leave None (overlay aggs are None too).
            hr = None
            try:
                if self.tmodel.height_ready():
                    hr = float(self.tmodel.resid_mean)
            except Exception:
                hr = None
            return candidates.MeProfile(number=self.my_number,
                                        height_resid_mean=hr, team=team)

        def _recompute_shortlist(self):
            """Re-classify every overlay track against the current MeProfile and
            cache obj_id -> Verdict + the summary. No-op (clears) when no overlay."""
            if not self._obj_aggs:
                self._obj_verdicts = {}
                self._obj_summary = {"me_likely": 0, "possible": 0,
                                     "ruled_out": 0, "total": 0}
                self._sync_candidates_label()
                return
            verdicts, summary = candidates.shortlist(
                list(self._obj_aggs.values()), self._me_profile())
            self._obj_verdicts = {v.obj_id: v for v in verdicts}
            self._obj_summary = summary
            self._sync_candidates_label()

        def _sync_candidates_label(self):
            """Refresh the CANDIDATES summary label, if it has been built."""
            lbl = getattr(self, "cand_lbl", None)
            if lbl is None:
                return
            if not self.has_objects_overlay():
                lbl.setText("candidates: (no objects.csv)")
                return
            s = self._obj_summary
            lbl.setText(f"candidates: {s['me_likely']} likely / "
                        f"{s['possible']} possible / {s['ruled_out']} out")

        def _obj_status_color(self, status, QtGui):
            """Box colour for an overlay obj by candidate status (None for
            ruled-out, which is hidden unless show_ruled_out)."""
            if status == "me-likely":
                return QtGui.QColor(0, 230, 0)        # bright green = highlighted
            if status == "possible":
                return QtGui.QColor(0, 200, 200)      # teal/cyan = normal
            return QtGui.QColor(150, 90, 90)          # ruled-out (only if shown)

        def objects_at(self, frame):
            """Overlay objects at `frame`: list of (obj_id, box, team, conf)."""
            return self.objects_by_frame.get(frame, [])

        def visible_objects_at(self, frame):
            """Overlay objects to DRAW at `frame`: ruled-out are hidden unless the
            'Show ruled-out' toggle is on. The active target's obj (if any) is
            always kept. Returns the same tuples as objects_at."""
            out = []
            fb = self.focus_box()
            for oid, box, team, conf in self.objects_at(frame):
                v = self._obj_verdicts.get(oid)
                status = v.status if v is not None else "possible"
                # HIDE OTHER TEAMS ('h'): drop opposition overlay players. This was
                # only wired to the OLD detection path, so the toggle did nothing
                # once the objects overlay loaded. Judge by the track's AGGREGATE
                # dominant team (robust to a single noisy/unsure frame), falling
                # back to this frame's label; "unsure" stays so the target is never
                # hidden by mistake.
                if self.hide_other_teams:
                    agg = self._obj_aggs.get(oid)
                    dom = agg.dominant_team() if agg is not None else team
                    if dom == "opp":
                        continue
                if status == "ruled-out" and not self.show_ruled_out:
                    continue
                # FOCUS: only trace/draw objects inside the 3x-ROI focus region
                if fb is not None and not bbox.contains(fb, *bbox.center(box)):
                    continue
                out.append((oid, box, team, conf, status))
            return out

        # ---- UI ----
        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            v = QtWidgets.QVBoxLayout(central)
            # small content inset so the central view is never flush against the
            # window/screen edge (esp. when maximized); docks keep their own margins.
            v.setContentsMargins(4, 4, 4, 4)

            # TOP status bar
            self.status_lbl = QtWidgets.QLabel()
            self.status_lbl.setStyleSheet(
                "background:#111;color:#eee;padding:5px;font-family:Consolas;font-size:12px;")
            v.addWidget(self.status_lbl)

            # CENTRE video
            self.view = VideoView(self)
            self.view.clicked.connect(self.on_view_click)
            v.addWidget(self.view, 1)

            # BOTTOM full-width scrubber + play/pause
            bottom = QtWidgets.QHBoxLayout()
            self.play_btn = QtWidgets.QPushButton("Play")
            self.play_btn.setFixedWidth(90)
            self.play_btn.clicked.connect(self.toggle_play)
            bottom.addWidget(self.play_btn)
            self.save_btn = QtWidgets.QPushButton("Save")
            self.save_btn.setFixedWidth(90)
            self.save_btn.setToolTip("Save the target track CSV (or press S)")
            self.save_btn.clicked.connect(self.do_save)
            bottom.addWidget(self.save_btn)
            self.track_btn = QtWidgets.QPushButton("Track: OFF")
            self.track_btn.setCheckable(True)
            self.track_btn.setFixedWidth(110)
            self.track_btn.setToolTip("ROI tracking on/off (or press R). Must be ON "
                                      "for Play to record + learn.")
            self.track_btn.clicked.connect(self.toggle_roi_tracking)
            bottom.addWidget(self.track_btn)
            self.manual_btn = QtWidgets.QPushButton("Manual")
            self.manual_btn.setCheckable(True)
            self.manual_btn.setFixedWidth(100)
            self.manual_btn.setToolTip("MANUAL cursor-follow (or press M): no CSRT/"
                                       "snap; ROI follows the mouse, each step "
                                       "records it. M again = resume auto-follow.")
            self.manual_btn.clicked.connect(self.toggle_manual_mode)
            bottom.addWidget(self.manual_btn)
            # frame-step + edit buttons (same actions as the arrow / d / c keys)
            self.prev_btn = QtWidgets.QPushButton("◀")
            self.prev_btn.setFixedWidth(40)
            self.prev_btn.setToolTip("Previous frame (same as Left arrow)")
            self.prev_btn.clicked.connect(lambda: self.step(-1))
            bottom.addWidget(self.prev_btn)
            self.next_btn = QtWidgets.QPushButton("▶")
            self.next_btn.setFixedWidth(40)
            self.next_btn.setToolTip("Next frame (same as Right arrow)")
            self.next_btn.clicked.connect(lambda: self.step(1))
            bottom.addWidget(self.next_btn)
            self.del_fwd_btn = QtWidgets.QPushButton("Del fwd")
            self.del_fwd_btn.setFixedWidth(70)
            self.del_fwd_btn.setToolTip("Delete the armed/target track forward from "
                                        "this frame (same as 'd')")
            self.del_fwd_btn.clicked.connect(self.do_delete_forward)
            bottom.addWidget(self.del_fwd_btn)
            self.clean_btn = QtWidgets.QPushButton("Clean jumps")
            self.clean_btn.setFixedWidth(100)
            self.clean_btn.setToolTip(
                f"Remove teleport/outlier frames from the target track over the "
                f"last {CLEAN_WINDOW_S:.0f}s (same as 'c'). One undo ('u') restores them.")
            self.clean_btn.clicked.connect(lambda: self.clean_jumps())
            bottom.addWidget(self.clean_btn)
            self.slider = QtWidgets.QSlider(Qt.Horizontal)
            self.slider.setMinimum(1)
            self.slider.setMaximum(max(1, self.n_frames))
            self.slider.valueChanged.connect(self.on_slider)
            bottom.addWidget(self.slider, 1)
            v.addLayout(bottom)

            # RIGHT collapsible dock: detection IDs visible this frame
            self.dock = QtWidgets.QDockWidget("Detections in frame", self)
            self.dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
            # keep the dock user-resizable (movable + floatable + closable); do
            # NOT lock it to a fixed width (the min width below is a floor only).
            self.dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable)
            self.det_list = QtWidgets.QListWidget()
            self.det_list.itemClicked.connect(self.on_det_list_click)
            # NAMING: double-click a detection OR right-click for a context menu
            # to assign it to a (registry) player. Both routes call the same
            # _assign_detection_dialog so the name is persisted in the DB.
            self.det_list.itemDoubleClicked.connect(self.on_det_list_double)
            self.det_list.setContextMenuPolicy(Qt.CustomContextMenu)
            self.det_list.customContextMenuRequested.connect(
                self.on_det_list_menu)
            # Don't clip/elide labels: size the min width to the WIDEST realistic
            # label (id + team + quoted name + conf) so IDs/names show fully, and
            # let a horizontal scrollbar appear if one still overflows.
            self.det_list.setTextElideMode(Qt.ElideNone)
            self.det_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            fm = QtGui.QFontMetrics(self.det_list.font())
            sample = '#142  blue&gold  "Pou Frew"  conf=0.97  [ROI]'
            sb = self.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
            det_min_w = fm.horizontalAdvance(sample) + sb + 28   # +scrollbar+padding
            self.det_list.setMinimumWidth(det_min_w)
            self.dock.setWidget(self.det_list)
            self.dock.setMinimumWidth(det_min_w + 8)             # frame/title chrome
            self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

            # LEFT dock: the full mouse-usable CONTROLS panel (superset of hotkeys)
            self._build_controls_panel()

            # menu bar - makes Save / Export / Quit discoverable (keys still work)
            filem = self.menuBar().addMenu("&File")
            a_save = filem.addAction("Save target track  (S)")
            a_save.triggered.connect(self.do_save)
            filem.addSeparator()
            a_open = filem.addAction("Open clip...  (Ctrl+O)")
            a_open.setShortcut(QtGui.QKeySequence("Ctrl+O"))
            a_open.setToolTip("Load another clip of the same game. The shared "
                              "calibration + the learned player profile carry over.")
            a_open.triggered.connect(self.open_clip)
            filem.addSeparator()
            a_quit = filem.addAction("Quit  (Q)")
            a_quit.triggered.connect(self.close)

            # View menu - two checkable DISPLAY FILTERS (keys h / g also toggle them)
            viewm = self.menuBar().addMenu("&View")
            self.act_hide_teams = viewm.addAction("Hide other teams (H)")
            self.act_hide_teams.setCheckable(True)
            self.act_hide_teams.setChecked(self.hide_other_teams)
            self.act_hide_teams.triggered.connect(self.toggle_hide_other_teams)
            self.act_hide_off = viewm.addAction("Hide off-field / bystanders (G)")
            self.act_hide_off.setCheckable(True)
            self.act_hide_off.setChecked(self.hide_off_field)
            self.act_hide_off.triggered.connect(self.toggle_hide_off_field)
            self.act_minimap = viewm.addAction("Pitch minimap (Shift+M)")
            self.act_minimap.setCheckable(True)
            self.act_minimap.setChecked(self.minimap_on)
            self.act_minimap.triggered.connect(self.toggle_minimap)

            # Track menu - MANUAL follow + RECOVERY toggle (keys m / v also toggle)
            trackm = self.menuBar().addMenu("&Track")
            self.act_manual = trackm.addAction("Manual follow (M)")
            self.act_manual.setCheckable(True)
            self.act_manual.setChecked(self.manual_mode)
            self.act_manual.triggered.connect(self.toggle_manual_mode)
            self.act_recovery = trackm.addAction("Recovery on loss (V) - else HOLD")
            self.act_recovery.setCheckable(True)
            self.act_recovery.setChecked(self.recovery_on)
            self.act_recovery.triggered.connect(self.toggle_recovery)

            # play timer
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self._set_timer_interval()

        # ---- LEFT "Controls" dock: every hotkey action as a labelled button ----
        def _scan_players(self):
            """Names of players that already have a profile on disk
            (output/player_<name>.json), sorted. The single-player default file
            (player_profile.json) is NOT a named player, so it is excluded here."""
            names = []
            try:
                out = ProjectPaths().output
                for p in sorted(out.glob("player_*.json")):
                    stem = p.stem                         # "player_<name>"
                    if stem == "player_profile":
                        continue                          # the no-name default file
                    name = stem[len("player_"):]
                    if name:
                        names.append(name)
            except Exception:
                pass
            return names

        # sentinel labels used in the player combo
        SINGLE_PLAYER_LABEL = "(single player)"
        NEW_PLAYER_LABEL = "New player..."

        def _build_player_combo(self):
            """Populate the player selector: every on-disk player_<name>.json, a
            "(single player)" entry for the default no-name profile, the current
            --player (in case it has no file yet), then a trailing "New player...".
            The CURRENT player is selected. Used by _build_controls_panel."""
            from PySide6 import QtWidgets
            combo = QtWidgets.QComboBox()
            names = self._scan_players()
            if self.player and self.player not in names:
                names.append(self.player)
            names = sorted(set(names))
            entries = [self.SINGLE_PLAYER_LABEL] + names + [self.NEW_PLAYER_LABEL]
            combo.addItems(entries)
            # select the current player (or "(single player)" when none)
            current = self.player if self.player else self.SINGLE_PLAYER_LABEL
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            self._player_combo_current = combo.currentText()
            combo.currentTextChanged.connect(self._on_player_changed)
            return combo

        def _on_player_changed(self, text):
            """Switch the tracked player. Selecting the already-current player does
            nothing. "New player..." prompts for a name. Any real switch offers to
            save the current track, then RELAUNCHES on the SAME video for the chosen
            player (mirrors open_clip) and closes this window."""
            from PySide6 import QtCore, QtWidgets
            prev = getattr(self, "_player_combo_current", None)
            if text == prev:
                return
            name = None                       # None => single-player (no --player)
            if text == self.SINGLE_PLAYER_LABEL:
                name = None
            elif text == self.NEW_PLAYER_LABEL:
                entered, ok = QtWidgets.QInputDialog.getText(
                    self, "New player", "Player name:")
                entered = (entered or "").strip()
                if not ok or not entered:
                    self._restore_player_combo()
                    return
                name = entered
            else:
                name = text
            # selecting the already-current player (via its real name) -> no-op
            if (name or None) == (self.player or None):
                self._restore_player_combo()
                return
            # offer to save the current track before relaunching
            if self.store.records:
                ans = QtWidgets.QMessageBox.question(
                    self, "Switch player",
                    "Save the current track before switching player?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    | QtWidgets.QMessageBox.Cancel)
                if ans == QtWidgets.QMessageBox.Cancel:
                    self._restore_player_combo()
                    return
                if ans == QtWidgets.QMessageBox.Yes:
                    self.do_save()
            # RELAUNCH on the SAME video for the chosen player (mirrors open_clip)
            app = pathlib.Path(__file__).resolve()
            relaunch_args = [str(app), "--video", str(self.video_path)]
            if name:
                relaunch_args += ["--player", name]
            # preserve the current target id + the detection-pool mode across the
            # relaunch (otherwise the new process resets to id 1 / ALL detections).
            relaunch_args += ["--id", str(self.store.target)]
            if self.kept_only:
                relaunch_args.append("--kept-only")
            # carry MY jersey number across the relaunch (also persisted in the
            # profile, but pass it so it is live immediately on the new process).
            if self.my_number is not None:
                relaunch_args += ["--my-number", str(self.my_number)]
            QtCore.QProcess.startDetached(sys.executable, relaunch_args)
            self.close()

        def _restore_player_combo(self):
            """Put the player combo back on the current player without re-firing
            the change handler (used when a switch is cancelled)."""
            combo = getattr(self, "player_combo", None)
            if combo is None:
                return
            current = self.player if self.player else self.SINGLE_PLAYER_LABEL
            idx = combo.findText(current)
            combo.blockSignals(True)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
            self._player_combo_current = combo.currentText()

        def _mkbtn(self, label, slot, tip, checkable=False):
            """Create a panel QPushButton wired to `slot`, tooltipped with `tip`.

            Buttons EXPAND to the panel width (Preferred height) so their full
            label is always readable -- the left Controls dock used to be too
            narrow and truncated labels ("Conf-adap", "Recover", ...). With a
            minimum-width dock (set in _build_controls_panel) + expanding buttons
            the text now fits."""
            from PySide6 import QtWidgets
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            if checkable:
                b.setCheckable(True)
            # expand horizontally to fill the column so the label is never clipped
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                            QtWidgets.QSizePolicy.Fixed)
            b.clicked.connect(lambda _checked=False, s=slot: s())
            return b

        def _build_controls_panel(self):
            from PySide6 import QtCore, QtGui, QtWidgets
            Qt = QtCore.Qt

            host = QtWidgets.QWidget()
            outer = QtWidgets.QVBoxLayout(host)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(8)

            def group(title):
                box = QtWidgets.QGroupBox(title)
                gl = QtWidgets.QVBoxLayout(box)
                gl.setContentsMargins(8, 6, 8, 8)
                gl.setSpacing(4)
                outer.addWidget(box)
                return gl

            # --- MY PLAYER (top): the prominent place to set WHO you are -----
            # the tracked player's name + YOUR jersey number. The number control
            # is the SAME 'My #' QSpinBox the LEARNING panel used (moved here, not
            # duplicated) so self.my_number stays the single source of truth and
            # set_my_number / _sync_my_number_spin keep working unchanged.
            gl = group("MY PLAYER")
            # "Tracking: <name>  #<n>" headline, refreshed via _sync_my_number_spin.
            self.myplayer_lbl = QtWidgets.QLabel()
            f = self.myplayer_lbl.font(); f.setBold(True)
            self.myplayer_lbl.setFont(f)
            gl.addWidget(self.myplayer_lbl)
            # NAME field: shows the current player. Renaming the LIVE player is
            # non-trivial (it comes from --player and namespaces the profile +
            # output files), so this is READ-ONLY and points the user at the
            # player selector below; the NUMBER is fully editable here.
            name_row = QtWidgets.QHBoxLayout(); name_row.setSpacing(4)
            name_row.addWidget(QtWidgets.QLabel("Name:"))
            self.myname_edit = QtWidgets.QLineEdit()
            self.myname_edit.setReadOnly(True)
            self.myname_edit.setText(self.player or "(single player)")
            self.myname_edit.setToolTip(
                "The player currently being tracked. To track a DIFFERENT player "
                "(or create one), use the PLAYER selector below -- switching there "
                "relaunches on this video for them. The jersey number is editable here.")
            name_row.addWidget(self.myname_edit, 1)
            gl.addLayout(name_row)
            # JERSEY NUMBER: the user's own shirt number (0 = unset). When set AND
            # tesseract is installed, candidates whose detected number is
            # confidently DIFFERENT are excluded. Persists in the player profile.
            mynum_row = QtWidgets.QHBoxLayout(); mynum_row.setSpacing(4)
            mynum_row.addWidget(QtWidgets.QLabel("My #:"))
            self.mynum_spin = QtWidgets.QSpinBox()
            self.mynum_spin.setRange(0, MAX_JERSEY)      # 0 = unset
            self.mynum_spin.setSpecialValueText("--")    # show "--" for 0/unset
            self.mynum_spin.setValue(int(self.my_number) if self.my_number else 0)
            self.mynum_spin.setToolTip(
                "YOUR jersey number (1..23; 0 = unset). When set and tesseract is "
                "installed, the tracker excludes candidates whose detected shirt "
                "number is confidently NOT yours, and prefers one that IS. "
                "Persists in the player profile.")
            self.mynum_spin.valueChanged.connect(
                lambda v: self.set_my_number(v if v else None))
            mynum_row.addWidget(self.mynum_spin)
            mynum_row.addStretch(1)
            gl.addLayout(mynum_row)

            # --- PLAYER (selector) ---
            gl = group("PLAYER")
            self.player_combo = self._build_player_combo()
            self.player_combo.setToolTip(
                "Switch which player is tracked. Picking another player (or "
                "'New player...') relaunches on this video for them.")
            gl.addWidget(self.player_combo)

            # --- PLAYBACK ---
            gl = group("PLAYBACK")
            self.btn_play = self._mkbtn(
                "Play / Pause", self.toggle_play, "Play / pause (Space)")
            gl.addWidget(self.btn_play)
            row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
            row.addWidget(self._mkbtn("-1s", lambda: self.step(-int(round(self.fps))),
                                      "Step back 1 second (,)"))
            row.addWidget(self._mkbtn("-1f", lambda: self.step(-1),
                                      "Step back 1 frame (Left)"))
            row.addWidget(self._mkbtn("+1f", lambda: self.step(1),
                                      "Step forward 1 frame (Right)"))
            row.addWidget(self._mkbtn("+1s", lambda: self.step(int(round(self.fps))),
                                      "Step forward 1 second (.)"))
            gl.addLayout(row)
            row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
            row.addWidget(self._mkbtn("-10s", lambda: self.step(-int(round(self.fps * 10))),
                                      "Step back 10 seconds ([)"))
            row.addWidget(self._mkbtn("+10s", lambda: self.step(int(round(self.fps * 10))),
                                      "Step forward 10 seconds (])"))
            gl.addLayout(row)
            row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
            row.addWidget(self._mkbtn("Speed -", lambda: self.change_speed(-1),
                                      "Slower playback (<)"))
            self.speed_lbl = QtWidgets.QLabel(f"{self.speed:g}x")
            self.speed_lbl.setAlignment(Qt.AlignCenter)
            row.addWidget(self.speed_lbl)
            row.addWidget(self._mkbtn("Speed +", lambda: self.change_speed(1),
                                      "Faster playback (>)"))
            gl.addLayout(row)
            self.btn_conf_speed = self._mkbtn(
                "Conf-adaptive speed", self.toggle_conf_speed,
                "Confidence-adaptive speed (E): during play, slow down on "
                "uncertain frames and run ahead on confident ones (ignores the "
                "manual speed while on).", checkable=True)
            gl.addWidget(self.btn_conf_speed)
            gl.addWidget(self._mkbtn("Fit zoom", self.reset_zoom,
                                     "Reset view to fit-to-window (0)"))

            # --- TRACKING ---
            gl = group("TRACKING")
            self.btn_track = self._mkbtn(
                "Track", self.toggle_roi_tracking,
                "ROI tracking on/off (R). Must be ON for Play to record.",
                checkable=True)
            gl.addWidget(self.btn_track)
            self.btn_manual = self._mkbtn(
                "Manual", self.toggle_manual_mode,
                "Manual cursor-follow (M): ROI follows the mouse; each step records.",
                checkable=True)
            gl.addWidget(self.btn_manual)
            self.btn_recovery = self._mkbtn(
                "Recovery on loss", self.toggle_recovery,
                "Recovery on loss (V). OFF = HOLD the ROI in place on loss.",
                checkable=True)
            gl.addWidget(self.btn_recovery)
            row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
            row.addWidget(self._mkbtn("ROI -", lambda: self.resize_roi(False),
                                      "Shrink the ROI ~10% (-)"))
            row.addWidget(self._mkbtn("ROI +", lambda: self.resize_roi(True),
                                      "Grow the ROI ~10% (+)"))
            gl.addLayout(row)
            gl.addWidget(self._mkbtn("Reset ROI -> target", self.reset_roi_to_target,
                                     "Reset the ROI to the current target box (Shift+R)"))
            self.btn_arrow = self._mkbtn(
                "Arrow marker", self.toggle_spotlight,
                "Toggle the arrow / spotlight marker above the target's head (O)",
                checkable=True)
            gl.addWidget(self.btn_arrow)
            self.btn_adaptive = self._mkbtn(
                "Adaptive marker", self.toggle_adaptive_marker,
                "Confidence-adaptive marker (B): tight body ellipse when "
                "confident, growing to a dashed search circle when not.",
                checkable=True)
            gl.addWidget(self.btn_adaptive)

            # --- SAM 3 (background re-track) ---
            gl = group("SAM 3")
            self.btn_sam3_mark = self._mkbtn(
                "Mark SAM3 start", self.mark_sam3_start,
                "Mark a loose IN-point for a SAM3 re-track (W key). Then scrub/play "
                "to the END and hit 'SAM3 re-track -> here'. If no start is marked "
                "the last ~3 seconds up to the current frame are used.")
            gl.addWidget(self.btn_sam3_mark)
            self.btn_sam3_run = self._mkbtn(
                "SAM3 re-track -> here", self.run_sam3_retrack,
                "Re-track the marked window up to the CURRENT frame with SAM 3 in "
                "the BACKGROUND (separate process; GUI stays responsive). SAM3 is "
                "high-quality and HOLDS the lock through rucks, but is SLOW "
                "(~2s/frame: a 12s window is ~6 min). The clean result is merged "
                "into your track as ONE undoable step (u to revert). (Shift+W)")
            gl.addWidget(self.btn_sam3_run)
            self.btn_sam3_cancel = self._mkbtn(
                "Cancel SAM3", self.cancel_sam3,
                "Cancel the running SAM3 background re-track (no merge).")
            self.btn_sam3_cancel.setEnabled(False)
            gl.addWidget(self.btn_sam3_cancel)
            self.sam3_lbl = QtWidgets.QLabel("SAM3: idle")
            self.sam3_lbl.setWordWrap(True)
            gl.addWidget(self.sam3_lbl)

            # --- LEARNING ---
            gl = group("LEARNING")
            gl.addWidget(self._mkbtn("Save profile", self.save_player_profile,
                                     "Save the persistent player appearance profile (P)"))
            gl.addWidget(self._mkbtn("Reset model", self.reset_target_model,
                                     "Reset the in-session learned model (Shift+L)"))
            gl.addWidget(self._mkbtn("Clear profile", self.clear_player_profile,
                                     "Clear the persistent player profile on disk (Shift+K)"))
            self.learned_lbl = QtWidgets.QLabel("learned: 0f")
            gl.addWidget(self.learned_lbl)
            # (The 'My #' jersey-number control now lives in the MY PLAYER group at
            # the top of this panel -- self.mynum_spin / set_my_number unchanged.)
            # CONFIDENCE: live label + auto-pause-on-low toggle
            self.conf_lbl = QtWidgets.QLabel("conf: --")
            gl.addWidget(self.conf_lbl)
            # ONLINE IDENTITY ("me vs not-me"): pos/neg counts; prob when ready.
            self.id_lbl = QtWidgets.QLabel("identity: 0+/0-")
            self.id_lbl.setToolTip(
                "Online me/not-me classifier: learns from confirmations "
                "(positives) and corrections (Rule out = hard negatives). Shows "
                "id NN% for the current target once it is ready.")
            gl.addWidget(self.id_lbl)
            self.btn_autopause = self._mkbtn(
                "Auto-pause on low conf", self.toggle_autopause_low,
                "Pause playback when a frame's confidence < "
                f"{AUTO_PAUSE_THRESH:.2f} (A)", checkable=True)
            gl.addWidget(self.btn_autopause)
            # JERSEY-NUMBER OCR soft signal (BONUS cue; default OFF, hotkey J).
            # No-ops with a status note when the tesseract binary is not installed.
            self.btn_ocr = self._mkbtn(
                "Jersey OCR", self.toggle_ocr,
                "Jersey-number OCR soft confidence cue (J): when the player's "
                "back faces the camera, OCR the number and nudge confidence. "
                "DEFAULT OFF; needs the tesseract binary "
                "(winget install UB-Mannheim.TesseractOCR).", checkable=True)
            gl.addWidget(self.btn_ocr)
            self.ocr_lbl = QtWidgets.QLabel("target #: --")
            gl.addWidget(self.ocr_lbl)
            # SAM 2 box refinement (tighten the target box to its mask; default OFF).
            self.btn_sam2 = self._mkbtn(
                "SAM2 silhouette", self.toggle_sam2,
                "SAM 2 (F): draw a body-TIGHT silhouette hugging the player and "
                "dim the grass inside the box (replaces the ellipse), and tighten "
                "the tracked box to the mask. DEFAULT OFF; ~130ms/frame on GPU. "
                "Uses the installed ultralytics SAM2 (downloads sam2.1_t.pt once).",
                checkable=True)
            gl.addWidget(self.btn_sam2)

            # --- RE-ANCHOR ---
            gl = group("RE-ANCHOR")
            self.btn_ruleout = self._mkbtn(
                "Rule out (not me)", self.toggle_ruleout_mode,
                "RULE-OUT mode (N): turn on, then CLICK any player to mark them NOT "
                "you - their box turns red, they're excluded from selection, and it "
                "teaches the identity classifier. Click more to rule out several; "
                "press again to stop.", checkable=True)
            gl.addWidget(self.btn_ruleout)
            gl.addWidget(self._mkbtn("Delete forward", self.do_delete_forward,
                                     "Delete the armed/target track forward from here (d)"))
            gl.addWidget(self._mkbtn("Nuke track", self.do_nuke,
                                     "Delete ALL rows for the armed id (Shift+D)"))
            gl.addWidget(self._mkbtn("Clean jumps", self.clean_jumps,
                                     "Remove teleport/outlier frames (c)"))
            gl.addWidget(self._mkbtn("Undo", self.do_undo,
                                     "Undo the last change (u)"))
            gl.addWidget(self._mkbtn("Bulk IDs...", self.do_bulk_dialog,
                                     "Open the bulk ID manager dialog (i)"))

            # --- VIEW ---
            gl = group("VIEW")
            self.btn_hide_teams = self._mkbtn(
                "Hide other teams", self.toggle_hide_other_teams,
                "Hide confidently non-tracked-team boxes (h)", checkable=True)
            gl.addWidget(self.btn_hide_teams)
            self.btn_hide_off = self._mkbtn(
                "Hide off-field", self.toggle_hide_off_field,
                "Hide detections whose feet are off the pitch (g)", checkable=True)
            gl.addWidget(self.btn_hide_off)
            self.btn_focus = self._mkbtn(
                "Focus box (ID-trace area)", self.toggle_focus,
                "Only ID-trace detections inside a 3x-ROI focus box around your "
                "marker (x): declutters + stops the target jumping onto far "
                "players. Default ON.", checkable=True)
            gl.addWidget(self.btn_focus)

            # --- CANDIDATES (objects.csv overlay: "who could be ME?") ---
            # Only meaningful when an overnight mark_all objects.csv was loaded;
            # the summary label says "(no objects.csv)" otherwise. The overlay
            # auto-classifies every tracked player as me-likely / possible /
            # ruled-out so you pick from a short list instead of the whole field.
            gl = group("CANDIDATES")
            self.cand_lbl = QtWidgets.QLabel("candidates: (no objects.csv)")
            self.cand_lbl.setWordWrap(True)
            self.cand_lbl.setToolTip(
                "Auto-shortlist from the overnight 'mark everyone' objects.csv: "
                "how many tracked players are me-likely (bright green) / possible "
                "(teal) / ruled-out (hidden). Click a me-likely/possible box or "
                "list row to make it your target.")
            gl.addWidget(self.cand_lbl)
            self.btn_show_ruled = self._mkbtn(
                "Show ruled-out", self.toggle_show_ruled_out,
                "Also draw tracks the shortlist RULED OUT (opposition, wrong "
                "number, role/zone contradiction). Default OFF so only me-likely "
                "and possible players are shown.", checkable=True)
            gl.addWidget(self.btn_show_ruled)
            gl.addWidget(self._mkbtn(
                "Re-shortlist", self.reshortlist,
                "Re-run the 'who could be ME?' classification against your "
                "current profile (jersey number, team, learned height)."))
            self._sync_candidates_label()

            # --- PITCH (M1): per-frame homography + top-down minimap ----------
            # A pitch calibration (output/<stem>.pitch.json, else game.pitch.json),
            # clicked on ONE anchor frame, is slid along the pan/zoom by camera
            # -motion compensation so the minimap is valid on the CURRENT frame.
            # The mapping is only trustworthy when playing FORWARD from the anchor
            # frame; jumps/seeks mark it STALE (the live label says so).
            gl = group("PITCH")
            self.pitch_lbl = QtWidgets.QLabel("pitch: (no calibration)")
            self.pitch_lbl.setWordWrap(True)
            self.pitch_lbl.setToolTip(
                "Live pitch-homography health. 'live (N inl)' = the minimap is "
                "trustworthy this frame (N = camera-motion inliers). 'STALE' = you "
                "jumped/seeked; play forward from the calibration's anchor frame to "
                "re-establish it, or re-calibrate.")
            gl.addWidget(self.pitch_lbl)
            self.btn_minimap = self._mkbtn(
                "Minimap", self.toggle_minimap,
                "Show the top-down pitch minimap (default ON when a pitch "
                "calibration is loaded). Plots every visible player + the gold "
                "target by their feet, using the live CMC homography.",
                checkable=True)
            gl.addWidget(self.btn_minimap)
            gl.addWidget(self._mkbtn(
                "Calibrate pitch...", self.calibrate_pitch,
                "Launch the pitch-homography calibration tool on this video "
                "(click known pitch landmarks on one frame). Saves "
                "output/<clip>.pitch.json; use 'Reload pitch' after."))
            gl.addWidget(self._mkbtn(
                "Reload pitch", self.reload_pitch,
                "Re-read the pitch calibration file and rebuild the live "
                "homography (use after calibrating, without restarting)."))

            # --- TACTICAL (M2): player circles + offside line -----------------
            # A "character selector": toggle CIRCLES on, then click overlay
            # players to ring them (true-metre ground ellipses via the live
            # homography). The OFFSIDE line is a constant pitch-X line set from the
            # single selected player (else the target). Both need a live pitch
            # calibration to draw in metres; circles fall back to a dim pixel
            # ellipse at the feet when the homography is unavailable / stale.
            gl = group("TACTICAL")
            self.btn_circles = self._mkbtn(
                "Player circles", self.toggle_circles,
                "Player circles (Z): ring chosen players with a true 1 m ground "
                "circle at their feet. While ON, click an overlay player to "
                "select/deselect it (instead of placing an ROI); the target is "
                "always circled.", checkable=True)
            gl.addWidget(self.btn_circles)
            gl.addWidget(self._mkbtn(
                "Clear circles", self.clear_circles,
                "Deselect all circled players (the target circle stays)."))
            gl.addWidget(self._mkbtn(
                "Set offside @ player", self.set_offside_at_player,
                "Draw the offside line at the single selected player's pitch X "
                "(if exactly one is selected, else the target's X). Needs a live "
                "pitch calibration."))
            gl.addWidget(self._mkbtn(
                "Clear offside", self.clear_offside,
                "Remove the offside line."))

            # --- FILE ---
            gl = group("FILE")
            gl.addWidget(self._mkbtn("Save track", self.do_save,
                                     "Save the target track CSV (s)"))
            gl.addWidget(self._mkbtn("Open clip...", self.open_clip,
                                     "Load another clip of the same game (Ctrl+O)"))

            outer.addStretch(1)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            # never show a HORIZONTAL scrollbar: the inner column must be wide
            # enough to show full button labels; vertical scrolling stays on.
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setWidget(host)
            # Give the inner column a sensible minimum width so labels fit (the
            # dock used to be too narrow and truncated "Conf-adaptive speed",
            # "Recovery on loss", etc.). The dock itself stays user-resizable.
            host.setMinimumWidth(220)

            self.controls_dock = QtWidgets.QDockWidget("Controls", self)
            self.controls_dock.setAllowedAreas(
                Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            self.controls_dock.setWidget(scroll)
            # a minimum dock width that comfortably fits the widest labels; the
            # dock remains resizable (the user can widen/narrow it from here).
            self.controls_dock.setMinimumWidth(230)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_dock)

            self._sync_buttons()

        def _sync_buttons(self):
            """Reflect window state onto the checkable panel buttons + dynamic
            labels, using blockSignals to avoid feedback loops. Called after the
            relevant toggles and once per frame in _refresh()."""
            pairs = [
                (getattr(self, "btn_track", None), self.roi_on),
                (getattr(self, "btn_manual", None), self.manual_mode),
                (getattr(self, "btn_recovery", None), self.recovery_on),
                (getattr(self, "btn_arrow", None), self.spotlight_on),
                (getattr(self, "btn_adaptive", None), self.adaptive_marker_on),
                (getattr(self, "btn_conf_speed", None), self.conf_speed),
                (getattr(self, "btn_hide_teams", None), self.hide_other_teams),
                (getattr(self, "btn_hide_off", None), self.hide_off_field),
                (getattr(self, "btn_focus", None), self.focus_on),
                (getattr(self, "btn_ruleout", None), self.ruleout_mode),
                (getattr(self, "btn_autopause", None), self.autopause_low),
                (getattr(self, "btn_ocr", None), self.ocr_on),
                (getattr(self, "btn_sam2", None), self.sam2_on),
                (getattr(self, "btn_show_ruled", None), self.show_ruled_out),
                (getattr(self, "btn_minimap", None), self.minimap_on),
                (getattr(self, "btn_circles", None), self.circles_on),
            ]
            for btn, on in pairs:
                if btn is None:
                    continue
                btn.blockSignals(True)
                btn.setChecked(bool(on))
                btn.blockSignals(False)
            lbl = getattr(self, "speed_lbl", None)
            if lbl is not None:
                lbl.setText(f"{self.speed:g}x")
            lbl = getattr(self, "learned_lbl", None)
            if lbl is not None:
                app_n, _ = self.tmodel.learned()
                lbl.setText(f"learned: {app_n}f")
            lbl = getattr(self, "conf_lbl", None)
            if lbl is not None:
                cv = self.conf_by_frame.get(self.frame, self.last_confidence)
                lbl.setText("conf: --" if cv is None else f"conf: {cv:.2f}")
            lbl = getattr(self, "id_lbl", None)
            if lbl is not None:
                ip, ineg = self.idmodel.counts()
                txt = f"identity: {ip}+/{ineg}-"
                if self.idmodel.ready():
                    tbox = self.store.target_box_at(self.frame)
                    if tbox is not None:
                        idp = self.idmodel.prob(
                            self._identity_features(self.frame, tbox))
                        txt += f"  id {idp * 100:.0f}%"
                    else:
                        txt += "  ready"
                if self.ruled_out:
                    txt += f"  ruled out {len(self.ruled_out)}"
                lbl.setText(txt)
            # PITCH live-homography health label (M1).
            lbl = getattr(self, "pitch_lbl", None)
            if lbl is not None:
                txt = self._live_health_text()
                lbl.setText(txt if txt is not None else "pitch: (no calibration)")
            # keep the 'My #' spin box reflecting self.my_number (e.g. set via the
            # 'This is me' assign path or carried over from the profile / CLI).
            self._sync_my_number_spin()
            lbl = getattr(self, "ocr_lbl", None)
            if lbl is not None:
                if not self.ocr_on:
                    lbl.setText("target #: off")
                elif self.target_number is None:
                    lbl.setText("target #: --")
                else:
                    last = "" if self._ocr_last is None else f"  (last {self._ocr_last[0]})"
                    lbl.setText(f"target #{self.target_number}{last}")

        def _effective_speed(self):
            """The speed multiplier driving the play timer this tick.

            CONFIDENCE-ADAPTIVE SPEED: when self.conf_speed is ON, the multiplier
            comes from the latest confidence (conf_speed_mult) so playback slows
            on uncertain frames and runs ahead on confident ones; the manual
            `speed` is IGNORED while it's on so they don't fight. When OFF, the
            manual `speed` applies unchanged (existing behaviour)."""
            if self.conf_speed:
                cv = self.conf_by_frame.get(self.frame, self.last_confidence)
                return conf_speed_mult(cv)
            return self.speed or 1.0

        def _set_timer_interval(self):
            base = 1000.0 / (self.fps or FPS_ASSUMED)
            self.timer.setInterval(max(1, int(base / (self._effective_speed() or 1.0))))

        def change_speed(self, delta):
            """Step the playback speed up (+1) or down (-1) the SPEEDS list."""
            try:
                idx = SPEEDS.index(self.speed)
            except ValueError:
                idx = SPEEDS.index(DEFAULT_SPEED)
            idx = max(0, min(len(SPEEDS) - 1, idx + delta))
            self.speed = SPEEDS[idx]
            self._set_timer_interval()
            self.status_msg = f"speed {self.speed:g}x"
            self._refresh()

        def reset_zoom(self):
            """Reset the view to fit-to-window (zoom = fit, pan = 0). Shared by
            the '0' hotkey and the PLAYBACK "Fit zoom" button."""
            self.view.vt.reset()
            self.status_msg = "view reset (fit-to-window)"
            self._refresh()

        # ---- detection pool helpers ----
        def dets_at(self, frame):
            """List of (x1,y1,x2,y2,conf) kept detections at frame."""
            return self.det_by_frame.get(frame, [])

        def dets_in_roi(self, frame):
            """(indices, boxes) of kept dets whose centre is inside the ROI."""
            idxs, boxes = [], []
            if self.roi is None:
                return idxs, boxes
            for i, d in enumerate(self.dets_at(frame)):
                box = d[:4]
                if bbox.contains(self.roi, *bbox.center(box)):
                    idxs.append(i); boxes.append(box)
            return idxs, boxes

        def _eligible_indices(self, frame):
            """Set of detection indices (into dets_at(frame)) ELIGIBLE to be the
            target: all EXCEPT detections CONFIDENTLY classified as a NON-tracked
            (opposition) team. Tracked-team + 'unsure' detections stay eligible;
            with no calibration everything is eligible. Wraps the pure module-
            level eligible_target_indices() over this frame's team labels.

            ALSO excludes any detection that maps (by overlap) to a track id the
            user has explicitly RULED OUT this session (self.ruled_out) - a hard
            "not me" correction, mirroring the team constraint."""
            eligible = eligible_target_indices(self.team_labels_for_frame(frame))
            dets = self.dets_at(frame)
            if self.ruled_out:
                eligible = {i for i in eligible
                            if self._det_track_hint(dets[i][:4])
                            not in self.ruled_out}
            # MY-NUMBER live exclusion (Feature 1): drop any candidate whose track
            # is CONFIDENTLY a DIFFERENT shirt number than mine (mirrors ruled_out
            # / the team constraint). INERT unless ocr.available() AND my_number is
            # set. NEVER drop the current active target's own box or the armed
            # detection (mirror the height-gate protect logic) so the target can't
            # be gated out of itself before the votes are reliable.
            excl = self._jersey_excluded_tids()
            if excl:
                protect = self._height_protected_indices(set(eligible))
                eligible = {i for i in eligible
                            if i in protect
                            or self._det_track_hint(dets[i][:4]) not in excl}
            # FOCUS box: only detections whose centre is inside the 3x-ROI focus
            # region are ID-traced/selectable (declutter + anti-jump). Never drops
            # the active target / armed detection (height-protected set).
            fb = self.focus_box()
            if fb is not None:
                protect = self._height_protected_indices(set(eligible))
                eligible = {i for i in eligible
                            if i in protect
                            or bbox.contains(fb, *bbox.center(dets[i][:4]))}
            return eligible

        def focus_box(self):
            """The FOCUS region (xyxy): FOCUS_MULT x the ROI, centred on it,
            clamped to the frame. ONLY detections inside it are ID-traced. None
            when focus is off or there is no ROI."""
            if not self.focus_on or self.roi is None:
                return None
            rx1, ry1, rx2, ry2 = self.roi
            cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
            hw = (rx2 - rx1) * FOCUS_MULT / 2.0
            hh = (ry2 - ry1) * FOCUS_MULT / 2.0
            return clamp_box((cx - hw, cy - hh, cx + hw, cy + hh), self.fw, self.fh)

        # ---- HEIGHT-CONSISTENCY (rule out wrong-height candidates) ----
        def _tracked_height_model(self):
            """The HeightModel of the TRACKED team (first team with .track), or
            None when there is no calibration / no tracked-team height model.
            When None, the whole height feature is INERT (gating off, the
            confidence height-term goes neutral)."""
            if self.calib is None:
                return None
            for t in getattr(self.calib, "teams", []):
                if getattr(t, "track", False):
                    return getattr(t, "height", None)
            return None

        def _box_height_residual(self, box, hmodel=None):
            """Residual = box_height - tracked_team.height.predict(centre_y) for
            an xyxy `box`, or None when there is no tracked-team height model.
            centre_y is the box vertical centre."""
            if hmodel is None:
                hmodel = self._tracked_height_model()
            if hmodel is None:
                return None
            x1, y1, x2, y2 = box
            cy = (y1 + y2) / 2.0
            try:
                return float((y2 - y1) - hmodel.predict(cy))
            except Exception:
                return None

        def _height_protected_indices(self, idxs):
            """Indices in `idxs` that height gating must NEVER exclude: the ARMED
            detection AND the current active target's own box (both matched by
            box). Protecting the target's own box stops the height gate from
            excluding it early in learning - before enough residual samples have
            been seen - which would let the gate drop the target from its own
            selection."""
            protect = set()
            dets = self.dets_at(self.frame)
            # the ARMED detection (matched by its box)
            if self.armed_box is not None:
                ab = tuple(self.armed_box)
                for i in idxs:
                    if 0 <= i < len(dets) and tuple(dets[i][:4]) == ab:
                        protect.add(i)
            # the current active TARGET's own box (the row already recorded for
            # this frame), so the target is never height-gated out of itself.
            tbox = self.store.target_box_at(self.frame)
            if tbox is not None:
                tb = tuple(float(v) for v in tbox)
                for i in idxs:
                    if 0 <= i < len(dets) and tuple(float(v) for v in dets[i][:4]) == tb:
                        protect.add(i)
            return protect

        def _height_filter_indices(self, frame, idxs, boxes, protect=None):
            """Apply HEIGHT-CONSISTENCY gating to a candidate (idxs, boxes) pool.

            Only gates once the model has >= HEIGHT_MIN_SAMPLES residual samples
            AND a tracked-team height model exists; otherwise returns idxs
            unchanged (feature inert). NEVER excludes the protected indices (the
            current active target + the armed detection). Returns the filtered
            list of indices (a subset of idxs)."""
            hmodel = self._tracked_height_model()
            if hmodel is None or not self.tmodel.height_ready():
                return list(idxs)
            residuals = {}
            for i, b in zip(idxs, boxes):
                residuals[i] = self._box_height_residual(b, hmodel)
            keep = height_gate_indices(idxs, residuals, self.tmodel.resid_mean,
                                       self.tmodel.height_tol(), protect=protect)
            return [i for i in idxs if i in keep]

        def _target_candidate_pool(self, frame):
            """The team-constrained, height-filtered, tracked-preferred candidate
            pool for `frame` as a list of (idx, box). This is the SINGLE source of
            truth for which in-ROI candidates the target may snap to, shared by
            active_target_idx() (which scores within it) and the _track_step snap
            fallback (which picks nearest-centre within it) so the two never
            disagree on the eligible pool. Returns [] when there is no ROI / no
            in-ROI candidate / nothing survives the team constraint."""
            if self.roi is None:
                return []
            idxs, boxes = self.dets_in_roi(frame)
            if not boxes:
                return []
            # TEAM CONSTRAINT (+ explicit RULE-OUT): drop confident-opposition
            # candidates AND anything the user has ruled out as "not me".
            labels = self.team_labels_for_frame(frame)
            eligible = self._eligible_indices(frame)
            keep = [(i, box) for i, box in zip(idxs, boxes) if i in eligible]
            if not keep:
                return []
            # HEIGHT-CONSISTENCY: once the residual model is ready, drop any
            # candidate whose height residual is clearly inconsistent with the
            # learned target (but never the armed detection / the target's own
            # box). Inert otherwise.
            keep_idxs = [i for i, _ in keep]
            keep_boxes = [b for _, b in keep]
            allowed = set(self._height_filter_indices(
                frame, keep_idxs, keep_boxes,
                protect=self._height_protected_indices(idxs)))
            hkeep = [(i, b) for i, b in keep if i in allowed]
            if hkeep:
                keep = hkeep
            # PREFER confidently-tracked-team candidates; only fall back to the
            # rest ('unsure' / no-team) if there is no tracked-team candidate.
            tracked = [(i, box) for i, box in keep
                       if i < len(labels) and labels[i][0] is not None
                       and labels[i][2] and labels[i][1]]
            return tracked if tracked else keep

        def active_target_idx(self):
            """Index (into dets_at(frame)) of the BEST in-ROI candidate.

            TEAM-CONSTRAINED: confident-opposition detections are excluded from
            the candidate pool so the ROI can NEVER snap onto the other team /
            far ref. Among the eligible candidates, confidently-tracked-team
            boxes are PREFERRED (they win over any 'unsure' box); only if there
            is no tracked-team candidate do we fall back to scoring the 'unsure'
            ones.

            When the ONLINE TARGET MODEL holds data, candidates are scored by a
            weighted blend of nearness-to-ROI-centre, nearness-to-predicted-centre
            and appearance similarity to the learned player, and the max-scoring
            candidate wins. When the model is empty this falls back to the exact
            original nearest-ROI-centre behaviour."""
            pool = self._target_candidate_pool(self.frame)
            if not pool:
                return None
            idxs = [i for i, _ in pool]
            boxes = [b for _, b in pool]

            cx, cy = bbox.center(self.roi)

            # model empty -> original nearest-centre behaviour (unchanged)
            if not self.tmodel.has_data():
                j = bbox.nearest(boxes, cx, cy)
                return idxs[j] if j is not None else None

            # ROI diagonal as the distance normaliser
            rx1, ry1, rx2, ry2 = self.roi
            roi_diag = ((rx2 - rx1) ** 2 + (ry2 - ry1) ** 2) ** 0.5
            if roi_diag <= 1e-6:
                roi_diag = 1.0

            pred = self.tmodel.predict_centre(self.frame)
            img = self._frame_img_for_appearance()

            best_j, best_score = None, None
            for k, box in enumerate(boxes):
                bx, by = bbox.center(box)
                d_c = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
                center_term = max(0.0, min(1.0, 1.0 - d_c / roi_diag))
                if pred is not None:
                    d_p = ((bx - pred[0]) ** 2 + (by - pred[1]) ** 2) ** 0.5
                    pred_term = max(0.0, min(1.0, 1.0 - d_p / roi_diag))
                else:
                    pred_term = 0.0
                if img is not None and self.tmodel.count > 0:
                    app_term = self.tmodel.appearance_sim(TargetModel._crop(img, box))
                else:
                    app_term = 0.0
                score = (TM_W_CENTER * center_term + TM_W_PRED * pred_term
                         + TM_W_APP * app_term)
                # IDENTITY tiebreaker: once the "me vs not-me" model is ready, add
                # a small bonus for candidates the classifier thinks are the
                # target (prob>0.5) and a small penalty for ones it doesn't. This
                # only NUDGES selection among the eligible pool; it never hard-
                # excludes (the rule-out + team + height do that).
                idp = self._id_prob(self.frame, box)
                if idp is not None:
                    score += ID_TIEBREAK * (idp - 0.5) * 2.0
                # MY-NUMBER preference (Feature 1): a candidate whose track has
                # DECIDED on MY shirt number gets a strong selection boost so the
                # target locks to the right player. INERT when the feature is off.
                if self._reads_my_number(box):
                    score += MYNUM_PREF_BOOST
                if best_score is None or score > best_score:
                    best_score, best_j = score, k
            return idxs[best_j] if best_j is not None else None

        def _frame_img_for_appearance(self):
            """BGR image for the CURRENT frame, used to crop in-ROI candidates for
            appearance scoring. Reuses the already-decoded frame when possible."""
            img = getattr(self, "_cur_bgr", None)
            if img is not None and getattr(self, "_cur_bgr_frame", None) == self.frame:
                return img
            return self.reader.frame(self.frame)

        # ---- TEAM LABELS (calibration) ----
        def team_labels_for_frame(self, frame):
            """Classify each detection at `frame` to the nearest CALIBRATED team by
            its perceptual fingerprint (rt2.features). Returns a list aligned with
            dets_at(frame); each entry is (team_name, is_tracked, confident):

              * team_name : nearest team's .name (or None when feature inert)
              * is_tracked: that team's .track flag (None when inert)
              * confident : False when there are <2 fingerprinted teams, OR the L1
                            margin between the best & 2nd-best team is tiny
                            (< TEAM_MARGIN_MIN) -> treat the box as "unsure".

            CACHED per frame (cleared whenever the frame / detection set changes) so
            classification runs once per frame rather than every repaint. If there is
            no calibration / no fingerprints, returns all-None entries (feature off).
            """
            cached = self._team_cache.get(frame)
            if cached is not None:
                return cached
            dets = self.dets_at(frame)
            fp_teams = self._team_fp_teams
            if not dets or not fp_teams:
                out = [(None, None, False) for _ in dets]
                self._team_cache[frame] = out
                return out
            multi = len(fp_teams) >= 2
            img = (self._cur_bgr if getattr(self, "_cur_bgr_frame", None) == frame
                   else self.reader.frame(frame))
            out = []
            for d in dets:
                box = d[:4]
                if img is None:
                    out.append((None, None, False))
                    continue
                try:
                    fv = features.feature_vector(img, box)
                    dists = [(features.distance(fv, t.fingerprint), t) for t in fp_teams]
                except Exception:
                    out.append((None, None, False))
                    continue
                dists.sort(key=lambda x: x[0])
                best_d, best_t = dists[0]
                confident = multi
                if multi:
                    margin = dists[1][0] - best_d
                    if margin < TEAM_MARGIN_MIN:
                        confident = False
                out.append((best_t.name, bool(best_t.track), confident))
            self._team_cache[frame] = out
            return out

        # ---- DISPLAY FILTERS (cosmetic declutter only) ----
        def toggle_show_ruled_out(self):
            """CANDIDATES: show / hide the overlay tracks the shortlist ruled out."""
            self.show_ruled_out = not self.show_ruled_out
            self.status_msg = ("showing ruled-out tracks" if self.show_ruled_out
                               else "hiding ruled-out tracks")
            self._refresh()

        def reshortlist(self):
            """CANDIDATES: re-run the 'who could be ME?' classification."""
            if not self.has_objects_overlay():
                self.status_msg = "no objects.csv overlay loaded"
                self._refresh(); return
            self._recompute_shortlist()
            s = self._obj_summary
            self.status_msg = (f"re-shortlisted: {s['me_likely']} likely / "
                               f"{s['possible']} possible / {s['ruled_out']} out")
            self._refresh()

        def toggle_minimap(self):
            """PITCH: show / hide the top-down minimap overlay."""
            self.minimap_on = not self.minimap_on
            act = getattr(self, "act_minimap", None)
            if act is not None:
                act.blockSignals(True)
                act.setChecked(self.minimap_on)
                act.blockSignals(False)
            if self.minimap_on and self.live_h is None:
                self.status_msg = ("minimap ON, but no pitch calibration loaded - "
                                   "use 'Calibrate pitch...' then 'Reload pitch'")
            else:
                self.status_msg = ("minimap ON" if self.minimap_on
                                   else "minimap OFF")
            self._refresh()

        def calibrate_pitch(self):
            """PITCH: launch the calibration tool DETACHED on this video. It writes
            output/<clip>.pitch.json; press 'Reload pitch' afterwards to pick it
            up without restarting."""
            from PySide6 import QtCore
            script = str(pathlib.Path(__file__).resolve().parent
                         / "calibrate_pitch.py")
            ok = QtCore.QProcess.startDetached(
                sys.executable, [script, "--video", str(self.video_path)])
            if ok:
                self.status_msg = ("launched pitch calibration - click landmarks on "
                                   "ONE frame, Compute & Save, then 'Reload pitch'")
            else:
                self.status_msg = "could not launch calibrate_pitch.py"
            self._refresh()

        def reload_pitch(self):
            """PITCH: re-resolve + reload the .pitch.json and rebuild the live
            homography (so a fresh calibration takes effect without a restart)."""
            had = self.live_h is not None
            self._load_pitch_calibration()
            if self.live_h is not None:
                self.status_msg = (f"pitch calibration loaded (anchor frame "
                                   f"{self._pitch_anchor}) - play forward from it")
            elif had:
                self.status_msg = "pitch calibration reload FAILED - keeping none"
            else:
                self.status_msg = "no pitch calibration found to load"
            self._refresh()

        # ---- TACTICAL (M2): player circles + offside line ----
        def toggle_circles(self):
            """TACTICAL: enter / leave the 'character selector'. While ON a left
            click on an overlay player toggles its circle (selected_objs) instead
            of placing an ROI; OFF restores normal click behaviour. The target is
            always circled regardless."""
            self.circles_on = not self.circles_on
            if self.circles_on:
                self.status_msg = ("player circles ON - click overlay players to "
                                   "ring/un-ring them")
            else:
                self.status_msg = "player circles OFF"
            self._sync_buttons()
            self._refresh()

        def clear_circles(self):
            """TACTICAL: deselect every circled player (the target circle stays)."""
            self.selected_objs.clear()
            self.status_msg = "circles cleared"
            self._refresh()

        def _toggle_obj_circle(self, oid):
            """TACTICAL: add/remove overlay obj `oid` from the circle selection."""
            if oid in self.selected_objs:
                self.selected_objs.discard(oid)
                self.status_msg = f"un-ringed obj #{oid}"
            else:
                self.selected_objs.add(oid)
                self.status_msg = f"ringed obj #{oid}"
            self._refresh()

        def _selected_pitch_x(self):
            """The pitch X (metres) for the offside line: the SINGLE selected
            player's foot X if exactly one is selected, else the target's. Returns
            None when nothing resolvable / no live mapping."""
            lh = self.live_h
            if lh is None or not lh.healthy or self._live_stale:
                return None
            box = None
            if len(self.selected_objs) == 1:
                oid = next(iter(self.selected_objs))
                box = self._obj_box_by_id(oid)
            if box is None:
                box = self.store.target_box_at(self.frame)
            if box is None:
                return None
            xy = lh.foot_point(box)
            return None if xy is None else xy[0]

        def set_offside_at_player(self):
            """TACTICAL: set the offside line at the selected player's pitch X
            (single selection wins; else the target's). Needs a live pitch."""
            x = self._selected_pitch_x()
            if x is None:
                self.status_msg = ("can't set offside - need a live pitch + a "
                                   "selected player or target here")
                self._refresh(); return
            self.offside_x = x
            self.status_msg = f"offside line set @ X={x:.1f} m"
            self._refresh()

        def clear_offside(self):
            """TACTICAL: remove the offside line."""
            self.offside_x = None
            self.status_msg = "offside line cleared"
            self._refresh()

        def toggle_hide_other_teams(self):
            self.hide_other_teams = not self.hide_other_teams
            if hasattr(self, "act_hide_teams"):
                self.act_hide_teams.blockSignals(True)
                self.act_hide_teams.setChecked(self.hide_other_teams)
                self.act_hide_teams.blockSignals(False)
            self.status_msg = ("hide other teams ON" if self.hide_other_teams
                               else "hide other teams OFF")
            self._refresh()

        def toggle_hide_off_field(self):
            self.hide_off_field = not self.hide_off_field
            if hasattr(self, "act_hide_off"):
                self.act_hide_off.blockSignals(True)
                self.act_hide_off.setChecked(self.hide_off_field)
                self.act_hide_off.blockSignals(False)
            self.status_msg = ("hide off-field ON" if self.hide_off_field
                               else "hide off-field OFF")
            self._refresh()

        def toggle_focus(self):
            """Toggle the FOCUS box (hotkey 'x' / panel button). When ON (default)
            only detections inside the 3x-ROI focus box around your marker are
            ID-traced/selectable + drawn; far players are ignored (anti-jump)."""
            self.focus_on = not self.focus_on
            self.status_msg = ("focus box ON (ID-tracing only near your marker)"
                               if self.focus_on else "focus box OFF (whole frame)")
            self._sync_buttons()
            self._refresh()

        def toggle_recovery(self):
            """Toggle HOLD-don't-guess vs RECOVERY on loss (hotkey 'v' / menu).
            DEFAULT is OFF (HOLD): on loss the ROI is held in place. ON enables
            the team-constrained search-box recovery (and the magenta box)."""
            self.recovery_on = not self.recovery_on
            if hasattr(self, "act_recovery"):
                self.act_recovery.blockSignals(True)
                self.act_recovery.setChecked(self.recovery_on)
                self.act_recovery.blockSignals(False)
            if not self.recovery_on:
                self.search_box = None
            self.status_msg = ("recovery ON (search-box re-acquire)" if self.recovery_on
                               else "recovery OFF (HOLD on loss)")
            self._refresh()

        def toggle_autopause_low(self):
            """Toggle AUTO-PAUSE on low confidence (hotkey 'a' / panel button).
            When ON and playing, a frame whose confidence < AUTO_PAUSE_THRESH
            pauses playback so the user can check the target."""
            self.autopause_low = not self.autopause_low
            self.status_msg = ("auto-pause on low conf ON" if self.autopause_low
                               else "auto-pause on low conf OFF")
            self._refresh()

        def toggle_manual_mode(self):
            """Toggle MANUAL CURSOR-FOLLOW MODE (hotkey 'm' / bottom-bar button).

            STUTTER-SAFE: this ONLY flips the flag (+ snaps the ROI to the cursor
            on entry / re-anchors the CSRT on exit). It does NOT stop/restart the
            play timer, re-seek, or re-show the frame, so toggling mid-play never
            hitches - the existing per-frame advance loop just branches on the flag.

            When ON there is NO CSRT / snap / recovery: the ROI FOLLOWS THE MOUSE
            cursor (move the mouse to point at yourself) and each forward step
            records wherever it sits (SOURCE_MANUAL). When toggled OFF the CSRT is
            re-anchored on the current ROI so AUTO tracking resumes from there."""
            self.manual_mode = not self.manual_mode
            if self.manual_mode:
                self.search_box = None
                # snap the ROI straight onto the last known cursor position so
                # there's no jump on the first mouse-move.
                if self._cursor_fp is not None:
                    self._roi_follow_cursor(self._cursor_fp)
                self.status_msg = "MANUAL cursor-follow ON - move mouse to point at yourself"
            else:
                # resume AUTO. SNAP-TO-NEAREST + LOCK (Feature 2): if an ELIGIBLE
                # detection sits close to where the ROI was left (within
                # MANUAL_SNAP_ROI_WIDTHS x the ROI width), snap the ROI onto it,
                # re-anchor the CSRT there, record it as THIS frame's target and
                # seed the jump gate (lock). Otherwise fall back to the old
                # behaviour (re-anchor the CSRT on the ROI as-is).
                snapped = self._snap_to_nearest_on_exit()
                if not snapped:
                    self._reset_csrt()
                    self.status_msg = "auto-follow resumed"
            self._sync_manual_btn()
            self._refresh()

        def _snap_to_nearest_on_exit(self):
            """On manual-mode EXIT, snap+lock the ROI onto the nearest ELIGIBLE
            detection if one is near enough. Returns True if it locked onto a
            player, False if nothing was near (caller then keeps old behaviour).

            Uses the team/height/rule-out/jersey-filtered candidate pool so it can
            never lock onto the opposition / a wrong-numbered player. Pads the ROI
            ~MANUAL_SNAP_PAD around the box, re-anchors the CSRT, records the box as
            this frame's target (SOURCE_DETECTION) and _mark_target_good (seeds the
            jump gate). ROBUST: any failure just returns False."""
            if self.roi is None:
                return False
            try:
                pool = self._target_candidate_pool(self.frame)
                boxes = [b for _, b in pool]
                chosen = manual_snap_choice(self.roi, boxes)
            except Exception:
                return False
            if chosen is None:
                return False
            # pad the ROI ~15% around the chosen box, clamped to the frame
            x1, y1, x2, y2 = (float(v) for v in chosen)
            pw, ph = (x2 - x1) * MANUAL_SNAP_PAD, (y2 - y1) * MANUAL_SNAP_PAD
            self.roi = clamp_box((x1 - pw, y1 - ph, x2 + pw, y2 + ph),
                                 self.fw, self.fh)
            self.roi_on = True
            self._sync_track_btn()
            self._reset_csrt()                 # re-anchor the CSRT on the new ROI
            try:
                self.store.set_target_box(self.frame, chosen, SOURCE_DETECTION)
                self._learn_target(self.frame, chosen)
            except Exception:
                pass
            self._mark_target_good(self.frame, chosen)   # LOCK + seed the jump gate
            self._cache_frame()
            self.status_msg = "locked onto nearest player"
            return True

        def _sync_manual_btn(self):
            """Keep the Manual button label/checked state in sync with manual_mode."""
            btn = getattr(self, "manual_btn", None)
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(self.manual_mode)
                btn.setText("Manual: ON" if self.manual_mode else "Manual")
                btn.blockSignals(False)
            if hasattr(self, "act_manual"):
                self.act_manual.blockSignals(True)
                self.act_manual.setChecked(self.manual_mode)
                self.act_manual.blockSignals(False)

        def _field_mask_for(self, frame):
            """Cached rt2.field.field_mask for `frame` (computed lazily, only used
            while hide_off_field is on). Returns the uint8 mask or None if the
            frame image is unavailable / computation fails."""
            cached = self._field_mask_cache.get(frame)
            if cached is not None:
                return cached
            img = (self._cur_bgr if getattr(self, "_cur_bgr_frame", None) == frame
                   else self.reader.frame(frame))
            if img is None:
                return None
            try:
                mask = field.field_mask(img)
            except Exception:
                return None
            # keep the cache bounded during long playback
            if len(self._field_mask_cache) > 300:
                self._field_mask_cache.clear()
            self._field_mask_cache[frame] = mask
            return mask

        def visible_det_indices(self, frame):
            """Set of detection indices (into dets_at(frame)) that should be SHOWN
            given the active DISPLAY FILTERS. Always keeps the active target det and
            the armed detection visible. Purely cosmetic - does not affect tracking."""
            dets = self.dets_at(frame)
            visible = set(range(len(dets)))
            if not dets:
                return visible

            # indices we must NEVER hide (target + armed)
            keep = set()
            if frame == self.frame:
                active = self.active_target_idx()
                if active is not None:
                    keep.add(active)
            if self.armed_box is not None:
                for i, d in enumerate(dets):
                    if tuple(d[:4]) == tuple(self.armed_box):
                        keep.add(i)
                        break

            # (A) hide confidently non-tracked teams (needs calibration team info)
            if self.hide_other_teams:
                labels = self.team_labels_for_frame(frame)
                for i in list(visible):
                    if i in keep:
                        continue
                    if i < len(labels):
                        name, is_tracked, confident = labels[i]
                        if name is not None and confident and not is_tracked:
                            visible.discard(i)

            # (B) hide off-field detections (feet outside the pitch mask)
            if self.hide_off_field:
                mask = self._field_mask_for(frame)
                if mask is not None:
                    for i in list(visible):
                        if i in keep:
                            continue
                        if not field.feet_in_field(mask, dets[i][:4]):
                            visible.discard(i)

            # (C) FOCUS box: only show/trace detections inside the 3x-ROI region
            fb = self.focus_box()
            if fb is not None:
                for i in list(visible):
                    if i in keep:
                        continue
                    if not bbox.contains(fb, *bbox.center(dets[i][:4])):
                        visible.discard(i)

            visible |= keep
            return visible

        # ---- ROI / CSRT ----
        def set_roi(self, box):
            self.roi = box
            self.view.update()

        def _roi_follow_cursor(self, pt):
            """MANUAL CURSOR-FOLLOW: recentre the CURRENT ROI on frame-point `pt`,
            keeping its width/height, clamped to the frame, and refresh. If no ROI
            exists yet, create a default-sized one centred on the cursor. Cheap:
            just moves the box + repaints (no seek / no CSRT / no timer touch)."""
            if pt is None:
                return
            fx, fy = pt
            if self.roi is None:
                half = DEFAULT_ROI / 2.0
                box = (fx - half, fy - half, fx + half, fy + half)
            else:
                x1, y1, x2, y2 = self.roi
                hw, hh = (x2 - x1) / 2.0, (y2 - y1) / 2.0
                box = (fx - hw, fy - hh, fx + hw, fy + hh)
            self.roi = clamp_box(box, self.fw, self.fh)
            self.view.update()

        def place_roi(self, fx, fy, side=DEFAULT_ROI):
            half = side / 2.0
            self.set_roi(clamp_box((fx - half, fy - half, fx + half, fy + half),
                                   self.fw, self.fh))
            self._reset_csrt()
            # a freshly PLACED ROI usually means a new / re-anchored target, so the
            # learned model no longer applies - start learning the new player clean.
            # (a small drag-correct of the existing ROI goes through set_roi, not here,
            #  so it KEEPS the model.)
            self.tmodel.reset()
            # a re-anchored ROI is a trusted fresh start: clear the jump gate so the
            # next pick (near the new ROI) is accepted instead of being held.
            self._last_good_centre = None
            self._last_good_frame = None
            self._jump_pending = 0
            # Placing an ROI auto-enables tracking so Play actually records --
            # otherwise the ROI just sits there ("ROI set" but not "ON").
            self.roi_on = True
            self._sync_track_btn()
            # In MANUAL mode there is no CSRT/snap - the ROI follows the cursor and
            # each advance records it; the "tracking ON" wording would mislead, so
            # branch the message on manual_mode.
            if self.manual_mode:
                self.status_msg = ("ROI placed - MANUAL: move the mouse to point "
                                   "at your player; Play records each frame.")
            else:
                self.status_msg = "ROI placed - tracking ON. Press Play to record."

        def _csrt_scaled(self, img):
            """The frame CSRT runs on - downscaled by CSRT_SCALE for speed."""
            if CSRT_SCALE >= 0.999 or img is None:
                return img
            import cv2
            return cv2.resize(img, None, fx=CSRT_SCALE, fy=CSRT_SCALE,
                              interpolation=cv2.INTER_AREA)

        def _csrt_init(self, img, roi_xyxy):
            """Create + init CSRT on the DOWNSCALED frame; the ROI box is scaled
            into downscaled coords. Sets self.csrt."""
            self.csrt = make_csrt()
            x, y, w, h = xyxy_to_xywh(roi_xyxy)
            s = CSRT_SCALE
            self.csrt.init(self._csrt_scaled(img),
                           (int(x * s), int(y * s),
                            max(1, int(w * s)), max(1, int(h * s))))

        def _csrt_update(self, img):
            """Update CSRT on the downscaled frame; return (ok, xyxy box in FULL
            frame coords)."""
            ok, rect = self.csrt.update(self._csrt_scaled(img))
            if not ok:
                return False, None
            s = CSRT_SCALE
            x, y, w, h = rect
            return True, (x / s, y / s, (x + w) / s, (y + h) / s)

        def _reset_csrt(self):
            """(Re)initialise the CSRT on the current frame + ROI."""
            self.csrt = None
            self.csrt_frame = None
            if self.roi is None:
                return
            img = self.reader.frame(self.frame)
            if img is None:
                return
            try:
                self._csrt_init(img, self.roi)
                self.csrt_frame = self.frame
            except Exception as e:           # pragma: no cover
                self.csrt = None
                self.status_msg = f"CSRT init failed: {e}"

        def toggle_roi_tracking(self):
            if self.roi is None:
                self.status_msg = "no ROI - click to place one first"
                self._sync_track_btn()
                self._refresh()
                return
            self.roi_on = not self.roi_on
            if self.roi_on:
                self._reset_csrt()
                self.status_msg = "ROI tracking ON - Play to record"
            else:
                self.search_box = None       # don't draw a stale search box
                self.status_msg = "ROI tracking OFF"
            self._sync_track_btn()
            self._refresh()

        def _sync_track_btn(self):
            """Keep the Track button label/checked state in sync with roi_on."""
            btn = getattr(self, "track_btn", None)
            if btn is None:
                return
            btn.blockSignals(True)
            btn.setChecked(self.roi_on)
            btn.setText("Track: ON" if self.roi_on else "Track: OFF")
            btn.blockSignals(False)

        def reset_roi_to_target(self):
            tb = self.store.target_box_at(self.frame)
            if tb is None:
                self.status_msg = "no target box on this frame to reset to"
            else:
                # grow a little so the CSRT has context around the player
                x1, y1, x2, y2 = tb
                pad = 0.25
                w, h = (x2 - x1), (y2 - y1)
                self.set_roi(clamp_box((x1 - w * pad, y1 - h * pad,
                                        x2 + w * pad, y2 + h * pad), self.fw, self.fh))
                self._reset_csrt()
                self.status_msg = "ROI reset to target box"
            self._refresh()

        # ---- hotkey resize (+/-) ----
        def resize_roi(self, grow):
            """Grow (grow=True) or shrink the ROI by ~10%, kept SQUARE and CENTRED
            on its current centre, clamped to the frame. Re-anchors the CSRT."""
            if self.roi is None:
                self.status_msg = "no ROI - click to place one first"
                self._refresh()
                return
            x1, y1, x2, y2 = self.roi
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            side = max(x2 - x1, y2 - y1)
            factor = RESIZE_STEP if grow else 1.0 / RESIZE_STEP
            side = max(MIN_SIDE, min(side * factor, float(min(self.fw, self.fh))))
            half = side / 2.0
            self.set_roi(clamp_box((cx - half, cy - half, cx + half, cy + half),
                                   self.fw, self.fh))
            self._reset_csrt()
            rx1, ry1, rx2, ry2 = self.roi
            self.status_msg = f"ROI {int(rx2 - rx1)}x{int(ry2 - ry1)}px"
            self._refresh()

        def resize_spotlight(self, grow):
            """Grow / shrink the spotlight square by ~10% (kept square)."""
            factor = RESIZE_STEP if grow else 1.0 / RESIZE_STEP
            self.spotlight_side = max(
                MIN_SIDE, min(self.spotlight_side * factor, float(min(self.fw, self.fh))))
            self.status_msg = f"arrow size {int(self.spotlight_side)}px"
            self._refresh()

        # ---- spotlight follow (Feature 3) ----
        def toggle_spotlight(self):
            self.spotlight_on = not self.spotlight_on
            if self.spotlight_on:
                # seed the spotlight size from the current target box if we have one
                tb = self.store.target_box_at(self.frame)
                if tb is not None:
                    side = max(tb[2] - tb[0], tb[3] - tb[1]) * SPOTLIGHT_TARGET_MULT
                    self.spotlight_side = max(MIN_SIDE, side)
                self._spot_cx = self._spot_cy = None   # reset smoothing
                self.status_msg = "arrow marker ON"
            else:
                self.status_msg = "arrow marker OFF"
            self._refresh()

        # ---- CONFIDENCE-ADAPTIVE MARKER (Feature 1) ----
        def toggle_adaptive_marker(self):
            self.adaptive_marker_on = not self.adaptive_marker_on
            self._adapt_size = None              # reset size smoothing
            self._adapt_kind = None              # reset kind tracking too
            self.status_msg = ("adaptive marker ON" if self.adaptive_marker_on
                               else "adaptive marker OFF")
            self._refresh()

        # ---- CONFIDENCE-ADAPTIVE SPEED (Feature 3) ----
        def toggle_conf_speed(self):
            self.conf_speed = not self.conf_speed
            # snap the timer to the new regime immediately (manual when OFF)
            self._set_timer_interval()
            self.status_msg = ("conf-adaptive speed ON" if self.conf_speed
                               else "conf-adaptive speed OFF")
            self._refresh()

        def _spotlight_target_centre(self):
            """Raw (un-smoothed) HEAD anchor (cx, top_y) for the arrow marker:
            target box, else active detection, else ROI, else None."""
            tb = self.store.target_box_at(self.frame)
            if tb is not None:
                return ((tb[0] + tb[2]) / 2.0, tb[1])
            idx = self.active_target_idx()
            if idx is not None:
                box = self.dets_at(self.frame)[idx][:4]
                return ((box[0] + box[2]) / 2.0, box[1])
            if self.roi is not None:
                return ((self.roi[0] + self.roi[2]) / 2.0, self.roi[1])
            return None

        def _spotlight_centre(self):
            """EMA-smoothed spotlight centre (frame px) so the square glides."""
            raw = self._spotlight_target_centre()
            if raw is None:
                return None
            rx, ry = raw
            if self._spot_cx is None:
                self._spot_cx, self._spot_cy = rx, ry
            else:
                a = SPOTLIGHT_EMA
                self._spot_cx = a * rx + (1.0 - a) * self._spot_cx
                self._spot_cy = a * ry + (1.0 - a) * self._spot_cy
            return (self._spot_cx, self._spot_cy)

        def _advance_csrt_to(self, frame):
            """Run the CSRT from its current frame up to `frame` (forward only).
            Returns the new ROI box or None on failure."""
            import numpy as np  # noqa
            if self.csrt is None or self.csrt_frame is None:
                return None
            f = self.csrt_frame
            box = self.roi
            while f < frame:
                f += 1
                img = self.reader.frame(f)
                if img is None:
                    return None
                ok, box = self._csrt_update(img)
                if not ok:
                    self.csrt_frame = f
                    return None
                box = clamp_box(box, self.fw, self.fh)
            self.csrt_frame = frame
            return box

        # ---- frame cache ----
        def _cache_frame(self):
            """Store the ROI + target box decided for the current frame."""
            entry = {}
            if self.roi is not None:
                entry["roi"] = tuple(self.roi)
            tb = self.store.target_box_at(self.frame)
            if tb is not None:
                entry["target"] = tuple(tb)
            self.frame_cache[self.frame] = entry

        def _restore_from_cache(self, frame):
            """When scrubbing onto a cached frame, restore its ROI so the user
            sees the past decision. Returns True if restored."""
            entry = self.frame_cache.get(frame)
            if entry and "roi" in entry:
                self.roi = entry["roi"]
                return True
            return False

        # ---- the per-frame tracking step (Phase 4) ----
        def _record_manual(self, frame):
            """MANUAL CURSOR-FOLLOW step: record the CURRENT ROI box as the target
            for `frame` (SOURCE_MANUAL) and learn from it. No CSRT / snap /
            recovery - the ROI is following the mouse cursor onto the player."""
            self.search_box = None
            if self.roi is not None:
                self.store.set_target_box(frame, self.roi, SOURCE_MANUAL)
                self._learn_target(frame, self.roi)
                self._learn_identity(frame, self.roi)
                self._compute_confidence(frame, self.roi, SOURCE_MANUAL)
                # manual is the user pointing at the player -> a trusted anchor for
                # the jump gate (and clears any pending-jump count).
                self._mark_target_good(frame, self.roi)
                self._jump_pending = 0
            self.status_msg = "MANUAL - recorded ROI"
            self._cache_frame()

        def _hold_roi(self, frame, reason):
            """HOLD-don't-guess: keep the ROI where it is and record the held box
            (SOURCE_CSRT) WITHOUT moving to any far detection. Used when recovery
            is OFF and CSRT loses lock / no eligible detection is in the ROI."""
            self.search_box = None              # don't draw a stale search box
            self.status_msg = reason
            if self.roi is not None:
                self.store.set_target_box(frame, self.roi, SOURCE_CSRT)
                self._learn_target(frame, self.roi)
                self._learn_identity(frame, self.roi)
                self._compute_confidence(frame, self.roi, SOURCE_CSRT)
            self._cache_frame()

        def _track_step(self, frame):
            """Carry the ROI forward via CSRT to `frame`, snap to the best ELIGIBLE
            (same-team / unsure) detection, and append to the target track.

            On loss (CSRT lost, or no eligible detection inside the ROI) the
            behaviour depends on self.recovery_on:
              * recovery OFF (DEFAULT) -> HOLD the ROI in place (record the held
                box, SOURCE_CSRT); never run the big search-box recovery and never
                draw the magenta search box.
              * recovery ON -> run the team-constrained RECOVERY over a larger
                search box (gated by RECOVER_MIN); draw the search box."""
            # MANUAL FOLLOW MODE: no CSRT / snap / recovery - record the ROI.
            if self.manual_mode:
                self._record_manual(frame)
                return

            # keep the drawn search box current for this frame ONLY when recovery
            # is enabled; otherwise leave it cleared (no magenta box in HOLD mode).
            if self.recovery_on:
                self.search_box, _ = self._search_box_for(frame)
            else:
                self.search_box = None
            new_roi = self._advance_csrt_to(frame)
            if new_roi is None:
                # CSRT lost it (or not ready).
                if self.recovery_on and self._try_recover(frame):
                    self._cache_frame()
                    return
                self._hold_roi(frame, "CSRT lost target - ROI held")
                return
            self.roi = new_roi
            # snap: pick the best candidate from the team-constrained, height-
            # filtered, tracked-preferred POOL (active_target_idx scores within it
            # via the ONLINE MODEL). The nearest-centre FALLBACK (when scoring
            # returns None) must snap over the SAME pool, not the raw eligible
            # boxes, so the height gate / tracked-preference are honoured both ways.
            pool = self._target_candidate_pool(frame)
            if pool:
                pool_idxs = [i for i, _ in pool]
                pool_boxes = [b for _, b in pool]
                best_idx = self.active_target_idx()
                if best_idx is not None and best_idx in pool_idxs:
                    tbox = pool_boxes[pool_idxs.index(best_idx)]
                else:
                    cx, cy = bbox.center(self.roi)
                    tbox = pool_boxes[bbox.nearest(pool_boxes, cx, cy)]
                # optional SAM2 refinement: tighten the chosen box to its mask
                tbox = self._maybe_sam2_refine(frame, tbox)
                # JUMP GATE: don't teleport onto a far player - HOLD and wait until
                # the jump persists (a real fast move) before accepting it.
                if self._jump_guard(frame, tbox):
                    return
                self.store.set_target_box(frame, tbox, SOURCE_DETECTION)
                self._learn_target(frame, tbox)
                self._learn_identity(frame, tbox)
                # MY-NUMBER live jersey votes (Feature 1): OCR a few in-ROI
                # candidates (throttled/capped) so wrong-numbered tracks get
                # excluded next frame. INERT unless ocr.available() & my_number.
                self._collect_jersey_votes(frame, tbox)
                # CONFIDENCE: dominant pick over the next-best pool candidate
                margin = self._snap_margin_term(tbox, pool_boxes)
                self._compute_confidence(frame, tbox, SOURCE_DETECTION,
                                         margin_term=margin)
                # re-centre the CSRT on the detection so it doesn't drift
                self._recenter_csrt(frame, tbox)
                self._mark_target_good(frame, tbox)
                self._cache_frame()
            else:
                # No eligible candidate survives the pool -> recover or HOLD.
                if self.recovery_on and self._try_recover(frame):
                    self._cache_frame()
                    return
                self._hold_roi(frame, "no same-team detection in ROI - held")

        def _jump_guard(self, frame, box):
            """JUMP GATE. Returns True (and HOLDS this frame) when `box` is an
            implausible teleport from the last confirmed target and the jump has
            not yet persisted JUMP_CONFIRM frames; returns False to ACCEPT the
            box (normal step, or a jump that has persisted long enough = a real
            fast move / re-acquire)."""
            if jump_too_far(self._last_good_centre, self._last_good_frame,
                            box, frame):
                self._jump_pending += 1
                if self._jump_pending < JUMP_CONFIRM:
                    self._hold_roi(
                        frame, f"big jump rejected - holding "
                        f"({self._jump_pending}/{JUMP_CONFIRM}); re-anchor if real")
                    return True
            self._jump_pending = 0
            return False

        def _mark_target_good(self, frame, box):
            """Record the last CONFIRMED target centre/frame for the jump gate."""
            self._last_good_centre = bbox.center(box)
            self._last_good_frame = frame

        def _snap_margin_term(self, chosen, cand_boxes):
            """MARGIN term in [0,1]: how dominant the `chosen` box is over the
            next-best eligible candidate. A lone candidate is fully dominant (1).
            Each candidate scores a center-closeness in [0,1] vs the ROI centre
            (matching the snap fallback); margin = chosen_score - 2nd_best_score,
            mapped to [0,1]. None (neutral) when the ROI is missing."""
            if self.roi is None or not cand_boxes:
                return None
            if len(cand_boxes) == 1:
                return 1.0
            rx1, ry1, rx2, ry2 = self.roi
            diag = ((rx2 - rx1) ** 2 + (ry2 - ry1) ** 2) ** 0.5
            if diag <= 1e-6:
                diag = 1.0
            cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0

            def closeness(b):
                bx, by = bbox.center(b)
                d = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
                return max(0.0, min(1.0, 1.0 - d / diag))

            scores = sorted((closeness(b) for b in cand_boxes), reverse=True)
            chosen_s = closeness(chosen)
            # the runner-up is the best score that is not the chosen one
            second = scores[1] if scores[0] <= chosen_s + 1e-9 else scores[0]
            gap = max(0.0, chosen_s - second)
            return max(0.0, min(1.0, gap))

        # ---- RECOVERY (larger search box) ----
        def _search_box_for(self, frame):
            """The RECOVERY search region for `frame`: a box centred on the
            predicted target centre (or the ROI centre when no prediction),
            SEARCH_MULT x the ROI's (w, h), clamped to the frame. Returns
            (search_box, (cx, cy)) or (None, None) when there is no ROI."""
            if self.roi is None:
                return None, None
            rx1, ry1, rx2, ry2 = self.roi
            rw, rh = (rx2 - rx1), (ry2 - ry1)
            pred = self.tmodel.predict_centre(frame)
            if pred is not None:
                cx, cy = pred
            else:
                cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
            half_w = rw * SEARCH_MULT / 2.0
            half_h = rh * SEARCH_MULT / 2.0
            sb = clamp_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h),
                           self.fw, self.fh)
            return sb, (cx, cy)

        def _score_recovery_candidates(self, cands, centre, search_box, img):
            """Score recovery candidates and return (best_box, best_score).

            cands : list of xyxy candidate boxes (centres already known inside the
                    search box). centre : (cx, cy) the search box is built around.
            Delegates the per-candidate blend to the module-level recovery_score()
            (W_RPOS*pos_term + W_RAPP*app_term) so the exact scoring is testable
            headlessly by --selftest."""
            if not cands:
                return None, None
            best_box, best_score = None, None
            for box in cands:
                score = recovery_score(box, centre, search_box, self.tmodel, img)
                if best_score is None or score > best_score:
                    best_score, best_box = score, box
            return best_box, best_score

        def _try_recover(self, frame):
            """Attempt to RE-ACQUIRE the target on `frame` over a wider search box.

            Considers detections whose centre is inside the search box, scores them
            (position + appearance), and if the learned model has data AND the best
            score >= RECOVER_MIN, re-anchors the ROI on that detection (padded ~15%),
            re-inits the CSRT there, records it as a DETECTION, and learns from it.
            Returns True on a successful re-acquire, else False (caller holds ROI)."""
            sb, centre = self._search_box_for(frame)
            self.search_box = sb
            if sb is None or not self.tmodel.has_data():
                return False
            # candidates = ELIGIBLE detections whose CENTRE is inside the search
            # box. TEAM-CONSTRAINED: confident-opposition detections are excluded
            # so recovery can NEVER re-acquire onto the other team / far ref.
            eligible = self._eligible_indices(frame)
            cand_idxs, cand_boxes = [], []
            for i, d in enumerate(self.dets_at(frame)):
                box = d[:4]
                if i in eligible and bbox.contains(sb, *bbox.center(box)):
                    cand_idxs.append(i); cand_boxes.append(box)
            if not cand_idxs:
                return False
            # HEIGHT-CONSISTENCY: drop clearly-wrong-height candidates from the
            # recovery pool too (never the armed detection). Inert until ready.
            allowed = set(self._height_filter_indices(
                frame, cand_idxs, cand_boxes,
                protect=self._height_protected_indices(cand_idxs)))
            filtered = [b for i, b in zip(cand_idxs, cand_boxes) if i in allowed]
            cands = filtered if filtered else cand_boxes
            img = self._frame_img_for_appearance()
            best_box, best_score = self._score_recovery_candidates(
                cands, centre, sb, img)
            if best_box is None or best_score is None or best_score < RECOVER_MIN:
                return False
            # RE-ACQUIRE: ROI = the detection box padded ~15%, re-anchor the CSRT.
            x1, y1, x2, y2 = best_box
            w, h = (x2 - x1), (y2 - y1)
            pad = 0.15
            self.roi = clamp_box((x1 - w * pad, y1 - h * pad,
                                  x2 + w * pad, y2 + h * pad), self.fw, self.fh)
            self._reset_csrt()
            self.store.set_target_box(frame, best_box, SOURCE_DETECTION)
            self._learn_target(frame, best_box)
            self._learn_identity(frame, best_box)
            # a deliberate recovery re-acquire is a trusted anchor for the jump gate.
            self._mark_target_good(frame, best_box)
            self._jump_pending = 0
            # CONFIDENCE: a recovered detection is still a same-team snap, but the
            # dominance margin comes from the recovery scores (best over 2nd-best).
            margin = None
            if len(cands) >= 2:
                rscores = sorted(
                    (recovery_score(b, centre, sb, self.tmodel, img) for b in cands),
                    reverse=True)
                margin = max(0.0, min(1.0, rscores[0] - rscores[1]))
            elif len(cands) == 1:
                margin = 1.0
            self._compute_confidence(frame, best_box, SOURCE_DETECTION,
                                     margin_term=margin)
            self.status_msg = f"recovered (score {best_score:.2f})"
            return True

        def _learn_target(self, frame, box):
            """Fold a CONFIRMED target box for `frame` into the online model
            (motion deque + accumulating appearance mean + height residual).

            The HEIGHT residual (box_height - tracked_team.height.predict(centre_y))
            is computed here and folded into the TargetModel's Welford stats; it
            is None (a no-op) when there is no tracked-team height model, keeping
            TargetModel decoupled from calibration."""
            self.tmodel.update(frame, box, self._frame_img_for_appearance())
            self.tmodel.update_height(self._box_height_residual(box))

        # ---- ONLINE IDENTITY ("me vs not-me") feature vector + learning ----
        def _identity_features(self, frame, box):
            """Build the FIXED-LENGTH identity feature vector for an xyxy `box` at
            `frame` from EXISTING signals (team fingerprint distances, height
            residual deviation, normalised position, motion-consistency, size,
            appearance similarity). Missing signals fall back to NEUTRAL 0.0 via
            identity_features(), so the dimension is ALWAYS IDENTITY_DIM."""
            # team-fingerprint distances (0 when no calibration / no fingerprints)
            d_tracked = d_other = 0.0
            if self._team_fp_teams:
                try:
                    vec = features.feature_vector(
                        self._frame_img_for_appearance(), box)
                    bt, bo = features.classify(vec, self._team_fp_teams)
                    d_tracked = 0.0 if bt == float("inf") else bt
                    d_other = 0.0 if bo == float("inf") else bo
                except Exception:
                    d_tracked = d_other = 0.0
            # height-residual deviation vs the target's learned mean (0 if none)
            resid_dev = 0.0
            r = self._box_height_residual(box)
            if r is not None and self.tmodel.resid_count > 0:
                resid_dev = abs(r - self.tmodel.resid_mean)
            # normalised centre + size
            bx, by = bbox.center(box)
            norm_cx = bx / self.fw if self.fw else 0.0
            norm_cy = by / self.fh if self.fh else 0.0
            box_h_frac = (box[3] - box[1]) / self.fh if self.fh else 0.0
            # motion-consistency: distance from the predicted centre / frame diag
            motion = 0.0
            pred = self.tmodel.predict_centre(frame)
            if pred is not None:
                diag = (self.fw ** 2 + self.fh ** 2) ** 0.5 or 1.0
                motion = (((bx - pred[0]) ** 2 + (by - pred[1]) ** 2) ** 0.5) / diag
            # appearance similarity to the learned player (0 if no model / image)
            app = 0.0
            img = self._frame_img_for_appearance()
            if img is not None and self.tmodel.count > 0:
                app = self.tmodel.appearance_sim(TargetModel._crop(img, box))
            return identity_features(
                d_tracked, d_other, resid_dev, norm_cx, norm_cy,
                motion, box_h_frac, app)

        def _id_prob(self, frame, box):
            """The identity classifier's prob(me) for `box` at `frame`, or None
            when the model is not ready() (so callers leave the blend unchanged)."""
            if not self.idmodel.ready():
                return None
            try:
                return self.idmodel.prob(self._identity_features(frame, box))
            except Exception:
                return None

        def _learn_identity(self, frame, target_box):
            """LEARN from a CONFIRMED target frame: add the target box as a POSITIVE
            and every OTHER eligible in-frame detection as a WEAK negative. Called
            from every confirmed-target path (snap / manual / hold-with-detection /
            recovery). A strict no-op if the target box is missing."""
            if target_box is None:
                return
            try:
                self.idmodel.add_positive(
                    self._identity_features(frame, target_box))
            except Exception:
                return
            # WEAK negatives: the OTHER eligible detections this frame (not the
            # target itself, not anything already ruled out). Cheap, bounded.
            try:
                tb = tuple(float(v) for v in target_box)
                eligible = self._eligible_indices(frame)
                for i, d in enumerate(self.dets_at(frame)):
                    if i not in eligible:
                        continue
                    box = d[:4]
                    if tuple(float(v) for v in box) == tb:
                        continue
                    self.idmodel.add_negative(
                        self._identity_features(frame, box), weight=ID_WEAK_NEG)
            except Exception:
                pass

        # ---- per-frame CONFIDENCE rating ----
        def _height_conf_term(self, box):
            """HEIGHT closeness term in [0,1] for the chosen `box`: 1 when the
            residual is within ~0.5*std of the learned mean, decaying linearly to
            0 by HEIGHT_TOL. Neutral (None -> 0.5 in the blend) when the residual
            model is not ready or there is no measurable height."""
            if not self.tmodel.height_ready():
                return None
            r = self._box_height_residual(box)
            if r is None:
                return None
            dev = abs(r - self.tmodel.resid_mean)
            inner = max(1e-6, 0.5 * self.tmodel.resid_std())
            tol = self.tmodel.height_tol()
            if dev <= inner:
                return 1.0
            if dev >= tol:
                return 0.0
            return max(0.0, min(1.0, 1.0 - (dev - inner) / (tol - inner)))

        def _app_conf_term(self, box):
            """APPEARANCE term in [0,1]: similarity of the chosen crop to the
            learned player, or None (neutral) when no appearance model / image."""
            if self.tmodel.count <= 0:
                return None
            img = self._frame_img_for_appearance()
            if img is None:
                return None
            return self.tmodel.appearance_sim(TargetModel._crop(img, box))

        # ---- JERSEY-NUMBER OCR soft signal (BONUS cue; default OFF) ----
        def _ocr_region(self, box):
            """Crop the target's UPPER-BACK/torso region for OCR: the vertical
            band OCR_REGION_TOP..OCR_REGION_BOT of the box, central
            OCR_REGION_CENTRAL horizontally. Returns a BGR crop or None."""
            img = self._frame_img_for_appearance()
            if img is None or box is None:
                return None
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w < 2 or h < 2:
                return None
            ry1 = y1 + OCR_REGION_TOP * h
            ry2 = y1 + OCR_REGION_BOT * h
            margin = (1.0 - OCR_REGION_CENTRAL) / 2.0
            rx1 = x1 + margin * w
            rx2 = x2 - margin * w
            crop = TargetModel._crop(img, (rx1, ry1, rx2, ry2))
            return crop

        def _ocr_active(self):
            """True only when the toggle is ON AND the tesseract binary is usable.
            Everything OCR-related is a strict NO-OP unless this is True."""
            return bool(self.ocr_on) and ocr.available()

        def _maybe_ocr(self, frame, box):
            """When OCR is ACTIVE, the box is tall enough, and this is an Nth
            tracked frame, OCR the upper-back region and fold a confident 1-2 digit
            read into the TARGET's Counter (-> self.target_number). NO-OP otherwise.
            Returns the fresh (digits, conf) read this frame, or None.

            This NEVER raises (rt2.ocr.read_number swallows all errors and returns
            None) and NEVER touches the track or candidate selection - it only
            updates the soft-signal bookkeeping used by _ocr_adjust."""
            if not self._ocr_active() or box is None:
                return None
            x1, y1, x2, y2 = box
            if (y2 - y1) < OCR_MIN_BOX_H:
                return None
            self._ocr_frame_tick += 1
            if self._ocr_frame_tick % OCR_EVERY_N != 0:
                return None
            res = ocr.read_number(self._ocr_region(box))
            if res is None:
                return None
            digits, conf = res
            if conf < OCR_CONF_FLOOR:
                return None
            self._ocr_last = (digits, conf)
            self._ocr_counter[digits] += 1
            # most-frequent confident read becomes the target's number
            self.target_number = self._ocr_counter.most_common(1)[0][0]
            return res

        def _ocr_adjust(self, frame, box, conf):
            """Apply the SMALL jersey-OCR soft signal to `conf` and return it.

            Strict NO-OP (returns conf unchanged) when the toggle is OFF or the
            tesseract binary is unavailable. When active, OCRs (throttled) and:
              * a fresh confident read MATCHING self.target_number -> +OCR_BOOST
              * a fresh confident read of a DIFFERENT number       -> -OCR_PENALTY
            Re-clamped to [0,1]. The weight is intentionally small so OCR only
            nudges - it never overrides motion/appearance/height/margin."""
            if not self._ocr_active():
                return conf
            res = self._maybe_ocr(frame, box)
            if res is None:
                return conf
            digits, _ = res
            if self.target_number is not None and digits == self.target_number:
                conf = conf + OCR_BOOST
            elif self.target_number is not None and digits != self.target_number:
                conf = conf - OCR_PENALTY
            return max(0.0, min(1.0, conf))

        def toggle_ocr(self):
            """Toggle the JERSEY-NUMBER OCR soft signal (hotkey 'j' / panel button).

            DEFAULT OFF. If the tesseract binary is not installed
            (rt2.ocr.available() False) this is a NO-OP with a status note - the
            flag is left OFF so confidence is never touched. Otherwise it flips the
            flag; when turned OFF the accumulated reads are cleared."""
            if not ocr.available():
                self.ocr_on = False
                self.status_msg = ("Jersey OCR unavailable: install tesseract "
                                   "(winget install UB-Mannheim.TesseractOCR)")
                self._refresh()
                return
            self.ocr_on = not self.ocr_on
            if not self.ocr_on:
                # forget the session's accumulated reads when switching off
                self._ocr_counter.clear()
                self.target_number = None
                self._ocr_last = None
            self.status_msg = ("Jersey OCR ON (soft cue)" if self.ocr_on
                               else "Jersey OCR OFF")
            self._refresh()

        # ---- MY-NUMBER live jersey exclusion (Feature 1) ----
        def set_my_number(self, n):
            """Set the user's OWN jersey number (the LEARNING-panel 'My #' spin box
            and the 'This is me' assign path call this). `n` is an int (1..23) or
            None / 0 to clear. Mirrors it onto the TargetModel so it persists in
            the profile, and refreshes the dependent jersey bookkeeping (a changed
            number invalidates which tracks count as 'someone else'). NO-OP-safe."""
            try:
                n = None if not n else int(n)
            except (TypeError, ValueError):
                n = None
            if n is not None and not (1 <= n <= MAX_JERSEY):
                n = None
            self.my_number = n
            self.tmodel.my_number = n          # persist with the profile
            # the decision boundary changed -> drop stale per-track votes so the
            # exclusion re-decides against the new number cleanly.
            self._track_numbers.clear()
            self._sync_my_number_spin()
            # a changed number re-decides which overlay tracks could be ME.
            self._recompute_shortlist()
            self.status_msg = (f"my # set to {n}" if n is not None
                               else "my # cleared")
            self._refresh()

        def _sync_my_number_spin(self):
            """Keep the 'My #' spin box (MY PLAYER group) in sync with
            self.my_number (0 = unset), and refresh the "Tracking: <name> #<n>"
            headline label alongside it."""
            spin = getattr(self, "mynum_spin", None)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(int(self.my_number) if self.my_number else 0)
                spin.blockSignals(False)
            lbl = getattr(self, "myplayer_lbl", None)
            if lbl is not None:
                who = self.player or "(single player)"
                num = f"#{self.my_number}" if self.my_number else "#--"
                lbl.setText(f"Tracking: {who}   {num}")

        def _mynum_active(self):
            """True only when the my-number jersey feature is LIVE: a number is
            configured AND the tesseract binary is usable. Everything in this
            feature is a strict NO-OP (zero perf cost) unless this is True."""
            return self.my_number is not None and ocr.available()

        def _collect_jersey_votes(self, frame, active_box):
            """When the my-number feature is LIVE, OCR a FEW eligible in-ROI
            candidate crops (THROTTLED + CAPPED) and tally confident 1..MAX_JERSEY
            reads into each track's vote Counter (weighted by OCR confidence).

            Strict NO-OP unless _mynum_active(). Throttle: only every
            MYNUM_OCR_EVERY_N tracked frames. Cap: at most MYNUM_OCR_MAX_CANDS of
            the tallest eligible boxes >= MYNUM_OCR_MIN_H px. Every OCR call is
            wrapped so it can never crash the app (rt2.ocr already swallows its own
            errors). The votes drive _eligible_indices' exclusion + the selection
            preference; this method itself NEVER touches the track."""
            if not self._mynum_active() or self.roi is None:
                return
            self._mynum_frame_tick += 1
            if self._mynum_frame_tick % MYNUM_OCR_EVERY_N != 0:
                return
            try:
                idxs, boxes = self.dets_in_roi(frame)
            except Exception:
                return
            if not boxes:
                return
            # tallest-first, only boxes big enough for a readable number, capped
            cands = [(b, (b[3] - b[1])) for b in boxes if (b[3] - b[1]) >= MYNUM_OCR_MIN_H]
            cands.sort(key=lambda t: t[1], reverse=True)
            cands = cands[:MYNUM_OCR_MAX_CANDS]
            if not cands:
                return
            img = self._frame_img_for_appearance()
            if img is None:
                return
            for box, _h in cands:
                try:
                    patch = regions.torso_patch(img, box)
                    res = ocr.read_number(patch)
                except Exception:
                    continue
                if res is None:
                    continue
                digits, conf = res
                if conf < MYNUM_OCR_CONF_FLOOR:
                    continue
                try:
                    val = int(digits)
                except (TypeError, ValueError):
                    continue
                if not (1 <= val <= MAX_JERSEY):
                    continue            # ignore reads outside plausible jersey range
                tid = self._det_track_hint(box)
                if tid is None:
                    continue
                # weight the vote by OCR confidence so shaky reads count for less
                self._track_numbers[tid][digits] += float(conf)

        def _jersey_excluded_tids(self):
            """Set of track ids that are CONFIDENTLY SOMEONE ELSE (top vote decided
            on a number that is not mine). Empty when the feature is inert. Used by
            _eligible_indices to drop those tracks (like ruled_out)."""
            if not self._mynum_active():
                return set()
            out = set()
            for tid, counter in self._track_numbers.items():
                if confidently_someone_else(counter, self.my_number):
                    out.add(tid)
            return out

        def _reads_my_number(self, box):
            """True when `box`'s track has DECIDED on MY number (so it should be
            preferred + bumped). NO-OP-safe -> False when the feature is inert."""
            if not self._mynum_active():
                return False
            tid = self._det_track_hint(box)
            if tid is None:
                return False
            decided = jersey_decided_number(self._track_numbers.get(tid, Counter()))
            if decided is None:
                return False
            try:
                return int(decided) == int(self.my_number)
            except (TypeError, ValueError):
                return False

        def toggle_sam2(self):
            """Toggle SAM 2 box refinement (hotkey 'f' / panel button).

            DEFAULT OFF. If ultralytics/SAM2 is not importable this is a NO-OP
            with a status note. On first enable the Sam2Box is created lazily
            (which imports torch AFTER pyarrow - see rt2/sam2track) and the tiny
            checkpoint auto-downloads."""
            if not sam2track.available():
                self.sam2_on = False
                self.status_msg = "SAM2 unavailable (ultralytics import failed)"
                self._refresh()
                return
            self.sam2_on = not self.sam2_on
            if self.sam2_on and self.sam2 is None:
                self.status_msg = (f"SAM2 loading {self.sam2_model} "
                                   "(first use; large model may download ~900MB)...")
                self._refresh()
                QtWidgets.QApplication.processEvents()
                self.sam2 = sam2track.Sam2Box(model=self.sam2_model, device="0")
            self.status_msg = ("SAM2 ON: body-tight silhouette + box refine"
                               if self.sam2_on else "SAM2 OFF")
            if not self.sam2_on:
                self._sil_cache.clear()
            self._refresh()

        def _maybe_sam2_refine(self, frame, box):
            """If SAM2 refinement is ON, return a tighter box from SAM2's mask of
            the player in `box`; otherwise (or on any failure) return `box`
            unchanged. `frame` is the frame NUMBER; the image is fetched lazily."""
            if not self.sam2_on or self.sam2 is None or box is None:
                return box
            img = self._frame_img_for_appearance()
            if img is None:
                return box
            refined = self.sam2.refine(img, box)
            return refined if refined is not None else box

        def target_silhouette(self, frame, tb):
            """SAM2 mask polygons (FRAME coords) for the target box on `frame`,
            cached per (frame, rounded-box). None when SAM2 is off / unavailable /
            the segmentation is implausible. Computed lazily (once per frame)."""
            if not self.sam2_on or self.sam2 is None or tb is None:
                return None
            key = (frame, tuple(int(round(v)) for v in tb))
            if key in self._sil_cache:
                return self._sil_cache[key]
            if len(self._sil_cache) > 96:
                self._sil_cache.clear()
            img = self._frame_img_for_appearance()
            polys = self.sam2.mask_polys(img, tb) if img is not None else None
            self._sil_cache[key] = polys
            return polys

        def _compute_confidence(self, frame, box, source, margin_term=None):
            """Compute, store (per-frame + last_confidence) and return the [0,1]
            tracking confidence for `frame`. Also fires AUTO-PAUSE on low conf."""
            conf = frame_confidence(
                source,
                height_term=self._height_conf_term(box),
                app_term=self._app_conf_term(box),
                margin_term=margin_term,
                id_term=self._id_prob(frame, box))
            # JERSEY-NUMBER OCR soft signal (BONUS). Strict NO-OP unless the
            # toggle is ON *and* the tesseract binary is available; _ocr_adjust
            # applies only a SMALL boost/penalty and re-clamps to [0,1].
            conf = self._ocr_adjust(frame, box, conf)
            # MY-NUMBER bump (Feature 1): if the chosen target box's track has
            # decided on MY shirt number, nudge confidence UP a little. Strict
            # NO-OP when the feature is inert. Re-clamped to [0,1].
            if self._reads_my_number(box):
                conf = max(0.0, min(1.0, conf + MYNUM_CONF_BUMP))
            self.last_confidence = conf
            self.conf_by_frame[frame] = conf
            # LOST-IN-THE-RUCK tracking (Feature 3): keep a run-length of
            # consecutive sub-threshold frames; recovering above the threshold
            # re-arms the prompt (so each distinct loss can prompt once).
            if conf < AUTO_PAUSE_THRESH:
                self._ruck_low_run += 1
            else:
                self._ruck_low_run = 0
                self._ruck_prompted = False
            # AUTO-PAUSE: when armed + playing, a low-confidence frame pauses.
            paused_now = False
            if (self.autopause_low and self.playing
                    and conf < AUTO_PAUSE_THRESH):
                self.set_playing(False)
                self.status_msg = "paused: low confidence - check the target"
                paused_now = True
            # LOST-IN-THE-RUCK prompt (Feature 3): OFF-safe (only when auto-pause
            # is ON). Prompt ONCE per loss when either we just auto-paused OR a run
            # of RUCK_LOST_FRAMES sub-threshold frames has built up. Only while we
            # were playing (or just paused this frame); never in manual mode.
            if (self.autopause_low and not self._ruck_prompted
                    and not self.manual_mode
                    and (paused_now or self._ruck_low_run >= RUCK_LOST_FRAMES)
                    and (self.playing or paused_now)):
                self._ruck_prompted = True
                self._prompt_lost_in_ruck()
            return conf

        def _prompt_lost_in_ruck(self):
            """Non-spammy prompt (Feature 3) asking whether to switch to MANUAL
            mode after the target was lost in a ruck. Pauses playback first so the
            frame holds while the user decides. On 'Manual' -> enter manual mode
            (flag + button sync). ROBUST: wrapped so a dialog failure can't crash
            the app, and a strict no-op if already in manual mode."""
            if self.manual_mode:
                return
            try:
                self.set_playing(False)        # hold the frame for the decision
                box = QtWidgets.QMessageBox(self)
                box.setWindowTitle("Lost the target")
                box.setIcon(QtWidgets.QMessageBox.Question)
                box.setText("Lost the target in the ruck - switch to manual mode?")
                manual_btn = box.addButton("Manual", QtWidgets.QMessageBox.AcceptRole)
                box.addButton("Stay auto", QtWidgets.QMessageBox.RejectRole)
                box.exec()
                if box.clickedButton() is manual_btn and not self.manual_mode:
                    self.toggle_manual_mode()  # flips flag + syncs buttons
                    self.status_msg = "MANUAL mode - point at your player"
            except Exception:
                pass

        def _recenter_csrt(self, frame, tbox):
            """Nudge the ROI to keep the snapped detection centred, and re-init the
            CSRT there so it tracks the player rather than drifting off.

            PERF: re-initialising the CSRT every frame is the per-tick cost. When
            the snapped box barely moved relative to the CSRT box (deviation <= ~25%
            of the ROI size) we just recentre the ROI and KEEP the existing tracker
            (re-init buys nothing then). Only on a real move (>25%) do we re-init."""
            cx, cy = bbox.center(tbox)
            # deviation of the snap from the CURRENT CSRT/ROI centre, as a fraction
            # of the ROI size (measure before we move the ROI).
            ocx, ocy = bbox.center(self.roi)
            rw_full = max(1.0, self.roi[2] - self.roi[0])
            rh_full = max(1.0, self.roi[3] - self.roi[1])
            dev = max(abs(cx - ocx) / rw_full, abs(cy - ocy) / rh_full)
            rw = rw_full / 2.0
            rh = rh_full / 2.0
            self.roi = clamp_box((cx - rw, cy - rh, cx + rw, cy + rh), self.fw, self.fh)
            # within tolerance and the tracker is alive -> keep it, skip the re-init.
            if dev <= 0.25 and self.csrt is not None:
                return
            img = self.reader.frame(frame)
            if img is None:
                return
            try:
                self._csrt_init(img, self.roi)
                self.csrt_frame = frame
            except Exception:                # pragma: no cover
                pass

        # ---- navigation ----
        def goto(self, frame, do_track=None):
            """Move to `frame`. If moving forward by 1 with ROI tracking on, run a
            tracking step; otherwise just display (restoring cached ROI if any)."""
            frame = max(1, min(self.n_frames, int(frame)))
            forward_one = (frame == self.frame + 1)
            # MANUAL FOLLOW MODE: any forward step records the CURRENT ROI for the
            # new frame (no CSRT needed). The user has dragged the ROI onto the
            # player; stepping commits wherever it sits.
            if self.manual_mode and forward_one and self.roi is not None:
                self.frame = frame
                self._record_manual(frame)
                self._show_frame()
                return
            if do_track is None:
                do_track = self.roi_on and forward_one
            if do_track and self.csrt is None:
                self._reset_csrt()          # re-init so a forward step keeps recording
            if do_track and self.csrt is not None:
                self.frame = frame
                self._track_step(frame)
            else:
                # plain seek; restore cached ROI if we have one for this frame
                self.frame = frame
                self._restore_from_cache(frame)
            self._show_frame()

        def step(self, delta):
            self.set_playing(False)
            target = self.frame + delta
            if delta == 1 and self.manual_mode and self.roi is not None:
                self.goto(target)
            elif delta == 1 and self.roi_on:
                self.goto(target)           # goto re-inits CSRT if it was lost
            else:
                self.goto(target, do_track=False)

        # ---- playback ----
        def toggle_play(self):
            self.set_playing(not self.playing)

        def set_playing(self, on):
            self.playing = on
            self.play_btn.setText("Pause" if on else "Play")
            if on:
                self.timer.start()
            else:
                self.timer.stop()

        def _tick(self):
            if self.frame >= self.n_frames:
                self.set_playing(False)
                return
            self.goto(self.frame + 1)
            # CONFIDENCE-ADAPTIVE SPEED: re-time the next tick from the confidence
            # just produced for the advanced-to frame. (When conf_speed is OFF the
            # interval is constant so this is a cheap no-op refresh of the manual
            # speed.)
            if self.conf_speed:
                self._set_timer_interval()

        # ---- slider ----
        def on_slider(self, val):
            if val != self.frame:
                self.set_playing(False)
                self.goto(int(val), do_track=False)

        # ---- clicks ----
        def on_view_click(self, fx, fy):
            """A non-drag left click in the video: arm a detection if one is hit,
            else place a fresh ROI centred on the click."""
            # OBJECTS OVERLAY: clicking a VISIBLE overlay box picks that track --
            # me-likely / possible become the target (place ROI + record + mark
            # good); ruled-out (only clickable when shown) just arms. Clicking
            # empty space places a fresh ROI (unchanged).
            if self.has_objects_overlay():
                hit_oid = hit_box = None
                best_area = None
                for oid, box, _team, _conf, _status in self.visible_objects_at(self.frame):
                    if bbox.contains(box, fx, fy):
                        a = bbox.area(box)
                        if best_area is None or a < best_area:
                            best_area = a; hit_oid = oid; hit_box = box
                # RULE-OUT mode: a click on a player marks them NOT me (red +
                # excluded); empty space is a no-op. Takes priority over picking.
                if self.ruleout_mode:
                    if hit_oid is not None:
                        self._rule_out_box(hit_box, hit_oid)
                    else:
                        self.status_msg = "rule-out ON - click a player (not empty space)"
                        self._refresh()
                    return
                # TACTICAL "character selector": while circles are ON a click on a
                # player TOGGLES its circle instead of (re)picking the target /
                # placing an ROI. A click on empty space is a no-op here.
                if self.circles_on:
                    if hit_oid is not None:
                        self._toggle_obj_circle(hit_oid)
                    else:
                        self.status_msg = "circles ON - click a player to ring it"
                        self._refresh()
                    return
                if hit_oid is not None:
                    v = self._obj_verdicts.get(hit_oid)
                    status = v.status if v is not None else "possible"
                    if status == "ruled-out":
                        self.armed = self._det_track_hint(hit_box)
                        self.armed_box = hit_box
                        self.status_msg = f"armed ruled-out obj #{hit_oid}"
                        self._refresh()
                    else:
                        self.pick_object_as_target(hit_oid, hit_box)
                else:
                    self.place_roi(fx, fy)
                    self.status_msg = "placed ROI"
                    self._refresh()
                return
            dets = self.dets_at(self.frame)
            hit = None
            best_area = None
            for d in dets:
                box = d[:4]
                if bbox.contains(box, fx, fy):
                    a = bbox.area(box)
                    if best_area is None or a < best_area:
                        best_area = a; hit = box
            if hit is not None and self.ruleout_mode:
                self._rule_out_box(hit, self._det_track_hint(hit))
                return
            if hit is not None:
                self.armed = self._det_track_hint(hit)
                self.armed_box = hit
                self.status_msg = (f"armed target id {self.armed}" if self.armed is not None
                                   else "armed detection (no track id)")
            else:
                self.place_roi(fx, fy)
                self.status_msg = "placed ROI"
            self._refresh()

        def _det_track_hint(self, box):
            """Detections from the pool have no track id; for merge/delete we arm
            against the consolidated target track. If a target row at this frame
            overlaps the click, arm the target id; otherwise arm the target id by
            default so 'd'/'D' operate on the consolidated track.

            OBJECTS OVERLAY: when an objects.csv is loaded each player has a
            persistent obj_id, so PREFER it -- match `box` to the nearest overlay
            box this frame by IoU (then centre) and return that obj_id. This is
            what lets detection-naming + the my-number lock attach to stable
            obj_ids (fixing the earlier 'no per-detection track id' limitation).
            Falls back to the old behaviour when there is no overlay / no match."""
            if self.has_objects_overlay():
                objs = self.objects_at(self.frame)
                best_oid, best_iou = None, 0.0
                for oid, obox, _team, _conf in objs:
                    i = bbox.iou(obox, box)
                    if i > best_iou:
                        best_iou, best_oid = i, oid
                if best_oid is not None and best_iou > 0.3:
                    return best_oid
                # no overlap -> nearest centre (still a stable obj_id)
                if objs:
                    cx, cy = bbox.center(box)
                    boxes = [o[1] for o in objs]
                    j = bbox.nearest(boxes, cx, cy)
                    if j is not None:
                        return objs[j][0]
                # fall through to the legacy hint when the overlay has nothing here
            for rid in self.store.by_frame.get(self.frame, []):
                d = self.store.records[rid]
                if bbox.iou((d["x1"], d["y1"], d["x2"], d["y2"]), box) > 0.3:
                    return d["tid"]
            return self.store.target

        def _obj_box_by_id(self, oid, frame=None):
            """The box for overlay obj `oid` at `frame` (default current), or None."""
            if frame is None:
                frame = self.frame
            for o, box, _team, _conf in self.objects_at(frame):
                if o == oid:
                    return box
            return None

        def on_det_list_click(self, item):
            data = item.data(Qt.UserRole)
            # OBJECTS OVERLAY: a row is ("obj", obj_id). Clicking a me-likely /
            # possible track PICKS it as the target (ruled-out rows just arm).
            if isinstance(data, tuple) and len(data) == 2 and data[0] == "obj":
                oid = data[1]
                box = self._obj_box_by_id(oid)
                if box is None:
                    return
                v = self._obj_verdicts.get(oid)
                status = v.status if v is not None else "possible"
                if status == "ruled-out":
                    self.armed_box = box
                    self.armed = self._det_track_hint(box)
                    self.status_msg = f"armed ruled-out obj #{oid}"
                    self._refresh()
                else:
                    self.pick_object_as_target(oid, box)
                return
            idx = data
            dets = self.dets_at(self.frame)
            if idx is not None and 0 <= idx < len(dets):
                self.armed_box = dets[idx][:4]
                self.armed = self._det_track_hint(self.armed_box)
                self.status_msg = f"armed det #{idx} (target id {self.armed})"
                self._refresh()

        def pick_object_as_target(self, oid, box):
            """SHORTLIST PICK: select overlay obj `oid` (box) as THE target. Places
            a padded ROI on it (auto-arms tracking), records it as the target at
            this frame + marks it good, so normal tracking / SAM3 re-track then
            follows THIS track. Reuses the existing place_roi / arm / record
            plumbing so it behaves exactly like a manual click-pick."""
            cx, cy = bbox.center(box)
            bw, bh = (box[2] - box[0]), (box[3] - box[1])
            # padded ROI side: comfortably larger than the box so the in-ROI
            # snap has room, but not the whole field.
            side = max(DEFAULT_ROI, 1.6 * max(bw, bh))
            self.place_roi(cx, cy, side=side)
            self.armed = self._det_track_hint(box)
            self.armed_box = box
            # record this box as the consolidated target at this frame + arm the
            # jump gate so the next pick near it is accepted.
            try:
                self.store.set_target_box(self.frame, box, source=SOURCE_MANUAL)
            except Exception:
                pass
            self._mark_target_good(self.frame, box)
            self.status_msg = f"picked obj #{oid} as target (ROI placed, tracking ON)"
            self._refresh()

        # ---- detection -> player NAMING (links to output/registry.sqlite) ----
        def _det_box_for_list_item(self, item):
            """The frame-px box of the detection backing a list item, or None."""
            if item is None:
                return None
            data = item.data(Qt.UserRole)
            # OBJECTS OVERLAY rows carry ("obj", obj_id); resolve to its box so
            # naming attaches to the persistent obj_id (via _det_track_hint).
            if isinstance(data, tuple) and len(data) == 2 and data[0] == "obj":
                return self._obj_box_by_id(data[1])
            idx = data
            dets = self.dets_at(self.frame)
            if idx is not None and 0 <= idx < len(dets):
                return dets[idx][:4]
            return None

        def on_det_list_double(self, item):
            """Double-click a detection -> open the Assign-to-player dialog."""
            box = self._det_box_for_list_item(item)
            if box is not None:
                self.assign_detection_to_player(box)

        def on_det_list_menu(self, pos):
            """Right-click a detection -> context menu with 'Assign to player...'."""
            item = self.det_list.itemAt(pos)
            box = self._det_box_for_list_item(item)
            if box is None:
                return
            menu = QtWidgets.QMenu(self.det_list)
            act = menu.addAction("Assign to player...")
            act.triggered.connect(
                lambda _checked=False, b=box: self.assign_detection_to_player(b))
            menu.exec(self.det_list.mapToGlobal(pos))

        def assign_detection_to_player(self, box):
            """Open a small dialog to assign `box`'s TRACK ID to a registry
            player -- either an EXISTING player or a NEW one (name + optional
            jersey number + 'is teammate'/'this is me'). On confirm we create or
            find the player in output/registry.sqlite and remember the mapping
            track_id -> (uuid, display_name) for this session so the name shows on
            the detection list + on the box. ROBUST: every registry call is
            wrapped; a missing/locked DB never crashes -- we just report it in the
            status bar (mirrors the OCR/optional-feature house style)."""
            tid = self._det_track_hint(box)
            if tid is None:
                self.status_msg = "assign: detection has no track id"
                self._refresh(); return

            # Pull the list of existing registry players up-front (read-only). If
            # the DB can't be opened we still allow creating a NEW player (the
            # write is attempted later and reported if it fails).
            existing = []
            try:
                with registry.PlayerRegistry() as reg:
                    existing = reg.list_players()
            except Exception as e:
                existing = []
                self.status_msg = f"registry read failed ({e}); you can still add a new name"

            dlg = _AssignPlayerDialog(self, existing, current_tid=tid)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return
            choice = dlg.result_value()
            if choice is None:
                return

            # Resolve the choice to a registry uuid + display name, persisting to
            # the DB. All registry mutations are wrapped: on failure we keep the
            # name in-session (so it still shows) but warn it wasn't saved.
            uuid = None
            display_name = None
            saved = False
            try:
                with registry.PlayerRegistry() as reg:
                    if choice["kind"] == "existing":
                        uuid = choice["uuid"]
                        p = reg.get_player(uuid)
                        display_name = (p or {}).get("display_name") or choice.get("name")
                    else:  # "new"
                        name = choice["name"]
                        number = choice.get("number")
                        if choice.get("is_me"):
                            uuid = reg.set_me(name, number=number)
                            # "This is me" WITH a number also configures my live
                            # jersey-exclusion number (Feature 1), persisted in the
                            # profile. Only when a valid 1..23 number was given.
                            if number is not None and 1 <= int(number) <= MAX_JERSEY:
                                self.my_number = int(number)
                                self.tmodel.my_number = self.my_number
                                self._track_numbers.clear()
                                self._sync_my_number_spin()
                        else:
                            uuid = reg.add_player(
                                display_name=name,
                                is_teammate=bool(choice.get("is_teammate", True)),
                                number=number)
                        display_name = name
                    saved = True
            except Exception as e:
                # keep whatever the dialog gave us so the in-session name still works
                if choice["kind"] == "existing":
                    uuid = choice.get("uuid")
                    display_name = choice.get("name")
                else:
                    uuid = None
                    display_name = choice.get("name")
                self.status_msg = f"registry save failed ({e}); name kept this session only"

            if not display_name:
                self.status_msg = "assign: no name chosen"
                self._refresh(); return

            # remember for THIS session keyed on the stable track id
            self.det_names[int(tid)] = (uuid, display_name)
            if saved:
                self.status_msg = (f'assigned track {tid} -> "{display_name}" '
                                   f"(saved to registry)")
            self._refresh()

        # ---- target id ----
        def set_target_id(self, tid):
            self.store.target = int(tid)
            self.status_msg = f"target id = {self.store.target}"
            self._refresh()

        # ---- re-anchor ops (Phase 5) ----
        def do_merge(self, dst):
            if self.armed is None:
                self.status_msg = "no id armed (click a box first)"
            else:
                n = self.store.merge_forward(self.armed, dst, self.frame)
                self.status_msg = f"merged id {self.armed} -> {dst} ({n} rows fwd)"
            self._refresh()

        def do_delete_forward(self):
            if self.armed is None:
                self.status_msg = "no id armed"
            elif self.armed == self.store.target and not self._confirm(
                    "Delete your own target ID forward?"):
                self.status_msg = "delete cancelled"
            else:
                n = self.store.delete_forward(self.armed, self.frame)
                self.status_msg = f"deleted id {self.armed} from frame {self.frame} ({n} rows)"
            self._refresh()

        def toggle_ruleout_mode(self):
            """RULE-OUT MODE ('n' / the panel button). When ON, LEFT-CLICKING a
            player marks them NOT me: their box turns RED, the track is excluded
            from selection, and it feeds the identity model a HARD NEGATIVE. Stays
            on so you can rule out several; toggle again to leave."""
            self.ruleout_mode = not self.ruleout_mode
            self.status_msg = ("RULE-OUT mode ON - click players to mark NOT me "
                               "(they turn red); n/button again to stop"
                               if self.ruleout_mode else "rule-out mode OFF")
            self._sync_buttons()
            self._refresh()

        def _rule_out_box(self, box, tid):
            """Mark `box` (track id `tid`) as NOT me: hard negative + add to the
            session ruled_out set (excluded from eligibility, drawn red)."""
            if box is None:
                return
            try:
                self.idmodel.add_negative(
                    self._identity_features(self.frame, box), weight=ID_HARD_NEG)
            except Exception:
                pass
            if tid is not None:
                self.ruled_out.add(int(tid))
                self.status_msg = (f"ruled out id {int(tid)} (not me) - "
                                   f"{len(self.ruled_out)} ruled out")
            else:
                self.status_msg = "ruled out detection (not me)"
            self._refresh()

        def do_rule_out(self):
            """Rule out the currently ARMED detection (legacy path / bulk use).
            The primary UX is now toggle_ruleout_mode + click."""
            if self.armed_box is None:
                self.status_msg = ("rule out: turn on RULE-OUT mode (n) and click a "
                                   "player, or click a detection to arm it first")
                self._refresh(); return
            tid = self.armed if self.armed is not None else \
                self._det_track_hint(self.armed_box)
            self._rule_out_box(self.armed_box, tid)
            self.armed = None
            self.armed_box = None
            self._refresh()

        def clean_jumps(self, window_s=CLEAN_WINDOW_S):
            """SMART-CLEAN: remove teleport/outlier frames from the target track.

            Collects the target id's (frame, centre, box) rows, scopes to the
            last `window_s` seconds up to the current frame, builds a robust
            median-smoothed trajectory and removes frames whose centre deviates
            from it by more than max(MIN_JUMP_PX, JUMP_K * median box larger-side)
            -- the ROI-snapped-to-ref teleports / short bad runs. All removed
            frames go in as ONE undoable op (a single 'u' restores them).
            """
            tid = self.store.target
            rows = []
            for fr, rid in self.store.by_id.get(tid, []):
                d = self.store.records[rid]
                box = (d["x1"], d["y1"], d["x2"], d["y2"])
                rows.append((fr, bbox.center(box), box))
            rows.sort(key=lambda r: r[0])
            if not rows:
                self.status_msg = "no target track to clean"
                self._refresh(); return
            # scope to the last window_s seconds up to the current frame
            lo = self.frame - window_s * self.fps
            scoped = [r for r in rows if lo <= r[0] <= self.frame]
            if len(scoped) < 5:
                scoped = rows          # too little in scope -> clean whole track
            outliers = find_jump_frames(scoped)
            n = self.store.remove_frames(outliers, tid=tid)
            if n == 0:
                self.status_msg = "no jumps found"
            else:
                self.status_msg = f"cleaned {n} jump frames (last {window_s:.0f}s)"
            self._refresh()

        # -----------------------------------------------------------------
        # SAM 3 BACKGROUND RE-TRACK (loose-mark live, SAM3 re-tracks behind)
        # -----------------------------------------------------------------
        def mark_sam3_start(self):
            """Store the current frame as the SAM3 re-track IN-point (W)."""
            self._sam3_start = self.frame
            self.status_msg = (f"SAM3 start marked @ frame {self.frame} - scrub to "
                               f"the end, then 'SAM3 re-track -> here'")
            self._sync_sam3_label()
            self._refresh()

        def _sam3_seed_box(self, start):
            """Seed box for SAM3 on the window's FIRST frame: the target box at
            `start` if present, else the current ROI, else the target box at the
            current frame. None if nothing usable."""
            tb = self.store.target_box_at(start)
            if tb is not None:
                return tb
            if self.roi is not None:
                return tuple(self.roi)
            return self.store.target_box_at(self.frame)

        def run_sam3_retrack(self):
            """Launch apps/sam3_segment.py in a SEPARATE PROCESS (QProcess) to
            re-track the window [start .. current frame] and merge the result.

            Window = marked start (else last ~3s), capped at SAM3_MAX_WINDOW_S.
            Non-blocking: progress is parsed from STDERR; the merge happens in
            _on_sam3_finished. torch never loads in THIS process."""
            from PySide6 import QtCore
            if self._sam3_proc is not None:
                self.status_msg = "SAM3 already running - Cancel it first"
                self._refresh(); return
            max_frames = max(1, int(round(SAM3_MAX_WINDOW_S * self.fps)))
            start, n = sam3_window(self.frame, self._sam3_start, self.fps, max_frames)
            seed = self._sam3_seed_box(start)
            if seed is None:
                self.status_msg = ("SAM3: no seed box (mark/lock the target at the "
                                   "window start first)")
                self._refresh(); return
            # warn (but proceed) if the user asked for a very long window
            req = self.frame - (self._sam3_start if self._sam3_start is not None
                                else start) + 1
            warn = ""
            if self._sam3_start is not None and req > max_frames:
                warn = (f" (capped from {req} to {n} frames; "
                        f"max {SAM3_MAX_WINDOW_S:.0f}s)")
            # script + venv python (this app already runs under the venv)
            script = str(pathlib.Path(__file__).resolve().parent / "sam3_segment.py")
            out = (ProjectPaths().output /
                   f"sam3_seg_{start}_{n}.json")
            x1, y1, x2, y2 = seed
            args = [script, "--video", str(self.video_path),
                    "--start", str(int(start)), "--n", str(int(n)),
                    "--box", str(float(x1)), str(float(y1)),
                    str(float(x2)), str(float(y2)),
                    "--out", str(out)]
            proc = QtCore.QProcess(self)
            self._sam3_proc = proc
            self._sam3_out = out
            self._sam3_window = (start, n)
            self._sam3_stderr_buf = ""
            proc.readyReadStandardError.connect(self._on_sam3_stderr)
            proc.finished.connect(self._on_sam3_finished)
            proc.errorOccurred.connect(self._on_sam3_error)
            proc.start(sys.executable, args)
            self.set_playing(False)
            self._sam3_set_running(True)
            est_min = n * 2.0 / 60.0
            self.status_msg = (f"SAM3 re-tracking frames {start}..{start + n - 1} "
                               f"({n} frames, ~{est_min:.1f} min){warn}")
            self._sync_sam3_label(f"SAM3 starting 0/{n}...")
            self._refresh()

        def _on_sam3_stderr(self):
            """Parse `PROGRESS i/n` lines from the runner's STDERR -> live label."""
            proc = self._sam3_proc
            if proc is None:
                return
            chunk = bytes(proc.readAllStandardError()).decode("utf-8", "replace")
            self._sam3_stderr_buf += chunk
            last = None
            for line in self._sam3_stderr_buf.splitlines():
                line = line.strip()
                if line.startswith("PROGRESS "):
                    last = line[len("PROGRESS "):]
            # keep only a short tail so the buffer doesn't grow unbounded
            self._sam3_stderr_buf = self._sam3_stderr_buf[-4096:]
            if last is not None:
                self._sync_sam3_label(f"SAM3 re-tracking {last}...")

        def _on_sam3_error(self, _err):
            """QProcess failed to start / crashed (caught separately from a clean
            non-zero exit, which `finished` handles)."""
            if self._sam3_proc is None:
                return
            self._sam3_set_running(False)
            self.status_msg = "SAM3 failed to start (separate-process launch error)"
            self._sync_sam3_label("SAM3: failed to start")
            self._sam3_cleanup()
            self._refresh()

        def _on_sam3_finished(self, exit_code, _status):
            """On clean exit (0) load the JSON and MERGE it as one undoable op.
            Exit 2 = the gated SAM3 model is missing -> friendly status."""
            proc = self._sam3_proc
            if proc is None:
                return
            out = self._sam3_out
            self._sam3_set_running(False)
            if exit_code == 2:
                self.status_msg = "SAM3 model missing - see models/sam3.pt"
                self._sync_sam3_label("SAM3: model missing (models/sam3.pt)")
                self._sam3_cleanup(); self._refresh(); return
            if exit_code != 0:
                self.status_msg = f"SAM3 re-track failed (exit {exit_code})"
                self._sync_sam3_label(f"SAM3: failed (exit {exit_code})")
                self._sam3_cleanup(); self._refresh(); return
            n_merged = self._merge_sam3_result(out)
            self._sam3_cleanup()
            if n_merged <= 0:
                self.status_msg = "SAM3 finished but produced no boxes to merge"
                self._sync_sam3_label("SAM3: nothing to merge")
            else:
                self.status_msg = f"SAM3 merged {n_merged} frames (u to undo)"
                self._sync_sam3_label(f"SAM3: merged {n_merged} frames")
            self._refresh()

        def _merge_sam3_result(self, out):
            """Load {frame: [x1,y1,x2,y2]} (1-based keys) and upsert each box onto
            the target track as ONE undoable op. Clamps to the frame, only touches
            the windowed frames, sets the jump gate at the last merged frame."""
            import json
            try:
                with open(out, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                self.status_msg = f"SAM3: could not read result ({e})"
                return 0
            items = []
            for k, v in data.items():
                try:
                    fr = int(k)
                    box = clamp_box((float(v[0]), float(v[1]),
                                     float(v[2]), float(v[3])), self.fw, self.fh)
                except (ValueError, TypeError, IndexError):
                    continue
                items.append((fr, box))
            if not items:
                return 0
            items.sort(key=lambda it: it[0])
            n = self.store.set_target_boxes(items, SOURCE_SAM2)
            # seed the jump gate at the LAST merged frame (mirrors a confirmation)
            last_fr, last_box = items[-1]
            self._mark_target_good(last_fr, last_box)
            return n

        def cancel_sam3(self):
            """Kill the running SAM3 background re-track (no merge)."""
            proc = self._sam3_proc
            if proc is None:
                self.status_msg = "SAM3: nothing running"
                self._refresh(); return
            try:
                proc.kill()
            except Exception:
                pass
            self._sam3_set_running(False)
            self.status_msg = "SAM3 re-track cancelled"
            self._sync_sam3_label("SAM3: cancelled")
            self._sam3_cleanup()
            self._refresh()

        def _sam3_set_running(self, running):
            """Enable/disable the SAM3 panel buttons for the running state."""
            btn = getattr(self, "btn_sam3_run", None)
            if btn is not None:
                btn.setEnabled(not running)
            btn = getattr(self, "btn_sam3_mark", None)
            if btn is not None:
                btn.setEnabled(not running)
            btn = getattr(self, "btn_sam3_cancel", None)
            if btn is not None:
                btn.setEnabled(running)

        def _sam3_cleanup(self):
            """Drop the QProcess handle + delete the temp JSON it wrote."""
            out = self._sam3_out
            self._sam3_proc = None
            self._sam3_out = None
            self._sam3_window = None
            self._sam3_stderr_buf = ""
            if out is not None:
                try:
                    pathlib.Path(out).unlink()
                except OSError:
                    pass

        def _sync_sam3_label(self, text=None):
            """Update the SAM3 progress label (idle / marked / live i/n / done)."""
            lbl = getattr(self, "sam3_lbl", None)
            if lbl is None:
                return
            if text is not None:
                lbl.setText(text)
            elif self._sam3_start is not None:
                lbl.setText(f"SAM3: start @ {self._sam3_start}")
            else:
                lbl.setText("SAM3: idle")

        def do_nuke(self):
            if self.armed is None:
                self.status_msg = "no id armed"
            elif self.armed == self.store.target and not self._confirm(
                    "Nuke your own target ID entirely?"):
                self.status_msg = "nuke cancelled"
            else:
                n = self.store.nuke(self.armed)
                self.status_msg = f"nuked id {self.armed} ({n} rows)"
            self._refresh()

        def do_bulk_dialog(self):
            self.set_playing(False)
            dlg = ManageIdsDialog(self.store, self)
            dlg.exec()
            if dlg.applied:
                n = self.store.bulk_delete(dlg.removed_ids)
                self.status_msg = (f"removed {len(dlg.removed_ids)} ids "
                                   f"({n} rows) as one undo")
            else:
                self.status_msg = "manage-ids cancelled"
            self._refresh()

        def do_undo(self):
            ok = self.store.undo()
            self.status_msg = "undo" if ok else "nothing to undo"
            self._refresh()

        def reset_target_model(self):
            self.tmodel.reset()
            self.status_msg = "online target model reset (forgot motion+appearance)"
            self._refresh()

        def save_player_profile(self):
            """Persist the appearance profile NOW (dedicated 'p' key) so the carry-
            over accumulates without needing a full track save."""
            try:
                self.tmodel.save_profile(self.profile_path)
                self.idmodel.save(self.id_path)
                app_n, _ = self.tmodel.learned()
                ip, ineg = self.idmodel.counts()
                self.status_msg = (f"player profile saved ({app_n} crops; "
                                   f"id {ip}+/{ineg}-)")
            except Exception as e:                       # pragma: no cover
                self.status_msg = f"profile save failed: {e}"
            self._refresh()

        def clear_player_profile(self):
            """Clear the PERSISTENT player profile (dedicated 'K' key): empties
            player_profile.json AND resets the in-session appearance mean. This is
            distinct from 'L' (session-only model reset, leaves the file intact)."""
            try:
                if self.profile_path.exists():
                    self.profile_path.unlink()
            except Exception as e:                       # pragma: no cover
                self.status_msg = f"could not delete profile: {e}"
                self._refresh()
                return
            # reset appearance (and motion - we are starting the player over)
            self.tmodel.reset()
            self.status_msg = "persistent player profile CLEARED"
            self._refresh()

        def do_save(self):
            # PERSIST CONFIDENCE (Feature 2): attach each row's per-frame
            # confidence under "confidence" so write_tracks records the column.
            # Default 1.0 when this frame has no rating (e.g. conf_by_frame empty
            # / scrubbed frames) so saving never breaks.
            rows = self.store.to_rows()
            for row in rows:
                row["confidence"] = float(
                    self.conf_by_frame.get(row["frame"], 1.0))
            out = write_tracks(self.tracks_path, rows)
            # also persist the appearance profile so it accumulates across clips
            try:
                self.tmodel.save_profile(self.profile_path)
            except Exception as e:                       # pragma: no cover
                print(f"[track] WARNING could not save player profile: {e}")
            # persist the ONLINE IDENTITY classifier alongside it (sibling file)
            try:
                self.idmodel.save(self.id_path)
            except Exception as e:                       # pragma: no cover
                print(f"[track] WARNING could not save identity model: {e}")
            n = len(rows)
            self.status_msg = f"saved -> {out}"
            self._refresh()
            msg = f"Target track saved:\n{out}"
            if n is not None:
                msg += f"\n\n{n} frames recorded."
            QtWidgets.QMessageBox.information(self, "Saved", msg)

        def open_clip(self):
            """Load another clip of the same game (multi-clip workflow).

            Offers to save the current track first, then picks a video and
            RELAUNCHES the tracker on it detached, closing this window. The
            shared calibration (game.calibration.json) + the on-disk
            player_profile.json carry over automatically, so the new clip opens
            with the same calibration + learned appearance.
            """
            if self.store.records:
                ans = QtWidgets.QMessageBox.question(
                    self, "Open clip",
                    "Save the current track before switching clips?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    | QtWidgets.QMessageBox.Cancel)
                if ans == QtWidgets.QMessageBox.Cancel:
                    return
                if ans == QtWidgets.QMessageBox.Yes:
                    self.do_save()
            start_dir = str(ProjectPaths().input)
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open clip", start_dir,
                "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)")
            if not chosen:
                return
            app = pathlib.Path(__file__).resolve()
            relaunch_args = [str(app), "--video", chosen]
            # MULTI-PLAYER: carry the player name across clips so the new clip
            # opens with this player's profile / track / outputs.
            if self.player:
                relaunch_args += ["--player", self.player]
            # preserve the current target id + the detection-pool mode across the
            # relaunch (otherwise the new clip resets to id 1 / ALL detections).
            relaunch_args += ["--id", str(self.store.target)]
            if self.kept_only:
                relaunch_args.append("--kept-only")
            # carry MY jersey number across the relaunch (also persisted in the
            # profile, but pass it so it is live immediately on the new process).
            if self.my_number is not None:
                relaunch_args += ["--my-number", str(self.my_number)]
            QtCore.QProcess.startDetached(sys.executable, relaunch_args)
            self.close()

        def _confirm(self, text):
            return QtWidgets.QMessageBox.question(self, "Confirm", text) \
                == QtWidgets.QMessageBox.Yes

        # ---- rendering ----
        def _show_frame(self):
            # keep the per-frame team-label cache bounded during long playback
            if len(self._team_cache) > 600:
                self._team_cache.clear()
            # conf_by_frame is the per-frame confidence (a single float per tracked
            # frame). It must NOT be cleared during playback: do_save() reads it to
            # persist the rating each frame was tracked at, so clearing it silently
            # loses real ratings and writes the default 1.0 instead. A float per
            # frame for a whole game is trivial memory, so it is allowed to grow.
            img = self.reader.frame(self.frame)
            if img is not None:
                self._cur_bgr = img
                self._cur_bgr_frame = self.frame
                rgb = img[:, :, ::-1].copy()             # BGR -> RGB (contiguous)
                h, w = rgb.shape[:2]
                qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
                self.view.set_image(qimg.copy())
            # PER-FRAME CMC: advance the live pitch homography to this displayed
            # frame (sequential-only; jumps mark it stale). Inert if no calibration.
            self._update_live_homography()
            self._sync_slider()
            self._refresh_det_list()
            self._refresh()

        def _sync_slider(self):
            self.slider.blockSignals(True)
            self.slider.setValue(self.frame)
            self.slider.blockSignals(False)

        def _refresh_det_list(self):
            self.det_list.clear()
            # OBJECTS OVERLAY: list the tracked players present this frame, grouped
            # /sorted by candidate status (me-likely, possible, then ruled-out),
            # each row like  #23  me-likely  (your team; central). Ruled-out rows
            # are listed too (greyed) but the boxes are hidden unless the toggle.
            if self.has_objects_overlay():
                order = {"me-likely": 0, "possible": 1, "ruled-out": 2}
                items = []
                for oid, box, team, conf in self.objects_at(self.frame):
                    v = self._obj_verdicts.get(oid)
                    status = v.status if v is not None else "possible"
                    reasons = v.reasons if v is not None else []
                    items.append((order.get(status, 1), oid, box, team,
                                  conf, status, reasons))
                items.sort(key=lambda t: (t[0], t[1]))
                for _o, oid, box, team, conf, status, reasons in items:
                    inside = (self.roi is not None
                              and bbox.contains(self.roi, *bbox.center(box)))
                    tag = " [ROI]" if inside else ""
                    why = f"  ({'; '.join(reasons)})" if reasons else ""
                    name_txt = ""
                    if oid in self.det_names:
                        name_txt = f'  "{self.det_names[oid][1]}"'
                    it = QtWidgets.QListWidgetItem(
                        f"#{oid}  {status}{name_txt}{why}{tag}")
                    it.setData(Qt.UserRole, ("obj", oid))
                    if status == "me-likely":
                        it.setForeground(QtGui.QColor(0, 230, 0))
                    elif status == "possible":
                        it.setForeground(QtGui.QColor(0, 200, 200))
                    else:
                        it.setForeground(QtGui.QColor(150, 150, 150))   # ruled-out
                    self.det_list.addItem(it)
                return
            labels = self.team_labels_for_frame(self.frame)
            visible = self.visible_det_indices(self.frame)
            for i, d in enumerate(self.dets_at(self.frame)):
                if i not in visible:
                    continue
                cx, cy = bbox.center(d[:4])
                inside = self.roi is not None and bbox.contains(self.roi, cx, cy)
                tag = " [ROI]" if inside else ""
                name, is_tracked, confident = (labels[i] if i < len(labels)
                                               else (None, None, False))
                # "#i  <Team>  conf=.."  (team omitted when feature inert)
                if name is None:
                    team_txt = ""
                    fg = None
                elif not confident:
                    team_txt = "  unsure"
                    fg = QtGui.QColor(170, 170, 170)        # grey
                else:
                    team_txt = f"  {name}"
                    fg = (QtGui.QColor(0, 200, 200) if is_tracked      # teal/cyan
                          else QtGui.QColor(255, 150, 40))             # orange
                # ASSIGNED PLAYER NAME (from the persistent track_id->player map):
                # look up by the detection's stable track id so the name survives
                # the per-frame rebuild. Shown as e.g.  #2 blue&gold "Pou" conf=..
                name_txt = ""
                tid = self._det_track_hint(d[:4])
                if tid is not None and tid in self.det_names:
                    name_txt = f'  "{self.det_names[tid][1]}"'
                it = QtWidgets.QListWidgetItem(
                    f"#{i}{team_txt}{name_txt}  conf={d[4]:.2f}{tag}")
                it.setData(Qt.UserRole, i)
                if fg is not None:
                    it.setForeground(fg)
                self.det_list.addItem(it)

        def _refresh(self):
            self.view.update()
            t = self.frame / self.fps if self.fps else 0
            roi_state = "ON" if self.roi_on else ("set" if self.roi else "none")
            armed = self.armed if self.armed is not None else "-"
            buf = ""
            if self.input_mode == "target" and self.input_buf:
                buf = f"  target>{self.input_buf}_"
            elif self.input_mode == "merge" and self.input_buf:
                buf = f"  merge>{self.input_buf}_"
            zoom_txt = f"zoom {self.view.vt.zoom:4.1f}x" if self.view.vt.zoom > 1.001 else "zoom  fit"
            spot_txt = "arrow ON" if self.spotlight_on else "arrow OFF"
            if self.adaptive_marker_on:
                spot_txt += " | adapt ON"
            app_n, mot_n = self.tmodel.learned()
            learned_txt = f"learned {app_n}f/{mot_n}m"
            view_bits = []
            if self.hide_other_teams:
                view_bits.append("team-only")
            if self.hide_off_field:
                view_bits.append("on-field")
            view_txt = f"   view: {'+'.join(view_bits)}" if view_bits else ""
            player_txt = f"player:{self.player}   " if self.player else ""
            # MANUAL stands out; otherwise show whether loss-recovery or HOLD is active.
            mode_txt = ("** MANUAL **" if self.manual_mode
                        else ("recover" if self.recovery_on else "HOLD"))
            # per-frame CONFIDENCE rating (rating of THIS frame if known, else the
            # last computed value); flag auto-pause-on-low when armed.
            conf_val = self.conf_by_frame.get(self.frame, self.last_confidence)
            conf_txt = "conf  --" if conf_val is None else f"conf {conf_val:.2f}"
            if self.autopause_low:
                conf_txt += " AP"
            # ONLINE IDENTITY: once ready, show the classifier's prob for the
            # current target box as "id NN%" (+ how many were ruled out).
            if self.idmodel.ready():
                tbox = self.store.target_box_at(self.frame)
                if tbox is not None:
                    idp = self.idmodel.prob(self._identity_features(self.frame, tbox))
                    conf_txt += f"   id {idp * 100:.0f}%"
                if self.ruled_out:
                    conf_txt += f" (ruled out {len(self.ruled_out)})"
            # JERSEY-NUMBER OCR (shown only when ON): the learned target number +
            # the most recent read. Empty string when OFF (feature fully inert).
            if self.ocr_on:
                num = self.target_number if self.target_number is not None else "?"
                last = "" if self._ocr_last is None else f"/{self._ocr_last[0]}"
                conf_txt += f"   target #{num}{last}"
            # MY JERSEY NUMBER (Feature 1): shown whenever configured. A trailing
            # '~' flags that the live exclusion is INERT because tesseract is
            # missing (number kept, just not enforced).
            if self.my_number is not None:
                live = "" if ocr.available() else "~"
                conf_txt += f"   my #{self.my_number}{live}"
            # CONFIDENCE-ADAPTIVE SPEED: show the conf-driven multiplier when ON,
            # otherwise the manual speed.
            if self.conf_speed:
                speed_txt = f"speed conf {self._effective_speed():g}x"
            else:
                speed_txt = f"speed {self.speed:g}x"
            txt = (f" {player_txt}frame {self.frame}/{self.n_frames} ({t:5.1f}s)   "
                   f"target {self.store.target}   merges {self.store.merges}   "
                   f"deletes {self.store.deletes}   armed {armed}   ROI {roi_state}   "
                   f"{mode_txt}   {conf_txt}   "
                   f"{learned_txt}   {zoom_txt}   {speed_txt}   "
                   f"{spot_txt}{view_txt}{buf}")
            pitch_txt = self._live_health_text()
            if pitch_txt is not None:
                txt += f"   {pitch_txt}"
            if self.status_msg:
                txt += f"    [{self.status_msg}]"
            self.status_lbl.setText(txt)
            # keep the left CONTROLS panel's checkable buttons + labels in sync
            # (e.g. when a hotkey toggled the state); cheap, runs once per frame.
            self._sync_buttons()

        # ---- keyboard ----
        def keyPressEvent(self, ev):
            key = ev.key()
            text = ev.text()
            self.status_msg = ""

            # typed-digit input modes (target id > 9, or merge destination)
            if self.input_mode is not None:
                if key in (Qt.Key_Return, Qt.Key_Enter):
                    if self.input_buf:
                        val = int(self.input_buf)
                        if self.input_mode == "target":
                            self.set_target_id(val)
                        else:                 # merge
                            self.do_merge(val)
                    self.input_buf = ""; self.input_mode = None
                    self._refresh(); return
                if key == Qt.Key_Escape:
                    self.input_buf = ""; self.input_mode = None
                    self._refresh(); return
                if key == Qt.Key_Backspace:
                    self.input_buf = self.input_buf[:-1]; self._refresh(); return
                if text.isdigit():
                    self.input_buf += text; self._refresh(); return
                # fall through for non-digit keys

            if key == Qt.Key_Space:
                # First (non-repeat) press arms Space-as-pan-modifier AND toggles
                # play, matching the original space=play/pause behaviour. While
                # held, left-drag pans (handled in VideoView).
                if not ev.isAutoRepeat():
                    self.view._space_down = True
                    self.toggle_play()
            elif key == Qt.Key_0:
                self.reset_zoom()
            elif key == Qt.Key_Left:
                if self.view._space_down and self.view.vt.zoom > 1.0:
                    self.view.vt.pan_by_widget(80, 0); self._refresh()
                else:
                    self.step(-1)
            elif key == Qt.Key_Right:
                if self.view._space_down and self.view.vt.zoom > 1.0:
                    self.view.vt.pan_by_widget(-80, 0); self._refresh()
                else:
                    self.step(1)
            elif key == Qt.Key_Up and self.view._space_down and self.view.vt.zoom > 1.0:
                self.view.vt.pan_by_widget(0, 80); self._refresh()
            elif key == Qt.Key_Down and self.view._space_down and self.view.vt.zoom > 1.0:
                self.view.vt.pan_by_widget(0, -80); self._refresh()
            elif text == ",":
                self.step(-int(round(self.fps)))
            elif text == ".":
                self.step(int(round(self.fps)))
            elif text == "[":
                self.step(-int(round(self.fps * 10)))
            elif text == "]":
                self.step(int(round(self.fps * 10)))
            elif text in "123456789":
                self.set_target_id(int(text))
            elif text == "t":
                self.input_mode = "target"; self.input_buf = ""
                self.status_msg = "type target id, Enter"; self._refresh()
            elif key in (Qt.Key_Return, Qt.Key_Enter):
                # Enter with an armed id begins a merge-destination entry
                if self.armed is not None:
                    self.input_mode = "merge"; self.input_buf = ""
                    self.status_msg = (f"merge {self.armed}-> type dest id + Enter "
                                       f"(blank=target)")
                    self._refresh()
            elif text == "i":
                self.do_bulk_dialog()
            elif text == "d":
                self.do_delete_forward()
            elif text == "D":
                self.do_nuke()
            elif text == "u":
                self.do_undo()
            elif text == "c":
                self.clean_jumps()
            elif text == "s":
                self.do_save()
            elif text == "r":
                self.toggle_roi_tracking()
            elif text == "R":
                self.reset_roi_to_target()
            elif text == "L":
                self.reset_target_model()
            elif text == "p":
                self.save_player_profile()
            elif text == "K":
                self.clear_player_profile()
            elif text in ("+", "="):
                if self.spotlight_on:
                    self.resize_spotlight(True)
                else:
                    self.resize_roi(True)
            elif text in ("-", "_"):
                if self.spotlight_on:
                    self.resize_spotlight(False)
                else:
                    self.resize_roi(False)
            elif text == "<" or key == Qt.Key_Less:
                self.change_speed(-1)
            elif text == ">" or key == Qt.Key_Greater:
                self.change_speed(1)
            elif text == "o":
                self.toggle_spotlight()
            elif text == "b":
                self.toggle_adaptive_marker()
            elif text == "e":
                self.toggle_conf_speed()
            elif text == "h":
                self.toggle_hide_other_teams()
            elif text == "g":
                self.toggle_hide_off_field()
            elif text == "x":
                self.toggle_focus()
            elif text == "z":
                self.toggle_circles()
            elif text == "M":
                self.toggle_minimap()
            elif text == "m":
                # HOLD-to-manual: pressing m ENTERS manual (ROI follows the cursor);
                # RELEASING m (keyReleaseEvent) returns to auto AND snaps to the
                # nearest id. Ignore key auto-repeat; only enter if not already manual.
                if not ev.isAutoRepeat() and not self.manual_mode:
                    self.toggle_manual_mode()
            elif text == "v":
                self.toggle_recovery()
            elif text == "a":
                self.toggle_autopause_low()
            elif text == "j":
                self.toggle_ocr()
            elif text == "n":
                self.toggle_ruleout_mode()
            elif text == "f":
                self.toggle_sam2()
            elif text == "w":
                self.mark_sam3_start()
            elif text == "W":
                self.run_sam3_retrack()
            elif text == "q":
                self.close()
            elif key == Qt.Key_Escape:
                self.armed = None; self.armed_box = None
                self.input_buf = ""; self.input_mode = None
                self.status_msg = "cleared"; self._refresh()
            else:
                super().keyPressEvent(ev)

        def keyReleaseEvent(self, ev):
            if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
                self.view._space_down = False
                # if a pan was in progress, end it cleanly
                if self.view._mode == "pan":
                    self.view._mode = None
                    self.view._pan_anchor = None
                    self.view.unsetCursor()
            # HOLD-to-manual release: leaving manual snaps to the nearest id (the
            # exit branch of toggle_manual_mode does the snap). Ignore auto-repeat.
            if (ev.key() == Qt.Key_M and not ev.isAutoRepeat()
                    and self.manual_mode):
                self.toggle_manual_mode()
            super().keyReleaseEvent(ev)

        def closeEvent(self, ev):
            self.set_playing(False)
            try:
                self.reader.release()
            except Exception:
                pass
            super().closeEvent(ev)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    # WINDOW GEOMETRY: size + centre the window inside the AVAILABLE screen area
    # (availableGeometry EXCLUDES the taskbar) so the window never spills off the
    # sides or sits under the taskbar. Inset by a margin and clamp to what fits.
    # Robust if there is no primary screen (headless / odd platforms): fall back
    # to the old fixed resize.
    screen = QtWidgets.QApplication.primaryScreen()
    if screen is not None:
        avail = screen.availableGeometry()
        margin = 24
        w = min(1280, avail.width() - 2 * margin)
        h = min(900, avail.height() - 2 * margin)
        # never demand more than the available area for the minimum size
        win.setMinimumSize(min(900, w), min(640, h))
        win.resize(w, h)
        # centre within the available area (its top-left already clears the taskbar)
        x = avail.x() + (avail.width() - w) // 2
        y = avail.y() + (avail.height() - h) // 2
        win.move(x, y)
    else:
        win.resize(1280, 820)
    # open MAXIMIZED (fills the available screen, still respects the taskbar) so
    # the video and the right-hand detections dock both get maximum room. The
    # resize/move above set the restored (un-maximize) geometry.
    win.showMaximized()
    # give the detections dock a comfortably wide width so IDs/names never crowd
    try:
        _dw = max(360, win.dock.minimumWidth())
        win.resizeDocks([win.dock], [_dw], QtCore.Qt.Horizontal)
    except Exception:
        pass
    # FIX: default the video view to fit-the-whole-frame, CENTRED. We reset AFTER
    # show() so the transform uses the real post-layout viewport size (with the
    # left/right docks already claiming their width), making the picture sit
    # fixed in the middle on launch rather than off-centre / zoomed.
    QtCore.QTimer.singleShot(0, win.reset_zoom)
    print(__doc__.split("Usage:")[0])
    print(f"[track] {win.n_frames} frames @ {win.fps:.2f}fps "
          f"({win.fw}x{win.fh}) | target id = {store.target} | "
          f"detections frames loaded: {len(det_by_frame)}")
    app.exec()


# ===========================================================================
# CLI
# ===========================================================================
def main():
    # The help banner / status lines contain unicode (arrows, etc.). When stdout
    # is redirected to a file (e.g. launched detached) Windows defaults it to
    # cp1252, which can't encode them and crashes the app on the first print.
    # Force UTF-8 with replacement so console output never takes the app down.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Interactive single-player CSRT ROI tracker + re-anchor toolkit")
    ap.add_argument("--video", help="source video (default: first ProjectPaths().videos())")
    ap.add_argument("--id", type=int, default=1, help="canonical target track id (default 1)")
    ap.add_argument("--player", default=None,
                    help="player name for MULTI-PLAYER mode: each player gets their own "
                         "profile (output/player_<name>.json), track CSV and outputs "
                         "(<stem>.<name>.tracks.csv etc.). Omit for single-player mode "
                         "(behaviour identical to before).")
    ap.add_argument("--calibration", default=None,
                    help="explicit calibration json to use; otherwise this clip's own "
                         "calibration, then the shared game.calibration.json, is used")
    ap.add_argument("--kept-only", action="store_true",
                    help="only show team-filtered (kept) detections; default shows ALL "
                         "detected players so the ROI can always snap to your target")
    ap.add_argument("--my-number", type=int, default=None,
                    help="YOUR jersey number (1..23). When set AND tesseract is "
                         "installed, candidates whose detected shirt number is "
                         "confidently DIFFERENT are excluded from the target so the "
                         "tracker can't cross onto them; a candidate reading YOUR "
                         "number is preferred. Persists in the player profile.")
    ap.add_argument("--sam2-model", default=None,
                    help="SAM2 checkpoint for the 'f' silhouette/refine (default "
                         "sam2.1_l.pt large/'complex'; sam2.1_t.pt for speed)")
    ap.add_argument("--selftest", action="store_true",
                    help="run headless logic test and exit (no GUI)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    pp = ProjectPaths()
    if args.video:
        video = pathlib.Path(args.video)
    else:
        vids = pp.videos()
        if not vids:
            sys.exit("No videos found in input/ and --video not given")
        video = vids[0]
    if not video.exists():
        sys.exit(f"Video not found: {video}")

    # candidate detection pool (Phase 3 output) - optional
    det_by_frame = {}
    det_path = pp.detections(video)
    if det_path.exists():
        try:
            df = read_detections(det_path)
            det_by_frame = kept_by_frame(df) if args.kept_only else all_by_frame(df)
            mode = "kept-only" if args.kept_only else "ALL players"
            total = sum(len(v) for v in det_by_frame.values())
            print(f"[track] loaded detections: {det_path} "
                  f"({len(det_by_frame)} frames, {total} boxes, {mode})")
        except Exception as e:
            print(f"[track] WARNING could not read detections {det_path}: {e}")
    else:
        print(f"[track] no detections parquet at {det_path} "
              f"(tracking will use the CSRT box only)")

    # consolidated target track - optionally seed from an existing tracks CSV.
    # In MULTI-PLAYER mode (--player) the track namespaces to this player so each
    # person's track saves/loads separately.
    store = TargetTrack(target_id=args.id)
    tracks_path = pp.tracks(video, args.player)
    if tracks_path.exists():
        try:
            store.load_rows(read_tracks(tracks_path))
            store.undo_stack.clear()          # loaded state is the baseline
            print(f"[track] loaded existing tracks: {tracks_path} "
                  f"({len(store.records)} rows)")
        except Exception as e:
            print(f"[track] WARNING could not read tracks {tracks_path}: {e}")

    # match calibration (team colour fingerprints) - optional; enables per-detection
    # TEAM LABELS in the panel + team-coloured boxes. Missing/unreadable -> feature off.
    calib = None
    # Resolve in priority order: explicit --calibration > this clip's own
    # calibration > shared game.calibration.json. This lets a clip with no
    # calibration of its own fall back to the once-per-game calibration so the
    # same team labels carry across every clip of the match. None -> feature off.
    cal_path = pp.resolve_calibration(video, args.calibration)
    if cal_path is not None:
        try:
            from rt2.calibration import MatchCalibration
            calib = MatchCalibration.load(cal_path)
            n_fp = sum(1 for t in calib.teams if getattr(t, "fingerprint", None))
            print(f"[track] loaded calibration: {cal_path} "
                  f"({len(calib.teams)} teams, {n_fp} with fingerprints)")
        except Exception as e:
            calib = None
            print(f"[track] WARNING could not read calibration {cal_path}: {e}")
    else:
        print("[track] no calibration found (own clip or shared game) "
              "(team labels disabled)")

    run_gui(str(video), store, det_by_frame, tracks_path, calib, args.player,
            args.kept_only, args.my_number, args.sam2_model)


if __name__ == "__main__":
    main()
