"""Headless smoke test for the rt2 shared library."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rt2 import bbox, paths
from rt2.calibration import ColorSignature, HeightModel, TeamProfile, MatchCalibration
from rt2.detections import write_detections, read_detections, kept_by_frame
from rt2.tracks_io import write_tracks, read_tracks, SOURCE_DETECTION

tmp = Path(__file__).resolve().parent / "_tmp_selftest"
tmp.mkdir(exist_ok=True)

# --- bbox ---
a = (10, 10, 30, 30); b = (20, 20, 40, 40)
assert bbox.area(a) == 400
assert abs(bbox.iou(a, b) - (100 / 700)) < 1e-9
assert bbox.contains(a, 15, 15) and not bbox.contains(a, 35, 35)
assert bbox.nearest([a, b], 39, 39) == 1
print("bbox: OK")

# --- ColorSignature learns from synthetic navy patches ---
navy = np.zeros((20, 20, 3), np.uint8); navy[:] = (110, 40, 30)   # BGR darkish blue
patches = [navy + np.random.randint(-5, 5, navy.shape, dtype=np.int16).astype(np.uint8)
           for _ in range(8)]
sig = ColorSignature.learn(patches)
assert sig is not None, "signature failed to learn"
frac = sig.fraction(navy)
assert frac > 0.5, f"navy should match its own signature, got {frac}"
# a white patch should NOT match the navy signature much
white = np.full((20, 20, 3), 240, np.uint8)
assert sig.fraction(white) < 0.2
print(f"ColorSignature: OK (self-match {frac:.2f})")

# --- HeightModel: synthetic perspective (taller lower in frame) ---
ys = np.linspace(200, 1000, 30)
hs = -0.15 * ys + 250 + np.random.randn(30) * 3   # higher y => taller
hm = HeightModel.learn(list(zip(ys, hs)))
assert hm is not None and hm.a < 0, "slope should be negative"
assert hm.plausible(600, hm.predict(600))
assert not hm.plausible(600, hm.predict(600) + 500)
print(f"HeightModel: OK (a={hm.a:.3f} b={hm.b:.1f})")

# --- calibration round-trip ---
team = TeamProfile(name="blue&gold", track=True, primary=sig, height=hm, n_samples=8)
cal = MatchCalibration(video="chunk_028.mp4", teams=[team])
cpath = tmp / "x.calibration.json"
cal.save(cpath)
cal2 = MatchCalibration.load(cpath)
assert cal2.teams[0].name == "blue&gold" and cal2.teams[0].height.a < 0
assert cal2.tracked_teams()[0].primary.fraction(navy) > 0.5
print("MatchCalibration save/load: OK")

# --- detections parquet round-trip ---
rows = [{"frame": 1, "x1": 10, "y1": 10, "x2": 30, "y2": 60, "conf": 0.9,
         "team_score": 0.8, "kept": True},
        {"frame": 1, "x1": 90, "y1": 10, "x2": 110, "y2": 60, "conf": 0.7,
         "team_score": 0.1, "kept": False}]
dpath = tmp / "x.detections.parquet"
write_detections(dpath, rows)
df = read_detections(dpath)
kbf = kept_by_frame(df)
assert len(kbf[1]) == 1, "only one kept detection expected"
print("detections parquet: OK")

# --- tracks CSV round-trip ---
trows = [{"frame": 1, "target_id": 1, "x1": 10, "y1": 10, "x2": 30, "y2": 60,
          "source": SOURCE_DETECTION}]
tpath = tmp / "x.tracks.csv"
write_tracks(tpath, trows)
back = read_tracks(tpath)
assert back[0]["target_id"] == 1 and back[0]["source"] == "detection"
print("tracks CSV: OK")

# --- paths convention ---
pp = paths.ProjectPaths()
v = Path("input/match-king-country-vs-x.mp4")
assert pp.calibration(v).name == "match-king-country-vs-x.calibration.json"
assert pp.clips(v).name == "match-king-country-vs-x.clips.mp4"
print("paths: OK")

# cleanup
for p in tmp.iterdir():
    p.unlink()
tmp.rmdir()
print("\nALL rt2 SELF-TESTS PASSED")
