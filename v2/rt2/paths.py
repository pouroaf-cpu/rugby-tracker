"""Per-video output path conventions.

For input/match-king-country-vs-x.mp4 the artifacts are:
  output/match-king-country-vs-x.calibration.json
  output/match-king-country-vs-x.detections.parquet
  output/match-king-country-vs-x.tracks.csv
  output/match-king-country-vs-x.annotated.mp4
  output/match-king-country-vs-x.clips.mp4
"""
from __future__ import annotations

from pathlib import Path

# Project root resolved from this file's location (v2/rt2/paths.py -> repo root),
# so the code is portable and carries no hardcoded user path.
DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
FPS_ASSUMED = 29.97
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class ProjectPaths:
    def __init__(self, root: Path | str = DEFAULT_PROJECT):
        self.root = Path(root)
        self.input = self.root / "input"
        self.output = self.root / "output"

    def ensure(self) -> "ProjectPaths":
        self.input.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        return self

    def videos(self) -> list[Path]:
        if not self.input.exists():
            return []
        return sorted(p for p in self.input.iterdir()
                      if p.suffix.lower() in VIDEO_EXTS)

    # --- per-video artifact paths (keyed by the video file stem) ---
    def _o(self, video: Path, suffix: str, player: str | None = None) -> Path:
        stem = Path(video).stem
        if player:
            return self.output / f"{stem}.{player}.{suffix}"
        return self.output / f"{stem}.{suffix}"

    def player_profile(self, player: str | None = None) -> Path:
        """Persistent per-PLAYER appearance profile (carries across clips).
        Default (no name) keeps the single-player file; a name namespaces it so
        different people each train their own tracker."""
        return self.output / (f"player_{player}.json" if player else "player_profile.json")

    def calibration(self, video: Path) -> Path:
        return self._o(video, "calibration.json")

    def game_calibration(self) -> Path:
        """Shared, game-level calibration reused across all clips of a match."""
        return self.output / "game.calibration.json"

    def resolve_calibration(self, video, override=None) -> "Path | None":
        """Find the calibration to use for a clip, in priority order:
        explicit override > this clip's own calibration > shared game calibration.
        Returns None if none exist. This is what lets you calibrate ONCE and run
        every clip of a game off it."""
        for cand in (override, self.calibration(video), self.game_calibration()):
            if cand and Path(cand).exists():
                return Path(cand)
        return None

    def detections(self, video: Path) -> Path:
        return self._o(video, "detections.parquet")

    def tracks(self, video: Path, player: str | None = None) -> Path:
        return self._o(video, "tracks.csv", player)

    def annotated(self, video: Path, player: str | None = None) -> Path:
        return self._o(video, "annotated.mp4", player)

    def clips(self, video: Path, player: str | None = None) -> Path:
        return self._o(video, "clips.mp4", player)
