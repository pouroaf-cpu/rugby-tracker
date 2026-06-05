"""calibrate.py - Phase 2 team CALIBRATION tool (PySide6).

Learn each team's colour signatures + perspective height model by dropping
sample boxes over players in wide Veo match footage, then save a
MatchCalibration JSON next to the video (output/<stem>.calibration.json).

WORKFLOW
  1. Scrub to a frame where a player of the ACTIVE team is clearly visible.
  2. Left-click-drag a rectangle tightly around that player. The box (in
     original 1920x1080 pixel coords) is added to the active team's samples.
  3. Repeat ~15-30 times per tracked team, across DIFFERENT frames and player
     sizes (near + far) so the height model spans the pitch.
  4. Switch the ACTIVE team / add more teams as needed.
  5. Press "Learn + Save" to build and write the calibration.

VIDEO CONTROLS
  Play / Pause button        toggle playback
  Prev / Next frame buttons  step one frame (also pauses)
  Frame slider               scrub to any frame
  Frame counter              shows "frame N / total"
  Keyboard:
    Space                    play / pause
    Left / ','               previous frame
    Right / '.'              next frame
    '[' / ']'                jump -30 / +30 frames
    '0' / 'f'                reset zoom (fit frame to window at 1x)

MOUSE
  Left-click-drag on video   draw a sample box for the ACTIVE team
                             (tiny accidental drags are ignored)
  Mouse WHEEL                zoom in / out, anchored under the cursor
                             (~1.25x per notch, clamped fit .. 12x)
  Middle-mouse drag          pan the view when zoomed in
  Space + left-drag          pan the view (hold Space, then left-drag)

ZOOM / PAN
  Wheel up / down            zoom in / out under the cursor
  Middle-drag OR Space-drag  pan when zoomed in (scrollbars appear)
  '0' or 'f' key             reset zoom: fit the whole frame at 1x
  "Fit" button               reset zoom: fit the whole frame at 1x
  Sample boxes always stay in ORIGINAL frame pixel coords (scene is native
  resolution; drags are mapped via mapToScene, so zoom/pan cannot distort them).

TEAMS PANEL
  Team list                  click a row to make it the ACTIVE team
                             (new boxes are assigned to the active team)
  "track this team?" check   per-team; ALL teams with samples are learned, the
                             flag selects which team(s) to follow (calibrate the
                             opposition too so the pre-render can tell them apart)
  Add team                   append a new team
  Rename team                rename the selected team
  Remove team                delete the selected team (keeps >= 1 team)
  Sample count               shown per team in the list ("name  [track]  N box")
  Undo last box              delete the most recently dropped box (active team)
  Clear boxes                delete all boxes for the active team

LEARN + SAVE
  For each TRACKED team, every sample box is re-cropped from its source frame:
    torso_patch -> primary ColorSignature
    shorts_patch -> shorts ColorSignature
    (box_center_y, box_height) -> HeightModel
  A TeamProfile is built per team and saved as a MatchCalibration to
  ProjectPaths().calibration(video). The SAME calibration is also saved to
  ProjectPaths().game_calibration() (output\\game.calibration.json) as a shared,
  game-level calibration that every OTHER clip of this game can reuse -- a
  calibration is reusable across ALL clips of a game.
  Teams with < 12 samples warn (not block).
  After a successful Learn + Save, the tool offers to auto-run the rest of the
  pipeline: it can run pre-render (apps\\prerender.py) as a subprocess, stream
  its output in a modal dialog, and on success launch the tracker
  (apps\\track.py) detached so it opens independently.

CLI
  python apps\\calibrate.py                       # first video from ProjectPaths
  python apps\\calibrate.py --video input\\chunk_028.mp4
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import Qt, QRectF, QTimer, QPointF, QProcess
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QSlider, QPushButton, QLabel,
    QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, QCheckBox,
    QInputDialog, QMessageBox, QSizePolicy, QDialog, QPlainTextEdit,
)

from rt2.paths import ProjectPaths
from rt2.video import VideoReader
from rt2 import regions, features
from rt2.calibration import (
    ColorSignature, HeightModel, TeamProfile, MatchCalibration,
)

MIN_DRAG_PX = 6          # ignore rubber-band drags smaller than this (px, scene)
LOW_SAMPLE_WARN = 12     # warn (don't block) below this many samples per team

APPS_DIR = pathlib.Path(__file__).resolve().parent   # directory holding the pipeline scripts


class Team:
    """A team being calibrated: a name, a track flag, and its sample boxes.

    Each sample is (frame_idx_1based, (x1, y1, x2, y2)) in ORIGINAL full-res
    frame pixel coordinates.
    """

    def __init__(self, name: str, track: bool = True):
        self.name = name
        self.track = track
        self.samples: list[tuple[int, tuple[float, float, float, float]]] = []

    def add(self, frame_idx: int, box):
        self.samples.append((int(frame_idx), tuple(float(v) for v in box)))

    def __len__(self):
        return len(self.samples)


class VideoView(QGraphicsView):
    """QGraphicsView showing a frame pixmap, with a rubber-band sample drag.

    The pixmap item lives at scene (0,0) at native resolution, so scene
    coordinates ARE original-frame pixel coordinates. mapToScene() therefore
    converts mouse/display positions back to original pixels regardless of how
    the widget is scaled OR zoomed/panned. Zooming changes only the view
    transform (not the scene), so sample boxes captured via mapToScene stay in
    original frame pixels under any zoom/pan; fitInView keeps it correct on
    resize.

    Zoom: mouse wheel scales the view ~1.25x per notch, anchored under the
    cursor, clamped to [fit .. ZOOM_MAX x native].
    Pan:  middle-mouse drag, OR hold Space and left-drag (ScrollHandDrag);
    ordinary left-drag still draws a sample box, so panning never steals it.
    Reset: reset_zoom() re-fits the whole frame to the window at 1x.
    """

    ZOOM_STEP = 1.25         # view scale factor per wheel notch
    ZOOM_MAX = 12.0          # max zoom relative to the fit-to-window scale

    def __init__(self, on_box_drawn):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._on_box_drawn = on_box_drawn
        self._frame_w = 0
        self._frame_h = 0

        # Zoom/pan state. _fit_scale is the view scale (uniform) at fit-to-
        # window; current zoom relative to it must stay within [1, ZOOM_MAX].
        self._fit_scale = 1.0
        self._space_held = False
        self._panning = False        # an active hand-drag pan is in progress

        self._drag_origin: QPointF | None = None
        self._rubber = QGraphicsRectItem()
        self._rubber.setPen(QPen(QColor(0, 230, 0), 2, Qt.DashLine))
        self._rubber.setBrush(QBrush(QColor(0, 230, 0, 40)))
        self._rubber.setVisible(False)
        self._scene.addItem(self._rubber)

        self.setRenderHints(self.renderHints())
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        # Anchor wheel-zoom under the cursor; show scrollbars when zoomed in.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, frame_bgr: np.ndarray | None):
        """Display a BGR ndarray (converts to RGB, fixes QImage stride)."""
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        # cv2 is BGR; QImage.Format_RGB888 wants RGB. Make contiguous so the
        # bytesPerLine (stride) we pass is exactly w*3.
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        # .copy() detaches from the numpy buffer (which is about to be freed).
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        if (w, h) != (self._frame_w, self._frame_h):
            self._frame_w, self._frame_h = w, h
            self._scene.setSceneRect(0, 0, w, h)
            self._fit()

    def _fit(self):
        """Fit the whole frame to the window at 1x and record the fit scale."""
        if self._frame_w:
            self.fitInView(QRectF(0, 0, self._frame_w, self._frame_h),
                           Qt.KeepAspectRatio)
            # After fitInView the transform is uniform; remember its scale so
            # wheel-zoom can be clamped relative to fit-to-window.
            self._fit_scale = self.transform().m11()

    def reset_zoom(self):
        """Public reset: re-fit the frame to the window at 1x zoom."""
        self._fit()

    def _zoom_factor(self) -> float:
        """Current zoom relative to fit-to-window (1.0 == fit)."""
        if self._fit_scale <= 0:
            return 1.0
        return self.transform().m11() / self._fit_scale

    def _is_zoomed(self) -> bool:
        return self._zoom_factor() > 1.001

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-fit only when at (or below) fit-to-window; otherwise preserve the
        # user's current zoom across the resize.
        if not self._is_zoomed():
            self._fit()

    def wheelEvent(self, event):
        if not self._frame_w:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
        target = self._zoom_factor() * factor
        # Clamp total zoom to [fit (1.0) .. ZOOM_MAX].
        target = min(max(target, 1.0), self.ZOOM_MAX)
        cur = self._zoom_factor()
        if abs(target - cur) < 1e-6:
            event.accept()
            return
        self.scale(target / cur, target / cur)
        event.accept()

    def _set_pan_cursor(self):
        if self._space_held:
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.viewport().unsetCursor()

    def _clamp_scene(self, pt: QPointF) -> QPointF:
        x = min(max(pt.x(), 0.0), float(self._frame_w))
        y = min(max(pt.y(), 0.0), float(self._frame_h))
        return QPointF(x, y)

    def _start_pan(self, event):
        """Begin a hand-drag pan by synthesising a left-button press for
        ScrollHandDrag (which only pans on the left button)."""
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._panning = True
        synth = QMouseEvent(
            event.type(), event.position(), event.globalPosition(),
            Qt.LeftButton, Qt.LeftButton, event.modifiers())
        super().mousePressEvent(synth)

    def mousePressEvent(self, event):
        # Pan with middle-mouse, or with Space held + left-drag. This must be
        # checked BEFORE the box-drop path so panning never steals a box draw
        # when the user explicitly asks to pan.
        if self._frame_w and (
            event.button() == Qt.MiddleButton
            or (event.button() == Qt.LeftButton and self._space_held)
        ):
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self._start_pan(event)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._frame_w:
            # Ordinary left-drag: draw a sample box (scene/native pixel coords).
            self._drag_origin = self._clamp_scene(self.mapToScene(event.position().toPoint()))
            self._rubber.setRect(QRectF(self._drag_origin, self._drag_origin))
            self._rubber.setVisible(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None:
            cur = self._clamp_scene(self.mapToScene(event.position().toPoint()))
            self._rubber.setRect(QRectF(self._drag_origin, cur).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            # End a hand-drag pan; ScrollHandDrag listens to the left button.
            synth = QMouseEvent(
                event.type(), event.position(), event.globalPosition(),
                Qt.LeftButton, Qt.NoButton, event.modifiers())
            super().mouseReleaseEvent(synth)
            self._panning = False
            self.setDragMode(QGraphicsView.NoDrag)
            self._set_pan_cursor()
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            # Box committed in SCENE coords (mapToScene), so it is in original
            # frame pixels regardless of the current zoom/pan transform.
            cur = self._clamp_scene(self.mapToScene(event.position().toPoint()))
            rect = QRectF(self._drag_origin, cur).normalized()
            self._drag_origin = None
            self._rubber.setVisible(False)
            if rect.width() >= MIN_DRAG_PX and rect.height() >= MIN_DRAG_PX:
                box = (rect.left(), rect.top(), rect.right(), rect.bottom())
                self._on_box_drawn(box)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self._set_pan_cursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._set_pan_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)


class CalibrateWindow(QMainWindow):
    def __init__(self, video_path: pathlib.Path, paths: ProjectPaths):
        super().__init__()
        self.video_path = pathlib.Path(video_path)
        self.paths = paths
        self.reader = VideoReader(self.video_path)
        self.n_frames = self.reader.n_frames
        self.frame_idx = 1                     # 1-based, matching VideoReader

        self.teams: list[Team] = [Team("blue&gold", track=True)]
        self.active_idx = 0

        self.setWindowTitle(f"Calibrate - {self.video_path.name}")
        self._build_ui()
        self._show_frame(self.frame_idx)
        self._refresh_team_list()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_tick)
        self._playing = False

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # --- left: video + transport ---
        left = QVBoxLayout()
        self.view = VideoView(self._on_box_drawn)
        left.addWidget(self.view, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(max(1, self.n_frames))
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self._on_slider)
        left.addWidget(self.slider)

        transport = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(lambda: self._step(+1))
        self.btn_fit = QPushButton("Fit (0)")
        self.btn_fit.setToolTip("Reset zoom: fit the whole frame to the window at 1x")
        self.btn_fit.clicked.connect(self.view.reset_zoom)
        self.lbl_frame = QLabel()
        transport.addWidget(self.btn_play)
        transport.addWidget(self.btn_prev)
        transport.addWidget(self.btn_next)
        transport.addWidget(self.btn_fit)
        transport.addStretch(1)
        transport.addWidget(self.lbl_frame)
        left.addLayout(transport)
        root.addLayout(left, 3)

        # --- right: teams panel ---
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Teams</b> (click to set active)"))
        self.team_list = QListWidget()
        self.team_list.currentRowChanged.connect(self._on_active_changed)
        right.addWidget(self.team_list, 1)

        self.chk_track = QCheckBox("track this team?")
        self.chk_track.stateChanged.connect(self._on_track_toggled)
        right.addWidget(self.chk_track)

        team_btns = QHBoxLayout()
        b_add = QPushButton("Add team")
        b_add.clicked.connect(self._add_team)
        b_ren = QPushButton("Rename")
        b_ren.clicked.connect(self._rename_team)
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(self._remove_team)
        team_btns.addWidget(b_add)
        team_btns.addWidget(b_ren)
        team_btns.addWidget(b_rm)
        right.addLayout(team_btns)

        box_btns = QHBoxLayout()
        b_undo = QPushButton("Undo last box")
        b_undo.clicked.connect(self._undo_box)
        b_clear = QPushButton("Clear boxes")
        b_clear.clicked.connect(self._clear_boxes)
        box_btns.addWidget(b_undo)
        box_btns.addWidget(b_clear)
        right.addLayout(box_btns)

        self.lbl_active = QLabel()
        self.lbl_active.setWordWrap(True)
        right.addWidget(self.lbl_active)

        right.addStretch(1)
        self.btn_learn = QPushButton("Learn + Save")
        self.btn_learn.clicked.connect(self._learn_and_save)
        right.addWidget(self.btn_learn)

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        right.addWidget(self.lbl_status)

        root.addLayout(right, 1)
        self.setCentralWidget(central)
        self.resize(1280, 760)

    # ---- frame display ---------------------------------------------------
    def _show_frame(self, idx: int):
        idx = max(1, min(self.n_frames, idx))
        self.frame_idx = idx
        frame = self.reader.frame(idx)
        self.view.set_frame(frame)
        self.lbl_frame.setText(f"frame {idx} / {self.n_frames}")
        # keep slider in sync without re-triggering a seek
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

    def _step(self, delta: int):
        self._stop_play()
        self._show_frame(self.frame_idx + delta)

    def _on_slider(self, value: int):
        self._show_frame(value)

    # ---- playback --------------------------------------------------------
    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._playing = True
            self.btn_play.setText("Pause")
            interval = int(1000.0 / (self.reader.fps or 29.97))
            self._play_timer.start(max(1, interval))

    def _stop_play(self):
        if self._playing:
            self._playing = False
            self.btn_play.setText("Play")
            self._play_timer.stop()

    def _on_tick(self):
        if self.frame_idx >= self.n_frames:
            self._stop_play()
            return
        self._show_frame(self.frame_idx + 1)

    # ---- sample boxes ----------------------------------------------------
    def _active_team(self) -> Team:
        return self.teams[self.active_idx]

    def _on_box_drawn(self, box):
        team = self._active_team()
        team.add(self.frame_idx, box)
        self._refresh_team_list()
        self._refresh_active_label()
        x1, y1, x2, y2 = (round(v) for v in box)
        self._set_status(
            f"Added box to '{team.name}' on frame {self.frame_idx}: "
            f"({x1},{y1})-({x2},{y2}). {len(team)} samples."
        )

    def _undo_box(self):
        team = self._active_team()
        if not team.samples:
            self._set_status(f"'{team.name}' has no boxes to undo.")
            return
        frame_idx, _ = team.samples.pop()
        self._refresh_team_list()
        self._refresh_active_label()
        self._set_status(f"Removed last box (frame {frame_idx}) from '{team.name}'. "
                         f"{len(team)} left.")

    def _clear_boxes(self):
        team = self._active_team()
        if not team.samples:
            self._set_status(f"'{team.name}' has no boxes.")
            return
        n = len(team)
        team.samples.clear()
        self._refresh_team_list()
        self._refresh_active_label()
        self._set_status(f"Cleared {n} boxes from '{team.name}'.")

    # ---- teams panel -----------------------------------------------------
    def _refresh_team_list(self):
        self.team_list.blockSignals(True)
        self.team_list.clear()
        for i, t in enumerate(self.teams):
            tag = "track" if t.track else "skip"
            QListWidgetItem(f"{t.name}  [{tag}]  {len(t)} box", self.team_list)
        self.team_list.setCurrentRow(self.active_idx)
        self.team_list.blockSignals(False)
        self._sync_track_checkbox()

    def _sync_track_checkbox(self):
        t = self._active_team()
        self.chk_track.blockSignals(True)
        self.chk_track.setChecked(t.track)
        self.chk_track.setText(f"track '{t.name}'?")
        self.chk_track.blockSignals(False)

    def _refresh_active_label(self):
        t = self._active_team()
        self.lbl_active.setText(
            f"Active team: <b>{t.name}</b><br>Samples: {len(t)}"
        )

    def _on_active_changed(self, row: int):
        if 0 <= row < len(self.teams):
            self.active_idx = row
            self._sync_track_checkbox()
            self._refresh_active_label()

    def _on_track_toggled(self, _state):
        t = self._active_team()
        t.track = self.chk_track.isChecked()
        self._refresh_team_list()

    def _add_team(self):
        name, ok = QInputDialog.getText(self, "Add team", "Team name:")
        if not ok:
            return
        name = name.strip() or f"team{len(self.teams) + 1}"
        self.teams.append(Team(name, track=True))
        self.active_idx = len(self.teams) - 1
        self._refresh_team_list()
        self._refresh_active_label()
        self._set_status(f"Added team '{name}'.")

    def _rename_team(self):
        t = self._active_team()
        name, ok = QInputDialog.getText(self, "Rename team", "New name:", text=t.name)
        if not ok:
            return
        name = name.strip()
        if name:
            t.name = name
            self._refresh_team_list()
            self._refresh_active_label()
            self._set_status(f"Renamed to '{name}'.")

    def _remove_team(self):
        if len(self.teams) <= 1:
            self._set_status("Cannot remove the last team.")
            return
        t = self._active_team()
        if t.samples:
            ans = QMessageBox.question(
                self, "Remove team",
                f"Remove '{t.name}' and its {len(t)} boxes?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
        del self.teams[self.active_idx]
        self.active_idx = min(self.active_idx, len(self.teams) - 1)
        self._refresh_team_list()
        self._refresh_active_label()
        self._set_status(f"Removed team '{t.name}'.")

    # ---- learn + save ----------------------------------------------------
    def _learn_and_save(self):
        if not any(t.track for t in self.teams):
            self._warn("No tracked teams", "Tick 'track this team?' for at least one team to follow.")
            return
        # Learn EVERY team that has samples -- the opposition profile is needed so
        # the pre-render can tell the teams apart (nearest-team classification).
        # The track flag only selects which team(s) we actually follow.
        to_learn = [t for t in self.teams if len(t) > 0]
        if not to_learn:
            self._warn("No samples", "Drop some sample boxes on the players first.")
            return

        low = [t for t in to_learn if len(t) < LOW_SAMPLE_WARN]
        if low:
            msg = "\n".join(f"  - {t.name}: {len(t)} (< {LOW_SAMPLE_WARN})" for t in low)
            ans = QMessageBox.warning(
                self, "Few samples",
                "These teams have few samples:\n" + msg +
                "\n\nSave anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                self._set_status("Save cancelled.")
                return

        profiles = []
        notes = []
        for t in to_learn:
            profile, note = self._learn_team(t)
            profiles.append(profile)
            if note:
                notes.append(note)

        calib = MatchCalibration(video=str(self.video_path), teams=profiles)
        out_path = self.paths.calibration(self.video_path)
        try:
            calib.save(out_path)
        except Exception as exc:  # noqa: BLE001 - surface any IO error to the user
            self._warn("Save failed", str(exc))
            return

        # Also save the SAME calibration as the shared, game-level calibration so
        # every other clip of this game can reuse it (a calibration is reusable
        # across ALL clips of a game).
        game_path = self.paths.game_calibration()
        try:
            calib.save(game_path)
        except Exception as exc:  # noqa: BLE001 - surface any IO error to the user
            self._warn("Save failed", str(exc))
            return

        summary = f"Saved {len(profiles)} team(s) -> {out_path}"
        summary += f"\nAlso saved as shared GAME calibration -> {game_path}"
        if notes:
            summary += "\n" + "\n".join(notes)
        self._set_status(summary)

        # AUTO-ADVANCE: close the calibration window and open the tracker on this
        # video (no prompt). Detections come from a separate overnight mark_all /
        # pre-render pass - we don't block the hand-off on it (a full game's
        # pre-render is too long to sit through here).
        self._set_status(summary + "\nOpening tracker...")
        QProcess.startDetached(
            sys.executable,
            [str(APPS_DIR / "track.py"), "--video", str(self.video_path)])
        self.close()

    # ---- auto-advance: pre-render -> tracker -----------------------------
    def _run_prerender(self):
        """Run apps\\prerender.py as a QProcess subprocess, streaming its merged
        stdout/stderr into a modal dialog. On exit code 0 launch the tracker
        detached and close calibrate; on failure leave the dialog open."""
        # Build the modal progress dialog with a read-only streaming log.
        dlg = QDialog(self)
        dlg.setWindowTitle("Pre-rendering...")
        dlg.setModal(True)
        dlg.resize(720, 420)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Running pre-render. This can take a few minutes..."))
        log = QPlainTextEdit()
        log.setReadOnly(True)
        layout.addWidget(log, 1)
        btn = QPushButton("Cancel")
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        # QProcess for pre-render. Store on self so it is not GC'd mid-run.
        proc = QProcess(self)
        self._pre_proc = proc
        proc.setProcessChannelMode(QProcess.MergedChannels)

        def _append(text: str):
            log.moveCursor(log.textCursor().End)
            log.insertPlainText(text)
            # Auto-scroll to the bottom.
            sb = log.verticalScrollBar()
            sb.setValue(sb.maximum())

        def _on_output():
            _append(bytes(proc.readAllStandardOutput()).decode(errors="replace"))

        def _on_finished(exit_code, _status):
            if exit_code == 0:
                _append("\nPre-render complete - opening tracker...")
                # Launch the tracker DETACHED so it opens independently of us.
                QProcess.startDetached(
                    sys.executable,
                    [str(APPS_DIR / "track.py"), "--video", str(self.video_path)],
                )
                dlg.accept()
                self.close()
            else:
                _append(f"\nPre-render FAILED (exit {exit_code})")
                # Repurpose the button to dismiss the dialog so the log stays
                # readable; do NOT close the calibration window.
                btn.setText("Close")
                try:
                    btn.clicked.disconnect()
                except (TypeError, RuntimeError):
                    pass
                btn.clicked.connect(dlg.reject)

        def _on_cancel():
            # While running, Cancel kills the process and closes the dialog.
            proc.kill()
            dlg.reject()

        proc.readyReadStandardOutput.connect(_on_output)
        proc.finished.connect(_on_finished)
        btn.clicked.connect(_on_cancel)

        program = sys.executable
        args = [str(APPS_DIR / "prerender.py"), "--video", str(self.video_path)]
        _append(f"$ {program} {' '.join(args)}\n")
        proc.start(program, args)
        dlg.exec()

    def _learn_team(self, team: Team):
        """Build a TeamProfile from a team's sample boxes.

        Re-reads each box's source frame, crops torso/shorts via rt2.regions,
        and learns ColorSignatures + a HeightModel. Returns (profile, note).
        """
        torso_patches = []
        shorts_patches = []
        height_samples = []
        feature_vectors = []
        for frame_idx, box in team.samples:
            frame = self.reader.frame(frame_idx)
            if frame is None:
                continue
            tp = regions.torso_patch(frame, box)
            sp = regions.shorts_patch(frame, box)
            if tp is not None:
                torso_patches.append(tp)
            if sp is not None:
                shorts_patches.append(sp)
            x1, y1, x2, y2 = box
            cy = (y1 + y2) / 2.0
            height_samples.append((cy, y2 - y1))
            feature_vectors.append(features.feature_vector(frame, box))

        primary = ColorSignature.learn(torso_patches)
        shorts = ColorSignature.learn(shorts_patches)
        height = HeightModel.learn(height_samples)
        fingerprint = features.learn_fingerprint(feature_vectors)

        profile = TeamProfile(
            name=team.name, track=team.track,
            primary=primary, secondary=None, shorts=shorts,
            height=height, n_samples=len(team), fingerprint=fingerprint,
        )

        missing = []
        if primary is None:
            missing.append("primary")
        if shorts is None:
            missing.append("shorts")
        if height is None:
            missing.append("height")
        note = ""
        if missing:
            note = (f"  '{team.name}': could not learn {', '.join(missing)} "
                    f"(need more / cleaner samples).")
        return profile, note

    # ---- helpers ---------------------------------------------------------
    def _set_status(self, text: str):
        self.lbl_status.setText(text)

    def _warn(self, title: str, text: str):
        QMessageBox.warning(self, title, text)

    # ---- keyboard --------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self._toggle_play()
        elif key in (Qt.Key_Left, Qt.Key_Comma):
            self._step(-1)
        elif key in (Qt.Key_Right, Qt.Key_Period):
            self._step(+1)
        elif key == Qt.Key_BracketLeft:
            self._step(-30)
        elif key == Qt.Key_BracketRight:
            self._step(+30)
        elif key in (Qt.Key_0, Qt.Key_F):
            self.view.reset_zoom()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop_play()
        self.reader.release()
        super().closeEvent(event)


def _resolve_video(arg_video: str | None, paths: ProjectPaths) -> pathlib.Path:
    if arg_video:
        p = pathlib.Path(arg_video)
        if not p.exists():
            sys.exit(f"Video not found: {p}")
        return p
    videos = paths.videos()
    if not videos:
        sys.exit(f"No videos found in {paths.input}. Pass --video <path>.")
    return videos[0]


def main():
    ap = argparse.ArgumentParser(description="Phase 2 team calibration tool (PySide6).")
    ap.add_argument("--video", help="source video (default: first ProjectPaths().videos())")
    args = ap.parse_args()

    paths = ProjectPaths()
    video = _resolve_video(args.video, paths)

    app = QApplication(sys.argv)
    win = CalibrateWindow(video, paths)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
