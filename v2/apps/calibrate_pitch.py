"""calibrate_pitch.py - PITCH HOMOGRAPHY calibration tool (PySide6).

Create the image<->pitch homography for a clip by clicking known pitch
landmarks (try-line corners, 22m line ends, halfway, goalpost bases) and mapping
each to its real-world pitch coordinate in metres. With >=4 non-collinear point
correspondences `cv2.findHomography` solves the plane mapping; we report the
reprojection error and save it for the tracker (minimap + metre-accurate
"who is me" zones).

PANNING FOOTAGE  (the reason this tool accumulates across frames)
  On Veo auto-follow footage only ~2 pitch landmarks are clearly visible in any
  single frame, so 4 correspondences can NEVER be collected on one frame. We use
  CAMERA-MOTION COMPENSATION (rt2.cmc.CameraTracker) to accumulate points across
  frames: the FIRST placed point fixes an ANCHOR frame A and anchors the tracker.
  As you PLAY/STEP FORWARD, the tracker folds in each new frame's camera motion;
  every later click is mapped BACK into frame A's pixel coordinates via
  `tracker.to_anchor`, so all accumulated points live in ONE frame. The
  homography is then fit from the anchor-frame points, and saved with
  `anchor_frame = A` so the tracker in track.py (which anchors at A) lines up.

  CMC is SEQUENTIAL: it tracks frame-to-frame, so you must STEP/PLAY FORWARD from
  the anchor frame. An arbitrary scrub-jump (or stepping backward) breaks the
  chain; when that happens the tracker is paused and clicks are refused until you
  step forward again from the anchor (or Clear to re-anchor).

WORKFLOW
  1. Pick a landmark in the "Pitch landmarks" list, then LEFT-CLICK its exact
     spot on the frame. The first click sets the ANCHOR frame.
  2. PLAY / STEP FORWARD. As the pitch pans into view, pick + click more
     landmarks (only ~2 are visible at once). Each is stored in anchor-frame
     coords. Markers track the pan so you can see what's already placed.
  3. Accumulate at least 4 (more + well-spread = better). "Remove selected point"
     drops one; "Clear all points" resets and frees the anchor.
  4. "Compute & Save" runs findHomography, shows the reprojection error (metres),
     and saves <stem>.pitch.json + the shared game.pitch.json.
  Once >=4 points exist, moving the mouse over the frame shows the live pitch
  (X, Y) in metres under the cursor (via to_anchor + the homography).

VIDEO CONTROLS  (same as calibrate.py)
  Play/Pause, Prev/Next frame, slider; keys: Space, Left/',', Right/'.',
  '['/']' jump -/+30, '0'/'f' reset zoom. Mouse wheel zooms under the cursor;
  middle-drag or Space+left-drag pans.

CLI
  python apps\\calibrate_pitch.py                       # first video from ProjectPaths
  python apps\\calibrate_pitch.py --video input\\chunk_028.mp4
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import Qt, QRectF, QTimer, QPointF
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QMouseEvent, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSlider, QPushButton, QLabel, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy,
)

from rt2.paths import ProjectPaths
from rt2.video import VideoReader
from rt2.homography import PitchHomography, PITCH
from rt2.cmc import CameraTracker


class PitchView(QGraphicsView):
    """Frame view that reports left-clicks (in native frame pixels) for placing
    landmarks, and mouse-move (for live pitch read-out). Pan/zoom mirror
    calibrate.py: wheel zooms under the cursor, middle-drag / Space+left-drag
    pans. Scene == native pixels, so mapToScene gives original-frame coords.
    """

    ZOOM_STEP = 1.25
    ZOOM_MAX = 12.0

    def __init__(self, on_click, on_move):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._on_click = on_click
        self._on_move = on_move
        self._frame_w = 0
        self._frame_h = 0
        self._fit_scale = 1.0
        self._space_held = False
        self._panning = False
        self._markers = []                 # transient QGraphicsItems for points

        self.setRenderHints(self.renderHints())
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, frame_bgr):
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        if (w, h) != (self._frame_w, self._frame_h):
            self._frame_w, self._frame_h = w, h
            self._scene.setSceneRect(0, 0, w, h)
            self._fit()

    # ---- point markers (drawn on the scene so they pan/zoom with the frame) --
    def set_points(self, placed):
        """`placed` = list of (label, (x, y)) in CURRENT-frame native px; redraw
        markers. Entries whose (x, y) is None (off-frame / CMC failed) are
        skipped so we never crash on a bad transform."""
        for it in self._markers:
            self._scene.removeItem(it)
        self._markers.clear()
        for i, (label, xy) in enumerate(placed):
            if xy is None:
                continue
            x, y = xy
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                continue
            r = 6.0
            dot = self._scene.addEllipse(
                x - r, y - r, 2 * r, 2 * r,
                QPen(QColor(0, 230, 0), 2), QBrush(QColor(0, 230, 0, 90)))
            txt = self._scene.addText(str(i + 1))
            txt.setDefaultTextColor(QColor(0, 230, 0))
            f = QFont(); f.setPointSizeF(10); txt.setFont(f)
            txt.setPos(x + r, y - r - 14)
            self._markers.extend([dot, txt])

    def _fit(self):
        if self._frame_w:
            self.fitInView(QRectF(0, 0, self._frame_w, self._frame_h),
                           Qt.KeepAspectRatio)
            self._fit_scale = self.transform().m11()

    def reset_zoom(self):
        self._fit()

    def _zoom_factor(self):
        if self._fit_scale <= 0:
            return 1.0
        return self.transform().m11() / self._fit_scale

    def _is_zoomed(self):
        return self._zoom_factor() > 1.001

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._is_zoomed():
            self._fit()

    def wheelEvent(self, event):
        if not self._frame_w:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
        target = min(max(self._zoom_factor() * factor, 1.0), self.ZOOM_MAX)
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

    def _clamp_scene(self, pt):
        x = min(max(pt.x(), 0.0), float(self._frame_w))
        y = min(max(pt.y(), 0.0), float(self._frame_h))
        return QPointF(x, y)

    def _start_pan(self, event):
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._panning = True
        synth = QMouseEvent(
            event.type(), event.position(), event.globalPosition(),
            Qt.LeftButton, Qt.LeftButton, event.modifiers())
        super().mousePressEvent(synth)

    def mousePressEvent(self, event):
        if self._frame_w and (
            event.button() == Qt.MiddleButton
            or (event.button() == Qt.LeftButton and self._space_held)
        ):
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self._start_pan(event)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._frame_w:
            pt = self._clamp_scene(self.mapToScene(event.position().toPoint()))
            self._on_click(pt.x(), pt.y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._frame_w and not self._panning:
            pt = self._clamp_scene(self.mapToScene(event.position().toPoint()))
            self._on_move(pt.x(), pt.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            synth = QMouseEvent(
                event.type(), event.position(), event.globalPosition(),
                Qt.LeftButton, Qt.NoButton, event.modifiers())
            super().mouseReleaseEvent(synth)
            self._panning = False
            self.setDragMode(QGraphicsView.NoDrag)
            self._set_pan_cursor()
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


class CalibratePitchWindow(QMainWindow):
    def __init__(self, video_path, paths: ProjectPaths):
        super().__init__()
        self.video_path = pathlib.Path(video_path)
        self.paths = paths
        self.reader = VideoReader(self.video_path)
        self.n_frames = self.reader.n_frames
        self.frame_idx = 1

        # label -> (anchor_x, anchor_y): placed landmarks stored in ANCHOR-FRAME
        # pixels (mapped back from the click frame via CMC). Drawing maps them to
        # the current frame via tracker.from_anchor.
        self.placed: dict[str, tuple[float, float]] = {}
        self.homog: PitchHomography | None = None

        # --- CMC accumulation state ------------------------------------------
        # The first placed point sets the ANCHOR frame A; the tracker is anchored
        # to frame A's image and fed each SEQUENTIAL +1 frame as the user steps
        # forward. Every click is mapped back into frame A pixels via to_anchor.
        self.anchor_frame: int | None = None
        self.tracker = CameraTracker(min_inliers=25)
        # True once the CMC chain is broken by a non-sequential (back/jump) step;
        # clicks are refused until the user steps forward from the anchor again.
        self._cmc_stale = False
        # Frame index the tracker has currently folded motion up to (== the frame
        # last fed via update; == anchor_frame right after anchoring).
        self._tracker_frame: int | None = None
        # The previously displayed frame index, to detect sequential +1 steps.
        self._prev_shown: int | None = None

        self.setWindowTitle(f"Calibrate pitch - {self.video_path.name}")
        self._build_ui()
        self._show_frame(self.frame_idx)
        self._refresh_landmark_list()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_tick)
        self._playing = False

    # ---- UI --------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        self.lbl_guide = QLabel(
            "Place a landmark, then PLAY/STEP FORWARD; as the pitch pans into "
            "view, place more (only ~2 are visible at once). Accumulate >=4, "
            "then Compute & Save. Don't scrub backwards.")
        self.lbl_guide.setWordWrap(True)
        left.addWidget(self.lbl_guide)

        self.view = PitchView(self._on_click, self._on_move)
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
        self.btn_fit.clicked.connect(self.view.reset_zoom)
        self.lbl_frame = QLabel()
        self.lbl_live = QLabel("pitch: -")
        self.lbl_live.setMinimumWidth(180)
        transport.addWidget(self.btn_play)
        transport.addWidget(self.btn_prev)
        transport.addWidget(self.btn_next)
        transport.addWidget(self.btn_fit)
        transport.addStretch(1)
        transport.addWidget(self.lbl_live)
        transport.addWidget(self.lbl_frame)
        left.addLayout(transport)
        root.addLayout(left, 3)

        # right: landmark template + actions
        right = QVBoxLayout()
        right.addWidget(QLabel(
            "<b>Pitch landmarks</b><br>pick one, then click its spot on the frame"))
        self.landmark_list = QListWidget()
        right.addWidget(self.landmark_list, 1)

        b_remove = QPushButton("Remove selected point")
        b_remove.clicked.connect(self._remove_point)
        right.addWidget(b_remove)
        b_clear = QPushButton("Clear all points")
        b_clear.clicked.connect(self._clear_points)
        right.addWidget(b_clear)

        self.lbl_count = QLabel()
        self.lbl_count.setWordWrap(True)
        right.addWidget(self.lbl_count)

        self.lbl_cmc = QLabel()
        self.lbl_cmc.setWordWrap(True)
        right.addWidget(self.lbl_cmc)

        right.addStretch(1)
        self.btn_compute = QPushButton("Compute & Save")
        self.btn_compute.clicked.connect(self._compute_and_save)
        right.addWidget(self.btn_compute)

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        right.addWidget(self.lbl_status)

        root.addLayout(right, 1)
        self.setCentralWidget(central)
        self.resize(1280, 760)

    # ---- frame display ---------------------------------------------------
    def _show_frame(self, idx):
        idx = max(1, min(self.n_frames, idx))
        frame = self.reader.frame(idx)
        prev = self._prev_shown
        self.frame_idx = idx
        self.view.set_frame(frame)
        self.lbl_frame.setText(f"frame {idx} / {self.n_frames}")
        # Advance the CMC chain at the SEQUENTIAL-step choke point: only when an
        # anchor exists, this is a strictly +1 step from the last shown frame,
        # and the tracker is sitting on (idx-1).
        self._advance_cmc(idx, prev, frame)
        self._prev_shown = idx
        self._sync_markers()
        self._refresh_cmc_status()
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

    def _advance_cmc(self, idx, prev, frame):
        """Maintain the camera tracker as frames are displayed.

        Only a strictly sequential +1 step (idx == prev + 1) that lands exactly
        one frame past where the tracker currently sits feeds the tracker. Any
        backward step or jump marks the chain stale (until the user steps forward
        from the anchor again, or clears). Re-displaying the anchor frame itself,
        or stepping +1 back onto the tracker's current frame, is a no-op."""
        if self.anchor_frame is None or frame is None:
            return
        if idx == self._tracker_frame:
            # already aligned here (e.g. anchor frame re-shown) - chain intact.
            self._cmc_stale = False
            return
        sequential = (
            prev is not None
            and idx == prev + 1
            and self._tracker_frame is not None
            and idx == self._tracker_frame + 1
        )
        if not sequential:
            # backward step or arbitrary jump breaks the frame-to-frame chain.
            self._cmc_stale = True
            return
        try:
            self.tracker.update(frame)
        except Exception:                              # noqa: BLE001 - never crash
            self._cmc_stale = True
            return
        self._tracker_frame = idx
        # Stay aligned but flag CMC unhealthy if the motion estimate was weak.
        self._cmc_stale = not self.tracker.healthy

    def _cmc_ready(self):
        """True if a point can be accepted on the CURRENT frame: an anchor exists,
        the chain is not stale, the tracker is healthy, and it sits on this frame."""
        if self.anchor_frame is None:
            return True                                # first click sets the anchor
        return (not self._cmc_stale
                and self.tracker.healthy
                and self._tracker_frame == self.frame_idx)

    def _refresh_cmc_status(self):
        if self.anchor_frame is None:
            self.lbl_cmc.setText("CMC: no anchor yet - first click sets it.")
            return
        if self._cmc_stale or not self.tracker.healthy:
            self.lbl_cmc.setText(
                f"CMC PAUSED (anchor frame {self.anchor_frame}). Stepped "
                f"non-sequentially or motion estimate weak - step FORWARD from "
                f"the anchor, or Clear to re-anchor. Points not accepted while "
                f"paused.")
        elif self._tracker_frame != self.frame_idx:
            self.lbl_cmc.setText(
                f"CMC anchor frame {self.anchor_frame}; tracker at frame "
                f"{self._tracker_frame}. Step FORWARD to {self._tracker_frame + 1} "
                f"to continue accumulating here.")
        else:
            self.lbl_cmc.setText(
                f"CMC healthy: anchor frame {self.anchor_frame}, tracking frame "
                f"{self.frame_idx} ({self.tracker.last_inliers} inliers, "
                f"{self.tracker.frames_since_anchor} frames since anchor).")

    def _step(self, delta):
        self._stop_play()
        self._show_frame(self.frame_idx + delta)

    def _on_slider(self, value):
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

    # ---- landmark list ---------------------------------------------------
    def _refresh_landmark_list(self):
        cur = self.landmark_list.currentRow()
        self.landmark_list.blockSignals(True)
        self.landmark_list.clear()
        for name in PITCH.names():
            X, Y = PITCH.named_points[name]
            mark = " [SET]" if name in self.placed else ""
            QListWidgetItem(f"{name}  ({X:.0f}, {Y:.0f} m){mark}", self.landmark_list)
        if cur < 0:
            cur = self._first_unplaced_row()
        self.landmark_list.setCurrentRow(max(0, min(cur, self.landmark_list.count() - 1)))
        self.landmark_list.blockSignals(False)
        n = len(self.placed)
        self.lbl_count.setText(f"{n} landmarks accumulated (need >= 4).")

    def _first_unplaced_row(self):
        names = PITCH.names()
        for i, name in enumerate(names):
            if name not in self.placed:
                return i
        return 0

    def _selected_name(self):
        row = self.landmark_list.currentRow()
        names = PITCH.names()
        if 0 <= row < len(names):
            return names[row]
        return None

    def _current_xy(self, anchor_xy):
        """Map an anchor-frame point to its CURRENT-frame pixel via from_anchor.
        On the anchor frame (tracker identity) this is the point itself. Returns
        (x, y) or None if the transform fails."""
        ax, ay = anchor_xy
        try:
            cur = self.tracker.from_anchor(ax, ay)
        except Exception:                              # noqa: BLE001
            return None
        return cur

    def _sync_markers(self):
        """Draw each accumulated point at its CURRENT-frame position (markers
        follow the pan). If the tracker can't map a point, it is skipped."""
        names = PITCH.names()
        placed = []
        for name in names:
            if name in self.placed:
                placed.append((name, self._current_xy(self.placed[name])))
        self.view.set_points(placed)

    # ---- point placement -------------------------------------------------
    def _on_click(self, x, y):
        name = self._selected_name()
        if name is None:
            self._set_status("Pick a landmark from the list first, then click.")
            return
        # First click sets the anchor frame and anchors the tracker on it.
        if self.anchor_frame is None:
            frame = self.reader.frame(self.frame_idx)
            if frame is None:
                self._set_status("Could not read this frame; cannot anchor here.")
                return
            self.anchor_frame = self.frame_idx
            self.tracker.anchor(frame)
            self._tracker_frame = self.frame_idx
            self._cmc_stale = False
            axy = (float(x), float(y))                 # on the anchor: identity
        else:
            if not self._cmc_ready():
                self._set_status(
                    "CMC paused/unhealthy - point not accepted. Step FORWARD "
                    "from the anchor frame (don't scrub back), or Clear to "
                    "re-anchor.")
                return
            axy = self.tracker.to_anchor(float(x), float(y))
            if axy is None or not (np.isfinite(axy[0]) and np.isfinite(axy[1])):
                self._set_status(
                    "Could not map this click back to the anchor frame (CMC) - "
                    "point not accepted. Try a healthier frame or re-anchor.")
                return
        self.placed[name] = (float(axy[0]), float(axy[1]))
        self._refresh_landmark_list()
        # auto-advance to the next unplaced landmark
        self.landmark_list.setCurrentRow(self._first_unplaced_row())
        self._sync_markers()
        self._rebuild_homography()
        self._set_status(
            f"Placed '{name}' (anchor px {axy[0]:.0f}, {axy[1]:.0f}). "
            f"{len(self.placed)} accumulated.")

    def _on_move(self, x, y):
        # Live pitch read-out: map current-frame pixel -> anchor -> pitch metres.
        if self.homog is not None and self.homog.ok:
            axy = None
            if self.anchor_frame is None or self._tracker_frame == self.frame_idx:
                # no anchor (shouldn't have a homography then) or on tracker frame
                try:
                    axy = self.tracker.to_anchor(x, y) if self.anchor_frame is not None else (x, y)
                except Exception:                      # noqa: BLE001
                    axy = None
            if axy is not None and np.isfinite(axy[0]) and np.isfinite(axy[1]):
                xy = self.homog.image_to_pitch(axy[0], axy[1])
                if xy is not None:
                    self.lbl_live.setText(f"pitch: ({xy[0]:.1f}, {xy[1]:.1f}) m")
                    return
        self.lbl_live.setText("pitch: -")

    def _remove_point(self):
        name = self._selected_name()
        if name and name in self.placed:
            del self.placed[name]
            if not self.placed:                # no points left -> free the anchor
                self._reset_cmc()
            self._refresh_landmark_list()
            self._sync_markers()
            self._rebuild_homography()
            self._refresh_cmc_status()
            self._set_status(f"Removed '{name}'.")

    def _clear_points(self):
        if not self.placed and self.anchor_frame is None:
            return
        self.placed.clear()
        self.homog = None
        self._reset_cmc()
        self._refresh_landmark_list()
        self._sync_markers()
        self._refresh_cmc_status()
        self._set_status("Cleared all points. Next click sets a new anchor frame.")

    def _reset_cmc(self):
        """Forget the anchor + CMC chain so the next click re-anchors fresh."""
        self.anchor_frame = None
        self.tracker = CameraTracker(min_inliers=25)
        self._tracker_frame = None
        self._cmc_stale = False

    def _ordered_correspondences(self):
        """Accumulated correspondences in ANCHOR-frame pixels + pitch metres."""
        names = PITCH.names()
        img, pit = [], []
        for name in names:
            if name in self.placed:
                img.append(self.placed[name])         # anchor-frame pixels
                pit.append(PITCH.named_points[name])
        return img, pit

    def _rebuild_homography(self):
        """Live-build the homography (for the cursor read-out) when >=4 points.
        image_points are ANCHOR-frame pixels (matching the saved format)."""
        img, pit = self._ordered_correspondences()
        if len(img) >= 4:
            self.homog = PitchHomography(image_points=img, pitch_points=pit,
                                         anchor_frame=self.anchor_frame)
        else:
            self.homog = None

    # ---- compute + save --------------------------------------------------
    def _compute_and_save(self):
        img, pit = self._ordered_correspondences()
        if len(img) < 4:
            self._warn("Need 4 points",
                       f"Accumulate at least 4 landmarks (have {len(img)}).")
            return
        h = PitchHomography(image_points=img, pitch_points=pit,
                            anchor_frame=self.anchor_frame)
        if not h.ok:
            self._warn("Degenerate points",
                       "Could not solve a homography from these points. They may "
                       "be collinear or too clustered -- spread them across the "
                       "pitch (near + far, both touchlines).")
            return
        self.homog = h
        err = h.reproj_error
        err_txt = f"{err:.2f} m" if err is not None else "n/a"

        out_path = self.paths.output / f"{self.video_path.stem}.pitch.json"
        game_path = self.paths.output / "game.pitch.json"
        try:
            self.paths.output.mkdir(parents=True, exist_ok=True)
            h.save(out_path)
            h.save(game_path)
        except Exception as exc:                       # noqa: BLE001
            self._warn("Save failed", str(exc))
            return

        msg = (f"Saved homography ({len(img)} points, reproj error {err_txt}, "
               f"anchor frame {self.anchor_frame}).\n"
               f"  {out_path}\n  {game_path} (shared game)")
        self._set_status(msg)
        warn = ""
        if err is not None and err > 3.0:
            warn = ("\n\nNote: reprojection error is high (> 3 m). Re-check the "
                    "clicked points are accurate and well spread for a better fit.")
        QMessageBox.information(
            self, "Pitch homography saved",
            f"Saved with reprojection error {err_txt} from {len(img)} points "
            f"(anchor frame {self.anchor_frame}).\n\n"
            f"Per-clip: {out_path}\nShared game: {game_path}{warn}")

    # ---- helpers ---------------------------------------------------------
    def _set_status(self, text):
        self.lbl_status.setText(text)

    def _warn(self, title, text):
        QMessageBox.warning(self, title, text)

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


def _resolve_video(arg_video, paths: ProjectPaths) -> pathlib.Path:
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
    ap = argparse.ArgumentParser(description="Pitch homography calibration tool (PySide6).")
    ap.add_argument("--video", help="source video (default: first ProjectPaths().videos())")
    args = ap.parse_args()

    paths = ProjectPaths()
    video = _resolve_video(args.video, paths)

    app = QApplication(sys.argv)
    win = CalibratePitchWindow(video, paths)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
