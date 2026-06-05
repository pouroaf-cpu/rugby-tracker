"""Training Analyse - dual-clip pose-overlay form review.

Load two clips (two angles of yourself, OR yourself vs a professional reference),
marry them to the same moment, draw a BlazePose skeleton on each, and play them
side by side. Export the synced, skeleton-annotated side-by-side video to feed
into another AI for body-positioning feedback.

Sync is "both": an audio cross-correlation auto-aligns clips that share a sound
(two angles of the same take), and a manual offset slider fine-tunes it - the
manual path also covers you-vs-pro, where the clips share no audio. Clips at
different frame rates are aligned on real time (seconds), not frame number.

Runs embedded in studio.py's "Training Analyse" tab, or standalone:
    python v2/apps/training.py
"""
from __future__ import annotations

import pathlib
import sys

# v2 on path so `from rt2.x import ...` works whether run standalone or embedded.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from rt2.video import VideoReader
from rt2.paths import ProjectPaths
from rt2 import sync as audiosync

A, B = "A", "B"
_SPEEDS = [0.25, 0.5, 1.0, 2.0]
_EXPORT_H = 720          # each view scaled to this height in the export
_GAP = 8                 # black gap between the two views (px)


def _bgr_to_qpixmap(frame_bgr, target_size: QtCore.QSize) -> QtGui.QPixmap:
    """Convert a BGR frame to a QPixmap scaled to fit target_size (keep aspect)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    pm = QtGui.QPixmap.fromImage(img)
    if target_size.width() > 0 and target_size.height() > 0:
        pm = pm.scaled(target_size, QtCore.Qt.KeepAspectRatio,
                       QtCore.Qt.SmoothTransformation)
    return pm


def _scale_to_height(frame_bgr, h: int):
    cur_h, cur_w = frame_bgr.shape[:2]
    if cur_h == h:
        return frame_bgr
    w = max(1, int(round(cur_w * h / cur_h)))
    return cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)


def _lazy_pose():
    """Import + construct the pose estimator lazily (loading the heavy model
    takes a moment, and we don't want to pay it until a clip is loaded)."""
    from rt2.pose import PoseOverlay
    return PoseOverlay()


# ---------------------------------------------------------------------------
# Export worker - renders the side-by-side annotated video off the UI thread.
# ---------------------------------------------------------------------------
class ExportWorker(QtCore.QThread):
    progress = QtCore.Signal(int)        # 0..100
    finished_ok = QtCore.Signal(str)     # output path
    failed = QtCore.Signal(str)          # error message

    def __init__(self, path_a, path_b, offset, t0, t1, fps_out, out_path, parent=None):
        super().__init__(parent)
        self.path_a = path_a
        self.path_b = path_b
        self.offset = offset
        self.t0 = t0
        self.t1 = t1
        self.fps_out = fps_out
        self.out_path = out_path
        self._abort = False

    def stop(self):
        self._abort = True

    def run(self):
        ra = rb = pose = writer = None
        try:
            ra = VideoReader(self.path_a)
            rb = VideoReader(self.path_b)
            pose = _lazy_pose()           # fresh estimator inside the worker thread

            # Probe the output frame size from the first composite.
            t = self.t0
            first = self._composite(ra, rb, pose, t)
            if first is None:
                self.failed.emit("Could not read frames at the start of the overlap.")
                return
            H, W = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.out_path, fourcc, self.fps_out, (W, H))
            if not writer.isOpened():
                self.failed.emit("Could not open the output video for writing.")
                return

            dt = 1.0 / self.fps_out
            total = max(1, int(round((self.t1 - self.t0) / dt)))
            n = 0
            while t <= self.t1 and not self._abort:
                frame = self._composite(ra, rb, pose, t)
                if frame is not None:
                    if frame.shape[0] != H or frame.shape[1] != W:
                        frame = cv2.resize(frame, (W, H))
                    writer.write(frame)
                n += 1
                t += dt
                self.progress.emit(min(100, int(100 * n / total)))

            if self._abort:
                self.failed.emit("Export cancelled.")
                return
            self.finished_ok.emit(self.out_path)
        except Exception as e:                       # pragma: no cover - surfaced in UI
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            for r in (ra, rb):
                try:
                    if r:
                        r.release()
                except Exception:
                    pass
            if writer:
                writer.release()
            if pose:
                pose.close()

    def _composite(self, ra, rb, pose, t):
        fa = ra.frame(int(round(t * ra.fps)) + 1)
        fb = rb.frame(int(round((t + self.offset) * rb.fps)) + 1)
        if fa is None or fb is None:
            return None
        fa = pose.draw(fa, pose.landmarks(fa))
        fb = pose.draw(fb, pose.landmarks(fb))
        fa = _scale_to_height(fa, _EXPORT_H)
        fb = _scale_to_height(fb, _EXPORT_H)
        gap = np.zeros((_EXPORT_H, _GAP, 3), np.uint8)
        return np.hstack([fa, gap, fb])


# ---------------------------------------------------------------------------
# The Training Analyse widget.
# ---------------------------------------------------------------------------
class TrainingWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.readers: dict[str, VideoReader | None] = {A: None, B: None}
        self.paths: dict[str, str | None] = {A: None, B: None}
        self._pose = None                 # PoseOverlay, lazily created
        self.offset = 0.0                 # seconds: b_time = a_time + offset
        self.master_t = 0.0               # current time on A's clock
        self.speed = 1.0
        self.show_skel = True
        self.loop = False
        self.loop_in: float | None = None    # times on A's clock; None => overlap edge
        self.loop_out: float | None = None
        self._export = None

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)        # ~30 fps UI tick
        self.timer.timeout.connect(self._on_tick)

    # -- UI -----------------------------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # Row 1: load clips
        load = QtWidgets.QHBoxLayout()
        self.btn_a = QtWidgets.QPushButton("Load Clip A")
        self.btn_b = QtWidgets.QPushButton("Load Clip B")
        self.lbl_a = QtWidgets.QLabel("(no clip)")
        self.lbl_b = QtWidgets.QLabel("(no clip)")
        self.lbl_a.setStyleSheet("color:#888")
        self.lbl_b.setStyleSheet("color:#888")
        self.btn_a.clicked.connect(lambda: self._load(A))
        self.btn_b.clicked.connect(lambda: self._load(B))
        load.addWidget(self.btn_a)
        load.addWidget(self.lbl_a, 1)
        load.addSpacing(16)
        load.addWidget(self.btn_b)
        load.addWidget(self.lbl_b, 1)
        root.addLayout(load)

        # Row 2: sync
        sync = QtWidgets.QHBoxLayout()
        self.btn_autosync = QtWidgets.QPushButton("Auto-sync (audio)")
        self.btn_autosync.clicked.connect(self._auto_sync)
        sync.addWidget(self.btn_autosync)
        sync.addWidget(QtWidgets.QLabel("Offset B (s):"))
        self.spin_offset = QtWidgets.QDoubleSpinBox()
        self.spin_offset.setRange(-120.0, 120.0)
        self.spin_offset.setSingleStep(0.05)
        self.spin_offset.setDecimals(2)
        self.spin_offset.valueChanged.connect(self._on_offset_changed)
        sync.addWidget(self.spin_offset)
        self.lbl_sync = QtWidgets.QLabel("")
        self.lbl_sync.setStyleSheet("color:#888")
        sync.addWidget(self.lbl_sync, 1)
        root.addLayout(sync)

        # Row 3: side-by-side views
        views = QtWidgets.QHBoxLayout()
        self.view_a = QtWidgets.QLabel("Clip A")
        self.view_b = QtWidgets.QLabel("Clip B")
        for v in (self.view_a, self.view_b):
            v.setAlignment(QtCore.Qt.AlignCenter)
            v.setMinimumSize(320, 240)
            v.setStyleSheet("background:#111; color:#555;")
            v.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                            QtWidgets.QSizePolicy.Expanding)
        views.addWidget(self.view_a, 1)
        views.addWidget(self.view_b, 1)
        root.addLayout(views, 1)

        # Row 4: transport
        trans = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("⏮")
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_start.clicked.connect(self._to_start)
        self.btn_play.clicked.connect(self._toggle_play)
        trans.addWidget(self.btn_start)
        trans.addWidget(self.btn_play)
        self.scrub = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.sliderMoved.connect(self._on_scrub)
        trans.addWidget(self.scrub, 1)
        self.lbl_time = QtWidgets.QLabel("0.00 / 0.00 s")
        trans.addWidget(self.lbl_time)
        trans.addWidget(QtWidgets.QLabel("Speed"))
        self.cmb_speed = QtWidgets.QComboBox()
        for s in _SPEEDS:
            self.cmb_speed.addItem(f"{s:g}x", s)
        self.cmb_speed.setCurrentIndex(_SPEEDS.index(1.0))
        self.cmb_speed.currentIndexChanged.connect(
            lambda i: setattr(self, "speed", self.cmb_speed.currentData()))
        trans.addWidget(self.cmb_speed)
        self.chk_skel = QtWidgets.QCheckBox("Skeleton")
        self.chk_skel.setChecked(True)
        self.chk_skel.toggled.connect(self._on_skel_toggled)
        trans.addWidget(self.chk_skel)
        # Loop a single rep (useful for you-vs-pro: trim each to one rep and repeat).
        self.btn_in = QtWidgets.QPushButton("Set In")
        self.btn_out = QtWidgets.QPushButton("Set Out")
        self.chk_loop = QtWidgets.QCheckBox("Loop")
        self.btn_clear_loop = QtWidgets.QPushButton("Clear")
        self.btn_in.clicked.connect(self._set_in)
        self.btn_out.clicked.connect(self._set_out)
        self.chk_loop.toggled.connect(lambda v: setattr(self, "loop", v))
        self.btn_clear_loop.clicked.connect(self._clear_loop)
        for wdg in (self.btn_in, self.btn_out, self.chk_loop, self.btn_clear_loop):
            trans.addWidget(wdg)
        root.addLayout(trans)

        # Row 5: export
        exp = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export side-by-side video")
        self.btn_export.clicked.connect(self._export_video)
        exp.addWidget(self.btn_export)
        self.lbl_export = QtWidgets.QLabel(
            "Skeleton preview runs slow-mo on CPU; export renders at full quality (silent).")
        self.lbl_export.setStyleSheet("color:#888")
        exp.addWidget(self.lbl_export, 1)
        root.addLayout(exp)

    # -- helpers ------------------------------------------------------------
    def _ensure_pose(self):
        if self._pose is None:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self._pose = _lazy_pose()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        return self._pose

    def _overlap(self):
        """(t0, t1) on A's clock where both clips have footage. If only one is
        loaded, the range is that single clip."""
        ra, rb = self.readers[A], self.readers[B]
        if ra and rb:
            t0 = max(0.0, -self.offset)
            t1 = min(ra.duration_s(), rb.duration_s() - self.offset)
            return t0, max(t0, t1)
        if ra:
            return 0.0, ra.duration_s()
        if rb:
            return -self.offset, rb.duration_s() - self.offset
        return 0.0, 0.0

    def _load(self, slot):
        start_dir = str(ProjectPaths().input)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load Clip {slot}", start_dir,
            "Video (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)")
        if not path:
            return
        try:
            reader = VideoReader(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            return
        old = self.readers[slot]
        if old:
            old.release()
        self.readers[slot] = reader
        self.paths[slot] = path
        lbl = self.lbl_a if slot == A else self.lbl_b
        lbl.setText(f"{pathlib.Path(path).name}  "
                    f"({reader.width}x{reader.height}, {reader.fps:.1f}fps, "
                    f"{reader.duration_s():.1f}s)")
        lbl.setStyleSheet("color:#ddd")
        self.master_t, _ = self._overlap()
        self._render()

    def _auto_sync(self):
        if not (self.paths[A] and self.paths[B]):
            self.lbl_sync.setText("Load both clips first.")
            return
        if not audiosync.have_ffmpeg():
            self.lbl_sync.setText("ffmpeg not found - use the manual offset.")
            return
        self.lbl_sync.setText("Analysing audio…")
        QtWidgets.QApplication.processEvents()
        off = audiosync.audio_offset_seconds(self.paths[A], self.paths[B])
        if off is None:
            self.lbl_sync.setText("No shared audio found - set the offset manually.")
            return
        self.spin_offset.setValue(round(off, 2))   # triggers _on_offset_changed
        self.lbl_sync.setText(f"Audio-synced: B offset {off:+.2f}s")

    def _on_offset_changed(self, val):
        self.offset = float(val)
        t0, t1 = self._overlap()
        self.master_t = min(max(self.master_t, t0), t1)
        self._render()

    def _on_skel_toggled(self, on):
        self.show_skel = on
        if on:
            self._ensure_pose()
        self._render()

    def _loop_bounds(self):
        """The active loop region on A's clock, defaulting to the full overlap."""
        t0, t1 = self._overlap()
        lo = self.loop_in if self.loop_in is not None else t0
        hi = self.loop_out if self.loop_out is not None else t1
        lo = min(max(lo, t0), t1)
        hi = min(max(hi, t0), t1)
        if hi <= lo:
            return t0, t1
        return lo, hi

    def _set_in(self):
        self.loop_in = self.master_t
        self._refresh_loop_label()

    def _set_out(self):
        self.loop_out = self.master_t
        self._refresh_loop_label()

    def _clear_loop(self):
        self.loop_in = self.loop_out = None
        self._refresh_loop_label()

    def _refresh_loop_label(self):
        lo, hi = self._loop_bounds()
        t0, _ = self._overlap()
        self.lbl_sync.setText(f"Loop region: {lo - t0:.2f}–{hi - t0:.2f} s")

    # -- transport ----------------------------------------------------------
    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ Play")
        else:
            if self.show_skel:
                self._ensure_pose()
            self.timer.start()
            self.btn_play.setText("⏸ Pause")

    def _to_start(self):
        self.master_t, _ = self._overlap()
        self._render()

    def _on_tick(self):
        self.master_t += (self.timer.interval() / 1000.0) * self.speed
        if self.loop:
            lo, hi = self._loop_bounds()
            if self.master_t >= hi:
                self.master_t = lo            # wrap back to the rep start
        else:
            _, t1 = self._overlap()
            if self.master_t >= t1:
                self.master_t = t1
                self.timer.stop()
                self.btn_play.setText("▶ Play")
        self._render()

    def _on_scrub(self, val):
        t0, t1 = self._overlap()
        self.master_t = t0 + (t1 - t0) * (val / 1000.0)
        self._render()

    # -- rendering ----------------------------------------------------------
    def _frame_for(self, slot, t_on_a):
        reader = self.readers[slot]
        if reader is None:
            return None
        t = t_on_a if slot == A else (t_on_a + self.offset)
        if t < 0 or t > reader.duration_s():
            return None
        frame = reader.frame(int(round(t * reader.fps)) + 1)
        if frame is None:
            return None
        if self.show_skel and self._pose is not None:
            pose = self._pose
            frame = pose.draw(frame, pose.landmarks(frame))
        return frame

    def _render(self):
        for slot, view in ((A, self.view_a), (B, self.view_b)):
            frame = self._frame_for(slot, self.master_t)
            if frame is None:
                if self.readers[slot] is None:
                    view.setText(f"Clip {slot}\n(load a video)")
                continue
            view.setPixmap(_bgr_to_qpixmap(frame, view.size()))

        t0, t1 = self._overlap()
        span = max(1e-6, t1 - t0)
        self.scrub.blockSignals(True)
        self.scrub.setValue(int(1000 * (self.master_t - t0) / span))
        self.scrub.blockSignals(False)
        self.lbl_time.setText(f"{self.master_t - t0:.2f} / {span:.2f} s")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._render()

    # -- export -------------------------------------------------------------
    def _export_video(self):
        if not (self.readers[A] and self.readers[B]):
            QtWidgets.QMessageBox.information(
                self, "Export", "Load both clips before exporting.")
            return
        t0, t1 = self._overlap()
        if t1 - t0 < 0.1:
            QtWidgets.QMessageBox.information(
                self, "Export", "The clips don't overlap - check the sync offset.")
            return
        out = ProjectPaths().ensure().output
        stem_a = pathlib.Path(self.paths[A]).stem
        stem_b = pathlib.Path(self.paths[B]).stem
        default = str(out / f"{stem_a}_vs_{stem_b}_training.mp4")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export side-by-side video", default, "MP4 video (*.mp4)")
        if not path:
            return
        fps_out = min(self.readers[A].fps, self.readers[B].fps)
        fps_out = max(10.0, min(60.0, fps_out))

        self.btn_export.setEnabled(False)
        dlg = QtWidgets.QProgressDialog(
            "Rendering side-by-side video…", "Cancel", 0, 100, self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)

        worker = ExportWorker(self.paths[A], self.paths[B], self.offset,
                              t0, t1, fps_out, path, self)
        self._export = worker
        worker.progress.connect(dlg.setValue)
        dlg.canceled.connect(worker.stop)

        def _ok(p):
            dlg.reset()
            self.btn_export.setEnabled(True)
            QtWidgets.QMessageBox.information(self, "Export complete", f"Saved:\n{p}")

        def _err(msg):
            dlg.reset()
            self.btn_export.setEnabled(True)
            QtWidgets.QMessageBox.warning(self, "Export", msg)

        worker.finished_ok.connect(_ok)
        worker.failed.connect(_err)
        worker.start()

    def shutdown(self):
        if self.timer.isActive():
            self.timer.stop()
        if self._export and self._export.isRunning():
            self._export.stop()
            self._export.wait(2000)
        for r in self.readers.values():
            if r:
                r.release()
        if self._pose:
            self._pose.close()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = TrainingWidget()
    w.setWindowTitle("Training Analyse")
    w.resize(1280, 820)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
