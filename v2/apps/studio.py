"""Rugby Tracker Studio - startup shell with two modes.

A thin tabbed launcher so the program opens into a choice of:

  * Game Analyse    - the existing match player-tracker (v2/apps/track.py),
                      launched unchanged in its own window.
  * Training Analyse - dual-clip pose-overlay form review (v2/apps/training.py):
                      load two clips (two angles of yourself, or you vs a pro),
                      sync them, draw a BlazePose skeleton, play side by side,
                      and export the annotated video for AI feedback.

The game tracker is a large standalone QMainWindow with its own docks/menubar, so
it's launched as a separate process rather than embedded - this keeps that app
completely untouched. The training tool is a plain QWidget, so it embeds directly.

Run:
    python v2/apps/studio.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

# Make both the apps dir (for `import training`) and v2 (for `from rt2...`) importable.
_APPS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_APPS))
sys.path.insert(0, str(_APPS.parent))

from PySide6 import QtCore, QtWidgets

from training import TrainingWidget
from rt2.paths import ProjectPaths


class GameLauncher(QtWidgets.QWidget):
    """Game Analyse tab: pick an optional clip and open the existing tracker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video: str | None = None
        self._track_script = _APPS / "track.py"

        lay = QtWidgets.QVBoxLayout(self)
        lay.addStretch(1)

        title = QtWidgets.QLabel("Game Analyse")
        title.setStyleSheet("font-size:20px; font-weight:bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(title)

        desc = QtWidgets.QLabel(
            "Track a player through match footage (detection, re-ID, re-anchor).\n"
            "Opens the full game tracker in its own window.")
        desc.setAlignment(QtCore.Qt.AlignCenter)
        desc.setStyleSheet("color:#aaa;")
        lay.addWidget(desc)
        lay.addSpacing(12)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_pick = QtWidgets.QPushButton("Choose video… (optional)")
        self.lbl_pick = QtWidgets.QLabel("uses the first clip in input/ if none chosen")
        self.lbl_pick.setStyleSheet("color:#888;")
        self.btn_pick.clicked.connect(self._pick)
        row.addWidget(self.btn_pick)
        row.addWidget(self.lbl_pick)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        row2.addStretch(1)
        self.btn_open = QtWidgets.QPushButton("Open Game Tracker")
        self.btn_open.setMinimumWidth(200)
        self.btn_open.clicked.connect(self._open)
        row2.addWidget(self.btn_open)
        row2.addStretch(1)
        lay.addLayout(row2)

        lay.addStretch(2)

    def _pick(self):
        start_dir = str(ProjectPaths().input)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose game video", start_dir,
            "Video (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)")
        if path:
            self._video = path
            self.lbl_pick.setText(pathlib.Path(path).name)
            self.lbl_pick.setStyleSheet("color:#ddd;")

    def _open(self):
        if not self._track_script.exists():
            QtWidgets.QMessageBox.critical(
                self, "Game Analyse", f"Tracker not found:\n{self._track_script}")
            return
        cmd = [sys.executable, str(self._track_script)]
        if self._video:
            cmd += ["--video", self._video]
        try:
            subprocess.Popen(cmd, cwd=str(self._track_script.parent))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Game Analyse", str(e))


class StudioWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rugby Tracker Studio")
        self.tabs = QtWidgets.QTabWidget()
        self.game = GameLauncher()
        self.training = TrainingWidget()
        self.tabs.addTab(self.game, "Game Analyse")
        self.tabs.addTab(self.training, "Training Analyse")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, ev):
        try:
            self.training.shutdown()
        except Exception:
            pass
        super().closeEvent(ev)


def main():
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = StudioWindow()
    screen = QtWidgets.QApplication.primaryScreen()
    if screen is not None:
        avail = screen.availableGeometry()
        win.resize(min(1280, avail.width() - 48), min(900, avail.height() - 48))
    else:
        win.resize(1280, 820)
    win.showMaximized()
    app.exec()


if __name__ == "__main__":
    main()
