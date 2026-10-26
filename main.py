# -*- coding: utf-8 -*-
"""Standalone launcher for LinkNex Module 5 - Punctate Organelle Dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow

from punctate_dynamics_module import PunctateDynamicsWidget


APP_TITLE = "LinkNex - Punctate Organelle Dynamics"


def resource_path(relative: str) -> Path:
    """Resolve resources in source mode and future PyInstaller mode."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        base = Path(__file__).resolve().parent
    return base / relative


class Module5Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)

        icon_path = resource_path("assets/linknex_app.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.module = PunctateDynamicsWidget(self)
        self.setCentralWidget(self.module)

        # In the full LinkNex application this signal returns to Home.
        # In the standalone project it simply closes the standalone window.
        self.module.backRequested.connect(self.close)

    def closeEvent(self, event):
        # Save an edited mask before closing, matching the module's normal
        # frame/view-change behavior.
        try:
            self.module._save_current_mask_if_dirty()
        except Exception as exc:
            print(f"[WARN] Could not auto-save current mask on exit: {exc}")

        # Ask running QThreads to stop cleanly when possible.
        try:
            for thread, _worker in list(getattr(self.module, "_threads", [])):
                if thread.isRunning():
                    thread.quit()
                    thread.wait(1500)
        except Exception:
            pass

        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    icon_path = resource_path("assets/linknex_app.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = Module5Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
