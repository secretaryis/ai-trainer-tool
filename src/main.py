import sys
import os
import datetime
import traceback
from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt5.QtGui import QPixmap, QTextCursor
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtCore
from gui.main_window import MainWindow
from utils import deps


def _log_line(text: str):
    timestamp = datetime.datetime.now().isoformat()
    with open("error.log", "a") as f:
        f.write(f"[{timestamp}] {text}\n")
    print(text, file=sys.stderr)


def handle_import_error(exc_type, exc, tb):
    if issubclass(exc_type, ImportError):
        msg = str(exc)
        _log_line(msg)
        if tb:
            with open("error.log", "a") as f:
                traceback.print_tb(tb, file=f)
        app = QApplication.instance()
        if app:
            QMessageBox.critical(None, "Missing dependency", msg)
    else:
        sys.__excepthook__(exc_type, exc, tb)


def main():
    sys.excepthook = handle_import_error
    if not os.environ.get("DISPLAY") and sys.platform != "win32":
        _log_line("Warning: DISPLAY not set; Qt GUI may not start.")

    # Register QTextCursor for queued connections before any threads/signals
    if hasattr(QtCore, "qRegisterMetaType"):
        try:
            QtCore.qRegisterMetaType(QTextCursor)
        except Exception:
            pass
    app = QApplication(sys.argv)

    splash = QSplashScreen(QPixmap(), Qt.WindowStaysOnTopHint)
    splash.showMessage("Loading AI Trainer Tool...", Qt.AlignCenter, Qt.white)
    splash.show()

    def launch():
        try:
            window = MainWindow()
            app.main_window = window  # prevent GC
            window.showMaximized()
            splash.finish(window)
        except Exception as e:
            _log_line(f"Startup failed: {e}")
            traceback.print_exc()

    QTimer.singleShot(800, launch)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
