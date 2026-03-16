import os
import tempfile
import requests
from urllib.parse import urlparse
from PyQt5.QtCore import QThread, pyqtSignal


def guess_suffix(url, content_type=None):
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return ".pdf"
    if path.endswith(".html") or path.endswith(".htm"):
        return ".html"
    if path.endswith(".txt"):
        return ".txt"
    if any(path.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]):
        return ".png"
    if content_type and "pdf" in content_type:
        return ".pdf"
    if content_type and "html" in content_type:
        return ".html"
    if content_type and "text" in content_type:
        return ".txt"
    if content_type and "image" in content_type:
        return ".png"
    return ".dat"


class DownloadThread(QThread):
    progress = pyqtSignal(int, int)  # downloaded, total
    finished = pyqtSignal(str, str)  # local_path, content_type
    error = pyqtSignal(str)

    def __init__(self, url, content_type="auto", timeout=30):
        super().__init__()
        self.url = url
        self.content_type = content_type
        self.timeout = timeout
        self.cancelled = False

    def run(self):
        try:
            with requests.get(self.url, stream=True, timeout=self.timeout) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                suffix = guess_suffix(self.url, resp.headers.get("content-type"))
                downloaded = 0
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if self.cancelled:
                            tmp.close()
                            os.unlink(tmp.name)
                            return
                        if chunk:
                            tmp.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                self.progress.emit(downloaded, total)
                    tmp_path = tmp.name
            final_type = self.content_type if self.content_type != "auto" else resp.headers.get("content-type", "auto")
            self.finished.emit(tmp_path, final_type)
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self.cancelled = True
