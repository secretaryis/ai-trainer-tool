import os
from utils.deps import require

pyt_avail, pyt_msg = require("pytesseract", "pytesseract", optional=True)
pil_avail, pil_msg = require("PIL", "Pillow", optional=True)
if pyt_avail:
    import pytesseract  # type: ignore
if pil_avail:
    from PIL import Image  # type: ignore


class OCRParser:
    def extract_text(self, image_path):
        if not (pyt_avail and pil_avail):
            raise ImportError(f"OCR unavailable: {pyt_msg if not pyt_avail else pil_msg}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        return pytesseract.image_to_string(Image.open(image_path))
