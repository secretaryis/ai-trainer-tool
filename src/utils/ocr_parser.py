import logging
from typing import Optional
from utils.deps import require

pyt_avail, pyt_msg = require("pytesseract", "pytesseract", optional=True)
pil_avail, pil_msg = require("PIL", "Pillow", optional=True)
easy_avail, easy_msg = require("easyocr", "easyocr", optional=True)

if pyt_avail:
    import pytesseract  # type: ignore
if pil_avail:
    from PIL import Image  # type: ignore
if easy_avail:
    import easyocr  # type: ignore


class OCRParser:
    def __init__(self, engine: str = "tesseract", languages: str = "ara+eng"):
        self.engine = engine
        self.languages = languages
        self._easy_reader = None

    def _get_easy_reader(self):
        if not easy_avail:
            raise ImportError(easy_msg)
        if self._easy_reader is None:
            langs = [lang.strip() for lang in self.languages.split("+") if lang.strip()]
            self._easy_reader = easyocr.Reader(langs, gpu=False)
        return self._easy_reader

    def image_to_text(self, image, lang: Optional[str] = None):
        lang = lang or self.languages
        if self.engine == "easyocr":
            return self._easyocr(image, lang)
        # default tesseract
        return self._tesseract(image, lang)

    def _tesseract(self, image, lang):
        if not (pyt_avail and pil_avail):
            raise ImportError(pyt_msg if not pyt_avail else pil_msg)
        config = f"--oem 3 --psm 6 -l {lang}"
        return pytesseract.image_to_string(image, config=config)

    def _easyocr(self, image, lang):
        reader = self._get_easy_reader()
        # easyocr expects filepath or ndarray; convert PIL to ndarray
        try:
            import numpy as np  # type: ignore
        except ImportError:
            raise ImportError("numpy required for easyocr")
        arr = np.array(image)
        results = reader.readtext(arr, detail=0)
        return "\n".join(results)
