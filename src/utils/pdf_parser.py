import os
import logging
import re
import subprocess
import tempfile
import shutil
import hashlib
import concurrent.futures
import unicodedata
from typing import List
from utils.deps import require, require_ocr
from utils.ocr_parser import OCRParser
try:
    import arabic_reshaper  # type: ignore
except ImportError:
    arabic_reshaper = None

try:
    from bidi.algorithm import get_display  # type: ignore
except ImportError:
    get_display = None

pdfplumber_available, _ = require("pdfplumber", "pdfplumber", optional=True)
pypdf_available, _ = require("PyPDF2", "PyPDF2", optional=True)
fitz_available, _ = require("fitz", "pymupdf", optional=True)
pdfplumber = None
PyPDF2 = None
fitz = None
if pdfplumber_available:
    import pdfplumber  # type: ignore
if pypdf_available:
    import PyPDF2  # type: ignore
if fitz_available:
    import fitz  # type: ignore


class PDFParser:
    def __init__(self):
        self.last_stats = None

    def extract_text(self, pdf_path, use_ocr=True, lang="ara+eng", backend="auto", use_cache=True, ocr_engine="tesseract", max_workers=None):
        """
        Extract text from PDF using multiple fallbacks with cleaning.
        If extraction looks poor and OCR is allowed, fall back to OCR.
        """
        self.last_stats = {
            "backend": backend,
            "ocr_used": False,
            "pages": 0,
            "poor_pages": 0,
            "cache_hit": False,
            "ocr_pages": 0,
            "elapsed": None,
        }

        if backend not in ("auto", "fast", "enhanced"):
            backend = "auto"

        cache_key = None
        if use_cache:
            cache_key = self._cache_key(pdf_path, lang, backend, ocr_engine)
            cached = self._load_cache(cache_key)
            if cached:
                self.last_stats["cache_hit"] = True
                return cached

        # fast path: existing flow
        text_chunks = []
        page_texts = []

        def _extract_with_pymupdf(path):
            local_chunks = []
            local_page_texts = []
            if fitz:
                try:
                    doc = fitz.open(path)
                    for page in doc:
                        txt = page.get_text("blocks") or page.get_text()
                        local_page_texts.append(txt or "")
                        if txt:
                            local_chunks.append(txt)
                except Exception as e:
                    logging.warning(f"pymupdf failed: {e}")
            return local_chunks, local_page_texts

        text_chunks, page_texts = _extract_with_pymupdf(pdf_path)
        self.last_stats["pages"] = len(page_texts) if page_texts else 0

        # 2) pdfplumber
        if not text_chunks and pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text()
                        if txt:
                            text_chunks.append(txt)
            except Exception as e:
                logging.warning(f"pdfplumber failed: {e}")

        # 3) PyPDF2
        if not text_chunks and PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        txt = page.extract_text()
                        if txt:
                            text_chunks.append(txt)
            except Exception as e:
                logging.warning(f"PyPDF2 failed: {e}")

        cleaned = self._clean("\n".join(text_chunks)) if text_chunks else ""
        extraction_poor = self._needs_ocr(cleaned, page_texts)

        # 4) Enhanced path (ocrmypdf) if requested or auto triggers
        should_use_enhanced = backend == "enhanced" or (backend == "auto" and extraction_poor)
        if should_use_enhanced and use_ocr:
            ok, msg = require_ocr()
            if not ok:
                logging.warning(msg)
            else:
                searchable_pdf = None
                try:
                    searchable_pdf = self._run_ocrmypdf(pdf_path, lang=lang)
                    if searchable_pdf:
                        self.last_stats["ocr_used"] = True
                        chunks2, pages2 = _extract_with_pymupdf(searchable_pdf)
                        if pages2:
                            self.last_stats["pages"] = len(pages2)
                        page_texts = pages2 or page_texts
                        cleaned = self._clean("\n".join(chunks2)) if chunks2 else cleaned
                        extraction_poor = False
                except Exception as e:
                    logging.warning(f"ocrmypdf failed: {e}")
                finally:
                    if searchable_pdf and os.path.exists(searchable_pdf):
                        try:
                            os.remove(searchable_pdf)
                        except Exception:
                            pass

        # 5) OCR fallback per-page images if still poor
        if (not cleaned or extraction_poor) and use_ocr:
            ok, msg = require_ocr()
            if not ok:
                logging.warning(msg)
            else:
                try:
                    ocr_text = self._ocr_pdf(pdf_path, lang=lang, engine=ocr_engine, max_workers=max_workers)
                    if ocr_text:
                        self.last_stats["ocr_used"] = True
                        cleaned = self._clean(ocr_text)
                        extraction_poor = False
                except Exception as e:
                    logging.warning(f"OCR fallback failed: {e}")

        if use_cache and cache_key and cleaned:
            self._store_cache(cache_key, cleaned)

        return cleaned if cleaned else None

    def _ocr_pdf(self, pdf_path, lang="ara+eng", engine="tesseract", max_workers=None):
        from pdf2image import convert_from_path
        from PIL import ImageOps, ImageFilter

        images = convert_from_path(pdf_path, dpi=250)
        total_pages = len(images)
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 2)
        if total_pages <= 1:
            max_workers = 1

        parser = OCRParser(engine=engine, languages=lang)

        def _do_ocr(idx_img):
            idx, img = idx_img
            gray = ImageOps.grayscale(img)
            enhanced = ImageOps.autocontrast(gray)
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            text = parser.image_to_text(enhanced, lang=lang)
            return idx, text

        if max_workers <= 1:
            results = [_do_ocr(pair) for pair in enumerate(images)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
                results = list(exe.map(_do_ocr, enumerate(images)))

        results.sort(key=lambda x: x[0])
        self.last_stats["ocr_pages"] = total_pages
        return "\n".join(r[1] for r in results if r[1])

    def _clean(self, text: str):
        text = text or ""
        text = re.sub(r"[\t\r]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = self._normalize_direction(text.strip())
        return text

    # Cache helpers
    def _cache_dir(self):
        path = os.path.join(".cache", "ocr")
        os.makedirs(path, exist_ok=True)
        return path

    def _cache_key(self, pdf_path, lang, backend, engine):
        stat = os.stat(pdf_path)
        h = hashlib.sha256()
        h.update(str(stat.st_mtime).encode())
        h.update(str(stat.st_size).encode())
        h.update(lang.encode())
        h.update(backend.encode())
        h.update(engine.encode())
        return h.hexdigest()

    def _cache_path(self, key):
        return os.path.join(self._cache_dir(), f"{key}.txt")

    def _load_cache(self, key):
        path = self._cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return None
        return None

    def _store_cache(self, key, text):
        try:
            with open(self._cache_path(key), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    def _is_arabic_char(self, c: str) -> bool:
        return (
            "\u0600" <= c <= "\u06FF" or
            "\u0750" <= c <= "\u077F" or
            "\u08A0" <= c <= "\u08FF"
        )

    def _arabic_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        arabic_count = sum(1 for c in letters if self._is_arabic_char(c))
        return arabic_count / len(letters)

    def _fix_arabic_line(self, line: str) -> str:
        ratio = self._arabic_ratio(line)
        # If mixed latin and arabic, avoid reordering; just cleanup
        has_latin = any("a" <= c.lower() <= "z" for c in line)
        base = line.replace("|", " ")
        if has_latin and 0.1 < ratio < 0.6:
            return re.sub(r" +", " ", base)

        cleaned = base
        tokens = cleaned.split()
        # Merge consecutive single-letter arabic tokens
        merged_tokens = []
        buffer = []
        for t in tokens:
            if len(t) == 1 and self._is_arabic_char(t):
                buffer.append(t)
            else:
                if buffer:
                    merged_tokens.append("".join(buffer))
                    buffer = []
                merged_tokens.append(t)
        if buffer:
            merged_tokens.append("".join(buffer))
        cleaned = " ".join(merged_tokens)
        # If shaping libs available, use them for proper glyph joining + RTL order
        shaped_ok = False
        if arabic_reshaper and get_display:
            try:
                reshaped = arabic_reshaper.reshape(cleaned)
                displayed = get_display(reshaped)
                cleaned = re.sub(r"[|]+", " ", re.sub(r" +", " ", displayed)).strip()
                shaped_ok = True
            except Exception:
                pass

        if shaped_ok:
            return cleaned.replace("|", " ")

        # Fallback: reverse token order to improve readability
        tokens = cleaned.split()
        result = " ".join(reversed(tokens)).replace("|", " ")
        return result

    def _normalize_direction(self, text: str) -> str:
        lines = text.split("\n")
        fixed = []
        for ln in lines:
            stripped = unicodedata.normalize("NFKC", ln).strip()
            if not stripped:
                fixed.append("")
                continue
            fixed.append(self._fix_arabic_line(stripped))
        return "\n".join(fixed)

    def _needs_ocr(self, text: str, page_texts: List[str]) -> bool:
        if self.last_stats is None:
            self.last_stats = {"poor_pages": 0}
        if not text or len(text.strip()) < 50:
            return True
        letters = sum(1 for c in text if c.isalpha())
        ratio = letters / max(len(text), 1)
        bad_chars = text.count("�") / max(len(text), 1)
        presentation = sum(1 for c in text if ("\uFB50" <= c <= "\uFDFF") or ("\uFE70" <= c <= "\uFEFF")) / max(len(text), 1)
        poor_pages = 0
        for ptxt in page_texts or []:
            if not ptxt or len(ptxt.strip()) < 10:
                poor_pages += 1
            elif ptxt.count("�") / max(len(ptxt), 1) > 0.2:
                poor_pages += 1
        self.last_stats["poor_pages"] = poor_pages
        if bad_chars > 0.1 or ratio < 0.2 or presentation > 0.05:
            return True
        if page_texts and poor_pages / max(len(page_texts), 1) > 0.3:
            return True
        return False

    def _run_ocrmypdf(self, pdf_path: str, lang: str = "ara+eng") -> str:
        """Run ocrmypdf to produce a searchable PDF; returns path to temp file."""
        if shutil.which("ocrmypdf") is None:
            raise RuntimeError("ocrmypdf CLI not found; install system package.")
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        cmd = ["ocrmypdf", "--quiet", "--force-ocr", "--rotate-pages", "--deskew", "-l", lang, pdf_path, tmp_path]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_path

    def get_page_count(self, pdf_path):
        """Get number of pages in PDF."""
        try:
            if PyPDF2:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return len(pdf_reader.pages)
            if fitz:
                return len(fitz.open(pdf_path))
            return 0
        except Exception:
            return 0
