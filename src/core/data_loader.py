try:
    from datasets import Dataset
except ImportError:
    class Dataset:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_dict(cls, data):
            return cls(list(data['text']))

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            for text in self.data:
                yield {"text": text}

        def select(self, indices):
            return Dataset([self.data[i] for i in indices])
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from .html_parser import HTMLParser
from .ocr_parser import OCRParser
from .online_loader import guess_suffix
from utils.deps import require
from utils.postprocess import SpellPostProcessor
import os
import csv
import sqlite3
import re
import importlib
import tempfile
import requests

try:
    from langdetect import detect as _lang_detect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False



class DataLoader:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()
        self.html_parser = HTMLParser(max_depth=1)
        self.ocr_parser = OCRParser()
        self.history = []
        self.redo_stack = []
        self.auto_clean = True
        self.clean_level = "medium"
        self.use_lang_detect = True
        self.last_clean_stats = None
        self.extract_backend = "auto"
        self.last_quality_stats = None
        self.use_cache = True
        self.max_workers = None
        self.ocr_engine = "tesseract"
        self.extract_tables = False
        self.post_spellcheck = False
        self.spell_processor = SpellPostProcessor()

    def load_text_data(self, text, auto_clean=None):
        """Load data from plain text."""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Split text into lines or paragraphs
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        data = {'text': lines}
        ds = Dataset.from_dict(data)

        if auto_clean is None:
            auto_clean = self.auto_clean
        if auto_clean:
            ds = self.clean_dataset(ds, level=self.clean_level, lang_detection=self.use_lang_detect)
        else:
            self.last_clean_stats = {"original": len(ds), "kept": len(ds), "removed": 0}
        if self.post_spellcheck and self.spell_processor:
            ds = self._spellcheck_dataset(ds)
        return ds

    def load_pdf_data(self, pdf_path, use_ocr=True, lang="ara+eng", auto_clean=None, backend=None):
        """Load data from PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        backend = backend or self.extract_backend
        try:
            text = self.pdf_parser.extract_text(
                pdf_path,
                use_ocr=use_ocr,
                lang=lang,
                backend=backend,
                use_cache=self.use_cache,
                ocr_engine=self.ocr_engine,
                max_workers=self.max_workers,
            )
        except TypeError:
            # backward compatibility for tests with mocked extract_text
            text = self.pdf_parser.extract_text(pdf_path, use_ocr=use_ocr, lang=lang, backend=backend)
        self.last_quality_stats = getattr(self.pdf_parser, "last_stats", None)
        if not text:
            raise ValueError("No text extracted from PDF")
        # tables
        if self.extract_tables:
            tables = self._extract_tables(pdf_path)
            if tables:
                text = text + "\n\n" + "\n\n".join(f"[TABLE] {t}" for t in tables)
        ds = self.load_text_data(text, auto_clean=auto_clean)
        if self.post_spellcheck and self.spell_processor:
            ds = self._spellcheck_dataset(ds)
        return ds

    def load_docx_data(self, docx_path):
        if not os.path.exists(docx_path):
            raise FileNotFoundError(docx_path)
        text = self.docx_parser.extract_text(docx_path)
        return self.load_text_data(text)

    def load_html_data(self, html_path):
        if not os.path.exists(html_path):
            raise FileNotFoundError(html_path)
        ok, msg = require("bs4", "BeautifulSoup (bs4)", optional=True)
        if not ok:
            raise ImportError(msg)
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            text = self.html_parser.extract_text(f.read())
        return self.load_text_data(text)

    def load_web_data(self, url, depth=1):
        ok, msg = require("bs4", "BeautifulSoup (bs4)", optional=True)
        if not ok:
            raise ImportError(msg)
        self.html_parser.max_depth = depth
        texts = self.html_parser.crawl(url, depth=0)
        return self.load_text_data("\n".join(texts))

    def load_csv_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        rows = []
        with open(csv_path, newline='', encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                joined = " ".join(row).strip()
                if joined:
                    rows.append(joined)
        return Dataset.from_dict({"text": rows})

    def load_excel_data(self, excel_path):
        if not os.path.exists(excel_path):
            raise FileNotFoundError(excel_path)
        ok, msg = require("pandas", "pandas", optional=True)
        if not ok:
            raise ImportError(msg)
        pd = importlib.import_module("pandas")
        df = pd.read_excel(excel_path)
        joined = df.astype(str).agg(" ".join, axis=1).tolist()
        return Dataset.from_dict({"text": joined})

    def load_sqlite_data(self, sqlite_path, table, text_columns):
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(sqlite_path)
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cols = ", ".join(text_columns)
        cur.execute(f"SELECT {cols} FROM {table}")
        rows = []
        for row in cur.fetchall():
            joined = " ".join([str(x) for x in row]).strip()
            if joined:
                rows.append(joined)
        conn.close()
        return Dataset.from_dict({"text": rows})

    def load_image_data(self, image_path):
        text = self.ocr_parser.extract_text(image_path)
        return self.load_text_data(text)

    def load_url(self, url, content_type="auto", use_ocr=True, lang="ara+eng", auto_clean=None, backend=None):
        """Download from URL then route to appropriate loader."""
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        suffix = guess_suffix(url, resp.headers.get("content-type"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        try:
            ctype = content_type
            path_lower = tmp_path.lower()
            if ctype == "auto":
                if path_lower.endswith(".pdf"):
                    ctype = "pdf"
                elif path_lower.endswith(".html") or path_lower.endswith(".htm"):
                    ctype = "html"
                elif path_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    ctype = "image"
                else:
                    ctype = "text"

            if ctype == "pdf":
                ds = self.load_pdf_data(tmp_path, use_ocr=use_ocr, lang=lang, auto_clean=auto_clean, backend=backend)
                return ds
            if ctype == "html":
                return self.load_html_data(tmp_path)
            if ctype == "image":
                return self.load_image_data(tmp_path)
            return self.load_text_data(open(tmp_path, "r", encoding="utf-8", errors="ignore").read(), auto_clean=auto_clean)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def preview_data(self, dataset, max_items=5):
        """Get preview of the dataset."""
        preview = []
        for i, item in enumerate(dataset):
            if i >= max_items:
                break
            preview.append(item['text'][:100] + "..." if len(item['text']) > 100 else item['text'])
        return preview

    def get_dataset_info(self, dataset):
        """Get information about the dataset."""
        columns = []
        if hasattr(dataset, "column_names"):
            try:
                columns = list(dataset.column_names)
            except Exception:
                columns = []
        elif len(dataset):
            columns = ["text"]
        return {
            'num_samples': len(dataset),
            'columns': columns,
        }

    # Cleaning utilities
    def clean_dataset(self, dataset, level="medium", lang_detection=True, return_stats=False):
        """Clean and deduplicate dataset lines; stores last_clean_stats."""
        original = len(dataset)
        if level == "off":
            stats = {"original": original, "kept": original, "removed": 0}
            self.last_clean_stats = stats
            return (dataset, stats) if return_stats else dataset
        kept = []
        removed = 0
        seen = set()

        for item in dataset:
            keep, cleaned = self._clean_line(item["text"], level=level, lang_detection=lang_detection)
            if not keep:
                removed += 1
                continue
            normalized = cleaned.strip()
            if normalized in seen:
                removed += 1
                continue
            seen.add(normalized)
            kept.append(normalized)

        cleaned_ds = Dataset.from_dict({"text": kept})
        stats = {"original": original, "kept": len(kept), "removed": removed}
        self.last_clean_stats = stats
        if return_stats:
            return cleaned_ds, stats
        return cleaned_ds

    def _spellcheck_dataset(self, dataset):
        if not self.spell_processor:
            return dataset
        corrected = []
        for item in dataset:
            txt = item["text"]
            ratio = self._arabic_ratio(txt)
            corrected.append(self.spell_processor.correct_line(txt, arabic_ratio=ratio))
        return Dataset.from_dict({"text": corrected})

    def _detect_lang(self, text: str) -> str:
        if not LANGDETECT_AVAILABLE:
            return "unknown"
        snippet = text.strip()
        if len(snippet) < 3:
            return "unknown"
        try:
            return _lang_detect(snippet)
        except Exception:
            return "unknown"

    def _arabic_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" or "\u08A0" <= c <= "\u08FF")
        return arabic / len(letters)

    def _shape_arabic_if_needed(self, text: str, ratio: float) -> str:
        if ratio < 0.35:
            return text
        try:
            import arabic_reshaper  # type: ignore
            from bidi.algorithm import get_display  # type: ignore
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text

    def _clean_line(self, text: str, level="medium", lang_detection=True):
        original_no_pipe = (text or "").replace("|", "").strip()
        txt = text or ""
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"http\S+", " ", txt)
        txt = re.sub(r"[\t\r]+", " ", txt)
        txt = re.sub(r" +", " ", txt)
        txt = txt.strip()
        if not txt:
            return False, ""

        lang = self._detect_lang(txt) if lang_detection else "unknown"
        ratio = self._arabic_ratio(txt)
        has_latin = any("a" <= c.lower() <= "z" for c in txt)
        if (lang == "ar" or ratio >= 0.35) and not (has_latin and ratio < 0.6):
            txt = self._shape_arabic_if_needed(txt, ratio)

        # Noise filters
        if level == "strong":
            if len(txt) < 4:
                return False, ""
            words = txt.split()
            if len(words) < 2:
                return False, ""
        if txt.count("�") / max(len(txt), 1) > 0.2:
            return False, ""

        txt = txt.replace("|", " ")
        if (lang == "ar" or ratio >= 0.35) and not has_latin:
            if len(txt) < len(original_no_pipe):
                txt = original_no_pipe
        return True, txt

    # Filtering utilities
    def filter_deduplicate(self, dataset):
        seen = set()
        filtered = []
        for item in dataset:
            text = item["text"]
            if text not in seen:
                seen.add(text)
                filtered.append(text)
        return Dataset.from_dict({"text": filtered})

    def filter_min_length(self, dataset, min_words=3):
        filtered = []
        for item in dataset:
            if len(item["text"].split()) >= min_words:
                filtered.append(item["text"])
        return Dataset.from_dict({"text": filtered})

    def filter_keywords(self, dataset, include=None, exclude=None):
        include = include or []
        exclude = exclude or []
        filtered = []
        for item in dataset:
            txt = item["text"].lower()
            if include and not any(k.lower() in txt for k in include):
                continue
            if exclude and any(k.lower() in txt for k in exclude):
                continue
            filtered.append(item["text"])
        return Dataset.from_dict({"text": filtered})

    def filter_clean_html_links(self, dataset):
        def clean(text):
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"<[^>]+>", "", text)
            return text.strip()
        return Dataset.from_dict({"text": [clean(item["text"]) for item in dataset]})

    def train_val_split(self, dataset, val_percent=0.1):
        split_idx = int(len(dataset) * (1 - val_percent))
        train = dataset.select(range(split_idx))
        val = dataset.select(range(split_idx, len(dataset)))
        return train, val

    def _extract_tables(self, pdf_path):
        tables_text = []
        ok_camelot, camelot_msg = require("camelot", "camelot-py", optional=True)
        if ok_camelot:
            import camelot  # type: ignore
            try:
                tables = camelot.read_pdf(pdf_path, pages="1-end")
                for t in tables:
                    try:
                        tables_text.append("\n".join([" ".join(map(str, row)) for row in t.data]))
                    except Exception:
                        continue
            except Exception:
                pass
        else:
            ok_tabula, tabula_msg = require("tabula", "tabula-py", optional=True)
            if ok_tabula:
                import tabula  # type: ignore
                try:
                    dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
                    for df in dfs:
                        tables_text.append(" ".join(df.astype(str).stack().tolist()))
                except Exception:
                    pass
        return tables_text

    def push_history(self, dataset):
        self.history.append(dataset)
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            ds = self.history.pop()
            self.redo_stack.append(ds)
            return ds
        return None

    def redo(self):
        if self.redo_stack:
            ds = self.redo_stack.pop()
            self.history.append(ds)
            return ds
        return None
