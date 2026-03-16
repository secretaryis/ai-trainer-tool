import types
import pytest

from core.data_loader import DataLoader
from utils.pdf_parser import PDFParser


def test_needs_ocr_detects_bad_chars():
    parser = PDFParser()
    text = "�� �� ��"
    assert parser._needs_ocr(text, ["", ""])


def test_needs_ocr_detects_short_text():
    parser = PDFParser()
    assert parser._needs_ocr("hi", [])


def test_data_loader_passes_backend(monkeypatch, tmp_path):
    dl = DataLoader()
    calls = {}

    def fake_extract(pdf_path, use_ocr=True, lang=None, backend=None):
        calls["backend"] = backend
        return "text"

    monkeypatch.setattr(dl.pdf_parser, "extract_text", fake_extract)
    dl.extract_backend = "fast"
    pdf = tmp_path / "file.pdf"
    pdf.write_text("stub")
    _ = dl.load_pdf_data(str(pdf), use_ocr=False, lang="eng")
    assert calls["backend"] == "fast"
