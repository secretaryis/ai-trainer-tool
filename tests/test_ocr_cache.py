import os
from utils.pdf_parser import PDFParser


def test_cache_store_and_load(tmp_path):
    parser = PDFParser()
    fake_pdf = tmp_path / "file.pdf"
    fake_pdf.write_text("dummy")
    key = parser._cache_key(str(fake_pdf), "eng", "auto", "tesseract")
    parser._store_cache(key, "hello")
    cached = parser._load_cache(key)
    assert cached == "hello"


def test_needs_ocr_presentation_forms():
    parser = PDFParser()
    sample = "ﺍ ﺯ ﺭ ﺓ " * 10
    assert parser._needs_ocr(sample, [])
