#!/usr/bin/env bash
set -e
echo "Installing OCR dependencies (pytesseract, pillow). External tesseract binary required."
python3 -m pip install pytesseract pillow pdf2image pymupdf
echo "Install system tesseract with your package manager, e.g.:"
echo "  sudo apt install tesseract-ocr"
