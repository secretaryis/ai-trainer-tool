#!/usr/bin/env bash
set -euo pipefail

# Single installer/runner for AI Trainer Tool.
# Options:
#   --with-ocr    install OCR python deps (pytesseract, pillow) — requires system tesseract binary.
#   --with-dev    install dev extras (pytest).
#   --run         launch app after install.

WITH_OCR=0
WITH_DEV=0
RUN_APP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ocr) WITH_OCR=1 ;;
    --with-dev) WITH_DEV=1 ;;
    --run) RUN_APP=1 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

if ! command -v python3 >/dev/null; then
  echo "python3 not found"; exit 1
fi
if ! command -v pip >/dev/null; then
  echo "pip not found"; exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ $WITH_OCR -eq 1 ]]; then
  pip install pytesseract pillow
  echo "Reminder: install system tesseract (e.g., sudo apt install tesseract-ocr)."
fi

if [[ $WITH_DEV -eq 1 ]]; then
  pip install pytest
fi

cat > ai-trainer-tool.desktop <<EOF
[Desktop Entry]
Name=AI Trainer Tool
Exec=$(pwd)/.venv/bin/python $(pwd)/src/main.py
Type=Application
Categories=Utility;Development;
EOF

echo "Install complete."
echo "Run: source .venv/bin/activate && python src/main.py"
echo "Desktop entry: $(pwd)/ai-trainer-tool.desktop"
echo "Optional deps: torch/transformers (models), PyPDF2 or pdfplumber (PDF), bs4 (HTML), pandas/openpyxl (Excel), pytesseract+pillow (+system tesseract) for OCR."

if [[ $RUN_APP -eq 1 ]]; then
  python src/main.py
fi
