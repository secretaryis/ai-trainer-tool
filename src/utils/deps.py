import importlib
import sys
from typing import Tuple

DEFAULT_HINTS = {
    "torch": "pip install torch --extra-index-url https://download.pytorch.org/whl/cpu",
    "transformers": "pip install transformers",
    "bs4": "pip install beautifulsoup4",
    "PyPDF2": "pip install PyPDF2",
    "pdfplumber": "pip install pdfplumber",
    "pytesseract": "pip install pytesseract ; install system tesseract-ocr",
    "pandas": "pip install pandas",
}


def require(module_name: str, friendly: str = None, install_hint: str = None, optional: bool = True) -> Tuple[bool, str]:
    """
    Attempt to import module_name.
    Returns (ok, message). If optional=False and missing, raises ImportError with hint.
    """
    friendly = friendly or module_name
    if install_hint is None:
        install_hint = DEFAULT_HINTS.get(module_name, f"pip install {module_name}")
    try:
        importlib.import_module(module_name)
        return True, f"{friendly} available"
    except (ImportError, ModuleNotFoundError) as e:
        msg = f"{friendly} not installed. Install via: {install_hint}"
        if optional:
            return False, msg
        raise ImportError(msg) from e


def require_ocr():
    ok, msg = require('pytesseract', friendly='Tesseract OCR', optional=True)
    if not ok:
        return ok, msg
    import shutil
    if shutil.which('tesseract') is None:
        return False, "لم يتم العثور على برنامج Tesseract في النظام. يرجى تثبيته عبر مدير الحزم."
    return True, None


def mark_missing_in_sys(module_name: str):
    """Utility for tests to mark a module missing."""
    sys.modules.pop(module_name, None)
