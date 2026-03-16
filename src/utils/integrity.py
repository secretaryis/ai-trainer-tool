import os
from typing import List, Tuple

REQUIRED_PATHS = [
    "src/core/pdf_parser.py",
    "src/core/docx_parser.py",
    "src/core/html_parser.py",
    "src/core/ocr_parser.py",
    "src/core/audio_parser.py",
    "src/utils/memory.py",
    "src/utils/network.py",
    "src/models/__init__.py",
]


def check_files(base_dir=".") -> List[Tuple[str, bool]]:
    results = []
    for rel in REQUIRED_PATHS:
        path = os.path.join(base_dir, rel)
        results.append((rel, os.path.exists(path)))
    return results


def missing(base_dir=".") -> List[str]:
    return [rel for rel, ok in check_files(base_dir) if not ok]


def env_warnings(base_dir=".") -> List[str]:
    """Collect non-fatal warnings about environment and filesystem health."""
    warnings = []
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        warnings.append("DISPLAY not set; Qt windows may not open.")
    cache_dir = os.path.join(base_dir, ".cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        test_path = os.path.join(cache_dir, "write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception:
        warnings.append(f"Cache directory not writable: {cache_dir}")
    return warnings
