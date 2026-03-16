import subprocess
import sys
import time
import shutil

DEFAULT_RETRIES = 3
BACKOFF_BASE = 2


def retryable(func, *args, retries=DEFAULT_RETRIES, **kwargs):
    """Execute func with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(BACKOFF_BASE ** attempt)


def pip_install(package, venv_prefix=None):
    """Install a package via pip inside current (or provided) venv."""
    python_exec = sys.executable if venv_prefix is None else f"{venv_prefix}/bin/python"
    cmd = [python_exec, "-m", "pip", "install", package]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def has_tool(name):
    """Check if a CLI tool exists."""
    return shutil.which(name) is not None
