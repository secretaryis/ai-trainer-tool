import sys
import importlib
from utils import deps


def test_require_missing_optional_returns_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "nonexistent_mod", None)
    ok, msg = deps.require("nonexistent_mod", "NonExistent", optional=True)
    assert not ok
    assert "pip install nonexistent_mod" in msg


def test_require_present(monkeypatch):
    monkeypatch.setitem(sys.modules, "math", importlib.import_module("math"))
    ok, msg = deps.require("math", "math", optional=False)
    assert ok
    assert "available" in msg
