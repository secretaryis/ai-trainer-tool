import builtins
import types

from utils import deps


def test_require_missing_optional_returns_message():
    ok, msg = deps.require("definitely_missing_mod_xyz", "MissingMod", optional=True)
    assert not ok
    assert "pip install" in msg
