import os
from integration.ollama_bridge import OllamaBridge


def test_create_modelfile(tmp_path):
    bridge = OllamaBridge()
    ok, msg = bridge.create_modelfile("model.gguf", "test-model", modelfile_path=tmp_path / "Modelfile")
    assert ok
    assert (tmp_path / "Modelfile").exists()
    content = (tmp_path / "Modelfile").read_text()
    assert "FROM model.gguf" in content
