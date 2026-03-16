import types
import sys
from core import exporter


class DummyModel:
    class Config:
        pass

    config = Config()

    def save_pretrained(self, path):
        return None


class DummyTokenizer:
    def save_pretrained(self, path):
        return None


class DummyOnnxConfig:
    DEFAULT_ONNX_OPSET = 13


def test_export_onnx_uses_features_manager(monkeypatch, tmp_path):
    calls = {}

    def fake_check_supported_model_or_raise(model, feature=None):
        calls["feature"] = feature
        return None, lambda cfg: DummyOnnxConfig()

    def fake_export(preprocessor, model, config, opset, output):
        calls["export"] = {"opset": opset, "output": str(output)}

    monkeypatch.setattr(exporter, "AutoTokenizer", object())
    fake_module = types.SimpleNamespace(FeaturesManager=types.SimpleNamespace(check_supported_model_or_raise=fake_check_supported_model_or_raise), export=fake_export)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(onnx=fake_module))
    monkeypatch.setitem(sys.modules, "transformers.onnx", fake_module)

    exp = exporter.ModelExporter(DummyModel(), DummyTokenizer(), "")
    ok, msg = exp.export_onnx(tmp_path)
    assert ok is True
    assert "model.onnx" in calls["export"]["output"]
    assert calls["feature"] == "text-generation"
