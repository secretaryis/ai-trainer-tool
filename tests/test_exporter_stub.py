from core.exporter import ModelExporter


def test_convert_script_path_is_string():
    exp = ModelExporter(None, None, "")
    assert exp.convert_script is None or isinstance(exp.convert_script, str)
