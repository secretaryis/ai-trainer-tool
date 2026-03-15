from core.model_manager import ModelManager


def test_license_cache_defaults():
    mm = ModelManager()
    lic = mm.fetch_license("gpt2")
    assert isinstance(lic, str)


def test_unload_clears_refs():
    mm = ModelManager()
    mm.loaded_model = object()
    mm.loaded_tokenizer = object()
    mm.unload_model()
    assert mm.loaded_model is None
    assert mm.loaded_tokenizer is None
