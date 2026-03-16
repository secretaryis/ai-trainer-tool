from utils.config import Config, DEFAULT_CONFIG


def test_config_round_trip(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg = Config(config_file=str(cfg_path))
    cfg.set("language", "ar")
    cfg.update_training_defaults(batch_size=8, grad_accum=2)
    cfg.update_inference_defaults(temperature=0.3, max_new_tokens=32)

    # reload
    cfg2 = Config(config_file=str(cfg_path))
    train = cfg2.get_training_defaults()
    infer = cfg2.get_inference_defaults()
    assert cfg2.get("language") == "ar"
    assert train["batch_size"] == 8
    assert train["grad_accum"] == 2
    assert infer["temperature"] == 0.3
    assert infer["max_new_tokens"] == 32
    # ensure defaults merged
    assert cfg2.get("data_clean_level", None) == DEFAULT_CONFIG["data_clean_level"]
