import json
import os
from copy import deepcopy

DEFAULT_CONFIG = {
    "language": "en",
    "theme": "light",
    "last_model": "",
    "training_defaults": {
        "batch_size": 4,
        "max_length": 128,
        "epochs": 3,
        "grad_accum": 1,
        "dataloader_workers": 0,
        "seed": 42,
    },
    "inference_defaults": {
        "temperature": 0.2,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "max_new_tokens": 64,
        "seed": 42,
    },
    "export_formats": ["pytorch", "safetensors"],
    "data_clean_level": "medium",
    "data_lang_detect": True,
    "auto_clean": True,
    "extract_backend": "auto",
    "extract_tables": False,
    "post_spellcheck": False,
    "ocr_engine": "tesseract",
    "use_cache": True,
    "max_workers": 0,
    "window_geometry": None,
}


class Config:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def _merge_defaults(self, loaded):
        merged = deepcopy(DEFAULT_CONFIG)
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
        return merged

    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    return self._merge_defaults(loaded)
            except Exception:
                pass
        return deepcopy(DEFAULT_CONFIG)

    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value
        self.save_config()

    def update_training_defaults(self, batch_size=None, max_length=None, epochs=None, grad_accum=None, dataloader_workers=None, seed=None):
        """Update training default settings."""
        train = self.config.setdefault("training_defaults", {})
        if batch_size is not None:
            train["batch_size"] = batch_size
        if max_length is not None:
            train["max_length"] = max_length
        if epochs is not None:
            train["epochs"] = epochs
        if grad_accum is not None:
            train["grad_accum"] = grad_accum
        if dataloader_workers is not None:
            train["dataloader_workers"] = dataloader_workers
        if seed is not None:
            train["seed"] = seed
        self.save_config()

    def update_inference_defaults(self, temperature=None, top_p=None, repetition_penalty=None, max_new_tokens=None, seed=None):
        """Update inference default settings."""
        inf = self.config.setdefault("inference_defaults", {})
        if temperature is not None:
            inf["temperature"] = float(temperature)
        if top_p is not None:
            inf["top_p"] = float(top_p)
        if repetition_penalty is not None:
            inf["repetition_penalty"] = float(repetition_penalty)
        if max_new_tokens is not None:
            inf["max_new_tokens"] = int(max_new_tokens)
        if seed is not None:
            inf["seed"] = int(seed)
        self.save_config()

    def get_training_defaults(self):
        """Get training default settings."""
        return self.config.get("training_defaults", {})

    def get_inference_defaults(self):
        """Get inference default settings."""
        return self.config.get("inference_defaults", {})
