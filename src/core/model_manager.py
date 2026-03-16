from huggingface_hub import HfApi
import importlib
from utils.deps import require
torch_ok, torch_msg = require("torch", "PyTorch", optional=True)
if torch_ok:
    import torch  # type: ignore
else:
    torch = type("TorchFallback", (), {
        "cuda": type("Cuda", (), {
            "is_available": staticmethod(lambda: False),
            "device_count": staticmethod(lambda: 0),
            "empty_cache": staticmethod(lambda: None)
        })(),
        "float32": None,
    })()
import os
import time
import gc
import requests

DEFAULT_RETRIES = 3
RETRY_BACKOFF = 2

class ModelManager:
    def __init__(self):
        self.api = HfApi()
        self.popular_models = [
            'gpt2',
            'gpt2-medium',
            'distilgpt2',
            'microsoft/DialoGPT-small',
            'microsoft/DialoGPT-medium',
            'EleutherAI/gpt-neo-125M',
            'EleutherAI/gpt-neo-1.3B',
        ]
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.license_cache = {}
        self.model_info_cache = {}

    def get_popular_models(self):
        """Return list of popular models."""
        return self.popular_models

    def search_models(self, query, limit=10):
        """Search for models on Hugging Face with retry."""
        for attempt in range(DEFAULT_RETRIES):
            try:
                models = self.api.list_models(search=query, limit=limit)
                return [model.modelId for model in models]
            except Exception as e:
                if attempt == DEFAULT_RETRIES - 1:
                    print(f"Error searching models: {e}")
                    return []
                time.sleep(RETRY_BACKOFF ** attempt)

    def load_model(self, model_name, progress_callback=None, progress_percent=None, device_preference=None):
        """Load model and tokenizer with retries and device hint."""
        device_map = None
        if torch.cuda.is_available() and device_preference == 'cuda':
            device_map = "auto"

        transformers_ok, transformers_msg = require("transformers", "transformers", optional=True)
        if not transformers_ok or not torch_ok:
            return False, f"Missing dependency: {transformers_msg if not transformers_ok else torch_msg}"
        transformers = importlib.import_module("transformers")
        AutoTokenizer = transformers.AutoTokenizer
        AutoModelForCausalLM = transformers.AutoModelForCausalLM

        for attempt in range(DEFAULT_RETRIES):
            try:
                if progress_callback:
                    progress_callback("Downloading tokenizer...")
                if progress_percent:
                    progress_percent(5)
                self.loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.loaded_tokenizer.pad_token is None:
                    self.loaded_tokenizer.pad_token = self.loaded_tokenizer.eos_token

                if progress_callback:
                    progress_callback("Downloading model...")
                if progress_percent:
                    progress_percent(20)
                self.loaded_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )
                if progress_percent:
                    progress_percent(100)
                return True, "Model loaded successfully."
            except Exception as e:
                if attempt == DEFAULT_RETRIES - 1:
                    return False, f"Error loading model: {str(e)}"
                time.sleep(RETRY_BACKOFF ** attempt)

    def get_model_info(self, model_name):
        """Get model information including size estimate."""
        if model_name in self.model_info_cache:
            return self.model_info_cache[model_name]
        try:
            model_info = self.api.model_info(model_name)
            # Estimate size (rough calculation)
            if hasattr(model_info, 'safetensors') and model_info.safetensors:
                size_gb = sum(f.size for f in model_info.safetensors) / (1024**3)
            else:
                size_gb = 0.5  # Default estimate
            info = {
                'name': model_name,
                'size_gb': round(size_gb, 2),
                'downloads': getattr(model_info, 'downloads', 0),
                'likes': getattr(model_info, 'likes', 0),
                'license': getattr(model_info, 'license', 'unknown'),
            }
            self.model_info_cache[model_name] = info
            return info
        except Exception as e:
            info = {
                'name': model_name,
                'size_gb': 0.5,  # Default
                'downloads': 0,
                'likes': 0,
                'license': 'unknown',
            }
            self.model_info_cache[model_name] = info
            return info

    def unload_model(self):
        """Unload the current model to free memory."""
        if self.loaded_model:
            del self.loaded_model
            self.loaded_model = None
        if self.loaded_tokenizer:
            del self.loaded_tokenizer
            self.loaded_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def fetch_license(self, model_name):
        """Fetch license text or spdx id for a model."""
        if model_name in self.license_cache:
            return self.license_cache[model_name]
        try:
            info = self.api.model_info(model_name)
            license_id = getattr(info, 'license', 'unknown')
            self.license_cache[model_name] = license_id
            return license_id
        except Exception:
            return 'unknown'

    def get_top_models(self, limit=10):
        """Return top models by downloads as suggestions."""
        try:
            models = self.api.list_models(sort="downloads", direction=-1, limit=limit)
            return [m.modelId for m in models]
        except Exception:
            return self.popular_models[:limit]
