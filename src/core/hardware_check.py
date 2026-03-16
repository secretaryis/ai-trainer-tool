import platform
from utils.deps import require
torch_ok, _ = require("torch", "PyTorch", optional=True)
if torch_ok:
    import torch  # type: ignore
else:
    torch = type("TorchFallback", (), {
        "cuda": type("Cuda", (), {
            "is_available": staticmethod(lambda: False),
            "device_count": staticmethod(lambda: 0),
            "empty_cache": staticmethod(lambda: None)
        })()
    })()
import shutil
import math
import gc
import os

try:
    import psutil
except ImportError:
    psutil = None

class HardwareCheck:
    def __init__(self):
        self.system_info = self.get_system_info()
        self.recommendations = self.get_recommendations()
        self.ram_summary = self._format_ram()

    def get_system_info(self):
        """Get basic hardware information."""
        cores = psutil.cpu_count(logical=False) if psutil else os.cpu_count() or 1
        threads = psutil.cpu_count(logical=True) if psutil else cores
        ram_total = psutil.virtual_memory().total if psutil else 2 * (1024**3)
        info = {
            'cpu_cores': cores,
            'cpu_threads': threads,
            'ram_gb': round(ram_total / (1024**3), 2),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'platform': platform.system(),
            'processor': platform.processor(),
            'disk_free_gb': round(shutil.disk_usage('/').free / (1024**3), 2),
        }
        return info

    def get_recommendations(self):
        """Provide recommendations based on hardware."""
        ram = self.system_info['ram_gb']
        cores = self.system_info['cpu_cores']
        gpu = self.system_info['gpu_available']

        # Batch size recommendations
        if ram < 4:
            batch_size = 1
        elif ram < 8:
            batch_size = 2
        elif ram < 16:
            batch_size = 4
        else:
            batch_size = 8

        # Max length for sequences
        if ram < 8:
            max_length = 128
        elif ram < 16:
            max_length = 256
        else:
            max_length = 512

        # Epochs based on cores
        if cores <= 2:
            recommended_epochs = 1
        elif cores <= 4:
            recommended_epochs = 2
        else:
            recommended_epochs = 3

        device_hint = 'cuda' if gpu else 'cpu'

        return {
            'batch_size': batch_size,
            'max_length': max_length,
            'epochs': recommended_epochs,
            'device': device_hint,
        }

    def check_model_compatibility(self, model_size_gb):
        """Check if a model can run on this hardware."""
        ram = self.system_info['ram_gb']
        disk_free = self.system_info['disk_free_gb']
        if model_size_gb > ram * 0.8:  # Use 80% of RAM as threshold
            return False, f"Model requires ~{model_size_gb}GB RAM, but system has {ram}GB. May be slow or fail."
        if model_size_gb > ram * 0.5:
            return True, f"Model may be slow on this hardware ({ram}GB RAM)."
        if disk_free < model_size_gb * 1.5:
            return False, f"Insufficient disk space (~{disk_free}GB free) for model size {model_size_gb}GB."
        return True, "Model should run efficiently on this hardware."

    def enough_disk(self, required_gb):
        return self.system_info['disk_free_gb'] >= required_gb

    def cleanup(self):
        """Force memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_ram(self):
        if psutil:
            vm = psutil.virtual_memory()
            used_gb = round(vm.used / (1024**3), 2)
            return f"{used_gb}GB used / {self.system_info['ram_gb']}GB total"
        return f"RAM approx {self.system_info['ram_gb']}GB (psutil not installed)"
