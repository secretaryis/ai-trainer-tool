import importlib
import gc
import psutil


def lazy_import(module_name):
    """Import a module only when needed."""
    return importlib.import_module(module_name)


def force_cleanup():
    """Force memory cleanup across CPU/GPU."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def ram_summary():
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "percent": vm.percent,
    }


def near_capacity(threshold_percent=90):
    """Check if RAM usage is near capacity."""
    return psutil.virtual_memory().percent >= threshold_percent
