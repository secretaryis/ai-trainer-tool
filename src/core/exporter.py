from utils.deps import require

torch_ok, torch_msg = require("torch", "PyTorch", optional=True)
transformers_ok, tf_msg = require("transformers", "transformers", optional=True)
if torch_ok:
    import torch  # type: ignore
else:
    torch = None
if transformers_ok:
    from transformers import AutoTokenizer  # type: ignore
else:
    AutoTokenizer = None
import os
import shutil
import subprocess
import requests
import tempfile
from pathlib import Path

CONVERT_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/examples/convert.py"


class ModelExporter:
    def __init__(self, model, tokenizer, model_path):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.convert_script = self._ensure_convert_script()
        self.convert_timeout = 1800  # seconds

    def _ensure_convert_script(self):
        cache_dir = Path.home() / ".cache" / "ai_trainer"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "convert.py"
        if target.exists():
            return str(target)
        try:
            resp = requests.get(CONVERT_URL, timeout=20)
            resp.raise_for_status()
            target.write_bytes(resp.content)
            return str(target)
        except Exception:
            return None

    def export_pytorch(self, output_path):
        """Export as PyTorch model."""
        if not torch_ok or AutoTokenizer is None:
            return False, f"Missing dependencies: {(tf_msg if AutoTokenizer is None else '')} {(torch_msg if not torch_ok else '')}".strip()
        try:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            return True, f"PyTorch model exported to {output_path}"
        except Exception as e:
            return False, f"PyTorch export failed: {str(e)}"

    def export_safetensors(self, output_path):
        """Export as SafeTensors."""
        if not torch_ok or AutoTokenizer is None:
            return False, f"Missing dependencies: {(tf_msg if AutoTokenizer is None else '')} {(torch_msg if not torch_ok else '')}".strip()
        try:
            self.model.save_pretrained(output_path, safe_serialization=True)
            self.tokenizer.save_pretrained(output_path)
            return True, f"SafeTensors model exported to {output_path}"
        except Exception as e:
            return False, f"SafeTensors export failed: {str(e)}"

    def export_onnx(self, output_path):
        """Export as ONNX using transformers' built-in exporter."""
        if AutoTokenizer is None:
            return False, tf_msg
        try:
            from pathlib import Path as _Path
            from transformers.onnx import FeaturesManager, export

            feature = "text-generation"
            _, onnx_config_cls = FeaturesManager.check_supported_model_or_raise(self.model, feature=feature)
            onnx_config = onnx_config_cls(self.model.config)
            onnx_path = _Path(output_path) / "model.onnx"
            export(
                preprocessor=self.tokenizer,
                model=self.model,
                config=onnx_config,
                opset=onnx_config.DEFAULT_ONNX_OPSET,
                output=onnx_path,
            )
            self.tokenizer.save_pretrained(output_path)
            return True, f"ONNX model exported to {onnx_path}"
        except ImportError:
            return False, "ONNX export requires 'transformers[onnx]' or 'optimum'. Install with: pip install transformers[onnx]"
        except Exception as e:
            return False, f"ONNX export failed: {str(e)}"

    def export_gguf(self, output_path):
        """Export as GGUF for Ollama with clearer errors and cleanup."""
        tmp_dir = None
        try:
            if not self.convert_script:
                return False, "convert.py not available. Check internet connection."

            tmp_dir = tempfile.mkdtemp(prefix="gguf_tmp_")
            self.model.save_pretrained(tmp_dir)
            self.tokenizer.save_pretrained(tmp_dir)

            os.makedirs(output_path, exist_ok=True)
            gguf_path = os.path.join(output_path, "model.gguf")

            cmd = ["python", self.convert_script, "--outfile", gguf_path, tmp_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.convert_timeout)
            if result.returncode != 0:
                stderr = result.stderr.strip() if result.stderr else "Unknown error"
                return False, f"GGUF convert failed: {stderr[:400]}"
            if not os.path.exists(gguf_path):
                return False, "GGUF convert finished without output file."
            return True, f"GGUF model exported to {gguf_path}"
        except subprocess.TimeoutExpired:
            return False, "GGUF convert timed out."
        except Exception as e:
            return False, f"GGUF export failed: {str(e)}"
        finally:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def export_all(self, base_path, formats=['pytorch', 'safetensors']):
        """Export model in multiple formats."""
        results = {}
        for fmt in formats:
            output_path = os.path.join(base_path, fmt)
            os.makedirs(output_path, exist_ok=True)
            
            if fmt == 'pytorch':
                success, msg = self.export_pytorch(output_path)
            elif fmt == 'safetensors':
                success, msg = self.export_safetensors(output_path)
            elif fmt == 'onnx':
                success, msg = self.export_onnx(output_path)
            elif fmt == 'gguf':
                success, msg = self.export_gguf(output_path)
            else:
                success, msg = False, f"Unknown format: {fmt}"
            
            results[fmt] = {'success': success, 'message': msg}
        
        return results
