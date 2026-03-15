import torch
from transformers import AutoTokenizer
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
        try:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            return True, f"PyTorch model exported to {output_path}"
        except Exception as e:
            return False, f"PyTorch export failed: {str(e)}"

    def export_safetensors(self, output_path):
        """Export as SafeTensors."""
        try:
            self.model.save_pretrained(output_path, safe_serialization=True)
            self.tokenizer.save_pretrained(output_path)
            return True, f"SafeTensors model exported to {output_path}"
        except Exception as e:
            return False, f"SafeTensors export failed: {str(e)}"

    def export_onnx(self, output_path):
        """Export as ONNX (basic implementation)."""
        try:
            # This is a simplified version. For full ONNX export, use optimum
            from transformers.onnx import export
            onnx_path = os.path.join(output_path, "model.onnx")
            export(self.model, self.tokenizer, onnx_path, "text-generation")
            self.tokenizer.save_pretrained(output_path)
            return True, f"ONNX model exported to {output_path}"
        except ImportError:
            return False, "ONNX export requires 'optimum' package. Install with: pip install optimum"
        except Exception as e:
            return False, f"ONNX export failed: {str(e)}"

    def export_gguf(self, output_path):
        """Export as GGUF for Ollama."""
        try:
            if not self.convert_script:
                return False, "convert.py not available. Check internet connection."

            # Save to temp pytorch dir first
            tmp_dir = tempfile.mkdtemp(prefix="gguf_tmp_")
            self.model.save_pretrained(tmp_dir)
            self.tokenizer.save_pretrained(tmp_dir)

            os.makedirs(output_path, exist_ok=True)
            gguf_path = os.path.join(output_path, "model.gguf")

            cmd = ["python", self.convert_script, "--outfile", gguf_path, tmp_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                return False, f"GGUF convert failed: {result.stderr[:200]}"
            return True, f"GGUF model exported to {gguf_path}"
        except subprocess.TimeoutExpired:
            return False, "GGUF convert timed out."
        except Exception as e:
            return False, f"GGUF export failed: {str(e)}"

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
