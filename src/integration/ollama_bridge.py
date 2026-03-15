import subprocess
import os
import platform

class OllamaBridge:
    def __init__(self):
        self.ollama_available = self.check_ollama_installed()

    def check_ollama_installed(self):
        """Check if Ollama is installed."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def get_install_instructions(self):
        """Get installation instructions for Ollama."""
        system = platform.system().lower()
        if system == "linux":
            return "Install Ollama by running: curl -fsSL https://ollama.ai/install.sh | sh"
        elif system == "darwin":
            return "Install Ollama from: https://ollama.ai/download"
        elif system == "windows":
            return "Install Ollama from: https://ollama.ai/download"
        else:
            return "Visit https://ollama.ai for installation instructions."

    def create_modelfile(self, model_path, model_name, modelfile_path="Modelfile"):
        """Create a Modelfile for the exported model."""
        try:
            with open(modelfile_path, 'w') as f:
                f.write(f"FROM {model_path}\n")
                f.write("PARAMETER temperature 0.7\n")
                f.write("PARAMETER top_p 0.9\n")
                f.write("SYSTEM \"You are a helpful AI assistant.\"\n")
            return True, f"Modelfile created at {modelfile_path}"
        except Exception as e:
            return False, f"Failed to create Modelfile: {str(e)}"

    def create_ollama_model(self, model_name, modelfile_path="Modelfile"):
        """Create model in Ollama."""
        if not self.ollama_available:
            return False, "Ollama is not installed."

        try:
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            if result.returncode == 0:
                return True, f"Model '{model_name}' created successfully in Ollama."
            else:
                return False, f"Failed to create model: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Model creation timed out."
        except Exception as e:
            return False, f"Error creating model: {str(e)}"

    def run_ollama_model(self, model_name):
        """Run the model in Ollama (opens interactive session)."""
        if not self.ollama_available:
            return False, "Ollama is not installed."

        try:
            # This will open an interactive session
            subprocess.Popen(["ollama", "run", model_name])
            return True, f"Started Ollama interactive session for model '{model_name}'"
        except Exception as e:
            return False, f"Failed to run model: {str(e)}"

    def list_ollama_models(self):
        """List available models in Ollama."""
        if not self.ollama_available:
            return []

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models
            else:
                return []
        except Exception:
            return []

    def delete_ollama_model(self, model_name):
        """Delete a model from Ollama."""
        if not self.ollama_available:
            return False, "Ollama is not installed."

        try:
            result = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return True, f"Model '{model_name}' deleted from Ollama."
            else:
                return False, f"Failed to delete model: {result.stderr}"
        except Exception as e:
            return False, f"Error deleting model: {str(e)}"
