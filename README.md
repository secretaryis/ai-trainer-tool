
# AI Trainer Tool

A comprehensive GUI desktop application for Linux that simplifies training small AI models on CPU for beginners, with seamless integration to run trained models locally via Ollama for interactive testing.

![Screenshot](screenshots/main.png)  
*Screenshots to be added – see the [screenshots](#screenshots) section below.*

---

## Table of Contents
- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start-cpu-friendly)
  - [Detailed Walkthrough](#detailed-walkthrough)
  - [Optional Flags](#optional-flags)
- [Configuration & Settings](#configuration--settings)
- [Exporting Models](#exporting-models)
- [Ollama Integration](#ollama-integration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Hardware Detection** – Automatically checks your system resources and provides optimised recommendations (batch size, max length, epochs, device).
- **Model Management** – Easily load popular models from Hugging Face (e.g., GPT-2, GPT-Neo) directly from the GUI.
- **Data Input** – Enter training text manually, upload PDF files (with OCR support for multiple languages, including Arabic+English), or import from a URL (HTML, TXT, PNG).
- **Training Modes** – Choose from **Simple**, **Full**, or **Partial** training with adjustable hyperparameters:
  - Batch size, epochs, max sequence length, gradient accumulation, number of data loader workers, seed.
- **Testing & Inference** – After training, test your model with interactive controls:
  - Temperature, top‑p, repetition penalty, max new tokens, seed – all adjustable to reduce hallucinations.
- **Export Formats** – Export your fine‑tuned model in multiple formats:
  - PyTorch, ONNX, Safetensors, GGUF (for Ollama and llama.cpp).
- **Ollama Integration** – One‑click export to GGUF and automatic creation of an Ollama model. If Ollama is not installed, the tool provides clear installation hints.
- **Data Cleaning** – Optional auto‑clean of input text, with adjustable cleaning level (light, medium, aggressive) and language detection.
- **User‑Friendly GUI** – Tab‑based wizard guides you through the entire process: Hardware check → Model selection → Training → Testing → Export.

---

## Screenshots

| Wizard – Step 1 | Hardware Recommendations | Training Settings |
|-----------------|---------------------------|--------------------|
| ![Wizard](screenshots/wizard.png) | ![Hardware](screenshots/hardware.png) | ![Training](screenshots/training.png) |

| Data Input (PDF + OCR) | Model Testing | Export to Ollama |
|------------------------|---------------|------------------|
| ![Data](screenshots/data.png) | ![Testing](screenshots/testing.png) | ![Export](screenshots/export.png) |

*(Screenshots are placeholders – replace with actual images from the `screenshots/` folder.)*

---

## Requirements

- **Operating System**: Linux (tested on Ubuntu 20.04+, Fedora, Arch)
- **Python**: 3.8 or higher
- **Disk Space**: At least 2 GB free for models and training data (recommended)
- **RAM**: 4 GB minimum, 8 GB+ recommended
- **Optional**:
  - NVIDIA GPU with proprietary drivers (for GPU acceleration)
  - [Ollama](https://ollama.com/) (for running exported models locally)
  - Tesseract OCR engine (for PDF OCR – install via your package manager, e.g., `sudo apt install tesseract-ocr tesseract-ocr-ara`)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/secretaryis/ai-trainer-tool.git
cd ai-trainer-tool
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # On some systems use `.venv/bin/activate.fish` for fish shell
```

### 3. Install dependencies
```bash
# With the virtual environment activated
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer not to activate the environment, you can use the full path:
```bash
.venv/bin/pip install -r requirements.txt
```

### 4. (Optional) Install OCR support
```bash
# Install system Tesseract (example for Debian/Ubuntu)
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ara   # Add other languages as needed

# Install Python OCR helpers
.venv/bin/pip install pytesseract pillow pdf2image pymupdf
```

### 5. Run the application
```bash
# With environment activated
python src/main.py

# Or using the full path (if not activated)
.venv/bin/python src/main.py
```

The GUI window should open. If you encounter any issues, see the [Troubleshooting](#troubleshooting) section.

---

## Usage

### Quick Start (CPU‑friendly)
1. Launch the tool (`python src/main.py` or `.venv/bin/python src/main.py`).
2. Go to the **Wizard** tab and pick a small model (e.g., `distilgpt2` or `gpt2`).
3. Paste a few lines of text into the **Data Input** area (or upload a PDF).
4. Switch to the **Training** tab. Leave mode set to **Simple** – the recommended batch size, epochs, etc. are pre‑filled based on your hardware.
5. Click **Start Training**. Training usually takes a few minutes on CPU.
6. Once training finishes, go to the **Testing** tab. Enter a prompt, adjust parameters (e.g., temperature 0.2, top‑p 0.8, max new tokens 64), and click **Generate Response**.
7. If you are satisfied, go to the **Export** tab. Choose **GGUF** and click **Export GGUF + Create in Ollama** to create a ready‑to‑run model with Ollama.

### Detailed Walkthrough
#### Hardware Tab
- Displays your system information (CPU cores, RAM, disk space, GPU availability).
- Shows recommended values for batch size, max length, epochs, and device.

#### Model Selection
- Choose a model from the dropdown (popular models like `gpt2`, `distilgpt2`, `EleutherAI/gpt-neo-125M`).
- Click **Load Model** to download and cache it (only once).

#### Data Input
- You can:
  - Type or paste text directly into the large text box.
  - Upload a PDF: check **Use OCR fallback for PDF**, select languages, and click **Upload PDF**.
  - Import from a URL: enter a link to a PDF, HTML, TXT, or PNG file and click **Import**.
- Adjust cleaning level and enable/disable auto‑clean as needed.

#### Training Settings
- **Mode**: Simple (recommended for beginners), Full (expose all parameters), or Partial.
- **Batch Size**, **Epochs**, **Max Length**, **Grad Accumulation**, **Data Loader Workers**, **Seed** – all configurable.
- Advanced options include learning rate, weight decay, and warmup steps (when Show advanced options is checked).

#### Testing
- After training, you can immediately test your model.
- Parameters:
  - **Temperature** – lower values (e.g., 0.2) make output more deterministic.
  - **Top‑p** – nucleus sampling (0.8 is a good default).
  - **Repetition penalty** – discourage repetitive text (1.1 is typical).
  - **Max new tokens** – limit response length.
  - **Seed** – for reproducibility.
- Click **Generate Response** to see the model’s output.

#### Export
- Select export format:
  - **PyTorch** – saves the model and tokenizer in PyTorch format.
  - **ONNX** – exports to ONNX (experimental).
  - **Safetensors** – safe serialisation format.
  - **GGUF** – format used by Ollama and llama.cpp.
- For GGUF, you can also click **Export GGUF + Create in Ollama** to automatically create an Ollama model. If Ollama is not installed, a popup will guide you.

### Optional Flags
When running the installation script `install.sh` (if provided), you can use:
```bash
./install.sh --with-ocr      # Install OCR dependencies
./install.sh --with-dev       # Install development and testing tools (pytest, etc.)
./install.sh --run            # Launch the tool after installation
```

---

## Configuration & Settings

The tool stores its configuration in `~/.config/ai-trainer-tool/config.json` (on Linux). You can manually edit this file to change:
- Default language
- Theme (Light/Dark)
- Accelerator backend (none, CUDA, ROCm, etc.)
- Paths to custom models

All settings are also accessible via the GUI under the **Settings** section.

---

## Exporting Models

### PyTorch
Saves the model and tokenizer in the standard PyTorch format. You can later load it with:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("path/to/saved/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/model")
```

### GGUF (for Ollama / llama.cpp)
1. After training, select **GGUF** and optionally click **Export GGUF + Create in Ollama**.
2. If Ollama is installed, the tool will:
   - Convert the model to GGUF using a conversion script.
   - Create a `Modelfile` and run `ollama create my-ollama-model -f Modelfile`.
3. You can then run the model with:
   ```bash
   ollama run my-ollama-model
   ```

If Ollama is not detected, the tool will display instructions for installing Ollama and the necessary conversion dependencies.

---

## Ollama Integration

The tool integrates deeply with Ollama to provide a seamless experience from training to local inference.

- **Automatic detection** – Checks if Ollama is installed and accessible.
- **One‑click creation** – After exporting to GGUF, you can immediately create an Ollama model with a single click.
- **Testing via Ollama** – Once the model is in Ollama, you can test it using the **Testing** tab (it will automatically use the Ollama model if available) or via the command line.

For manual creation, the tool generates a `Modelfile` in the export directory with the correct parameters.

---

## Troubleshooting

### "python: command not found" or "No module named ..."
- Ensure your virtual environment is activated: `source .venv/bin/activate`.
- Or use the full path: `.venv/bin/python src/main.py`.
- Check that all dependencies are installed: `.venv/bin/pip install -r requirements.txt`.

### GPU not detected
- Make sure you have NVIDIA drivers installed and `nvidia-smi` works.
- Install the CUDA version of PyTorch: follow instructions at [pytorch.org](https://pytorch.org/).

### PDF OCR fails
- Install Tesseract system package and the required language packs (e.g., `tesseract-ocr-ara` for Arabic).
- Install Python OCR dependencies: `pip install pytesseract pillow pdf2image pymupdf`.
- If using a headless system, you may need to install `poppler-utils` for `pdf2image`.

### Ollama export fails
- Ensure Ollama is installed (see [ollama.com](https://ollama.com/)).
- The conversion to GGUF requires `llama.cpp` and its conversion script. The tool attempts to download it automatically; if it fails, manual steps are printed.

### Training is very slow
- Reduce batch size or max length.
- If you have a GPU, make sure it is selected in the Hardware tab.
- Close other applications to free up RAM.

### Any other issues
Please open an issue on [GitHub](https://github.com/secretaryis/ai-trainer-tool/issues) with a description of the problem and the log output (if any).

---

## Contributing

Contributions are welcome! If you would like to improve the tool, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Make your changes and ensure the code passes tests (run `pytest`).
4. Commit your changes with clear commit messages.
5. Push to your fork and open a pull request.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

### Development Setup
- Install development dependencies: `pip install -r requirements-dev.txt`.
- Run tests: `pytest`.
- Format code with `black` and `isort`.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Ollama](https://ollama.com/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- All contributors and users who provide feedback.

---

**Happy training!** 🚀
