# AI Trainer Tool

A comprehensive GUI desktop application for Linux that simplifies training small AI models on CPU for beginners, with seamless integration to run trained models locally via Ollama for interactive testing.

## Features

- **Hardware Detection**: Automatic hardware check and optimization recommendations
- **Model Management**: Easy loading of models from Hugging Face
- **Data Handling**: Support for text input and PDF uploads
- **Training Modes**: Simple, Full, and Partial training options
- **Export Formats**: PyTorch, ONNX, Safetensors, GGUF
- **Ollama Integration**: Direct integration for running trained models, plus one-click GGUF + `ollama create`
- **Safer Inference**: Temperature/top-p/repetition controls with deterministic option for reduced hallucinations
- **Cleaning Controls**: Toggle auto-clean, choose level, and language detection

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-trainer-tool.git
   cd ai-trainer-tool
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
 .venv/bin/python src/main.py
```

### Optional flags
- Install OCR helpers: `./install.sh --with-ocr`
- Install dev/test deps: `./install.sh --with-dev`
- Auto-run after install: `./install.sh --run`

## Requirements

- Python 3.8+
- Linux OS
- Ollama (optional, for model running)
- Optional deps: torch/transformers (modeling), PyPDF2/pdfplumber (PDF), bs4 (HTML), pandas/openpyxl (Excel), pytesseract+pillow (+system tesseract for OCR), pdf2image, pymupdf for stronger PDF/OCR fallback

## Screenshots

(Add screenshots here)

## Quick start (CPU-friendly)
1. Launch `python src/main.py`.
2. Wizard tab: pick a small model (e.g., `distilgpt2`), paste a few lines of text.
3. Training tab: leave mode = Simple. Advanced settings let you set max_length, grad accumulation, and seed.
4. Testing tab: lower temperature (0.2) / top-p (0.8) for safer replies; set max new tokens to ~64.
5. Export tab: choose `GGUF` then click “Export GGUF + Create in Ollama” to get a ready-to-run model (or a Modelfile with install hints if Ollama is missing).

## Tests

Run unit tests (requires `pytest`):
```bash
pytest
```

## Release checklist
- Bump version in `setup.py`.
- Run `pytest`.
- Verify GGUF export on a small model; if Ollama missing, confirm install hint appears.
- Update docs/screenshots if UI changed.

## License

MIT License
