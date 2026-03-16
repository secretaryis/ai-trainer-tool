# AI Trainer Tool User Guide (EN)

1. Start the app (`python src/main.py`).
2. Wizard tab: follow steps to select model, load data, train, export, and run with Ollama.
3. Tabs: Hardware, Model & Data, Training, Testing & Export, Ollama for advanced control.
4. Data tab: choose cleaning level (off/medium/strong) and toggle language detection; OCR backend and cache controls are available.
5. Testing tab: tune temperature/top-p/repetition penalty/max tokens/seed for safer outputs.
6. Exports: PyTorch, SafeTensors, ONNX, GGUF (for Ollama). “Export GGUF + Create in Ollama” builds a Modelfile and calls `ollama create` when available (otherwise shows install hint).
7. Troubleshooting: ensure internet for downloads; keep at least 2GB free disk when exporting GGUF.
