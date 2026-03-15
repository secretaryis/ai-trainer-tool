# AI Trainer Tool

A comprehensive GUI desktop application for Linux that simplifies training small AI models on CPU for beginners, with seamless integration to run trained models locally via Ollama for interactive testing.

## Features

- **Hardware Detection**: Automatic hardware check and optimization recommendations
- **Model Management**: Easy loading of models from Hugging Face
- **Data Handling**: Support for text input and PDF uploads
- **Training Modes**: Simple, Full, and Partial training options
- **Export Formats**: PyTorch, ONNX, Safetensors, GGUF
- **Ollama Integration**: Direct integration for running trained models

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
   python src/main.py
   ```

## Requirements

- Python 3.8+
- Linux OS
- Ollama (optional, for model running)

## Screenshots

(Add screenshots here)

## License

MIT License
