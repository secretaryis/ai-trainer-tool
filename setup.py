from setuptools import setup, find_packages

setup(
    name="ai-trainer-tool",
    version="1.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PyQt5",
        "transformers",
        "torch",
        "datasets",
        "psutil",
        "PyPDF2",
        "onnx",
        "safetensors",
        "llama-cpp-python",
        "accelerate",
        "huggingface_hub",
    ],
    entry_points={
        "console_scripts": [
            "ai-trainer=gui.main_window:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A GUI tool for training small AI models with Ollama integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-trainer-tool",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
