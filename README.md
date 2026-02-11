# Open Game Dubber

**Open Game Dubber** is a modular, high-performance Python application designed to automatically transcribe, translate, and dub game audio clips locally.

## Features

- **Local-First**: Runs entirely offline for privacy and zero cost.
- **SOTA AI Models**:
    - **STT**: Faster-Whisper (large-v3-turbo)
    - **Translation**: Ollama (Llama-3.1 / Qwen3)
    - **TTS**: Qwen3-TTS (High Quality) & Fish Speech (Low Latency)
    - **Audio Processing**: Demucs (Source Separation) & DeepFilterNet (Denoising)
- **Batch Processing**: Optimized for multi-core CPUs and NVIDIA GPUs.

## Getting Started

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.1+ (Recommended)
- [uv](https://github.com/astral-sh/uv) (Recommended for dependency management)

### Installation

1.  Clone the repository:


2.  Install dependencies using uv:
    
### Model Setup

Before running the pipeline, you need to download the required AI models (Faster-Whisper, Demucs, etc.). We provide a convenient CLI command for this:

### Usage

**CLI Batch Mode**:

 Usage: python -m src.interface.cli [OPTIONS] COMMAND [ARGS]...

 Open Game Dubber CLI

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.      │
│ --show-completion             Show completion for the current shell, to copy │
│                               it or customize the installation.              │
│ --help                        Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ download  Download all required AI models (Faster-Whisper, Demucs, etc.)     │
│ hello     Test command to verify CLI is working.                             │
╰──────────────────────────────────────────────────────────────────────────────╯

## Documentation

- [Product Requirements Document (PRD)](PRD.md)
- [Implementation Plan](PLAN.md)

## License

MIT
