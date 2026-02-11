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
    ```bash
    git clone https://github.com/yourusername/open-game-dubber.git
    cd open-game-dubber
    ```

2.  Install dependencies using `uv`:
    ```bash
    uv sync
    ```

### Usage

**CLI Batch Mode**:
```bash
uv run dub batch --input-dir "path/to/wavs" --output-dir "path/to/output"
```

## Documentation

- [Product Requirements Document (PRD)](PRD.md)
- [Implementation Plan](PLAN.md)

## License

MIT
