# Open Game Dubber

**Open Game Dubber** is a modular, high-performance Python application designed to automatically transcribe, translate, and dub game audio clips locally.

## Features

- **Local-First**: Runs entirely offline for privacy and zero cost.
- **SOTA AI Models**:
    - **STT**: Faster-Whisper (large-v3-turbo)
    - **Translation**: Ollama (Llama-3.1)
    - **TTS**: Qwen3-TTS-12Hz (High Quality Zero-Shot Voice Cloning)
    - **Audio Processing**: Demucs (Source Separation)
- **Batch Processing**: Optimized for multi-core CPUs and NVIDIA GPUs.

## Examples

Compare the original English game audio with the locally generated dub:

<table>
<tr>
<th>Original Audio (English)</th>
<th>Dubbed Audio (Portuguese)</th>
</tr>
<tr>
<td>
<audio controls>
  <source src="docs/artifacts/original_sample.wav" type="audio/wav">
  Your browser does not support the audio element. <a href="docs/artifacts/original_sample.wav">Download</a>
</audio>
</td>
<td>
<audio controls>
  <source src="docs/artifacts/dubbed_sample.wav" type="audio/wav">
  Your browser does not support the audio element. <a href="docs/artifacts/dubbed_sample.wav">Download</a>
</audio>
</td>
</tr>
</table>

## Getting Started

### Prerequisites

- **Python 3.12+**
- **FFmpeg**: Required for audio processing.
- **SoX**: Required for TTS synthesis.
    - Windows: `scoop install sox` or `choco install sox`
    - Linux: `sudo apt install sox`
    - Mac: `brew install sox`
- **Ollama**: Required for local translation.
    - Install from [ollama.com](https://ollama.com/) and ensure it's running.
- **NVIDIA GPU with CUDA 12.1+**: (Recommended for performance)
- **[uv](https://github.com/astral-sh/uv)**: (Recommended for dependency management)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yuriolive/open-game-dubber.git
    cd open-game-dubber
    ```

2.  **Install Python dependencies**:
    ```bash
    uv sync
    ```

### Model Setup

Before running the pipeline, you need to download the required AI models (Faster-Whisper, Demucs, Qwen3-TTS) and pull the translation model in Ollama.

1.  **Download AI Models**:
    ```bash
    uv run dub download
    ```
    *Note: This will also attempt to pull `llama3.1` from your local Ollama instance.*

2.  **Manual Ollama Pull** (if step 1 fails):
    ```bash
    ollama pull llama3.1
    ```

### Usage

#### Batch Processing
Process all WAV files in a directory (default: `samples/`):
```bash
uv run dub dub-batch --target-lang "Brazilian Portuguese"
```

**Options**:
- `--input-dir`: Directory containing source WAV files (default: `samples`).
- `--output-dir`: Directory to save dubbed files (default: `output`).
- `--target-lang`: Target language for dubbing (e.g., "Portuguese", "Spanish", "Japanese").
- `--limit`: Limit the number of files to process.

#### Verification
Check if the CLI and basic dependencies are working:
```bash
uv run dub hello
```

## Documentation

- [Product Requirements Document](PRD.md)

## License

MIT
