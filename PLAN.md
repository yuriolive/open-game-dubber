# Implementation Plan: Open Game Dubber

# Goal Description
Build a local, high-performance Python application for batch dubbing game audio from English to any target language using SOTA AI models (Faster-Whisper, Ollama, Qwen3-TTS).

## User Review Required
> [!IMPORTANT]
> **Hardware Dependency**: The system is optimized for NVIDIA GPUs (CUDA 12.1+). Performance on AMD/Intel GPUs or CPU-only setups will be significantly lower.
> **Model Weights**: Users must accept licenses for Qwen3, Fish Speech, and Llama 3.1 models (typically Apache 2.0 or Community License).

## Proposed Changes

### Project Structure Setup
Create a robust project structure to support modular development.
#### [NEW] `pyproject.toml`
- Define dependencies: `typer`, `gradio`, `faster-whisper`, `ctranslate2`, `torch`, `torchaudio`, `librosa`, `ollama`, `demucs`, `deepfilternet`.
- Configure build system.
#### [NEW] `src/`
- `core/`: Main logic for pipeline orchestration.
- `models/`: Wrappers for Whisper, Ollama, TTS engines.
- `utils/`: Audio processing (Librosa), file I/O.
- `interface/`: CLI (Typer) and GUI (Gradio) entry points.

### Core Pipeline Implementation
#### [NEW] `src/models/stt.py`
- Implement `FasterWhisperTranscriber` class.
- Optimize for `large-v3-turbo` on GPU.
#### [NEW] `src/utils/audio_processor.py`
- Implement `AudioCleaner` class.
- Integrate `Demucs` for splitting (keep background, discard original voice).
- Integrate `DeepFilterNet` for cleaning reference audio.
#### [NEW] `src/models/translator.py`
- Implement `OllamaTranslator` class.
- Setup prompt templates for game dialogue style (configurable target language).
#### [NEW] `src/models/tts.py`
- Implement `QwenTTSWrapper` and `FishSpeechWrapper`.
- Handle voice cloning logic (reference audio loading).
#### [NEW] `src/core/pipeline.py`
- Implement `DubbingPipeline` class.
- Manage multiprocessing pool and GPU job queue.
#### [NEW] `src/core/state_manager.py`
- Implement `StateManager` class.
- valid/invalid state tracking using a persistent JSON manifest.
- `is_processed(file_path)` and `mark_processed(file_path)` methods.

### CLI Development
#### [NEW] `src/interface/cli.py`
- Implement `dub-batch` command using Typer.
- precise argument parsing for directories and worker counts.

## Verification Plan

### Automated Tests
- **Unit Tests**: Test individual model wrappers with mock data/short audio clips.
- **Integration Tests**: Run the full pipeline on a sample set of 5 audio files.

### Manual Verification
- **Performance**: Run `dub-batch` with `--workers 24` and monitor GPU VRAM/Util and CPU load via Task Manager/`nvtop`.
- **Quality**: Listen to generated clips to verify voice cloning similarity and translation naturalness.
