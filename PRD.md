# Product Requirements Document (PRD): Open Game Dubber

## 1. Executive Summary
**Open Game Dubber** is a modular, high-performance Python application designed to automatically transcribe, translate, and dub game audio clips from any source language to any target language. The system prioritizes offline local execution, leveraging modern hardware and state-of-the-art (SOTA) open-source AI models. The goal is a privacy-first, zero-cost, pro-quality dubbing pipeline superior to existing cloud-reliant or inefficient node-based solutions.

## 2. Problem Statement
Current solutions for game dubbing often suffer from:
- **High Latency/Cost**: Reliance on cloud APIs (ElevenLabs, OpenAI) which are expensive at scale and introduce network latency.
- **Complex Workflows**: Node-based UIs (ComfyUI) are powerful but cumbersome for batch processing thousands of files.
- **Poor Localization**: Generic TTS models lack specific training or fine-tuning for game dialogue in specific target languages, often missing emotional nuance or specific gaming terminology.
- **Hardware Underutilization**: Many tools do not fully leverage available high-end consumer hardware (high core count CPUs, modern NVIDIA GPUs).

## 3. Scope & Objectives
### Primary Objectives
- **Automated Pipeline**: End-to-end processing (WAV in -> WAV out) with minimal manual intervention.
- **High Quality**: Achieve SOTA voice cloning and natural-sounding translations using Qwen3-TTS and fine-tuned LLMs.
- **Performance**: Process ~100 clips/minute on target hardware.
- **Local-First**: 100% offline capability for privacy and zero operational cost.

### In-Scope
- Batch processing of WAV files.
- Speech-to-Text (STT) via Faster-Whisper.
- Text translation via local LLM (Ollama).
- Text-to-Speech (TTS) & Voice Cloning via Qwen3-TTS (primary) and Fish Speech (fallback).
- Audio post-processing (normalization, format conversion).
- CLI and simple GUI interfaces.

### Out-of-Scope (for v1)
- Real-time in-game overlay dubbing.
- Cloud-based API integration (except optional fallbacks).
- Complex multi-speaker separation (diarization is a future enhancement).

## 4. Technical Architecture
### Core Stack
- **Language**: Python 3.12
- **UI**: Typer (CLI) / Gradio (Web GUI)
- **Concurrency**: `multiprocessing` for CPU tasks, batch queuing for GPU.

### AI Models
| Component | Model / Tool | Justification |
| :--- | :--- | :--- |
| **STT** | **Faster-Whisper (large-v3-turbo)** | SOTA accuracy/speed, optimized CTranslate2 backend, handled 99+ langs. ~6GB VRAM. |
| **Translation** | **Ollama (Llama-3.1-8B / Qwen3-8B)** | Local, efficient, fine-tunable via prompts for "game dialogue" style. ~5GB VRAM. |
| **Source Sep / Denoise** | **Demucs (htdemucs) / DeepFilterNet** | Hybrid Transformer Demucs is SOTA for music/stem separation. DeepFilterNet cleans artifacts. |
| **TTS / Cloning** | **Qwen3-TTS (1.7B)** | Primary. SOTA open-source (Jan 2026), multilingual support, 3s zero-shot cloning, emotion transfer. ~4.5GB VRAM (fp16). |
| **Fallback TTS** | **Fish Speech V1.6** | Secondary. Ultra-low latency, robust generalization, highly efficient DualAR arch. |

### Pipeline Workflow
1.  **Input**: Load source audio (WAV).
2.  **Audio Pre-processing**: Apply **Demucs** to separate vocals from background SFX/Music. Apply **DeepFilterNet** to denoise vocals for better STT/Cloning.
3.  **STT**: Transcribe cleaned English vocals to text.
4.  **Translate**: LLM converts source text to target language (Context-aware prompting).
5.  **Synthesis**: Generate audio using reference voice (Voice Cloning).
6.  **Post-Process**: Mix new vocals with extracted background. Normalize, trim, match format (24kHz mono).
7.  **Output**: Save dubbed file & metadata.

## 5. Hardware Optimization
Target System: **12GB VRAM**, **12c/24t**.
- **VRAM Strategy**: Models run in fp16/bf16. 
  - STT and Translation can be offloaded or run sequentially if VRAM is tight.
  - TTS (Qwen3) occupies ~4.5GB, allowing concurrent STT or larger batch sizes.
- **CPU Utilization**: Heavy use of `multiprocessing` for audio I/O, post-processing (Librosa), and management overhead to keep the GPU fed.

## 6. Functional Requirements
### FR1: Codebase & Environment
- Modular Python package structure.
- dependency management via `requirements.txt` / `pyproject.toml`.
- GPU acceleration check on startup.

### FR2: CLI Operation
- Command: `dub-batch --input-dir <DIR> --ref-dir <DIR> --output-dir <DIR> --workers <N>`
- Progress logs (tqdm).

### FR4: Resilience
- **State Tracking**: Maintain a local manifest (JSON/SQLite) of processed files.
- **Resume Capability**: On startup, skip files already marked as 'completed' in the manifest.
- **Error Handling**: Log specific errors per file without crashing the entire batch.

## 7. Roadmap
- **Phase 1**: Core Pipeline & CLI (MVP). Support for generic WAV clips.
- **Phase 2**: Gradio GUI & refinement of translation prompts.
- **Phase 3**: Character identification/ref-matching automation.
- **Phase 4**: Packaging (Docker/Exe) for broader distribution.
