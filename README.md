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


2.  Install dependencies using  (or pip):
    Processing /app
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting faster-whisper>=1.0.0 (from open-game-dubber==0.1.0)
  Using cached faster_whisper-1.2.1-py3-none-any.whl.metadata (16 kB)
Collecting ctranslate2>=4.0.0 (from open-game-dubber==0.1.0)
  Using cached ctranslate2-4.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (10 kB)
Collecting torch>=2.4.0 (from open-game-dubber==0.1.0)
  Using cached torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (31 kB)
Collecting torchaudio>=2.4.0 (from open-game-dubber==0.1.0)
  Using cached torchaudio-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (6.9 kB)
Collecting librosa>=0.10.0 (from open-game-dubber==0.1.0)
  Using cached librosa-0.11.0-py3-none-any.whl.metadata (8.7 kB)
Collecting ollama>=0.3.0 (from open-game-dubber==0.1.0)
  Using cached ollama-0.6.1-py3-none-any.whl.metadata (4.3 kB)
Collecting demucs>=4.0.0 (from open-game-dubber==0.1.0)
  Using cached demucs-4.0.1.tar.gz (1.2 MB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting deepfilternet>=0.5.0 (from open-game-dubber==0.1.0)
  Using cached deepfilternet-0.5.6-py3-none-any.whl.metadata (1.7 kB)
Collecting typer>=0.12.0 (from open-game-dubber==0.1.0)
  Using cached typer-0.22.0-py3-none-any.whl.metadata (16 kB)
Collecting gradio>=4.0.0 (from open-game-dubber==0.1.0)
  Using cached gradio-6.5.1-py3-none-any.whl.metadata (16 kB)
Collecting setuptools (from ctranslate2>=4.0.0->open-game-dubber==0.1.0)
  Using cached setuptools-82.0.0-py3-none-any.whl.metadata (6.6 kB)
Collecting numpy (from ctranslate2>=4.0.0->open-game-dubber==0.1.0)
  Using cached numpy-2.4.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (6.6 kB)
Collecting pyyaml<7,>=5.3 (from ctranslate2>=4.0.0->open-game-dubber==0.1.0)
  Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting appdirs<2.0,>=1.4 (from deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached appdirs-1.4.4-py2.py3-none-any.whl.metadata (9.0 kB)
Collecting deepfilterlib==0.5.6 (from deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached DeepFilterLib-0.5.6.tar.gz (112 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting loguru>=0.5 (from deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached loguru-0.7.3-py3-none-any.whl.metadata (22 kB)
Collecting numpy (from ctranslate2>=4.0.0->open-game-dubber==0.1.0)
  Using cached numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Collecting packaging<24.0,>=23.0 (from deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached packaging-23.2-py3-none-any.whl.metadata (3.2 kB)
Collecting requests<3.0,>=2.27 (from deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting sympy>=1.6 (from deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting charset_normalizer<4,>=2 (from requests<3.0,>=2.27->deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached charset_normalizer-3.4.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (37 kB)
Collecting idna<4,>=2.5 (from requests<3.0,>=2.27->deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3.0,>=2.27->deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
Collecting certifi>=2017.4.17 (from requests<3.0,>=2.27->deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached certifi-2026.1.4-py3-none-any.whl.metadata (2.5 kB)
Collecting dora-search (from demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached dora_search-0.1.12.tar.gz (87 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting einops (from demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached einops-0.8.2-py3-none-any.whl.metadata (13 kB)
Collecting julius>=0.2.3 (from demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached julius-0.2.7.tar.gz (59 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting lameenc>=1.2 (from demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached lameenc-1.8.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (9.9 kB)
Collecting openunmix (from demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached openunmix-1.3.0-py3-none-any.whl.metadata (17 kB)
Collecting tqdm (from demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached tqdm-4.67.3-py3-none-any.whl.metadata (57 kB)
Collecting huggingface-hub>=0.21 (from faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached huggingface_hub-1.4.1-py3-none-any.whl.metadata (13 kB)
Collecting tokenizers<1,>=0.13 (from faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting onnxruntime<2,>=1.14 (from faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached onnxruntime-1.24.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (4.9 kB)
Collecting av>=11 (from faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached av-16.1.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (4.6 kB)
Collecting flatbuffers (from onnxruntime<2,>=1.14->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
Collecting protobuf (from onnxruntime<2,>=1.14->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached protobuf-6.33.5-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)
Collecting filelock (from huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached filelock-3.20.3-py3-none-any.whl.metadata (2.1 kB)
Collecting fsspec>=2023.5.0 (from huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached fsspec-2026.2.0-py3-none-any.whl.metadata (10 kB)
Collecting hf-xet<2.0.0,>=1.2.0 (from huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
Collecting httpx<1,>=0.23.0 (from huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting shellingham (from huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
Collecting typer-slim (from huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached typer_slim-0.22.0-py3-none-any.whl.metadata (4.1 kB)
Requirement already satisfied: typing-extensions>=4.1.0 in /home/jules/.pyenv/versions/3.12.12/lib/python3.12/site-packages (from huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0) (4.15.0)
Collecting anyio (from httpx<1,>=0.23.0->huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached anyio-4.12.1-py3-none-any.whl.metadata (4.3 kB)
Collecting httpcore==1.* (from httpx<1,>=0.23.0->huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting h11>=0.16 (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub>=0.21->faster-whisper>=1.0.0->open-game-dubber==0.1.0)
  Using cached h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting aiofiles<25.0,>=22.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached aiofiles-24.1.0-py3-none-any.whl.metadata (10 kB)
Collecting brotli>=1.1.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached brotli-1.2.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.1 kB)
Collecting fastapi<1.0,>=0.115.2 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached fastapi-0.128.7-py3-none-any.whl.metadata (30 kB)
Collecting ffmpy (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached ffmpy-1.0.0-py3-none-any.whl.metadata (3.0 kB)
Collecting gradio-client==2.0.3 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached gradio_client-2.0.3-py3-none-any.whl.metadata (7.1 kB)
Collecting groovy~=0.1 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached groovy-0.1.2-py3-none-any.whl.metadata (6.1 kB)
Collecting jinja2<4.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting markupsafe<4.0,>=2.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
Collecting orjson~=3.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached orjson-3.11.7-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (41 kB)
Collecting pandas<4.0,>=1.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached pandas-3.0.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (79 kB)
Collecting pillow<13.0,>=8.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached pillow-12.1.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting pydantic<=3.0,>=2.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
Collecting pydub (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached pydub-0.25.1-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting python-multipart>=0.0.18 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached python_multipart-0.0.22-py3-none-any.whl.metadata (1.8 kB)
Collecting pytz>=2017.2 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting safehttpx<0.2.0,>=0.1.7 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached safehttpx-0.1.7-py3-none-any.whl.metadata (4.2 kB)
Collecting semantic-version~=2.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached semantic_version-2.10.0-py2.py3-none-any.whl.metadata (9.7 kB)
Collecting starlette<1.0,>=0.40.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached starlette-0.52.1-py3-none-any.whl.metadata (6.3 kB)
Collecting tomlkit<0.14.0,>=0.12.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached tomlkit-0.13.3-py3-none-any.whl.metadata (2.8 kB)
Collecting uvicorn>=0.14.0 (from gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached uvicorn-0.40.0-py3-none-any.whl.metadata (6.7 kB)
Collecting typing-inspection>=0.4.2 (from fastapi<1.0,>=0.115.2->gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
Collecting annotated-doc>=0.0.2 (from fastapi<1.0,>=0.115.2->gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached annotated_doc-0.0.4-py3-none-any.whl.metadata (6.6 kB)
Collecting python-dateutil>=2.8.2 (from pandas<4.0,>=1.0->gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting annotated-types>=0.6.0 (from pydantic<=3.0,>=2.0->gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.41.5 (from pydantic<=3.0,>=2.0->gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached pydantic_core-2.41.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting click>=8.0.0 (from typer>=0.12.0->open-game-dubber==0.1.0)
  Using cached click-8.3.1-py3-none-any.whl.metadata (2.6 kB)
Collecting rich>=10.11.0 (from typer>=0.12.0->open-game-dubber==0.1.0)
  Using cached rich-14.3.2-py3-none-any.whl.metadata (18 kB)
Collecting audioread>=2.1.9 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached audioread-3.1.0-py3-none-any.whl.metadata (9.0 kB)
Collecting numba>=0.51.0 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached numba-0.63.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.9 kB)
Collecting scipy>=1.6.0 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached scipy-1.17.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting scikit-learn>=1.1.0 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached scikit_learn-1.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (11 kB)
Collecting joblib>=1.0 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting decorator>=4.3.0 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached decorator-5.2.1-py3-none-any.whl.metadata (3.9 kB)
Collecting soundfile>=0.12.1 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached soundfile-0.13.1-py2.py3-none-manylinux_2_28_x86_64.whl.metadata (16 kB)
Collecting pooch>=1.1 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached pooch-1.9.0-py3-none-any.whl.metadata (10 kB)
Collecting soxr>=0.3.2 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached soxr-1.0.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.6 kB)
Collecting lazy_loader>=0.1 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached lazy_loader-0.4-py3-none-any.whl.metadata (7.6 kB)
Collecting msgpack>=1.0 (from librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached msgpack-1.1.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.1 kB)
Collecting llvmlite<0.47,>=0.46.0dev0 (from numba>=0.51.0->librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached llvmlite-0.46.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (5.0 kB)
Collecting platformdirs>=2.5.0 (from pooch>=1.1->librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached platformdirs-4.5.1-py3-none-any.whl.metadata (12 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas<4.0,>=1.0->gradio>=4.0.0->open-game-dubber==0.1.0)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting markdown-it-py>=2.2.0 (from rich>=10.11.0->typer>=0.12.0->open-game-dubber==0.1.0)
  Using cached markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich>=10.11.0->typer>=0.12.0->open-game-dubber==0.1.0)
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=10.11.0->typer>=0.12.0->open-game-dubber==0.1.0)
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Collecting threadpoolctl>=3.2.0 (from scikit-learn>=1.1.0->librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting cffi>=1.0 (from soundfile>=0.12.1->librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached cffi-2.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.6 kB)
Collecting pycparser (from cffi>=1.0->soundfile>=0.12.1->librosa>=0.10.0->open-game-dubber==0.1.0)
  Using cached pycparser-3.0-py3-none-any.whl.metadata (8.2 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.6->deepfilternet>=0.5.0->open-game-dubber==0.1.0)
  Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting networkx>=2.5.1 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached networkx-3.6.1-py3-none-any.whl.metadata (6.8 kB)
Collecting cuda-bindings==12.9.4 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (2.6 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime-cu12==12.8.90 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti-cu12==12.8.90 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cublas-cu12==12.8.4.1 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufft-cu12==11.3.3.83 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-curand-cu12==10.3.9.90 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cusolver-cu12==11.7.3.90 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparse-cu12==12.5.8.93 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparselt-cu12==0.7.1 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl.metadata (7.0 kB)
Collecting nvidia-nccl-cu12==2.27.5 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)
Collecting nvidia-nvshmem-cu12==3.4.5 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
Collecting nvidia-nvtx-cu12==12.8.90 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvjitlink-cu12==12.8.93 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufile-cu12==1.13.1.3 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting triton==3.6.0 (from torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
Collecting cuda-pathfinder~=1.1 (from cuda-bindings==12.9.4->torch>=2.4.0->open-game-dubber==0.1.0)
  Using cached cuda_pathfinder-1.3.3-py3-none-any.whl.metadata (1.9 kB)
Collecting omegaconf (from dora-search->demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached omegaconf-2.3.0-py3-none-any.whl.metadata (3.9 kB)
Collecting retrying (from dora-search->demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached retrying-1.4.2-py3-none-any.whl.metadata (5.5 kB)
Collecting submitit (from dora-search->demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached submitit-1.5.4-py3-none-any.whl.metadata (7.4 kB)
Collecting treetable (from dora-search->demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached treetable-0.2.6-py3-none-any.whl.metadata (5.3 kB)
Collecting antlr4-python3-runtime==4.9.* (from omegaconf->dora-search->demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached antlr4-python3-runtime-4.9.3.tar.gz (117 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting cloudpickle>=1.2.1 (from submitit->dora-search->demucs>=4.0.0->open-game-dubber==0.1.0)
  Using cached cloudpickle-3.1.2-py3-none-any.whl.metadata (7.1 kB)
Using cached ctranslate2-4.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (39.0 MB)
Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (807 kB)
Using cached deepfilternet-0.5.6-py3-none-any.whl (113 kB)
Using cached appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
Using cached numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
Using cached packaging-23.2-py3-none-any.whl (53 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Using cached charset_normalizer-3.4.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (153 kB)
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached urllib3-2.6.3-py3-none-any.whl (131 kB)
Using cached certifi-2026.1.4-py3-none-any.whl (152 kB)
Using cached faster_whisper-1.2.1-py3-none-any.whl (1.1 MB)
Using cached onnxruntime-1.24.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.1 MB)
Using cached tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
Using cached huggingface_hub-1.4.1-py3-none-any.whl (553 kB)
Using cached hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
Using cached httpx-0.28.1-py3-none-any.whl (73 kB)
Using cached httpcore-1.0.9-py3-none-any.whl (78 kB)
Using cached av-16.1.0-cp312-cp312-manylinux_2_28_x86_64.whl (41.2 MB)
Using cached fsspec-2026.2.0-py3-none-any.whl (202 kB)
Using cached gradio-6.5.1-py3-none-any.whl (24.2 MB)
Using cached gradio_client-2.0.3-py3-none-any.whl (55 kB)
Using cached aiofiles-24.1.0-py3-none-any.whl (15 kB)
Using cached anyio-4.12.1-py3-none-any.whl (113 kB)
Using cached fastapi-0.128.7-py3-none-any.whl (103 kB)
Using cached groovy-0.1.2-py3-none-any.whl (14 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (22 kB)
Using cached orjson-3.11.7-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (133 kB)
Using cached pandas-3.0.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (10.9 MB)
Using cached pillow-12.1.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (7.0 MB)
Using cached pydantic-2.12.5-py3-none-any.whl (463 kB)
Using cached pydantic_core-2.41.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
Using cached safehttpx-0.1.7-py3-none-any.whl (9.0 kB)
Using cached semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)
Using cached starlette-0.52.1-py3-none-any.whl (74 kB)
Using cached tomlkit-0.13.3-py3-none-any.whl (38 kB)
Using cached typer-0.22.0-py3-none-any.whl (56 kB)
Using cached annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)
Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
Using cached brotli-1.2.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.4 MB)
Using cached click-8.3.1-py3-none-any.whl (108 kB)
Using cached h11-0.16.0-py3-none-any.whl (37 kB)
Using cached lameenc-1.8.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (249 kB)
Using cached librosa-0.11.0-py3-none-any.whl (260 kB)
Using cached audioread-3.1.0-py3-none-any.whl (23 kB)
Using cached decorator-5.2.1-py3-none-any.whl (9.2 kB)
Using cached joblib-1.5.3-py3-none-any.whl (309 kB)
Using cached lazy_loader-0.4-py3-none-any.whl (12 kB)
Using cached loguru-0.7.3-py3-none-any.whl (61 kB)
Using cached msgpack-1.1.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (427 kB)
Using cached numba-0.63.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
Using cached llvmlite-0.46.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (56.3 MB)
Using cached ollama-0.6.1-py3-none-any.whl (14 kB)
Using cached pooch-1.9.0-py3-none-any.whl (67 kB)
Using cached platformdirs-4.5.1-py3-none-any.whl (18 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached python_multipart-0.0.22-py3-none-any.whl (24 kB)
Using cached pytz-2025.2-py2.py3-none-any.whl (509 kB)
Using cached rich-14.3.2-py3-none-any.whl (309 kB)
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Using cached markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Using cached scikit_learn-1.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (8.9 MB)
Using cached scipy-1.17.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (35.0 MB)
Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached soundfile-0.13.1-py2.py3-none-manylinux_2_28_x86_64.whl (1.3 MB)
Using cached cffi-2.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (219 kB)
Using cached soxr-1.0.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (238 kB)
Using cached sympy-1.14.0-py3-none-any.whl (6.3 MB)
Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Using cached torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl (915.7 MB)
Using cached cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (12.2 MB)
Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl (594.3 MB)
Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (10.2 MB)
Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (88.0 MB)
Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (954 kB)
Using cached nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl (706.8 MB)
Using cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (193.1 MB)
Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl (63.6 MB)
Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl (267.5 MB)
Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (288.2 MB)
Using cached nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl (287.2 MB)
Using cached nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (322.3 MB)
Using cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.3 MB)
Using cached nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (139.1 MB)
Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)
Using cached triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (188.3 MB)
Using cached cuda_pathfinder-1.3.3-py3-none-any.whl (27 kB)
Using cached networkx-3.6.1-py3-none-any.whl (2.1 MB)
Using cached torchaudio-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl (1.9 MB)
Using cached tqdm-4.67.3-py3-none-any.whl (78 kB)
Using cached typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Using cached uvicorn-0.40.0-py3-none-any.whl (68 kB)
Using cached einops-0.8.2-py3-none-any.whl (65 kB)
Using cached ffmpy-1.0.0-py3-none-any.whl (5.6 kB)
Using cached filelock-3.20.3-py3-none-any.whl (16 kB)
Using cached flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
Using cached omegaconf-2.3.0-py3-none-any.whl (79 kB)
Using cached openunmix-1.3.0-py3-none-any.whl (40 kB)
Using cached protobuf-6.33.5-cp39-abi3-manylinux2014_x86_64.whl (323 kB)
Using cached pycparser-3.0-py3-none-any.whl (48 kB)
Using cached pydub-0.25.1-py2.py3-none-any.whl (32 kB)
Using cached retrying-1.4.2-py3-none-any.whl (10 kB)
Using cached setuptools-82.0.0-py3-none-any.whl (1.0 MB)
Using cached submitit-1.5.4-py3-none-any.whl (76 kB)
Using cached cloudpickle-3.1.2-py3-none-any.whl (22 kB)
Using cached treetable-0.2.6-py3-none-any.whl (7.4 kB)
Using cached typer_slim-0.22.0-py3-none-any.whl (3.4 kB)
Building wheels for collected packages: open-game-dubber, deepfilterlib, demucs, julius, dora-search, antlr4-python3-runtime
  Building wheel for open-game-dubber (pyproject.toml): started
  Building wheel for open-game-dubber (pyproject.toml): finished with status 'done'
  Created wheel for open-game-dubber: filename=open_game_dubber-0.1.0-py3-none-any.whl size=4714 sha256=fd16fa88ed80f875c860ae0fdd829948788524c3eaa41492288617da1cd2d456
  Stored in directory: /tmp/pip-ephem-wheel-cache-5uza1qhx/wheels/54/1b/b7/aa63e25c8f14f4f2ae7b04e6097bdecb770e455c5c1ee0a600
  Building wheel for deepfilterlib (pyproject.toml): started
  Building wheel for deepfilterlib (pyproject.toml): finished with status 'done'
  Created wheel for deepfilterlib: filename=DeepFilterLib-0.5.6-cp312-cp312-linux_x86_64.whl size=656603 sha256=a98acb81e1b76c83e553d1ffb75c3e11f791b07cee0f0ccca950d9d10174472d
  Stored in directory: /home/jules/.cache/pip/wheels/f5/e0/c1/b4bc8d99fa145f2444a0186a5c6355fc26b5ba26ca6665c975
  Building wheel for demucs (pyproject.toml): started
  Building wheel for demucs (pyproject.toml): finished with status 'done'
  Created wheel for demucs: filename=demucs-4.0.1-py3-none-any.whl size=78481 sha256=65051938bb0d221be0620bac662ce93fb407ff03181bef13b650d3d43e4ecc20
  Stored in directory: /home/jules/.cache/pip/wheels/1b/0c/20/a3b3daa1f9b65c8b0445729f94740ec335d0f86f1066c5c414
  Building wheel for julius (pyproject.toml): started
  Building wheel for julius (pyproject.toml): finished with status 'done'
  Created wheel for julius: filename=julius-0.2.7-py3-none-any.whl size=21967 sha256=fc6e18603917e648be87c13d15da82ba7b31b46c05311debc80d974af1cd3ffa
  Stored in directory: /home/jules/.cache/pip/wheels/de/c1/ca/544dafe48401e8e2e17064dfe465a390fca9e8720ffa12e744
  Building wheel for dora-search (pyproject.toml): started
  Building wheel for dora-search (pyproject.toml): finished with status 'done'
  Created wheel for dora-search: filename=dora_search-0.1.12-py3-none-any.whl size=75194 sha256=8d7ab7ea3dcd3710e3bbd19ccc4bd8743afae6e5bf230f6a75fbc29b1d179f02
  Stored in directory: /home/jules/.cache/pip/wheels/55/40/4c/3478187386b56625ab5aa8a9070ff1eea704ddb27d15abe73f
  Building wheel for antlr4-python3-runtime (pyproject.toml): started
  Building wheel for antlr4-python3-runtime (pyproject.toml): finished with status 'done'
  Created wheel for antlr4-python3-runtime: filename=antlr4_python3_runtime-4.9.3-py3-none-any.whl size=144591 sha256=18b95c630ff48e52f95fe1b2ed0b7f58ee26745c5918de40a737f21df1d404a6
  Stored in directory: /home/jules/.cache/pip/wheels/1f/be/48/13754633f1d08d1fbfc60d5e80ae1e5d7329500477685286cd
Successfully built open-game-dubber deepfilterlib demucs julius dora-search antlr4-python3-runtime
Installing collected packages: pytz, pydub, nvidia-cusparselt-cu12, mpmath, lameenc, flatbuffers, brotli, appdirs, antlr4-python3-runtime, urllib3, typing-inspection, triton, treetable, tqdm, tomlkit, threadpoolctl, sympy, six, shellingham, setuptools, semantic-version, retrying, pyyaml, python-multipart, pygments, pydantic-core, pycparser, protobuf, platformdirs, pillow, packaging, orjson, nvidia-nvtx-cu12, nvidia-nvshmem-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, msgpack, mdurl, markupsafe, loguru, llvmlite, joblib, idna, hf-xet, h11, groovy, fsspec, filelock, ffmpy, einops, decorator, cuda-pathfinder, cloudpickle, click, charset_normalizer, certifi, av, audioread, annotated-types, annotated-doc, aiofiles, uvicorn, submitit, soxr, scipy, requests, python-dateutil, pydantic, onnxruntime, omegaconf, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, numba, markdown-it-py, lazy_loader, jinja2, httpcore, deepfilterlib, cuda-bindings, ctranslate2, cffi, anyio, starlette, soundfile, scikit-learn, rich, pooch, pandas, nvidia-cusolver-cu12, httpx, deepfilternet, typer, torch, safehttpx, ollama, librosa, fastapi, typer-slim, torchaudio, julius, dora-search, openunmix, huggingface-hub, tokenizers, gradio-client, demucs, gradio, faster-whisper, open-game-dubber

Successfully installed aiofiles-24.1.0 annotated-doc-0.0.4 annotated-types-0.7.0 antlr4-python3-runtime-4.9.3 anyio-4.12.1 appdirs-1.4.4 audioread-3.1.0 av-16.1.0 brotli-1.2.0 certifi-2026.1.4 cffi-2.0.0 charset_normalizer-3.4.4 click-8.3.1 cloudpickle-3.1.2 ctranslate2-4.7.1 cuda-bindings-12.9.4 cuda-pathfinder-1.3.3 decorator-5.2.1 deepfilterlib-0.5.6 deepfilternet-0.5.6 demucs-4.0.1 dora-search-0.1.12 einops-0.8.2 fastapi-0.128.7 faster-whisper-1.2.1 ffmpy-1.0.0 filelock-3.20.3 flatbuffers-25.12.19 fsspec-2026.2.0 gradio-6.5.1 gradio-client-2.0.3 groovy-0.1.2 h11-0.16.0 hf-xet-1.2.0 httpcore-1.0.9 httpx-0.28.1 huggingface-hub-1.4.1 idna-3.11 jinja2-3.1.6 joblib-1.5.3 julius-0.2.7 lameenc-1.8.1 lazy_loader-0.4 librosa-0.11.0 llvmlite-0.46.0 loguru-0.7.3 markdown-it-py-4.0.0 markupsafe-3.0.3 mdurl-0.1.2 mpmath-1.3.0 msgpack-1.1.2 networkx-3.6.1 numba-0.63.1 numpy-1.26.4 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.5 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvshmem-cu12-3.4.5 nvidia-nvtx-cu12-12.8.90 ollama-0.6.1 omegaconf-2.3.0 onnxruntime-1.24.1 open-game-dubber-0.1.0 openunmix-1.3.0 orjson-3.11.7 packaging-23.2 pandas-3.0.0 pillow-12.1.1 platformdirs-4.5.1 pooch-1.9.0 protobuf-6.33.5 pycparser-3.0 pydantic-2.12.5 pydantic-core-2.41.5 pydub-0.25.1 pygments-2.19.2 python-dateutil-2.9.0.post0 python-multipart-0.0.22 pytz-2025.2 pyyaml-6.0.3 requests-2.32.5 retrying-1.4.2 rich-14.3.2 safehttpx-0.1.7 scikit-learn-1.8.0 scipy-1.17.0 semantic-version-2.10.0 setuptools-82.0.0 shellingham-1.5.4 six-1.17.0 soundfile-0.13.1 soxr-1.0.0 starlette-0.52.1 submitit-1.5.4 sympy-1.14.0 threadpoolctl-3.6.0 tokenizers-0.22.2 tomlkit-0.13.3 torch-2.10.0 torchaudio-2.10.0 tqdm-4.67.3 treetable-0.2.6 triton-3.6.0 typer-0.22.0 typer-slim-0.22.0 typing-inspection-0.4.2 urllib3-2.6.3 uvicorn-0.40.0

### Model Setup

Before running the pipeline, you need to download the required AI models (Faster-Whisper, Demucs, etc.). We provide a convenient CLI command for this:

Starting download of models to models...
Downloading: "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th" to /home/jules/.cache/torch/hub/checkpoints/955717e8-8726e21a.th
Download process completed.

You can specify a custom directory or model size:

Starting download of models to /path/to/models...

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
