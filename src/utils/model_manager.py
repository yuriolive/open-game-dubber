import logging
import os
import subprocess

# Try imports, handle gracefully if not installed
try:
    from faster_whisper import download_model as download_whisper
except ImportError:
    download_whisper = None

try:
    from demucs.pretrained import get_model as get_demucs_model
except ImportError:
    get_demucs_model = None

try:
    from src.models.translator import OllamaTranslator
except ImportError:
    OllamaTranslator = None

try:
    # DeepFilterNet handles downloads internally usually
    import deepfilternet  # noqa: F401
except ImportError:
    deepfilternet = None

logger = logging.getLogger(__name__)


def download_all_models(output_dir: str = "models", model_size: str = "large-v3-turbo"):
    """
    Downloads all necessary models for the pipeline.

    Args:
        output_dir (str): Base directory to store models.
        model_size (str): Whisper model size to download.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting model downloads...")

    # 1. Download Faster-Whisper
    if download_whisper:
        logger.info(f"Downloading Faster-Whisper model: {model_size}...")
        try:
            # faster-whisper downloads to cache by default, but we can specify output_dir if needed
            whisper_output_dir = os.path.join(output_dir, "faster-whisper")
            # TODO: Validate checksum to ensure file integrity
            if os.path.exists(os.path.join(whisper_output_dir, "model.bin")):
                logger.info(f"Faster-Whisper model found at: {whisper_output_dir}. Skipping download.")
            else:
                model_path = download_whisper(model_size, output_dir=whisper_output_dir)
                logger.info(f"Faster-Whisper model downloaded to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download Faster-Whisper model: {e}")
    else:
        logger.warning("faster-whisper library not found. Skipping download.")

    # 2. Download Demucs (htdemucs)
    if get_demucs_model:
        logger.info("Downloading Demucs (htdemucs) model...")
        try:
            # This triggers download to torch hub cache
            get_demucs_model("htdemucs")
            logger.info("Demucs model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download Demucs model: {e}")
    else:
        logger.warning("demucs library not found. Skipping download.")

    # 3. DeepFilterNet (downloads on first run usually)
    logger.info("DeepFilterNet models are typically downloaded on first use.")

    # 4. Qwen3-TTS (downloads from HuggingFace)
    logger.info("Checking Qwen3-TTS model...")
    try:
        import torch
        from qwen_tts import Qwen3TTSModel

        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        logger.info(f"Downloading/Verifying Qwen3-TTS model: {model_id}...")
        # This triggers download to HF cache
        Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cpu",  # Use CPU for download to avoid VRAM issues during prep
            torch_dtype=torch.float32,
        )
        logger.info("Qwen3-TTS model ready.")
    except ImportError:
        logger.warning("qwen-tts library not installed. Skipping model download.")
    except Exception as e:
        logger.error(f"Failed to download Qwen3-TTS model: {e}")

    # 5. Ollama models
    logger.info("Checking Ollama model: llama3.1...")
    if OllamaTranslator:
        try:
            translator = OllamaTranslator(model="llama3.1")
            # We use the API directly first as it's more reliable within our environment
            if translator.pull_model():
                logger.info("Ollama model llama3.1 ready.")
            else:
                # Fallback to CLI if API fails for some reason
                logger.info("Pulling llama3.1 model via Ollama CLI...")
                subprocess.run(["ollama", "pull", "llama3.1"], capture_output=True, check=True)
                logger.info("Ollama model llama3.1 ready.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(
                "Ollama command not found and API pull failed. Please ensure Ollama is installed and running."
            )
        except Exception as e:
            logger.error(f"Unexpected error pulling Ollama model: {e}")
    else:
        logger.warning("OllamaTranslator class not found. Skipping Ollama model download.")

    logger.info("All model downloads completed (or attempted).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all_models()
