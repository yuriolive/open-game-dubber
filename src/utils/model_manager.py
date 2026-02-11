import logging
import os

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
            model_path = download_whisper(model_size, output_dir=os.path.join(output_dir, "faster-whisper"))
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

    # 3. DeepFilterNet (downloads on first run usually, but we can try to trigger it)
    # DeepFilterNet specific download logic might be needed if it doesn't expose a clean API
    logger.info("DeepFilterNet models are typically downloaded on first use.")

    logger.info("All model downloads completed (or attempted).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all_models()
