import logging
import os
import subprocess
from typing import Optional

import soundfile as sf

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles audio manipulation, separation, and denoising.
    """

    def __init__(self):
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    def _resolve_demucs_paths(self, output_dir: str, audio_path: str) -> Optional[dict[str, str]]:
        """Resolves the output paths for Demucs separation."""
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        # htdemucs is the default model name, but hdemucs might be used in some versions
        for model in ["htdemucs", "hdemucs"]:
            model_dir = os.path.join(output_dir, model, filename)
            vocal_path = os.path.join(model_dir, "vocals.wav")
            bg_path = os.path.join(model_dir, "no_vocals.wav")

            if os.path.exists(vocal_path) and os.path.exists(bg_path):
                return {"vocals": vocal_path, "background": bg_path}
        return None

    def separate_vocals(self, audio_path: str, output_dir: str) -> Optional[dict[str, str]]:
        """
        Uses Demucs to separate vocals from background music/sfx.
        Returns a dictionary with 'vocals' and 'background' paths.
        Now runs in a separate process to avoid monkey-patching torchaudio.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        logger.info(f"Separating vocals for: {audio_path}")
        try:
            # We use subprocess instead of in-process call to avoid global monkey-patching
            # and to isolate dependency issues (like the backend problems on Windows).
            # The demucs CLI is expected to be in the environment.
            cmd = [
                "uv",
                "run",
                "python",
                os.path.join(os.path.dirname(__file__), "demucs_wrapper.py"),
                "--two-stems",
                "vocals",
                "-o",
                output_dir,
                audio_path,
            ]

            logger.info(f"Running Demucs via subprocess: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Demucs separation failed with exit code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return None

            return self._resolve_demucs_paths(output_dir, audio_path)
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}", exc_info=True)
            return None

    def denoise_vocals(self, vocal_path: str, output_path: str) -> Optional[str]:
        """
        Uses DeepFilterNet to clean vocal audio.
        """
        if not os.path.exists(vocal_path):
            return None

        logger.info(f"Denoising vocals: {vocal_path}")
        try:
            # DeepFilterNet typically provides a CLI 'df-process'
            # We must use 'uv run' to ensure it's found in the environment
            cmd = ["uv", "run", "df-process", "--", vocal_path, "-o", os.path.dirname(output_path)]
            logger.info(f"Running DeepFilterNet: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"DeepFilterNet output: {result.stdout}")

            base_name = os.path.splitext(os.path.basename(vocal_path))[0]
            out_dir = os.path.dirname(output_path)

            # DeepFilterNet usually appends _DeepFilterNet3 to the filename or similar
            expected_suffix = "_DeepFilterNet3.wav"
            potential_path = os.path.join(out_dir, base_name + expected_suffix)

            if os.path.exists(output_path):
                return output_path
            elif os.path.exists(potential_path):
                os.replace(potential_path, output_path)
                return output_path

            return output_path if os.path.exists(output_path) else None
        except subprocess.CalledProcessError as e:
            logger.error(f"DeepFilterNet denoising failed with exit code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"DeepFilterNet denoising failed: {e}")
            return None

    def mix_audio(self, vocal_path: str, background_path: str, output_path: str):
        """
        Combines vocals and background tracks using torchaudio.
        """
        if not torch or not torchaudio:
            logger.error("torch/torchaudio not found. Mixing is not possible without these dependencies.")
            return

        logger.info(f"Mixing audio to: {output_path}")
        vocal_data, sr_v = sf.read(vocal_path, always_2d=True)
        # sf reads as (frames, channels), torchaudio expects (channels, frames)
        vocal = torch.from_numpy(vocal_data.T).float()

        bg_data, sr_b = sf.read(background_path, always_2d=True)
        bg = torch.from_numpy(bg_data.T).float()

        # Ensure same sample rate and length (trim/pad if necessary)
        if sr_v != sr_b:
            bg = torchaudio.transforms.Resample(sr_b, sr_v)(bg)

        # Handle channel mismatch (e.g., mono vocals + stereo background)
        if vocal.shape[0] != bg.shape[0]:
            logger.info(f"Channel mismatch: vocals={vocal.shape[0]}, bg={bg.shape[0]}. Leveling...")
            if vocal.shape[0] == 1 and bg.shape[0] == 2:
                vocal = vocal.expand(2, -1)
            elif vocal.shape[0] == 2 and bg.shape[0] == 1:
                bg = bg.expand(2, -1)
            else:
                logger.warning("Unusual channel counts, simple expansion might not work perfectly.")

        min_len = min(vocal.shape[1], bg.shape[1])
        # Prevent digital clipping by reducing gain (simple additive mix can exceed 1.0)
        mixed = (vocal[:, :min_len] + bg[:, :min_len]) * 0.5

        # Use soundfile for saving to avoid torchaudio/torchcodec issues on Windows
        data = mixed.cpu().detach().numpy().T
        sf.write(output_path, data, sr_v)


if __name__ == "__main__":
    # Basic test logic
    logging.basicConfig(level=logging.INFO)
    processor = AudioProcessor()
    # print(processor.device)
