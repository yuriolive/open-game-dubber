import logging
import os
import subprocess
from typing import Optional

import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles audio manipulation, separation, and denoising.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def separate_vocals(self, audio_path: str, output_dir: str) -> Optional[str]:
        """
        Uses Demucs to separate vocals from background music/sfx.
        Returns the path to the extracted vocal track.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        logger.info(f"Separating vocals for: {audio_path}")
        try:
            # We monkeypatch torchaudio to avoid broken torchcodec on Windows

            original_save = torchaudio.save
            original_load = torchaudio.load

            def soundfile_save_patch(filepath, src, sample_rate, **kwargs):
                data = src.cpu().detach().numpy()
                if len(data.shape) == 2:
                    data = data.T
                sf.write(filepath, data, sample_rate)

            def soundfile_load_patch(filepath, **kwargs):
                data, sr = sf.read(filepath, always_2d=True)
                # soundfile returns (frames, channels), torchaudio expects (channels, frames)
                tensor = torch.from_numpy(data.T).float()
                return tensor, sr

            # Apply patches
            torchaudio.save = soundfile_save_patch
            torchaudio.load = soundfile_load_patch

            from demucs.separate import main as demucs_main

            # Prepare arguments for demucs
            args = ["--two-stems", "vocals", "-o", output_dir, audio_path]

            logger.info(f"Running Demucs in-process with args: {args}")
            try:
                demucs_main(args)
            except SystemExit as e:
                if e.code != 0:
                    logger.error(f"Demucs in-process failed with exit code {e.code}")
                    return None
            finally:
                # Restore originals
                torchaudio.save = original_save
                torchaudio.load = original_load

            # Demucs structure: output_dir/htdemucs/filename/vocals.wav
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            # htdemucs is the default model name
            vocal_path = os.path.join(output_dir, "htdemucs", filename, "vocals.wav")

            if os.path.exists(vocal_path):
                return vocal_path

            # Try falling back to older hdemucs if htdemucs folder missing
            vocal_path_alt = os.path.join(output_dir, "hdemucs", filename, "vocals.wav")
            if os.path.exists(vocal_path_alt):
                return vocal_path_alt

            return None
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
            cmd = ["df-process", vocal_path, "-o", os.path.dirname(output_path)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"DeepFilterNet output: {result.stdout}")

            # DeepFilterNet usually appends _DeepFilterNet3 to the filename or similar
            # Check for suffixes if output_path doesn't exist directly
            if os.path.exists(output_path):
                return output_path

            # Scan directory for any file starting with original name and ending with .wav
            base_name = os.path.splitext(os.path.basename(vocal_path))[0]
            out_dir = os.path.dirname(output_path)
            for f in os.listdir(out_dir):
                if f.startswith(base_name) and "DeepFilterNet" in f and f.endswith(".wav"):
                    found_path = os.path.join(out_dir, f)
                    # Rename to expected output_path for consistency
                    os.rename(found_path, output_path)
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
        logger.info(f"Mixing audio to: {output_path}")
        vocal_data, sr_v = sf.read(vocal_path, always_2d=True)
        # sf reads as (frames, channels), torchaudio expects (channels, frames)
        vocal = torch.from_numpy(vocal_data.T).float()

        bg_data, sr_b = sf.read(background_path, always_2d=True)
        bg = torch.from_numpy(bg_data.T).float()

        # Ensure same sample rate and length (trim/pad if necessary)
        if sr_v != sr_b:
            bg = torchaudio.transforms.Resample(sr_b, sr_v)(bg)

        # Simple additive mix (ensure they are the same shape)
        # In a real scenario, we'd handle channel matching/length matching more robustly
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
