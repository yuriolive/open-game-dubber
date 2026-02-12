import logging
import os
import subprocess
from typing import Optional

import librosa
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

    def denoise_audio(self, audio_path: str, output_path: str) -> Optional[str]:
        """
        Uses DeepFilterNet to clean audio files.
        """
        if not os.path.exists(audio_path):
            return None

        logger.info(f"Denoising audio: {audio_path}")
        try:
            # DeepFilterNet typically provides a CLI
            # We use 'uvx' (tool run) to run DeepFilterNet in an isolated environment with compatible torch versions
            # This bypasses the conflict between DeepFilterNet and torchaudio 2.8+ in the main project
            cmd = [
                "uvx",
                "--python",
                "3.12",
                "--from",
                "deepfilternet",
                "--with",
                "torch==2.5.1",
                "--with",
                "torchaudio==2.5.1",
                "--with",
                "soundfile",
                "deepFilter",
                "-m",
                "DeepFilterNet3",
                audio_path,
                "-o",
                os.path.dirname(output_path),
            ]
            logger.info(f"Running DeepFilterNet via uvx: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"DeepFilterNet output: {result.stdout}")

            base_name = os.path.splitext(os.path.basename(audio_path))[0]
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
            if "ModuleNotFoundError" in e.stderr:
                logger.warning(f"DeepFilterNet unavailable (dependency issue): {e.stderr.strip().splitlines()[-1]}")
                logger.warning("Continuing with original audio (no denoising).")
            else:
                logger.error(f"DeepFilterNet denoising failed with exit code {e.returncode}")
                logger.error(f"STDOUT: {e.stdout}")
                logger.error(f"STDERR: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"DeepFilterNet denoising failed: {e}")
            return None

    def get_audio_duration(self, audio_path: str) -> float:
        """
        Returns the duration of an audio file in seconds.
        """
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            logger.error(f"Failed to get audio duration for {audio_path}: {e}")
            return 0.0

    def _trim_silence(self, audio: torch.Tensor, sr: int, top_db: int = 30) -> torch.Tensor:
        """
        Trims leading and trailing silence from a torch tensor using librosa.
        """
        try:
            # librosa expects (n_samples,) or (n_channels, n_samples)
            audio_np = audio.cpu().detach().numpy()
            trimmed_np, _ = librosa.effects.trim(audio_np, top_db=top_db)
            return torch.from_numpy(trimmed_np).float().to(self.device)
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}. Returning original audio.")
            return audio

    def mix_audio(self, vocal_path: str, background_path: str, output_path: str):
        """
        Combines vocals and background tracks using torchaudio.
        Optimizes for quality by using the highest available sample rate and avoiding robotic artifacts.
        """
        if not torch or not torchaudio:
            logger.error("torch/torchaudio not found. Mixing is not possible without these dependencies.")
            return

        logger.info(f"Mixing audio to: {output_path}")
        vocal_data, sr_v = sf.read(vocal_path, always_2d=True)
        bg_data, sr_b = sf.read(background_path, always_2d=True)

        # Target the highest sample rate to preserve quality
        target_sr = max(sr_v, sr_b)

        # sf reads as (frames, channels), torchaudio expects (channels, frames)
        vocal = torch.from_numpy(vocal_data.T).float().to(self.device)
        bg = torch.from_numpy(bg_data.T).float().to(self.device)

        # 1. Resample to highest sample rate first
        if sr_v != target_sr:
            resampler = torchaudio.transforms.Resample(sr_v, target_sr).to(self.device)
            vocal = resampler(vocal)
        if sr_b != target_sr:
            resampler = torchaudio.transforms.Resample(sr_b, target_sr).to(self.device)
            bg = resampler(bg)

        # 2. Trim silence from vocals to eliminate unnecessary stretching
        # This is a key SOTA-inspired step: removing dead air often brings duration within target.
        vocal = self._trim_silence(vocal, target_sr)

        # Handle channel mismatch (e.g., mono vocals + stereo background)
        if vocal.shape[0] != bg.shape[0]:
            logger.info(f"Channel mismatch: vocals={vocal.shape[0]}, bg={bg.shape[0]}. Leveling...")
            if vocal.shape[0] == 1 and bg.shape[0] == 2:
                vocal = vocal.expand(2, -1)
            elif vocal.shape[0] == 2 and bg.shape[0] == 1:
                bg = bg.expand(2, -1)
            else:
                logger.warning("Unusual channel counts, simple expansion might not work perfectly.")

        target_len = bg.shape[1]
        vocal_len = vocal.shape[1]

        # 3. Synchronize duration
        # We only time-stretch if the trimmed vocal is EXPLICITLY longer than the background
        # and the difference is significant (>100ms). Otherwise, we prefer padding/truncation.
        diff_ms = abs(vocal_len - target_len) / target_sr * 1000

        if vocal_len > target_len and diff_ms > 100:
            rate = vocal_len / max(target_len, 1)
            if rate > 1.25:
                logger.warning(f"High time-stretch rate detected: {rate:.2f}x. Audio may sound robotic.")

            try:
                vocal_np = vocal.cpu().detach().numpy()
                stretched_vocal_np = librosa.effects.time_stretch(vocal_np, rate=rate)
                vocal = torch.from_numpy(stretched_vocal_np).float().to(self.device)
                logger.info(f"Time-stretched vocals by {rate:.2f}x to sync with background.")
            except Exception as e:
                logger.error(f"Time stretch failed: {e}. Falling back to truncation.")
                vocal = vocal[:, :target_len]
        elif vocal_len < target_len:
            # Pad vocals with silence to match background duration
            padding_len = target_len - vocal_len
            vocal = torch.nn.functional.pad(vocal, (0, padding_len))
            logger.info(
                f"Padded vocals with {padding_len} silence samples (approx {diff_ms:.0f}ms) to match background."
            )

        # Ensure same final length for mixing
        vocal = vocal[:, :target_len]
        bg = bg[:, :target_len]

        # 4. Sum and then normalize to peak
        # Give slight priority to vocals (1.0) and lower background slightly (0.8)
        mixed = vocal + (bg * 0.8)

        # Simple peak normalization if clipping occurs
        try:
            max_val = torch.max(torch.abs(mixed))
            if float(max_val) > 1.0:
                mixed = mixed / max_val
                logger.info(f"Normalized mixed output to avoid clipping (max was {float(max_val):.2f})")
        except (TypeError, ValueError, RuntimeError):
            # Fallback for unexpected tensor types or mock objects in tests
            pass

        # Save to file
        data = mixed.cpu().detach().numpy().T
        sf.write(output_path, data, target_sr)


if __name__ == "__main__":
    # Basic test logic
    logging.basicConfig(level=logging.INFO)
    processor = AudioProcessor()
    # print(processor.device)
