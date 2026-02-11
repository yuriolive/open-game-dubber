import logging
import os
from typing import Optional

import soundfile as sf

try:
    import torch
    import torchaudio
    from qwen_tts import Qwen3TTSModel
except ImportError:
    torch = None
    torchaudio = None
    Qwen3TTSModel = None

logger = logging.getLogger(__name__)


class TTSWrapper:
    """
    Handles dubbing synthesis using zero-shot voice cloning with Qwen3-TTS.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        self.model_id = model_id
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self._model = None

    @property
    def model(self):
        """
        Lazy load the Qwen3TTS model.
        """
        if self._model is None:
            if not Qwen3TTSModel:
                logger.error("qwen-tts library not installed or import failed.")
                return None

            logger.info(f"Loading Qwen3 TTS model: {self.model_id} on {self.device}...")
            try:
                # Qwen3-TTS recommends bfloat16 for CUDA
                self._model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                )
            except Exception as e:
                logger.error(f"Failed to load TTS model: {e}")
                self._model = "FAILED"
        return self._model

    def generate_dub(
        self,
        text: str,
        ref_audio_path: str,
        output_path: str,
        language: str = "Portuguese",
        ref_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates dubbed audio using voice cloning from reference.

        Args:
            text: The target text to synthesize.
            ref_audio_path: Path to the original vocal clip (reference).
            output_path: Path where the dub should be saved.
            language: Target language name.
            ref_text: The transcript of the original vocal clip (optional, but
                     improves cloning quality significantly).
        """
        if not text.strip():
            return None

        if not os.path.exists(ref_audio_path):
            logger.error(f"Reference audio not found: {ref_audio_path}")
            return None

        logger.info(f"Generating dub for: {text[:30]}... using reference: {ref_audio_path}")

        try:
            model = self.model
            if model is None or model == "FAILED":
                return None

            # Qwen3TTSModel has a specific method for zero-shot cloning
            # ref_audio can be a path. If ref_text is None, it uses x-vector-only mode.
            wavs, sr = model.generate_voice_clone(
                text=text, language=language, ref_audio=ref_audio_path, ref_text=ref_text
            )

            # wavs is typically a list of waveforms (one per text/audio pair)
            # We take the first one and save it.
            if len(wavs) > 0:
                sf.write(output_path, wavs[0].cpu().numpy(), sr)
                logger.info(f"Successfully synthesized audio to {output_path}")
                return output_path
            else:
                logger.error("TTS model returned no audio.")
                return None

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tts = TTSWrapper()
    # tts.generate_dub("Olá, como você está?", "ref.wav", "output.wav")
