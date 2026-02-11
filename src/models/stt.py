import os
from typing import List

import torch

# We use a try-except block to allow linting/testing without heavy dependencies if needed,
# but in production this should be a hard dependency.
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

class FasterWhisperTranscriber:
    """
    A wrapper for the Faster-Whisper model to handle audio transcription.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16" if torch.cuda.is_available() else "int8",
    ):
        """
        Initialize the transcriber.

        Args:
            model_size (str): The size of the whisper model (e.g., "large-v3", "medium").
            device (str): Device to run on ("cuda" or "cpu").
            compute_type (str): Quantization type ("float16", "int8_float16", "int8").
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def model(self):
        """
        Lazy load the model.
        """
        if self._model is None:
            if WhisperModel is None:
                raise ImportError("faster_whisper is not installed.")

            # We delay import or model loading until needed
            print(f"Loading Faster-Whisper model: {self.model_size} on {self.device}...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio_path: str, language: str = "en") -> List[dict]:
        """
        Transcribe an audio file.

        Args:
            audio_path (str): Path to the WAV file.
            language (str): Language code (default "en").

        Returns:
            List[dict]: A list of segments with 'start', 'end', and 'text'.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            vad_filter=True
        )

        result = []
        # segments is a generator, so we iterate
        for segment in segments:
            result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        return result
