import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before importing the module under test
# This allows running tests without installing heavy ML libraries
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.cuda.is_available"] = MagicMock(return_value=False)
sys.modules["faster_whisper"] = MagicMock()

# Now import the module
# We need to ensure the module sees our mocks
from src.models.stt import FasterWhisperTranscriber  # noqa: E402


class TestFasterWhisperTranscriber(unittest.TestCase):
    def setUp(self):
        # Reset the mock for each test
        sys.modules["faster_whisper"].reset_mock()
        self.transcriber = FasterWhisperTranscriber(device="cpu")

    def test_initialization(self):
        self.assertEqual(self.transcriber.device, "cpu")
        self.assertIsNone(self.transcriber._model)

    def test_lazy_loading(self):
        # We need to inject our mock into the class's global scope or mock where it's used
        # Since we mocked sys.modules['faster_whisper'], the import in stt.py got the mock.
        # But stt.py does 'from faster_whisper import WhisperModel'.
        # If that import succeeded (because we mocked it), WhisperModel in stt.py is the mock class.

        # Let's inspect what stt.WhisperModel is
        from src.models import stt

        # We want to verify that accessing .model triggers instantiation
        with patch.object(stt, "WhisperModel", return_value=MagicMock()) as mock_model_class:
            _ = self.transcriber.model
            mock_model_class.assert_called_once()

    def test_transcribe(self):
        # Mock the model instance and its transcribe method
        mock_model_instance = MagicMock()

        # Setup segment return
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.5
        mock_segment.text = "Hello world"

        # transcribe returns (segments_generator, info)
        mock_model_instance.transcribe.return_value = ([mock_segment], None)

        # Inject the mock model into the transcriber
        self.transcriber._model = mock_model_instance

        with patch("os.path.exists", return_value=True):
            result = self.transcriber.transcribe("test_audio.wav")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Hello world")
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 1.5)

        mock_model_instance.transcribe.assert_called_with("test_audio.wav", beam_size=5, language="en", vad_filter=True)

    def test_transcribe_file_not_found(self):
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.transcriber.transcribe("nonexistent.wav")


if __name__ == "__main__":
    unittest.main()
