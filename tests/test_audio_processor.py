import os

# Use the same mocking strategy as test_stt.py to avoid heavy dependencies in basic tests
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

sys.modules["torchaudio"] = MagicMock()
sys.modules["demucs"] = MagicMock()
sys.modules["demucs.separate"] = MagicMock()

from src.utils.audio_processor import AudioProcessor  # noqa: E402


class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor()
        self.vocal_path = "test_vocal.wav"
        self.bg_path = "test_bg.wav"
        self.output_path = "test_mixed.wav"

    def tearDown(self):
        for p in [self.vocal_path, self.bg_path, self.output_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_mix_audio_channel_matching_mono_stereo(self):
        """Tests that mono vocals and stereo background are mixed to stereo output."""
        # Create dummy mono vocal (100 samples, 1 channel)
        vocal_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        # Create dummy stereo background (100 samples, 2 channels)
        bg_data = np.random.uniform(-1, 1, (100, 2)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        # Mock torchaudio resample to return the input (or something simple)
        with patch("torchaudio.transforms.Resample", return_value=lambda x: x):
            self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)

        # Verify output exists and is stereo
        self.assertTrue(os.path.exists(self.output_path))
        data, sr = sf.read(self.output_path)
        self.assertEqual(data.shape[1], 2)
        self.assertEqual(sr, 16000)

    def test_mix_audio_channel_matching_stereo_mono(self):
        """Tests that stereo vocals and mono background are mixed to stereo output."""
        # Create dummy stereo vocal (100 samples, 2 channels)
        vocal_data = np.random.uniform(-1, 1, (100, 2)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        # Create dummy mono background (100 samples, 1 channel)
        bg_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        with patch("torchaudio.transforms.Resample", return_value=lambda x: x):
            self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))
        data, sr = sf.read(self.output_path)
        self.assertEqual(data.shape[1], 2)

    def test_separate_vocals_returns_dict(self):
        """Tests that separate_vocals returns the expected dictionary format."""
        audio_path = "input.wav"
        output_dir = "out"

        # Mocking dependencies
        # Since these are local imports, we can patch where they are expected to be imported from
        with patch("demucs.separate.main"), patch("os.path.exists") as mock_exists:
            # Simulate files being created
            mock_exists.side_effect = lambda p: True

            # Also mock the internal patch context manager to avoid errors
            with patch("unittest.mock.patch"):
                result = self.processor.separate_vocals(audio_path, output_dir)

            self.assertIsInstance(result, dict)
            self.assertIn("vocals", result)
            self.assertIn("background", result)
            self.assertTrue(result["vocals"].endswith("vocals.wav"))
            self.assertTrue(result["background"].endswith("no_vocals.wav"))


if __name__ == "__main__":
    unittest.main()
