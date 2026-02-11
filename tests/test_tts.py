import os
import unittest

import numpy as np
import soundfile as sf

from src.models.tts import TTSWrapper


class TestTTSWrapper(unittest.TestCase):
    def setUp(self):
        self.tts = TTSWrapper()
        self.ref_path = "test_ref.wav"
        self.output_path = "test_gen.wav"

        # Create dummy reference audio
        data = np.zeros(24000)
        sf.write(self.ref_path, data, 24000)

    def tearDown(self):
        for p in [self.ref_path, self.output_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_generate_dub_mock_success(self):
        """Tests that generate_dub succeeds and creates a valid audio file in mock mode."""
        text = "Test synthesis"
        result = self.tts.generate_dub(text, self.ref_path, self.output_path)

        self.assertEqual(result, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

        # Verify it's a valid audio file
        data, sr = sf.read(self.output_path)
        self.assertEqual(sr, 24000)
        self.assertGreater(len(data), 0)

    def test_generate_dub_missing_ref(self):
        """Tests that generate_dub returns None if reference path is missing."""
        result = self.tts.generate_dub("text", "non_existent.wav", self.output_path)
        self.assertIsNone(result)

    def test_generate_dub_empty_text(self):
        """Tests that generate_dub returns None for empty text."""
        result = self.tts.generate_dub("", self.ref_path, self.output_path)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
