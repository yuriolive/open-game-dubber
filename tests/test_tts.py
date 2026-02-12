import unittest

from src.models.tts import TTSWrapper


class TestTTSWrapper(unittest.TestCase):
    def setUp(self):
        from unittest.mock import MagicMock, patch

        self.mock_qwen = MagicMock()
        self.mock_model_instance = MagicMock()
        self.mock_qwen.from_pretrained.return_value = self.mock_model_instance

        def exists_side_effect(path):
            if path == "non_existent.wav":
                return False
            return True

        # Patching multiple targets
        self.patchers = [
            patch("src.models.tts.Qwen3TTSModel", self.mock_qwen),
            patch("src.models.tts.torch", MagicMock()),
            patch("src.models.tts.torchaudio", MagicMock()),
            patch("src.models.tts.sf", MagicMock()),
            patch("src.models.tts.os.path.exists", side_effect=exists_side_effect),
        ]
        for p in self.patchers:
            p.start()

        self.tts = TTSWrapper()
        self.ref_path = "test_ref.wav"
        self.output_path = "test_gen.wav"

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    def test_generate_dub_mock_success(self):
        """Tests that generate_dub succeeds and creates a valid audio file in mock mode."""
        from unittest.mock import MagicMock

        text = "Test synthesis"

        # Setup mock behavior
        # model.generate_voice_clone returns (wavs, sr)
        # We need to return a list with at least one element
        self.mock_model_instance.generate_voice_clone.return_value = ([MagicMock()], 24000)

        result = self.tts.generate_dub(text, self.ref_path, self.output_path)

        self.assertEqual(result, self.output_path)
        self.mock_model_instance.generate_voice_clone.assert_called_once()

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
