import unittest
from unittest.mock import patch


class TestModelManager(unittest.TestCase):

    @patch("src.utils.model_manager.download_whisper")
    @patch("src.utils.model_manager.get_demucs_model")
    @patch("os.makedirs")
    def test_download_all_models_success(self, mock_makedirs, mock_get_demucs, mock_download_whisper):
        from src.utils.model_manager import download_all_models

        download_all_models(output_dir="test_models", model_size="tiny")

        mock_makedirs.assert_called_with("test_models", exist_ok=True)
        mock_download_whisper.assert_called_with("tiny")
        mock_get_demucs.assert_called_with("htdemucs")

    @patch("src.utils.model_manager.download_whisper", side_effect=Exception("Network error"))
    @patch("src.utils.model_manager.get_demucs_model")
    @patch("os.makedirs")
    def test_download_all_models_failure(self, mock_makedirs, mock_get_demucs, mock_download_whisper):
        from src.utils.model_manager import download_all_models

        # Should not raise exception, just log error
        with self.assertLogs(level='ERROR') as cm:
            download_all_models()

        self.assertTrue(any("Failed to download Faster-Whisper model" in r for r in cm.output))
        # Demucs should still try to download even if whisper failed
        mock_get_demucs.assert_called()

if __name__ == "__main__":
    unittest.main()
