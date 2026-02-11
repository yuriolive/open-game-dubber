import unittest
from unittest.mock import patch


class TestModelManager(unittest.TestCase):
    @patch("src.utils.model_manager.download_whisper")
    @patch("src.utils.model_manager.get_demucs_model")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_download_all_models_success(self, mock_exists, mock_makedirs, mock_get_demucs, mock_download_whisper):
        from src.utils.model_manager import download_all_models

        download_all_models(output_dir="test_models", model_size="tiny")

        import os

        mock_makedirs.assert_called_with("test_models", exist_ok=True)
        # Should be called because exists returns False
        mock_download_whisper.assert_called_with("tiny", output_dir=os.path.join("test_models", "faster-whisper"))
        mock_get_demucs.assert_called_with("htdemucs")

    @patch("src.utils.model_manager.download_whisper")
    @patch("src.utils.model_manager.get_demucs_model")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_download_skip_existing(self, mock_exists, mock_makedirs, mock_get_demucs, mock_download_whisper):
        from src.utils.model_manager import download_all_models

        with self.assertLogs(level="INFO") as cm:
            download_all_models(output_dir="test_models", model_size="tiny")

        # Should NOT be called because exists returns True
        mock_download_whisper.assert_not_called()
        self.assertTrue(any("Faster-Whisper model found" in r for r in cm.output))
        # Demucs still called as we only added logic for whisper
        mock_get_demucs.assert_called_with("htdemucs")

    @patch("src.utils.model_manager.download_whisper", side_effect=Exception("Network error"))
    @patch("src.utils.model_manager.get_demucs_model")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_download_all_models_failure(self, mock_exists, mock_makedirs, mock_get_demucs, mock_download_whisper):
        from src.utils.model_manager import download_all_models

        # Should not raise exception, just log error
        with self.assertLogs(level="ERROR") as cm:
            download_all_models()

        self.assertTrue(any("Failed to download Faster-Whisper model" in r for r in cm.output))
        # Demucs should still try to download even if whisper failed
        mock_get_demucs.assert_called()


if __name__ == "__main__":
    unittest.main()
