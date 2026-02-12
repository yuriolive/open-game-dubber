import os
import shutil

# Mocking modules before importing DubbingPipeline to avoid actual init
import sys
import unittest
from unittest.mock import MagicMock, patch

# Removed global sys.modules patching
# from src.core.pipeline import DubbingPipeline will be done in setUp


class TestDubbingPipeline(unittest.TestCase):
    def setUp(self):
        # Patch modules before importing DubbingPipeline
        self.modules_patcher = patch.dict(
            "sys.modules",
            {
                "src.models.stt": MagicMock(),
                "src.models.translator": MagicMock(),
                "src.models.tts": MagicMock(),
                "src.utils.audio_processor": MagicMock(),
                "src.core.state_manager": MagicMock(),
            },
        )
        self.modules_patcher.start()

        if "src.core.pipeline" in sys.modules:
            del sys.modules["src.core.pipeline"]
        from src.core.pipeline import DubbingPipeline

        self.output_dir = "test_output_dir"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.pipeline = DubbingPipeline(self.output_dir, "Portuguese")

        # Reset mocks
        self.pipeline.stt = MagicMock()
        self.pipeline.translator = MagicMock()
        self.pipeline.tts = MagicMock()
        self.pipeline.processor = MagicMock()
        self.pipeline.state = MagicMock()

        # Default behavior: processed check returns False
        self.pipeline.state.is_processed.return_value = False

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        self.modules_patcher.stop()
        if "src.core.pipeline" in sys.modules:
            del sys.modules["src.core.pipeline"]

    @patch("src.core.pipeline.shutil")
    @patch("src.core.pipeline.os.path.exists")
    @patch("src.core.pipeline.os.makedirs")
    @patch("tempfile.TemporaryDirectory")
    def test_process_file_success_flow(self, mock_temp_dir, mock_makedirs, mock_exists, mock_shutil):
        """
        Verifies the full successful flow.
        """
        audio_path = "input/sample.wav"

        # Setup temp dir
        mock_temp_dir.return_value.__enter__.return_value = "temp_dir_path"

        # Setup mocks
        self.pipeline.processor.separate_vocals.return_value = {
            "vocals": "temp/vocals.wav",
            "background": "temp/bg.wav",
        }
        self.pipeline.processor.denoise_vocals.return_value = "temp/vocals_clean.wav"
        self.pipeline.stt.transcribe.return_value = [{"text": "Hello world"}]
        self.pipeline.translator.translate.return_value = {
            "text": "Olá mundo",
            "tts_instruction": "Brazilian Portuguese accent and pronunciation",
        }
        self.pipeline.tts.generate_dub.return_value = "temp/dub.wav"

        # Ensure os.path.exists returns True so we enter the mix_audio branch
        mock_exists.return_value = True

        result = self.pipeline.process_file(audio_path)

        self.assertTrue(result)

        # 1. Verify separate_vocals called
        self.pipeline.processor.separate_vocals.assert_called_once()

        # 2. Verify denoise_vocals called
        self.pipeline.processor.denoise_vocals.assert_called_once()

        # 3. Verify transcribe called
        self.pipeline.stt.transcribe.assert_called_once()

        # 4. Verify translate called
        self.pipeline.translator.translate.assert_called_once()

        # 5. Verify mix_audio called (since bg exists)
        self.pipeline.processor.mix_audio.assert_called_once()

        # 6. Verify success
        self.pipeline.state.mark_completed.assert_called_once()

    @patch("src.core.pipeline.shutil")
    @patch("tempfile.TemporaryDirectory")
    def test_process_file_denoise_failure_fallback(self, mock_temp_dir, mock_shutil):
        """
        Tests that if denoise_vocals fails (returns None), we fallback to original vocals.
        """
        audio_path = "input/sample.wav"

        # Mocks setup
        self.pipeline.processor.separate_vocals.return_value = {
            "vocals": "temp/vocals.wav",
            "background": "temp/bg.wav",
        }
        # Simulate denoise failure
        self.pipeline.processor.denoise_vocals.return_value = None

        self.pipeline.stt.transcribe.return_value = [{"text": "Hello"}]
        self.pipeline.translator.translate.return_value = {"text": "Olá", "tts_instruction": ""}
        self.pipeline.tts.generate_dub.return_value = "temp/dub.wav"

        mock_temp_dir.return_value.__enter__.return_value = "temp_dir_path"

        self.pipeline.process_file(audio_path)

        # Verify transcribe called with ORIGINAL vocals
        self.pipeline.stt.transcribe.assert_called_once_with("temp/vocals.wav")


if __name__ == "__main__":
    unittest.main()
