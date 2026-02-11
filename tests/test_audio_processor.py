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

    def _create_dummy_and_mix(self, vocal_channels, bg_channels, expected_channels):
        """Helper to create dummy audio, mix them, and verify output."""
        # Create dummy data
        vocal_data = np.random.uniform(-1, 1, (100, vocal_channels)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, bg_channels)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        # Mock torchaudio resample
        with patch("torchaudio.transforms.Resample", return_value=lambda x: x):
            self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)

        # Verify output
        self.assertTrue(os.path.exists(self.output_path))
        data, sr = sf.read(self.output_path)
        self.assertEqual(data.shape[1], expected_channels)
        self.assertEqual(sr, 16000)

    def test_mix_audio_channel_matching_mono_stereo(self):
        """Tests that mono vocals and stereo background are mixed to stereo output."""
        self._create_dummy_and_mix(vocal_channels=1, bg_channels=2, expected_channels=2)

    def test_mix_audio_channel_matching_stereo_mono(self):
        """Tests that stereo vocals and mono background are mixed to stereo output."""
        self._create_dummy_and_mix(vocal_channels=2, bg_channels=1, expected_channels=2)

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

    def test_torchaudio_patch_restores_original(self):
        """Tests that _TorchaudioPatch context manager restores original functions."""
        import torchaudio

        original_save = torchaudio.save
        original_load = torchaudio.load

        with self.processor._TorchaudioPatch():
            # Should be patched to our helpers
            self.assertNotEqual(torchaudio.save, original_save)
            self.assertNotEqual(torchaudio.load, original_load)

        # Should be restored
        self.assertEqual(torchaudio.save, original_save)
        self.assertEqual(torchaudio.load, original_load)

    def test_resolve_demucs_paths(self):
        """Tests the logic for resolving Demucs output paths."""
        output_dir = "test_denoise_out"
        audio_path = "input_audio.wav"

        # Test case: htdemucs (default)
        # Structure: output_dir/htdemucs/input_audio/vocals.wav
        with patch("os.path.exists", return_value=True):
            paths = self.processor._resolve_demucs_paths(output_dir, audio_path)
            self.assertEqual(paths["vocals"], os.path.join(output_dir, "htdemucs", "input_audio", "vocals.wav"))
            self.assertEqual(paths["background"], os.path.join(output_dir, "htdemucs", "input_audio", "no_vocals.wav"))

    def test_denoise_vocals_subprocess_call(self):
        """Tests that denoise_vocals calls df-process with correct security flags."""
        vocal_path = "vocal.wav"
        output_path = "vocal_clean.wav"

        with patch("os.path.exists", side_effect=[True, True]), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Success")

            self.processor.denoise_vocals(vocal_path, output_path)

            # Verify the call pattern
            args, kwargs = mock_run.call_args
            cmd_list = args[0]
            self.assertEqual(cmd_list[0], "df-process")
            self.assertEqual(cmd_list[1], "--")
            self.assertIn(vocal_path, cmd_list)


if __name__ == "__main__":
    unittest.main()
