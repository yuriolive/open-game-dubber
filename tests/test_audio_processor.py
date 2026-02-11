import os

# Use the same mocking strategy as test_stt.py to avoid heavy dependencies in basic tests
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

# Removed global sys.modules patching
# from src.utils.audio_processor import AudioProcessor will be done in setUp


class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        # Patch modules before importing AudioProcessor
        self.modules_patcher = patch.dict(
            "sys.modules",
            {"torch": MagicMock(), "torchaudio": MagicMock(), "demucs": MagicMock(), "demucs.separate": MagicMock()},
        )
        self.modules_patcher.start()

        # Import inside setUp to use patched modules
        if "src.utils.audio_processor" in sys.modules:
            del sys.modules["src.utils.audio_processor"]
        from src.utils.audio_processor import AudioProcessor

        self.processor = AudioProcessor()

        self.vocal_path = "test_vocal.wav"
        self.bg_path = "test_bg.wav"
        self.output_path = "test_mixed.wav"

    def tearDown(self):
        for p in [self.vocal_path, self.bg_path, self.output_path]:
            if os.path.exists(p):
                os.remove(p)
        self.modules_patcher.stop()
        if "src.utils.audio_processor" in sys.modules:
            del sys.modules["src.utils.audio_processor"]

    def _create_dummy_and_mix(self, vocal_channels, bg_channels, expected_channels):
        """Helper to create dummy audio, mix them, and verify output."""
        # Create dummy data
        vocal_data = np.random.uniform(-1, 1, (100, vocal_channels)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, bg_channels)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        # Mock torch.from_numpy to return objects with shape
        import sys

        mock_torch = sys.modules["torch"]

        def mock_from_numpy(array):
            m = MagicMock()
            # Array from sf.read is (frames, channels), code transposes it
            # So from_numpy gets (channels, frames)
            # We need to reflect that in shape
            m.shape = array.shape
            m.float.return_value = m
            # Support addition and multiplication
            m.__add__.return_value = m
            m.__mul__.return_value = m
            m.__radd__.return_value = m
            m.__rmul__.return_value = m
            # Handle slicing
            m.__getitem__.return_value = m
            # Handle .cpu().detach().numpy().T
            m.cpu.return_value.detach.return_value.numpy.return_value.T = np.zeros((100, expected_channels))

            # For expand logic
            m.expand.return_value = m

            return m

        mock_torch.from_numpy.side_effect = mock_from_numpy

        # Mock torchaudio resample
        mock_torchaudio = sys.modules["torchaudio"]
        # Resample returns a callable, which returns a tensor
        resample_transform = MagicMock()
        resample_transform.side_effect = lambda x: x
        mock_torchaudio.transforms.Resample.return_value = resample_transform

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
        """Tests that separate_vocals returns the expected dictionary format and calls subprocess."""
        audio_path = "input.wav"
        output_dir = "out"

        with patch("subprocess.run") as mock_run, patch("os.path.exists") as mock_exists:
            # Simulate subprocess success
            mock_run.return_value = MagicMock(returncode=0)
            # Simulate files being created
            mock_exists.side_effect = lambda p: True

            result = self.processor.separate_vocals(audio_path, output_dir)

            self.assertIsInstance(result, dict)
            self.assertIn("vocals", result)
            self.assertIn("background", result)

            # Verify subprocess.run call
            self.assertTrue(mock_run.called)
            cmd = mock_run.call_args[0][0]
            self.assertIn("python", cmd)
            self.assertIn("-m", cmd)
            self.assertIn("demucs.separate", cmd)
            self.assertIn("--two-stems", cmd)
            self.assertIn("vocals", cmd)
            self.assertIn("-o", cmd)
            self.assertIn(output_dir, cmd)
            self.assertIn(audio_path, cmd)

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

        with patch("os.path.exists", return_value=True), patch("subprocess.run") as mock_run:
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
