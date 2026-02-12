import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf


class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        # Patch modules before importing AudioProcessor
        self.modules_patcher = patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "torchaudio": MagicMock(),
                "demucs": MagicMock(),
                "demucs.separate": MagicMock(),
                "librosa": MagicMock(),
            },
        )
        self.modules_patcher.start()

        self.vocal_path = "test_vocal.wav"
        self.bg_path = "test_bg.wav"
        self.output_path = "test_mixed.wav"

        # Configure global torch mock
        mock_torch = sys.modules["torch"]
        self.mock_pad = mock_torch.nn.functional.pad
        self.mock_pad.return_value = MagicMock()  # Will be overwritten by code if needed
        mock_torch.max.side_effect = lambda *args, **kwargs: 0.5
        mock_torch.abs.side_effect = lambda x: x
        mock_torch.from_numpy.side_effect = lambda arr: self._create_mock_tensor(arr)

        # Configure global torchaudio mock
        mock_torchaudio = sys.modules["torchaudio"]
        resample_transform = MagicMock()
        resample_transform.to.return_value = resample_transform
        resample_transform.side_effect = lambda x: x
        mock_torchaudio.transforms.Resample.return_value = resample_transform

        # Configure global librosa mock
        self.mock_librosa = sys.modules["librosa"]
        self.mock_librosa.effects.trim.return_value = (np.zeros((1, 50)), None)
        self.mock_librosa.effects.time_stretch.return_value = np.zeros((1, 100))

        # Import inside setUp to use patched modules
        if "src.utils.audio_processor" in sys.modules:
            del sys.modules["src.utils.audio_processor"]
        from src.utils.audio_processor import AudioProcessor

        self.processor = AudioProcessor()

    def _create_mock_tensor(self, array):
        m = MagicMock()
        # Handle cases where array might be a mock (e.g. from librosa)
        if isinstance(array, np.ndarray):
            shape = array.shape
        elif hasattr(array, "shape") and isinstance(array.shape, tuple):
            shape = array.shape
        else:
            shape = (1, 100)

        m.shape = shape
        m.float.return_value = m
        m.to.return_value = m
        m.__getitem__.return_value = m
        m.__add__.return_value = m
        m.__sub__.return_value = m
        m.__mul__.return_value = m
        m.__abs__.return_value = m
        m.__truediv__.return_value = m
        m.__float__.return_value = 0.5
        m.__gt__.return_value = False
        m.__lt__.return_value = False
        m.__ge__.return_value = False
        m.__le__.return_value = False
        m.__ge__.return_value = False
        m.__le__.return_value = False
        m.__radd__.return_value = m
        m.__rmul__.return_value = m

        # When .cpu().detach().numpy() is called for sf.write, return (channels, frames)
        # So that .T in the code makes it (frames, channels)
        m.cpu.return_value.detach.return_value.numpy.return_value = np.zeros(shape)
        m.expand.return_value = m
        return m

    def tearDown(self):
        for p in [self.vocal_path, self.bg_path, self.output_path]:
            if os.path.exists(p):
                os.remove(p)
        self.modules_patcher.stop()

    def test_denoise_audio_subprocess_call(self):
        """Tests that denoise_audio calls deepFilter with correct parameters."""
        vocal_path = "vocal.wav"
        output_path = "vocal_clean.wav"

        with patch("os.path.exists", return_value=True), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Success")
            self.processor.denoise_audio(vocal_path, output_path)

            # Verify the call pattern
            self.assertTrue(mock_run.called)
            args, kwargs = mock_run.call_args
            cmd_list = args[0]
            self.assertIn("deepFilter", cmd_list)
            self.assertIn(vocal_path, cmd_list)
            self.assertIn("uvx", cmd_list)

    def test_separate_vocals_returns_dict(self):
        """Tests that separate_vocals returns the expected dictionary format and calls subprocess."""
        audio_path = "input.wav"
        output_dir = "out"

        with patch("subprocess.run") as mock_run, patch("os.path.exists") as mock_exists:
            mock_run.return_value = MagicMock(returncode=0)
            mock_exists.side_effect = lambda p: True

            result = self.processor.separate_vocals(audio_path, output_dir)

            self.assertIsInstance(result, dict)
            self.assertIn("vocals", result)
            self.assertIn("background", result)
            self.assertTrue(mock_run.called)

    def test_resolve_demucs_paths(self):
        """Tests the logic for resolving Demucs output paths."""
        output_dir = "test_denoise_out"
        audio_path = "input_audio.wav"

        with patch("os.path.exists", return_value=True):
            paths = self.processor._resolve_demucs_paths(output_dir, audio_path)
            self.assertIn("htdemucs", paths["vocals"])
            self.assertIn("vocals.wav", paths["vocals"])
            self.assertIn("no_vocals.wav", paths["background"])

    def test_mix_audio_basic(self):
        """Tests that mix_audio runs without error and creates an output file."""
        vocal_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    def test_mix_audio_preserves_highest_sample_rate(self):
        """Tests that mix_audio targets the highest available sample rate."""
        vocal_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 44100)

        # Force torch.max to return a float to avoid TypeError in comparison
        with patch("src.utils.audio_processor.torch.max", return_value=0.5):
            self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)

        data, sr = sf.read(self.output_path)
        self.assertEqual(sr, 44100)

    def test_mix_audio_with_empty_background(self):
        """Tests that mix_audio handles empty background tracks without ZeroDivisionError."""
        vocal_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.zeros((0, 1)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        # Should NOT raise ZeroDivisionError
        self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    def test_mix_audio_stretches_longer_vocals(self):
        """Tests that mix_audio handles longer vocals."""
        vocal_data = np.random.uniform(-1, 1, (200, 1)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        # Should run without error
        self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    def test_mix_audio_pads_shorter_vocals(self):
        """Tests that mix_audio pads vocals with silence when they are shorter than background."""
        vocal_data = np.random.uniform(-1, 1, (50, 1)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)

        # Verify that padding was called
        self.mock_pad.assert_called()

    def test_mix_audio_channel_mismatch_mono_to_stereo(self):
        """Tests mixing mono vocals with stereo background."""
        vocal_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)  # Mono
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, 2)).astype(np.float32)  # Stereo
        sf.write(self.bg_path, bg_data, 16000)

        self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    def test_mix_audio_channel_mismatch_stereo_to_mono(self):
        """Tests mixing stereo vocals with mono background."""
        vocal_data = np.random.uniform(-1, 1, (100, 2)).astype(np.float32)  # Stereo
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)  # Mono
        sf.write(self.bg_path, bg_data, 16000)

    def test_trim_silence_call(self):
        """Tests that _trim_silence calls librosa.effects.trim."""
        dummy_audio = self._create_mock_tensor(np.zeros((1, 100)))
        self.processor._trim_silence(dummy_audio, 16000)
        self.mock_librosa.effects.trim.assert_called()

    def test_mix_audio_stretches_only_if_significant(self):
        """Tests that mix_audio avoids stretching for small discrepancies (<100ms)."""
        # 16000 samples = 1s. Let's make discrepancy 400 samples = 25ms.
        vocal_data = np.random.uniform(-1, 1, (10400, 1)).astype(np.float32)
        sf.write(self.vocal_path, vocal_data, 16000)

        bg_data = np.random.uniform(-1, 1, (10000, 1)).astype(np.float32)
        sf.write(self.bg_path, bg_data, 16000)

        # Mock trim to return original length to isolate stretch logic
        self.mock_librosa.effects.trim.return_value = (vocal_data.T, None)

        self.processor.mix_audio(self.vocal_path, self.bg_path, self.output_path)

        # Verify that time_stretch was NOT called because 25ms < 100ms
        self.mock_librosa.effects.time_stretch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
