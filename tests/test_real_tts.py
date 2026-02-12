import numpy as np
import pytest

from src.models.tts import TTSWrapper


@pytest.fixture
def tts_wrapper(mocker):
    # Mocking the heavy model loading for unit tests
    mocker.patch("qwen_tts.Qwen3TTSModel.from_pretrained")
    return TTSWrapper()


def test_tts_wrapper_initialization(tts_wrapper):
    assert tts_wrapper.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert tts_wrapper._model is None


def test_generate_dub_basic(tts_wrapper, tmp_path, mocker):
    # Create a mock reference audio file
    ref_audio = tmp_path / "ref.wav"
    import soundfile as sf

    sf.write(str(ref_audio), np.zeros(16000), 16000)

    output_audio = tmp_path / "output.wav"

    # Mock model behavior
    mock_model = mocker.Mock()
    # model.generate_voice_clone returns (wavs, sr)
    mock_model.generate_voice_clone.return_value = ([np.zeros(16000)], 24000)
    tts_wrapper._model = mock_model

    # Mock soundfile.write
    mock_sf_write = mocker.patch("soundfile.write")

    result = tts_wrapper.generate_dub("Hello", str(ref_audio), str(output_audio), ref_text="Original Text")

    assert result == str(output_audio)
    mock_model.generate_voice_clone.assert_called_once_with(
        text="Hello",
        language="Portuguese",
        ref_audio=str(ref_audio),
        ref_text="Original Text",
        instruct=None,
        x_vector_only_mode=False,
    )
    mock_sf_write.assert_called_once()


def test_generate_dub_missing_ref(tts_wrapper, tmp_path):
    result = tts_wrapper.generate_dub("Hello", "non_existent.wav", str(tmp_path / "out.wav"))
    assert result is None
