import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.getcwd())

from src.models.tts import TTSWrapper  # noqa: E402


def test_tts():
    tts = TTSWrapper()
    print("Testing 'portuguese'...")
    try:
        # Mocking generate_voice_clone to just print args if we could, but we can't
        # easily mock the internal library here without more setup.
        # So we will rely on the error message.
        # We'll pass a dummy file for ref_audio if needed, or just see if validation happens first.

        # Create a dummy ref audio file
        with open("dummy.wav", "wb") as f:
            f.write(b"RIFF....WAVEfmt ....data....")

        tts.generate_dub(
            text="Olá mundo",
            ref_audio_path="dummy.wav",
            output_path="output_debug.wav",
            language="portuguese",
            instruct="Brazilian Portuguese accent",
        )
    except Exception as e:
        logger.error(f"Failed with 'portuguese': {e}")

    print("\nTesting 'Brazilian Portuguese'...")
    try:
        tts.generate_dub(
            text="Olá mundo",
            ref_audio_path="dummy.wav",
            output_path="output_debug_fail.wav",
            language="Brazilian Portuguese",
            instruct="Brazilian Portuguese accent",
        )
    except Exception as e:
        logger.error(f"Failed with 'Brazilian Portuguese': {e}")

    if os.path.exists("dummy.wav"):
        os.remove("dummy.wav")


if __name__ == "__main__":
    test_tts()
