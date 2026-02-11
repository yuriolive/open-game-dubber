import logging
import os
import shutil

from src.core.state_manager import StateManager
from src.models.stt import FasterWhisperTranscriber
from src.models.translator import OllamaTranslator
from src.models.tts import TTSWrapper
from src.utils.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class DubbingPipeline:
    """
    Coordinates the end-to-end dubbing flow.
    """

    def __init__(self, output_dir: str, target_lang: str = "Portuguese"):
        self.output_dir = output_dir
        self.target_lang = target_lang

        # Initialize components
        self.stt = FasterWhisperTranscriber()
        self.translator = OllamaTranslator()
        self.tts = TTSWrapper()
        self.processor = AudioProcessor()
        self.state = StateManager(output_dir)

    def process_file(self, audio_path: str) -> bool:
        """
        Processes a single audio file through the full pipeline.
        """
        filename = os.path.basename(audio_path)

        if self.state.is_processed(audio_path):
            logger.info(f"Skipping already processed file: {filename}")
            return True

        logger.info(f"--- Processing: {filename} ---")

        try:
            # 1. Separate Vocals
            vocal_dir = os.path.join(self.output_dir, "temp", "vocals")
            separated = self.processor.separate_vocals(audio_path, vocal_dir)
            if not separated:
                raise Exception("Vocal separation failed")

            vocal_path = separated["vocals"]
            bg_path = separated["background"]

            # 2. Transcribe
            # Note: We transcribe the cleaned vocals for better accuracy
            segments = self.stt.transcribe(vocal_path)
            if not segments:
                raise Exception("Transcription returned no segments")

            original_text = " ".join([seg["text"] for seg in segments])
            logger.info(f"Transcription: {original_text}")

            # 3. Translate
            translated_text = self.translator.translate(original_text, self.target_lang)
            logger.info(f"Translation: {translated_text}")

            # 4. Synthesize Dub
            dub_output_path = os.path.join(self.output_dir, "temp", "dubs", filename)
            os.makedirs(os.path.dirname(dub_output_path), exist_ok=True)

            # Using original vocals as reference for cloning
            synthesized_path = self.tts.generate_dub(
                translated_text, vocal_path, dub_output_path, language=self.target_lang
            )
            if not synthesized_path:
                raise Exception("TTS synthesis failed")

            # 5. Mix with background
            final_output_path = os.path.join(self.output_dir, filename)

            if bg_path and os.path.exists(bg_path):
                self.processor.mix_audio(synthesized_path, bg_path, final_output_path)
            else:
                # If no background, just copy the dub
                shutil.copy(synthesized_path, final_output_path)

            # 6. Mark success
            self.state.mark_completed(audio_path, {"original_text": original_text, "translated_text": translated_text})
            logger.info(f"Successfully dubbed: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            self.state.mark_failed(audio_path, str(e))
            return False
        finally:
            # Cleanup temporary files
            temp_dir = os.path.join(self.output_dir, "temp")
            if os.path.exists(temp_dir):
                try:
                    logger.info("Cleaning up temporary files...")
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup temporary files: {cleanup_err}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # pipeline = DubbingPipeline("output", "Portuguese")
    # pipeline.process_file("samples/sample1.wav")
