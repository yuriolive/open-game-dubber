import logging
import os
import shutil
import tempfile

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

    def __init__(self, output_dir: str, target_lang: str = "Portuguese", debug: bool = False):
        self.output_dir = output_dir
        self.target_lang = target_lang
        self.debug = debug

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
            # Use a unique temporary directory for this file processing task
            with tempfile.TemporaryDirectory(dir=self.output_dir, prefix="dub_tmp_") as temp_dir:
                # 0. Denoise Input (Pre-processing)
                # Denoising the full input ensures both vocals and background stems are clean.
                denoised_input_path = os.path.join(temp_dir, "input_clean.wav")
                logger.info(f"Pre-denoising full input: {filename}")
                audio_path = self.processor.denoise_audio(audio_path, denoised_input_path) or audio_path

                # 1. Separate Vocals
                vocal_root = os.path.join(temp_dir, "vocals")
                separated = self.processor.separate_vocals(audio_path, vocal_root)
                if not separated:
                    raise Exception("Vocal separation failed")

                vocal_path = separated["vocals"]
                bg_path = separated["background"]

                # 2. Transcribe
                segments = self.stt.transcribe(vocal_path)
                if not segments:
                    raise Exception("Transcription returned no segments")

                original_text = " ".join([seg["text"] for seg in segments])
                logger.info(f"Transcription: {original_text}")

                # 3. Translate with duration awareness
                vocal_duration = self.processor.get_audio_duration(vocal_path)
                logger.info(f"Target duration: {vocal_duration:.2f}s")

                translation_result = self.translator.translate(
                    original_text, self.target_lang, target_duration=vocal_duration
                )
                translated_text = translation_result["text"]
                tts_instruction = translation_result.get("tts_instruction", "")
                target_language_response = translation_result.get("target_language", "")
                logger.info(f"Translation: {translated_text}")
                if tts_instruction:
                    logger.info(f"TTS instruction: {tts_instruction}")

                # 4. Synthesize Dub
                dub_output_path = os.path.join(temp_dir, "dubs", filename)
                os.makedirs(os.path.dirname(dub_output_path), exist_ok=True)

                # Normalize language to base language name for TTS compatibility
                if target_language_response:
                    base_language = target_language_response
                    logger.info(f"Using LLM-provided base language: '{base_language}'")
                else:
                    # Fallback to simple split if LLM fails to return target_language
                    base_language = self.target_lang.split()[-1].lower()
                    logger.warning(f"LLM did not return target_language. Falling back to heuristic: '{base_language}'")

                synthesized_path = self.tts.generate_dub(
                    translated_text,
                    vocal_path,
                    dub_output_path,
                    language=base_language,
                    ref_text=original_text,
                    instruct=tts_instruction,
                )
                if not synthesized_path:
                    raise Exception("TTS synthesis failed")

                # 5. Mix with background
                final_output_path = os.path.join(self.output_dir, filename)

                if bg_path and os.path.exists(bg_path):
                    self.processor.mix_audio(synthesized_path, bg_path, final_output_path)
                else:
                    shutil.copy(synthesized_path, final_output_path)

                # 5.5 Save intermediate files if debug is enabled
                if self.debug:
                    debug_dir = os.path.join(self.output_dir, "debug", filename)
                    os.makedirs(debug_dir, exist_ok=True)
                    # Copy everything from the root of temp_dir
                    for item in os.listdir(temp_dir):
                        s = os.path.join(temp_dir, item)
                        d = os.path.join(debug_dir, item)
                        if os.path.isdir(s):
                            shutil.copytree(s, d, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s, d)
                    logger.info(f"Intermediate files saved to: {debug_dir}")

                # 6. Mark success
                self.state.mark_completed(
                    audio_path, {"original_text": original_text, "translated_text": translated_text}
                )
                logger.info(f"Successfully dubbed: {filename}")
                return True

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            self.state.mark_failed(audio_path, str(e))
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # pipeline = DubbingPipeline("output", "Portuguese")
    # pipeline.process_file("samples/sample1.wav")
