import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaTranslator:
    """
    Translates text using local Ollama instance.
    """

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def translate(self, text: str, target_lang: str, context: Optional[str] = None) -> str:
        """
        Translate text to target language with optional context.
        """
        if not text.strip():
            return ""

        prompt = f"""
        Translate the following game dialogue from English to {target_lang}.
        Maintain the character's tone, emotion, and any specific gaming terminology.
        DO NOT include any explanation or extra text, ONLY the translation.

        {f"Context: {context}" if context else ""}

        Text to translate:
        ###
        {text}
        ###
        Translation:
        """

        logger.info(f"Translating text: {text[:50]}...")

        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}

            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            translated_text = result.get("response", "").strip().strip('"')

            return translated_text
        except Exception as e:
            logger.error(f"Ollama translation failed: {e}")
            # Fallback or return original if translation fails?
            # For now return original to avoid breaking the pipeline, but log the error.
            return text


if __name__ == "__main__":
    # Basic test (requires Ollama running)
    logging.basicConfig(level=logging.INFO)
    translator = OllamaTranslator()
    # print(translator.translate("Stay alert, the enemies are approaching!", "Portuguese"))
