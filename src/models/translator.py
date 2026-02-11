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
        self.pull_url = f"{base_url}/api/pull"

    def pull_model(self) -> bool:
        """
        Pull the required model via Ollama API with streaming progress.
        """
        logger.info(f"Pulling model '{self.model}' via Ollama API...")
        try:
            payload = {"model": self.model, "stream": True}
            # Large models can take substantial time to download; use a long timeout (1 hour) for the streaming pull
            response = requests.post(self.pull_url, json=payload, stream=True, timeout=3600)
            response.raise_for_status()

            import json

            for line in response.iter_lines():
                if line:
                    status = json.loads(line.decode("utf-8"))
                    if "status" in status:
                        # Log progress to keep connection alive and inform user
                        current = status.get("completed", 0)
                        total = status.get("total", 0)
                        if total > 0:
                            percent = (current / total) * 100
                            logger.info(f"Ollama Pull [{self.model}]: {status['status']} ({percent:.1f}%)")
                        else:
                            logger.info(f"Ollama Pull [{self.model}]: {status['status']}")

            logger.info(f"Successfully pulled model '{self.model}'")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model '{self.model}' via API: {e}")
            return False

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

            response = requests.post(self.api_url, json=payload, timeout=120)

            if response.status_code == 404:
                logger.error(f"Ollama model '{self.model}' not found. Please run 'ollama pull {self.model}'")
                return text

            response.raise_for_status()

            result = response.json()
            translated_text = result.get("response", "").strip().strip('"')

            if not translated_text:
                logger.warning("Ollama returned empty translation. Using original text.")
                return text

            logger.info(f"Translation successful: {translated_text[:50]}...")
            return translated_text
        except Exception as e:
            logger.error(f"Ollama translation failed: {e}")
            return text


if __name__ == "__main__":
    # Basic test (requires Ollama running)
    logging.basicConfig(level=logging.INFO)
    translator = OllamaTranslator()
    # print(translator.translate("Stay alert, the enemies are approaching!", "Portuguese"))
