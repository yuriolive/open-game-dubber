import json
import unittest
from unittest.mock import MagicMock, patch

from src.models.translator import OllamaTranslator


class TestOllamaTranslator(unittest.TestCase):
    def setUp(self):
        self.translator = OllamaTranslator()

    @patch("requests.post")
    def test_translate_prompt_contains_constraints(self, mock_post):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({"text": "Olá mundo", "tts_instruction": "Happy", "target_language": "portuguese"})
        }
        mock_post.return_value = mock_response

        text = "Hello world"
        target_lang = "Portuguese"

        # Call translate
        self.translator.translate(text, target_lang)

        # Check if constraints are in the prompt
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        prompt = payload["prompt"]

        self.assertIn("CRITICAL DUBBING INSTRUCTION", prompt)
        self.assertIn("LENGTH CONSTRAINT", prompt)
        self.assertIn("syllable count", prompt)
        self.assertIn(f'Original Text: "{text}"', prompt)
        self.assertIn("concise phrasing", prompt)

    @patch("requests.post")
    def test_translate_handles_json_error(self, mock_post):
        # Mock bad JSON response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Raw text response"}
        mock_post.return_value = mock_response

        result = self.translator.translate("Hello", "Portuguese")

        self.assertEqual(result["text"], "Raw text response")
        self.assertEqual(result["tts_instruction"], "")


if __name__ == "__main__":
    unittest.main()
