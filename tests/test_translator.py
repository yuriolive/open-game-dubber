import unittest
from unittest.mock import MagicMock, patch

from src.models.translator import OllamaTranslator


class TestOllamaTranslator(unittest.TestCase):
    def setUp(self):
        self.translator = OllamaTranslator()

    @patch("requests.post")
    def test_translate_uses_delimiters(self, mock_post):
        # Mock response from requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hola mundo"}
        mock_post.return_value = mock_response

        text = 'Hello "world"'
        result = self.translator.translate(text, "Spanish")

        self.assertEqual(result, "Hola mundo")

        # Verify the prompt contains our unique delimiters
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        prompt = payload["prompt"]
        self.assertIn("###", prompt)
        self.assertIn(text, prompt)
        self.assertIn("Spanish", prompt)


if __name__ == "__main__":
    unittest.main()
