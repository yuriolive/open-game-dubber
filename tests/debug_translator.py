import logging

from src.models.translator import OllamaTranslator

logging.basicConfig(level=logging.INFO)
translator = OllamaTranslator()
text = "Can we speak a moment?"
target_lang = "Portuguese"
result = translator.translate(text, target_lang)

print(f"\nOriginal: {text}")
print(f"Target Lang: {target_lang}")
print(f"Translated: {result}")
