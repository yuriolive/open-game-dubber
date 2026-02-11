import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages the state of the batch processing job to allow resumes.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """
        Loads state from manifest.json if it exists.
        """
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
                return {}
        return {}

    def _save_state(self):
        """
        Saves current state to manifest.json.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def is_processed(self, file_path: str) -> bool:
        """
        Checks if a file is already marked as 'completed' in the manifest.
        """
        filename = os.path.basename(file_path)
        return self.state.get(filename, {}).get("status") == "completed"

    def mark_completed(self, file_path: str, metadata: Dict[str, Any] = None):
        """
        Marks a file as completed and saves metadata.
        """
        filename = os.path.basename(file_path)
        self.state[filename] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save_state()

    def mark_failed(self, file_path: str, error: str):
        """
        Marks a file as failed with an error message.
        """
        filename = os.path.basename(file_path)
        self.state[filename] = {
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        self._save_state()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = StateManager("test_output")
    # manager.mark_completed("sample1.wav", {"text": "Hello"})
    # print(manager.is_processed("sample1.wav"))
