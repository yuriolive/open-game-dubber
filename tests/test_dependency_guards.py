import subprocess
import unittest


class TestDependencyGuards(unittest.TestCase):
    def test_audio_processor_imports_without_torch(self):
        """
        Verifies that AudioProcessor can be imported even if torch/torchaudio are missing.
        This runs in a separate subprocess to ensure a clean sys.modules state.
        """
        script = """
import sys
# Simulate missing dependencies
sys.modules['torch'] = None
sys.modules['torchaudio'] = None

try:
    import src.utils.audio_processor as ap_module
    from src.utils.audio_processor import AudioProcessor
    print("SUCCESS: Imported AudioProcessor")
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)

# Verify torch is indeed None in the module
if ap_module.torch is not None:
    print("FAILURE: ap_module.torch should be None")
    sys.exit(1)
"""
        # Run the script using uv run to ensure dependencies like soundfile are available
        result = subprocess.run(
            ["uv", "run", "python", "-c", script],
            capture_output=True,
            text=True,
            cwd=".",  # Assuming we run from project root
        )

        if result.returncode != 0:
            self.fail(f"Guard test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        self.assertIn("SUCCESS: Imported AudioProcessor", result.stdout)


if __name__ == "__main__":
    unittest.main()
