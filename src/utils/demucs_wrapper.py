import sys

import soundfile as sf
import torch
import torchaudio


# Monkey-patch torchaudio.save to use soundfile
# This bypasses the TorchCodec requirement on Windows which often fails to load DLLs
def patched_save(filepath, src, sample_rate, channels_first=True, **kwargs):
    """
    Patched version of torchaudio.save using soundfile.
    """
    if torch.is_tensor(src):
        data = src.cpu().detach().numpy()
    else:
        data = src

    if channels_first:
        if data.ndim == 2:
            data = data.T

    sf.write(filepath, data, sample_rate)


# Apply the patch
torchaudio.save = patched_save

# Import and run demucs
try:
    from demucs.separate import main

    if __name__ == "__main__":
        main()
except ImportError:
    print("Error: Demucs not found in the current environment.")
    sys.exit(1)
