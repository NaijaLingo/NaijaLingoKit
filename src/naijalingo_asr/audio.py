from __future__ import annotations
from typing import Union

import numpy as np
import librosa
from pathlib import Path
import os

TARGET_SAMPLE_RATE = 16000


def load_audio_mono_16k(source: Union[str, np.ndarray], target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load an audio source into a mono float32 numpy array at 16 kHz.

    - If `source` is a file path, uses librosa.load with sr=16k and mono=True.
    - If `source` is already a numpy array, assumes it is PCM float/float32 and
      returns a mono array. If 2D, averages the channels. Does not resample.
    """
    if isinstance(source, str):
        # Normalize and expand paths across OSes (Windows, Linux, macOS)
        normalized = str(Path(os.path.expanduser(source)))
        audio, _ = librosa.load(normalized, sr=target_sr, mono=True)
        return audio.astype(np.float32, copy=False)

    audio = np.asarray(source)
    if audio.ndim == 2:
        # Convert stereo/multi-channel to mono by averaging channels
        audio = audio.mean(axis=-1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio