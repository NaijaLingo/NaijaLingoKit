from __future__ import annotations
from typing import Union

import numpy as np
import librosa
from pathlib import Path
import os
import subprocess

from .logging_utils import get_logger

TARGET_SAMPLE_RATE = 16000
_LOG = get_logger(__name__)


def load_audio_mono_16k(source: Union[str, np.ndarray], target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load an audio source into a mono float32 numpy array at 16 kHz.

    - If `source` is a file path, uses librosa.load with sr=16k and mono=True.
    - If `source` is already a numpy array, assumes it is PCM float/float32 and
      returns a mono array. If 2D, averages the channels. Does not resample.
    """
    if isinstance(source, str):
        # Normalize and expand paths across OSes (Windows, Linux, macOS)
        normalized = str(Path(os.path.expanduser(source)))
        try:
            audio, _ = librosa.load(normalized, sr=target_sr, mono=True)
            return audio.astype(np.float32, copy=False)
        except Exception:
            _LOG.info("librosa backends unavailable for %s; falling back to ffmpeg decode", normalized)
            return _ffmpeg_decode_to_float32_mono(normalized, target_sr)

    audio = np.asarray(source)
    if audio.ndim == 2:
        # Convert stereo/multi-channel to mono by averaging channels
        audio = audio.mean(axis=-1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio


def _ffmpeg_decode_to_float32_mono(path: str, target_sr: int) -> np.ndarray:
    """Decode audio with ffmpeg into float32 mono at target_sr.

    Requires ffmpeg to be installed and on PATH.
    """
    ffmpeg_exe = None
    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        _LOG.debug("Using bundled ffmpeg at %s", ffmpeg_exe)
    except Exception:
        pass
    if ffmpeg_exe is None:
        from shutil import which

        ffmpeg_exe = which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is required to decode this file. Install it or add imageio-ffmpeg dependency.")
    cmd = [
        ffmpeg_exe,
        "-v",
        "error",
        "-nostdin",
        "-i",
        path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode file: {path}\n{proc.stderr.decode(errors='ignore')}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("ffmpeg produced no audio data. The input file may be unsupported or corrupt.")
    return audio