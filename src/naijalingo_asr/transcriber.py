from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile

from .logging_utils import configure_logging, suppress_external_warnings, get_logger
from .service_client import SpeechServiceClient, SpeechServiceConfig


_LOG = get_logger(__name__)


class ASRTranscriber:
    """
    Thin wrapper that now proxies to the Speech Service.

    Kept for backward compatibility. Parameters map to service call:
      - language -> form field
      - device   -> 'cpu'|'gpu'
      - compute_type is ignored (handled server-side)
    """

    def __init__(self, language: str, device: str = "cpu", compute_type: str = "auto") -> None:
        self.language = language
        self.device = "gpu" if device in {"cuda", "gpu"} else "cpu"
        self.client = SpeechServiceClient()
        self._log = get_logger(self.__class__.__name__)
        self._log.info("ASRTranscriber proxying to Speech Service at %s", self.client.config.base_url)

    def transcribe(self, audio_source, **kwargs) -> str:
        backend = kwargs.pop("backend", "whisper")
        task = kwargs.pop("task", "transcribe")
        language = kwargs.pop("language", self.language)

        # Only support file path input in this thin SDK for now
        if not isinstance(audio_source, str):
            raise TypeError("ASRTranscriber now expects a file path string for audio_source.")
        path = Path(audio_source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        self._log.debug("Calling speech service: backend=%s device=%s task=%s language=%s", backend, self.device, task, language)
        result = self.client.transcribe(
            str(path),
            backend=backend,
            device=self.device,
            task=task,
            language=language,
        )
        return (result.get("text") or "").strip()


def transcribe(audio_path: str, language: str, **kwargs) -> str:
    """
    Convenience function for one-shot transcription via Speech Service.

    Example:
        text = transcribe(\"/path/file.wav\", language=\"yo\", device=\"gpu\", backend=\"finetuned\")
    """
    if "log_level" in kwargs:
        configure_logging(kwargs.pop("log_level"))
    suppress_external_warnings()

    device = kwargs.pop("device", "cpu")
    backend = kwargs.pop("backend", "whisper")
    task = kwargs.pop("task", "transcribe")

    transcriber = ASRTranscriber(language=language, device=device)
    return transcriber.transcribe(audio_path, backend=backend, task=task, language=language)
