from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import requests

from .logging_utils import get_logger


@dataclass
class SpeechServiceConfig:
    base_url: str = os.environ.get("NAIJALINGO_SPEECH_SERVICE_URL", "http://localhost:8000")
    timeout: int = 120


class SpeechServiceClient:
    """
    Minimal client SDK to call NaijaLingo Speech Service STT endpoint.
    """

    def __init__(self, config: Optional[SpeechServiceConfig] = None) -> None:
        self.config = config or SpeechServiceConfig()
        self._log = get_logger(self.__class__.__name__)

    def transcribe(
        self,
        audio_path: str,
        *,
        backend: str = "whisper",
        device: str = "cpu",
        task: str = "transcribe",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call POST /api/stt with multipart form.

        Args:
            audio_path: path to audio file (wav recommended)
            backend: 'whisper' | 'facebook' | 'finetuned'
            device: 'cpu' | 'gpu'
            task: 'transcribe' | 'translate'
            language: optional language code (required by some backends)

        Returns:
            JSON dict from the service.
        """
        url = f"{self.config.base_url.rstrip('/')}/api/stt"
        self._log.info("POST %s backend=%s device=%s task=%s language=%s", url, backend, device, task, language)
        data = {
            "backend": backend,
            "device": device,
            "task": task,
        }
        if language is not None:
            data["language"] = language

        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path) or "audio.wav", f, "audio/wav")}
            resp = requests.post(url, data=data, files=files, timeout=self.config.timeout)

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Speech service error {resp.status_code}: {detail}")
        return resp.json()


