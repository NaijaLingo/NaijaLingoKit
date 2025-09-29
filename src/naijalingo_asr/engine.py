from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Any

from faster_whisper import WhisperModel
from .logging_utils import get_logger


@dataclass
class EngineConfig:
    device: str = "auto"  # "auto", "cpu", "cuda"
    compute_type: str = "auto"  # "auto", "float16", "int8", "int8_float16"


class WhisperEngine:
    def __init__(self, model_repo: str, config: Optional[EngineConfig] = None) -> None:
        self.model_repo = model_repo
        self.config = config or EngineConfig()
        self._model: Optional[WhisperModel] = None
        self._log = get_logger(self.__class__.__name__)

    def load(self) -> WhisperModel:
        if self._model is None:
            self._log.info("Loading model from %s (device=%s, compute_type=%s). First use may download files...",
                           self.model_repo, self.config.device, self.config.compute_type)
            self._model = WhisperModel(
                self.model_repo,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: Any,
        language: str,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        temperature: float = 0.0,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Iterable, object]:
        model = self.load()
        self._log.debug("Starting transcription: language=%s beam_size=%s vad_filter=%s temperature=%s", language, beam_size, vad_filter, temperature)
        segments, info = model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            temperature=temperature,
            initial_prompt=initial_prompt,
            **kwargs,
        )
        self._log.debug("Transcription complete: duration=%.2fs", getattr(info, "duration", -1))
        return segments, info
