from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .registry import get_repo_for_language
from .engine import WhisperEngine, EngineConfig
from .audio import load_audio_mono_16k
from .logging_utils import configure_logging, get_logger, suppress_external_warnings


@dataclass
class TranscribeOptions:
    language: str
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    temperature: float = 0.0
    initial_prompt: Optional[str] = None
    # passthrough for future faster-whisper args
    extra: Optional[Dict[str, Any]] = None


class ASRTranscriber:
    def __init__(self, language: str, device: str = "auto", compute_type: str = "auto") -> None:
        repo = get_repo_for_language(language)
        print(f"repo: {repo}")
        print("-----------------------------------------------------------------------")
        self.language = language
        self.engine = WhisperEngine(repo, EngineConfig(device=device, compute_type=compute_type))
        self._log = get_logger(self.__class__.__name__)

    def transcribe(self, audio_source, **kwargs) -> str:
        beam_size = kwargs.pop("beam_size", 5)
        vad_filter = kwargs.pop("vad_filter", True)
        temperature = kwargs.pop("temperature", 0.0)
        initial_prompt = kwargs.pop("initial_prompt", None)

        # Accept path-like or numpy array; always feed mono 16k float32 to engine
        audio_array = load_audio_mono_16k(audio_source) if isinstance(audio_source, str) else audio_source
        self._log.debug("Transcribe called: language=%s input_type=%s", self.language, type(audio_source).__name__)

        # Business rule: decode Igbo with English tokenizer (library does not accept 'ig')
        decode_language = "en" if self.language == "ig" else self.language
        if decode_language != self.language:
            self._log.info("Mapping requested language '%s' to '%s' for decoding", self.language, decode_language)
        segments, _ = self.engine.transcribe(
            audio_array,
            language=decode_language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            temperature=temperature,
            initial_prompt=initial_prompt,
            **kwargs,
        )
        return " ".join(seg.text for seg in segments)


def transcribe(audio, language: str, **kwargs) -> str:
    """Convenience function for one-shot transcription.

    Example:
        text = transcribe("/path/file.wav", language="yo", device="cuda", compute_type="float16")
        # or, using a numpy array preloaded at 16k mono:
        # text = transcribe(audio_array, language="yo")
    """
    # optional: allow caller to set log level via kwarg or env
    if "log_level" in kwargs:
        configure_logging(kwargs.pop("log_level"))
    # Quiet noisy upstream warnings unless caller overrides env
    suppress_external_warnings()
    transcriber = ASRTranscriber(
        language=language,
        device=kwargs.pop("device", "auto"),
        compute_type=kwargs.pop("compute_type", "auto"),
    )
    return transcriber.transcribe(audio, **kwargs)
