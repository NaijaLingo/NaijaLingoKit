from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import requests

from naijalingo_asr.logging_utils import get_logger  # reuse common logger


@dataclass
class SpeechServiceConfig:
    base_url: str = os.environ.get("NAIJALINGO_SPEECH_SERVICE_URL", "http://localhost:8000")
    timeout: int = 180


class SpeechServiceTTSClient:
    """
    Minimal SDK client for NaijaLingo Speech Service TTS endpoints.
    """

    def __init__(self, config: Optional[SpeechServiceConfig] = None) -> None:
        self.config = config or SpeechServiceConfig()
        self._log = get_logger(self.__class__.__name__)

    def synthesize_finetuned(
        self,
        *,
        text: str,
        model_name: Optional[str] = "chukypedro/hausa-tts-400m-0.3-pt",
        speaker_id: Optional[str] = None,
        device: str = "cpu",
        max_new_tokens: int = 1200,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call POST /api/tts/naijalingo-finetuned. Returns JSON with 'file_path', 'model_name', 'speaker_id'.
        """
        url = f"{self.config.base_url.rstrip('/')}/api/tts/naijalingo-finetuned"
        payload = {
            "text": text,
            "model_name": model_name,
            "speaker_id": speaker_id,
            "device": device,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "output_filename": output_filename,
        }
        self._log.info("POST %s model=%s device=%s speaker=%s", url, model_name, device, speaker_id)
        resp = requests.post(url, json=payload, timeout=self.config.timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"TTS service error {resp.status_code}: {detail}")
        return resp.json()

    def get_finetuned_voices(self) -> Dict[str, Any]:
        """
        Get default speaker list for finetuned TTS.
        """
        url = f"{self.config.base_url.rstrip('/')}/api/tts/naijalingo-finetuned/voices"
        self._log.info("GET %s", url)
        resp = requests.get(url, timeout=self.config.timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"TTS service error {resp.status_code}: {detail}")
        return resp.json()

    def get_finetuned_models(self) -> Dict[str, Any]:
        """
        Get default model mapping per language for finetuned TTS.
        """
        url = f"{self.config.base_url.rstrip('/')}/api/tts/naijalingo-finetuned/models"
        self._log.info("GET %s", url)
        resp = requests.get(url, timeout=self.config.timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"TTS service error {resp.status_code}: {detail}")
        return resp.json()

    def synthesize_open_source(
        self,
        *,
        text: str,
        language: str,
        device: str = "cpu",
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call POST /api/tts/open-source (MMS TTS).
        language: 'yoruba' | 'hausa' | 'pidgin' (as per service)
        """
        url = f"{self.config.base_url.rstrip('/')}/api/tts/open-source"
        payload = {
            "text": text,
            "language": language,
            "device": device,
            "output_filename": output_filename,
        }
        self._log.info("POST %s language=%s device=%s", url, language, device)
        resp = requests.post(url, json=payload, timeout=self.config.timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"TTS service error {resp.status_code}: {detail}")
        return resp.json()

    def synthesize_gemini(
        self,
        *,
        text: str,
        language: str,
        voice_name: str = "Zephyr",
        style_prompt: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call POST /api/tts/germini (Gemini TTS).
        Requires GEMINI_API_KEY on the service.
        """
        url = f"{self.config.base_url.rstrip('/')}/api/tts/germini"
        payload = {
            "text": text,
            "language": language,
            "voice_name": voice_name,
            "style_prompt": style_prompt,
            "output_filename": output_filename,
        }
        self._log.info("POST %s language=%s voice=%s", url, language, voice_name)
        resp = requests.post(url, json=payload, timeout=self.config.timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"TTS service error {resp.status_code}: {detail}")
        return resp.json()


