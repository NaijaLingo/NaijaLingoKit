"""Developer SDK for NaijaLingo TTS (service client wrapper)."""

from typing import Optional, Dict, Any
import logging
import warnings

from .service_client import SpeechServiceTTSClient, SpeechServiceConfig


def suppress_all_logs():
	"""
	Suppress logging output from common libraries. This keeps notebooks/SDK quiet.
	"""
	warnings.filterwarnings("ignore")
	try:
		import transformers  # type: ignore
		transformers.logging.set_verbosity_error()
		transformers.logging.disable_progress_bar()
	except Exception:
		pass
	for name in ("nemo", "nemo_logger", "torch", "pytorch", "numba", "matplotlib", "PIL"):
		try:
			logging.getLogger(name).setLevel(logging.ERROR)
		except Exception:
			pass
	logging.getLogger().setLevel(logging.ERROR)


class NaijaLingoTTSClient:
	"""
	Thin convenience wrapper over SpeechServiceTTSClient for finetuned TTS.
	"""

	def __init__(self, base_url: Optional[str] = None, timeout: int = 180, suppress_logs: bool = True) -> None:
		if suppress_logs:
			suppress_all_logs()
		cfg = SpeechServiceConfig(base_url=base_url or SpeechServiceConfig().base_url, timeout=timeout)
		self.client = SpeechServiceTTSClient(cfg)

	def synthesize(
		self,
		text: str,
		*,
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
		Calls the speech service finetuned TTS and returns its JSON response.
		"""
		return self.client.synthesize_finetuned(
			text=text,
			model_name=model_name,
			speaker_id=speaker_id,
			device=device,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			top_p=top_p,
			repetition_penalty=repetition_penalty,
			output_filename=output_filename,
		)
