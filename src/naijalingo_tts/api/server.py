from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from enum import Enum
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
	from naijalingo_tts import NaijaTTS
except Exception as exc:
	raise RuntimeError("Failed to import naijalingo_tts. Ensure the src path is on sys.path.") from exc


# ------------------------------------------------------------------------------
# Configuration and Enums
# ------------------------------------------------------------------------------

class LanguageEnum(str, Enum):
	hausa = "hausa"
	igbo = "igbo"
	yoruba = "yoruba"
	pidgin = "pidgin"


class VoiceEnum(str, Enum):

	######HAUSA VOICES#########
	Aliyu = "Aliyu"
	Talatu = "Talatu"
	Bashir = "Bashir"
	Asabe = "Asabe"
	Adamu = "Adamu"
	Jumai = "Jumai"
	Yakub = "Yakub"
	Aisha = "A’isha"
	Rahmah = "Rahmah"
	Kyauta = "Kyauta"
	Aminah = "Aminah"
	Yusuf = "Yusuf"
	Hauwa = "Hauwa"

	######YORUBA VOICES#########
	Oyewale = "Oyewale"
	Kehinde = "Kehinde"
	Oladipo = "Oladipo"
	Bolanle = "Bolanle"
	Adedamola = "Adedamola"
	Abiola = "Abiola"
	Ifeoluwa = "Ifeoluwa"
	Abiodun = "Abiodun"
	Taiwo = "Taiwo"
	Ayomide = "Ayomide"
	Adebowale = "Adebowale"
	Bankole = "Bankole"
	Monife = "Monife"
	Ademola = "Ademola"
	Femi = "Femi"
	Adunni = "Adunni"
	Damilola = "Damilola"
	Lolade = "Lolade"

	######IGBO VOICES#########
	Adaeze = "Adaeze"
	Amarachi = "Amarachi"
	Uchechi = "Uchechi"
	Obinna = "Obinna"
	Adaugo = "Adaugo"
	Ifeoma = "Ifeoma"

	######PIDGIN VOICES#########
	Joy = "Joy"
	Samuel = "Samuel"
	Josphine = "Josphine"
	Kate = "Kate"
	Henry = "Henry"
	Simon = "Simon"
	Anita = "Anita"
	Ibrahim = "Ibrahim"
	Musa = "Musa"
	Sandra = "Sandra"



class ModelEnum(str, Enum):
	# For now all default to the same model; can be customized per language later
	hausa = "chukypedro/hausa-tts-400m-0.3-pt"
	igbo = "chukypedro/igbo-tts-400m-0.3-pt"
	yoruba = "chukypedro/yoruba-tts-400m-0.3-pt"
	pidgin = "chukypedro/pidgin-tts-400m-0.3-pt"


DEFAULT_SPEAKERS: List[str] = [
		"Aliyu", "Talatu", "Bashir", "Asabe", "Adamu", "Jumai", "Yakub",
		"A’isha", "Rahmah", "Kyauta", "Aminah", "Yusuf", "Hauwa",
		"Oyewale", "Kehinde", "Oladipo", "Bolanle", "Adedamola", "Abiola", "Ifeoluwa", "Abiodun", "Taiwo", "Ayomide", "Adebowale", "Bankole", "Monife", "Ademola", "Femi", "Adunni", "Damilola", "Lolade",
		"Adaeze", "Amarachi", "Uchechi", "Obinna", "Adaugo", "Ifeoma",
		"Joy", "Samuel", "Josphine", "Kate", "Henry", "Simon", "Anita", "Ibrahim", "Musa", "Sandra",
]

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def generate_filename(prefix: str = "tts", ext: str = "wav") -> str:
	return f"{prefix}_{uuid.uuid4().hex}.{ext}"


# ------------------------------------------------------------------------------
# Lazy model registry (one model instance per language)
# ------------------------------------------------------------------------------

_model_lock = threading.Lock()
_models_by_language: Dict[LanguageEnum, NaijaTTS] = {}


def get_or_create_tts(language: LanguageEnum) -> NaijaTTS:
	model_name = getattr(ModelEnum, language.value).value
	with _model_lock:
		if language in _models_by_language:
			return _models_by_language[language]
		# Instantiate with sensible defaults; logs are suppressed except our explicit prints
		print(f"Loading NaijaTTS model for language={language.value}: {model_name}")
		tts = NaijaTTS(
			model_name=model_name,
			default_speakers=DEFAULT_SPEAKERS,
			temperature=0.95,
			top_p=0.92,
			max_new_tokens=1500,
			repetition_penalty=1.1,
			device_map="auto",
			suppress_logs=True,
			show_info=False,
		)
		_models_by_language[language] = tts
		return tts


# ------------------------------------------------------------------------------
# Helpers: text chunking and audio IO
# ------------------------------------------------------------------------------

def _split_text_for_tts(text: str, max_chars: int = 220, hard_max_chars: int = 400) -> List[str]:
	import re

	text = text.strip()
	if not text:
		return []
	sentences = re.split(r"(?<=[.!?])\s+", text)
	chunks: List[str] = []
	buf = ""
	for s in sentences:
		s = s.strip()
		if not s:
			continue
		if len(s) > hard_max_chars:
			for i in range(0, len(s), hard_max_chars):
				part = s[i : i + hard_max_chars].strip()
				if part:
					if buf:
						chunks.append(buf.strip())
						buf = ""
					chunks.append(part)
			continue
		if len(buf) + (1 if buf else 0) + len(s) <= max_chars:
			buf = f"{buf} {s}".strip() if buf else s
		else:
			if buf:
				chunks.append(buf.strip())
			buf = s
	if buf:
		chunks.append(buf.strip())
	return chunks


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
	try:
		import soundfile as sf
	except Exception as exc:
		raise HTTPException(status_code=500, detail="soundfile not installed; cannot write WAV") from exc
	mem = BytesIO()
	sf.write(mem, audio.astype(np.float32), samplerate=sample_rate, format="WAV", subtype="PCM_16")
	return mem.getvalue()


def _word_count(text: str) -> int:
	return len([w for w in text.strip().split() if w])


# ------------------------------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------------------------------

class TTSRequestModel(BaseModel):
	text: str = Field(..., description="Input text to synthesize")
	language: LanguageEnum = Field(LanguageEnum.hausa, description="Language of the text")
	speaker_id: Optional[VoiceEnum] = Field(None, description="Optional speaker/voice id")


class TTSResponseModel(BaseModel):
	download_url: str
	filename: str
	sample_rate: int


# ------------------------------------------------------------------------------
# FastAPI app and endpoints
# ------------------------------------------------------------------------------

app = FastAPI(title="NaijaLingo TTS API", version="0.1.0")


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.get("/download/{filename}")
def download_file(filename: str) -> FileResponse:
	target = MEDIA_DIR / filename
	if not target.exists():
		raise HTTPException(status_code=404, detail="File not found")
	return FileResponse(
		path=str(target),
		media_type="audio/wav",
		filename=filename,
		headers={"Content-Disposition": f'attachment; filename="{filename}"'},
	)


@app.post("/tts/synthesize", response_model=TTSResponseModel)
def tts_synthesize(body: TTSRequestModel, req: Request) -> dict:
	"""
	Text-to-speech for short text (<= 40 words).
	Returns a JSON with a downloadable URL to a WAV file.
	"""
	if _word_count(body.text) > 40:
		raise HTTPException(status_code=422, detail="Text exceeds 40-word limit for this endpoint.")
	if not body.text.strip():
		raise HTTPException(status_code=422, detail="Text cannot be empty.")

	tts = get_or_create_tts(body.language)
	speaker = body.speaker_id.value if body.speaker_id else None

	audio, _ = tts.generate(body.text, speaker_id=speaker)

	filename = generate_filename(prefix=f"tts_{body.language.value}", ext="wav")
	filepath = MEDIA_DIR / filename

	try:
		import soundfile as sf
		sf.write(str(filepath), audio.astype(np.float32), tts.sample_rate)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Failed to write audio: {exc}") from exc

	download_url = str(req.url_for("download_file", filename=filename))
	return {"download_url": download_url, "filename": filename, "sample_rate": tts.sample_rate}


@app.post("/tts/stream")
def tts_stream(body: TTSRequestModel, req: Request) -> StreamingResponse:
	"""
	Streaming TTS for long text (no word limit).
	Streams JSON Server-Sent Events (SSE) with base64 WAV chunks per segment.
	Also saves a full WAV on disk and sends final event with download_url.
	"""
	if not body.text.strip():
		raise HTTPException(status_code=422, detail="Text cannot be empty.")

	tts = get_or_create_tts(body.language)
	speaker = body.speaker_id.value if body.speaker_id else None
	filename = generate_filename(prefix=f"tts_stream_{body.language.value}", ext="wav")
	filepath = MEDIA_DIR / filename

	def event_stream() -> Iterable[bytes]:
		# Open file for append while streaming
		try:
			import soundfile as sf
			sf_handle = sf.SoundFile(str(filepath), mode="w", samplerate=tts.sample_rate, channels=1, subtype="PCM_16")
		except Exception as exc:
			yield f"event: error\ndata: {json.dumps({'error': f'Cannot open output file: {exc}'})}\n\n".encode("utf-8")
			return

		start = time.time()
		chunks = _split_text_for_tts(body.text, max_chars=220, hard_max_chars=400)
		total = max(1, len(chunks))
		yield f"event: start\ndata: {json.dumps({'chunks': total, 'filename': filename})}\n\n".encode("utf-8")

		try:
			for idx, chunk in enumerate(chunks, 1):
				audio, _ = tts.generate(chunk, speaker_id=speaker)
				# write to disk
				sf_handle.write(audio.astype(np.float32))
				# encode a small WAV for this chunk
				wav_bytes = _audio_to_wav_bytes(audio, tts.sample_rate)
				b64 = base64.b64encode(wav_bytes).decode("ascii")
				payload = {"index": idx, "total": total, "text": chunk, "audio_wav_b64": b64}
				yield f"event: chunk\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
		except Exception as exc:
			yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")
		finally:
			try:
				sf_handle.close()
			except Exception:
				pass
		elapsed = round(time.time() - start, 3)
		download_url = str(req.url_for("download_file", filename=filename))
		yield f"event: done\ndata: {json.dumps({'elapsed_sec': elapsed, 'download_url': download_url})}\n\n".encode("utf-8")

	return StreamingResponse(event_stream(), media_type="text/event-stream")


