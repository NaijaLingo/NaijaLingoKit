# NaijaLingo ASR SDK (Speech Service Client)

Thin Python SDK for calling the NaijaLingo Speech Service Speech‑to‑Text (STT) API.
This SDK proxies requests to your service (no local ASR inference).

## Requirements

- Python 3.10+
- Speech service running (default URL: `http://localhost:8000`)
- Install dependency:

```bash
pip install requests
```

Optional: configure the service base URL via env var:

```bash
export NAIJALINGO_SPEECH_SERVICE_URL="http://localhost:8000"
```

## Quick Start (one‑shot)

```python
from naijalingo_asr import transcribe

# Transcribe with your preferred backend
text = transcribe(
    "/path/to/audio.wav",
    language="ha",        # yo | ig | ha | en (aliases supported by service)
    device="gpu",         # "cpu" or "gpu"
    backend="finetuned",  # "whisper" | "facebook" | "finetuned"
    # task="transcribe",  # default
)
print(text)
```

## Using the high‑level class

```python
from naijalingo_asr import ASRTranscriber

stt = ASRTranscriber(language="yo", device="cpu")
text = stt.transcribe("/path/to/audio.wav", backend="whisper")
print(text)
```

## Using the low‑level client

```python
from naijalingo_asr import SpeechServiceClient, SpeechServiceConfig

client = SpeechServiceClient(SpeechServiceConfig(base_url="http://localhost:8000"))
result = client.transcribe(
    "/path/to/audio.wav",
    backend="facebook",   # or "whisper" / "finetuned"
    device="cpu",
    task="transcribe",
    language="ha",        # required for facebook
)
print(result["text"])
```

## Backends

- **whisper**: Faster‑Whisper models (language optional; service can detect).
- **facebook**: Facebook MMS CTC; requires language (`ig`, `yo`, `ha`, `pcm`).
- **finetuned**: Your CT2 Whisper repos; language maps internally:
  - `yoruba|yo` → `chukypedro/ct2_whisper_yo`, language=`yo`
  - `hausa|ha` → `chukypedro/ct2_whisper_ha`, language=`ha`
  - `igbo|ig|ibo` → `chukypedro/ct2_whisper_ig`, language=`en` (decode via English)
  - `nigeria-english|nigerian-english|ng-en|en` → `chukypedro/ct2_whisper_ng-en`, language=`en`

## Notes

- This SDK expects a file path for the audio argument.
- Errors from the service are raised as `RuntimeError` with the server response.
- Set `NAIJALINGO_SPEECH_SERVICE_URL` to point to a remote deployment if needed.


