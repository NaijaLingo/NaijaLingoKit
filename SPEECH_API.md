# NaijaLingo Speech Service API

Base URL:
```
https://<host>:<port>/api
```

---

## 1. TTS (Facebook MMS)
**POST** `/tts/open-source`
```json
{
  "text": "Bawo ni, kaabo si NaijaLingo!",
  "language": "yoruba",
  "device": "cpu",
  "output_filename": null
}
```
**Response**
```json
{
  "file_path": "tts_result/tts_yoruba.wav",
  "language": "yoruba",
  "device": "cpu"
}
```

---

## 2. TTS (Gemini Voice)
**POST** `/tts/germini`
```json
{
  "text": "Life na better gift wey God dash us.",
  "language": "yoruba",
  "style_prompt": "Warm Yoruba radio host vibe",
  "voice_name": "Zephyr",
  "output_filename": "yoruba_gemini.wav"
}
```
**Response**
```json
{
  "file_path": "tts_result/yoruba_gemini.wav",
  "language": "yoruba",
  "voice_name": "Zephyr"
}
```

---

## 3. TTS (NaijaLingo Finetuned)
**POST** `/tts/naijalingo-finetuned`
```json
{
  "text": "Nigeria bụ obodo nwere ọtụtụ mmadụ.",
  "model_name": "chukypedro/igbo-tts-400m-0.3-v3-pt",
  "speaker_id": "Chioma",
  "device": "cpu",
  "max_new_tokens": 1200,
  "temperature": 1.0,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "output_filename": null
}
```
**Response**
```json
{
  "file_path": "tts_result/tts_naijalingo_Chioma.wav",
  "model_name": "chukypedro/igbo-tts-400m-0.3-v3-pt",
  "speaker_id": "Chioma"
}
```

### Streaming Variant
**POST** `/tts/naijalingo-finetuned/stream`
- Same body as above.
- Server-Sent Events: `start`, `chunk`, `error`, `done`. Each `chunk` contains a base64 WAV snippet plus metadata.

---

## 4. Speech-to-Text
**POST** `/stt`
- Multipart form fields:
  - `file`: audio file (WAV recommended)
  - `device`: `cpu` or `gpu`
  - `task`: `transcribe` or `translate`
  - `language`: optional (`yor`, `ig`, `ha`, `en`, `pcm`)
  - `backend`: `whisper`, `facebook`, or `finetuned`

```bash
curl -X POST http://localhost:8000/api/stt \
  -F "file=@sample.wav" \
  -F "device=cpu" \
  -F "task=transcribe" \
  -F "language=yor" \
  -F "backend=whisper"
```
**Response**
```json
{
  "text": "Bawo ni, kaabo si NaijaLingo!",
  "language": "yor",
  "language_probability": 0.97,
  "duration": 4.2
}
```

---

## 5. Helper Endpoints
| Endpoint | Description | Sample |
|----------|-------------|--------|
| `GET /tts/naijalingo-finetuned/voices` | Returns available speaker IDs. | `{ "speakers": ["Aliyu", "Chioma", ...] }` |
| `GET /tts/naijalingo-finetuned/models` | Maps language labels to HF repos. | `{ "igbo": "chukypedro/igbo-tts-400m-0.3-v3-pt", ... }` |
| `GET /tts/download/{filename}` | Downloads a generated WAV from `tts_result/`. | Binary WAV |

---

## 6. Environment Checklist
- `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) for private/gated finetuned models.
- `GEMINI_API_KEY` for Gemini text-to-speech.
- Generated audio is written to `tts_result/`; use download endpoint when exposing files externally.

