# NaijaLingo TTS SDK (Service Client)

Developer SDK for testing all NaijaLingo TTS pipelines through the Speech Service:
- NaijaLingo finetuned TTS
- Open-source MMS TTS (Facebook)
- Gemini TTS (requires GEMINI_API_KEY on the service)

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (WSL2 supported) or CPU

Install packages:

```bash
pip install requests soundfile
```

Notes:
- The heavy TTS deps (transformers, NeMo, torch) are on the service side.
- The SDK just calls the service.

## Quick Start (Finetuned)

```python
from naijalingo_tts import SpeechServiceTTSClient, SpeechServiceConfig, suppress_all_logs

suppress_all_logs()
client = SpeechServiceTTSClient(SpeechServiceConfig(base_url="http://localhost:8000"))

resp = client.synthesize_finetuned(
    text="Sannu, ya ya kake?",
    model_name="chukypedro/hausa-tts-400m-0.3-pt",
    speaker_id="Aliyu",
    device="cpu",
    output_filename="tts_naijalingo_sample.wav",
)
print(resp["file_path"])
```

### Discover speakers and models

```python
voices = client.get_finetuned_voices()   # {"speakers": [...]}
models = client.get_finetuned_models()   # {"hausa": "...", "igbo": "...", ...}
print(voices)
print(models)
```

## Open-source MMS TTS

```python
resp = client.synthesize_open_source(
    text="Ẹ káàrọ̀ o, báwo ni ìlera rẹ̀?",
    language="yoruba",   # 'hausa' | 'yoruba' | 'pidgin'
    device="cpu",
    output_filename="tts_mms_yoruba.wav",
)
print(resp["file_path"])
```

## Gemini TTS

```python
resp = client.synthesize_gemini(
    text="How are you today?",
    language="pidgin",    # 'yoruba' | 'igbo' | 'hausa' | 'pidgin'
    voice_name="Zephyr",
    style_prompt=None,
    output_filename="tts_gemini_pidgin.wav",
)
print(resp["file_path"])
```

## API Summary (SDK)

- `SpeechServiceTTSClient(SpeechServiceConfig(base_url))`
  - `synthesize_finetuned(text, model_name, speaker_id, device, ...) -> {"file_path", ...}`
  - `get_finetuned_voices() -> {"speakers": [...] }`
  - `get_finetuned_models() -> {...}`
  - `synthesize_open_source(text, language, device, ...) -> {"file_path", ...}`
  - `synthesize_gemini(text, language, voice_name, style_prompt, ...) -> {"file_path", ...}`
- `suppress_all_logs()` – suppress noisy logs in notebooks

## Troubleshooting

- Ensure the speech service is running and reachable at `BASE_URL`.
- Gemini TTS requires `GEMINI_API_KEY` on the service.
- Finetuned TTS `speaker_id` must be one of the allowed voices (see `get_finetuned_voices()`).

