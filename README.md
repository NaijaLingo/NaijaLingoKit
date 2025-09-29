# NaijaLingo ASR SDK

ASR SDK for Nigerian languages using CTranslate2-converted Whisper models.

## Install

```bash
pip install naijalingo-asr
```

## Quickstart

```python
from naijalingo_asr import transcribe

text = transcribe("/path/to/audio.wav", language="yo")
print(text)

text = transcribe("/path/to/igbo_audio.wav", language="ig")
print(text)
```

## CLI

```bash
naijalingo-asr --audio_path /path/to/audio.wav --language yo
```

### CLI options

- `--language` (required): Language code (`yo`, `ig`, `ha`, `en`).
- `--device` (optional): `auto` (default), `cpu`, or `cuda`.
- `--compute-type` (optional): `auto` (default), `float16`, `int8`, `int8_float16`.
- `--beam-size` (optional): Beam search size (default: `5`).
- `--no-vad` (optional flag): Disable VAD filter (enabled by default).
- `--temperature` (optional): Sampling temperature (default: `0.0`).
- `--initial-prompt` (optional): Context prompt to prime decoding.
- `--log-level` (optional): `DEBUG`, `INFO`, `WARNING` (default), `ERROR`.

Examples:

```bash
# GPU with float16
naijalingo-asr --audio_path /path/to/audio.wav --language yo \
  --device cuda --compute-type float16 --beam-size 5 --log-level INFO

# CPU with int8
naijalingo-asr --audio_path /path/to/audio.wav --language ig \
  --device cpu --compute-type int8 --no-vad

# With an initial prompt context
naijalingo-asr --audio_path /path/to/audio.wav --language ha \
  --initial-prompt "Medical conversation in Hausa"
```

## Docker

Docker support has been removed. Use the CLI or Python API instead.

## Supported languages

- yo: Yoruba
- ig: Igbo
- ha: Hausa
- en: Nigerian-accented English

## Notes
- Uses faster-whisper (CTranslate2 backend)
- Accepts file paths (mp3/wav/m4a/etc.) via librosa, or a numpy array (mono 16k)
- Task is transcription only; set `task="transcribe"` and the language code.

## Logging

Set via CLI `--log-level INFO` or env `NAIJALINGO_ASR_LOG=INFO`.


