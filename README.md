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
```

## CLI

```bash
naijalingo-asr --audio /path/to/audio.wav --language yo
```

## Docker usage (no build required)

Pull the image (CPU):

```bash
docker pull chukypedro15/naijalingo-asr:cpu:latest
```

Run transcription via Docker CMD:

```bash
docker run --rm -v $(pwd):/data chukypedro15/naijalingo-asr:cpu:latest \
  naijalingo-asr --audio /data/audio.wav --language yo
```

For GPU (CUDA), a separate CUDA-enabled image will be provided later.

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


