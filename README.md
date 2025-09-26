# NaijaLingo ASR SDK

ASR SDK for Nigerian languages using CTranslate2-converted Whisper models.

## Install

```bash
pip install naijaligo-asr
```

## Quickstart

```python
from naijaligo_asr import transcribe

text = transcribe("/path/to/audio.wav", language="yo")
print(text)
```

## CLI

```bash
naijaligo-asr --audio /path/to/audio.wav --language yo
```

## Docker usage

Build the image (CPU):

```bash
docker build -t naijaligo-asr:cpu .
```

Run transcription via CLI:

```bash
docker run --rm -v $(pwd):/data naijaligo-asr:cpu \
  naijaligo-asr --audio /data/audio.mp3 --language yo
```

For GPU (CUDA), use a CUDA-enabled base and `--gpus all` (to be added later).

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

## Publishing to PyPI

1. Ensure metadata in `pyproject.toml` is correct.
2. Build:
   ```bash
   pip install build twine
   python -m build
   ```
3. Upload:
   ```bash
   twine upload dist/*
   ```
4. Install test:
   ```bash
   pip install naijaligo-asr
   ```

No `setup.py` is needed; this project uses PEP 621 `pyproject.toml` with setuptools.
