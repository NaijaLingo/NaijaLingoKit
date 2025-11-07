# NaijaLingo TTS (Local Pipeline)

A local, Kani-TTS–compatible pipeline to run your fine-tuned model
`chukypedro/hausa-tts-400m-0.3-pt` inside this repository without publishing to PyPI.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (WSL2 supported) or CPU

Install packages (pinning versions compatible with LFM2 and NeMo):

```bash
pip install "transformers==4.57.1"
pip install "torch>=2.0.0"  # install the CUDA-specific wheel if you have a GPU
pip install "nemo-toolkit[all]==2.4.0"
pip install numpy>=1.24.0 scipy>=1.10.0 librosa>=0.10.0 omegaconf>=2.3.0 soundfile>=0.12.0
```

Notes:
- If you have a GPU, install the correct Torch build for your CUDA version.
- NeMo pulls a number of extras; make sure your environment has enough disk.

## Quick Start

```python
from naijalingo_tts import NaijaTTS

# Optional: a display-only set of speakers (your model still accepts any string)
default_speakers = ["david", "kore", "maria"]

tts = NaijaTTS(
    "chukypedro/hausa-tts-400m-0.3-pt",
    default_speakers=default_speakers,  # optional, for banner + show_speakers()
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=1200,
    repetition_penalty=1.1,
    device_map="auto",
)

audio, text = tts("Sannu, ya ya kake?", speaker_id="david")

tts.save_audio(audio, "hausa_david.wav")
print("Saved to hausa_david.wav")
```

## Multi-speaker Usage

- If your model config does not contain `speaker_settings`, you can still pass a `speaker_id` string.
- To show a friendly list in the banner and via `.show_speakers()`, provide `default_speakers` when constructing `NaijaTTS`.

```python
print(tts.status)        # 'multispeaker' if default_speakers provided, else 'singlspeaker'
print(tts.speaker_list)  # the default_speakers you passed in (for display)

tts.show_speakers()     # nicely formatted list if you set default_speakers
```

## API Summary

- `NaijaTTS(model_name, ..., default_speakers=None)`
  - `model_name`: e.g., `"chukypedro/hausa-tts-400m-0.3-pt"`
  - `default_speakers`: optional list for display; generation accepts any `speaker_id`
- `tts(text, speaker_id=None) -> (audio: np.ndarray, text: str)`
- `tts.generate(text, speaker_id=None)` – same as calling
- `tts.save_audio(audio, path)` – write WAV at 22,050 Hz
- `tts.sample_rate` – `22050`
- `tts.show_model_info()` – print banner
- `tts.show_speakers()` – list display speakers if provided

## Troubleshooting

- If you see "Special speech tokens not exist!", your generated sequence did not contain the speech markers; reduce `temperature`, increase `max_new_tokens`, or verify your model/tokenizer alignment.
- On CPU, generation will be slow. Prefer a CUDA build of Torch on a GPU.
- For WSL2 + GPU, ensure NVIDIA drivers and CUDA are correctly configured and use the matching Torch wheel.


