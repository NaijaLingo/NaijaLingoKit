"""Simple local API for NaijaLingo TTS (based on Kani-TTS API)."""

from typing import Tuple, Optional, List
import logging
import warnings

import numpy as np

from .core import TTSConfig, NemoAudioPlayer, NaijaModel


def suppress_all_logs():
    """
    Suppress logging output from transformers, NeMo, PyTorch, and other libraries.
    Only print() statements from user code will be visible.
    """
    warnings.filterwarnings("ignore")

    try:
        import transformers  # type: ignore

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except Exception:
        pass

    logging.getLogger("nemo").setLevel(logging.ERROR)
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("pytorch").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


class NaijaTTS:
    """
    Simple interface to run your fine-tuned Kani-style TTS model locally.

    Example:
        >>> tts = NaijaTTS('chukypedro/hausa-tts-400m-0.3-pt', default_speakers=["speaker1", "speaker2"])
        >>> audio, text = tts("Sannu!", speaker_id="speaker1")
        >>> tts.save_audio(audio, "out.wav")
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        max_new_tokens: int = 1200,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        tokeniser_length: int = 64400,
        suppress_logs: bool = True,
        show_info: bool = True,
        default_speakers: Optional[List[str]] = None,
    ) -> None:
        if suppress_logs:
            suppress_all_logs()

        self.config = TTSConfig(
            device_map=device_map,
            tokeniser_length=tokeniser_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        self.model_name = model_name
        self.player = NemoAudioPlayer(self.config)
        self.model = NaijaModel(self.config, model_name, self.player, default_speakers=default_speakers)

        self.status = self.model.status
        self.speaker_list = self.model.speaker_list
        self.sample_rate = self.config.sample_rate

        if show_info:
            self.show_model_info()

    def __call__(self, text: str, speaker_id: Optional[str] = None) -> Tuple[np.ndarray, str]:
        return self.generate(text, speaker_id)

    def generate(self, text: str, speaker_id: Optional[str] = None) -> Tuple[np.ndarray, str]:
        return self.model.run_model(text, speaker_id)

    def save_audio(self, audio: np.ndarray, output_path: str) -> None:
        try:
            import soundfile as sf  # type: ignore

            sf.write(output_path, audio, self.sample_rate)
        except ImportError as exc:
            raise ImportError(
                "soundfile is required to save audio. Install it with: pip install soundfile"
            ) from exc

    def show_model_info(self) -> None:
        print()
        print("╔════════════════════════════════════════════════════════════╗")
        print("║                                                            ║")
        print("║                     N A I J A  L I N G O                   ║")
        print("║                            T T S                           ║")
        print("║                                                            ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()
        print("              /\\_/\\  ")
        print("             ( o.o )")
        print("              > ^ <")
        print()
        print("─" * 62)

        model_display = self.model_name
        if len(model_display) > 50:
            model_display = "..." + model_display[-47:]
        print(f"  Model: {model_display}")

        import torch  # local import to avoid import order surprises

        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"  Device: {device}")

        if self.status == "multispeaker":
            print(f"  Mode: Multi-speaker ({len(self.speaker_list)} speakers)")
            if self.speaker_list and len(self.speaker_list) <= 5:
                speakers_str = ", ".join(self.speaker_list)
                print(f"  Speakers: {speakers_str}")
            elif self.speaker_list:
                print(
                    f"  Speakers: {self.speaker_list[0]}, {self.speaker_list[1]}, ... (use .show_speakers() to see all)"
                )
        else:
            print("  Mode: Single-speaker (speaker_id still accepted)")

        print()
        print("  Configuration:")
        print(f"    • Sample Rate: {self.sample_rate} Hz")
        print(f"    • Temperature: {self.config.temperature}")
        print(f"    • Top-p: {self.config.top_p}")
        print(f"    • Max Tokens: {self.config.max_new_tokens}")
        print(f"    • Repetition Penalty: {self.config.repetition_penalty}")
        print("─" * 62)
        print()
        print("  Ready to generate speech! 🎵")
        print()

    def show_speakers(self) -> None:
        print("=" * 50)
        if self.status == "multispeaker":
            print("Available Speakers:")
            print("-" * 50)
            if self.speaker_list:
                for i, speaker in enumerate(self.speaker_list, 1):
                    print(f"  {i}. {speaker}")
            else:
                print("  No speakers configured")
        else:
            print("Single-speaker model (speaker_id still accepted)")
        print("=" * 50)


