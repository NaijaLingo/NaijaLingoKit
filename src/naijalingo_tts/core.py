"""Core components for NaijaLingo TTS audio generation (local pipeline).

This mirrors the public Kani-TTS core, with a key difference:
- It gracefully handles models without `config.speaker_settings` by allowing
  arbitrary `speaker_id` strings to be passed in, or an optional default list
  to expose via API for convenience.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
from nemo.collections.tts.models import AudioCodecModel
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class TTSConfig:
    """Configuration for TTS model."""

    device_map: str = "auto"
    tokeniser_length: int = 64400
    start_of_text: int = 1
    end_of_text: int = 2
    max_new_tokens: int = 1200
    temperature: float = 1
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    nanocodec_model: str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
    sample_rate = 22050


class NemoAudioPlayer:
    """Handles audio codec operations using NVIDIA NeMo."""

    def __init__(self, config: TTSConfig, text_tokenizer_name: Optional[str] = None) -> None:
        self.conf = config
        print(f"Downloading NeMo codec model: {self.conf.nanocodec_model} ...")
        self.nemo_codec_model = (
            AudioCodecModel.from_pretrained(self.conf.nanocodec_model).eval()
        )
        print("NeMo codec model loaded.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nemo_codec_model.to(self.device)

        self.text_tokenizer_name = text_tokenizer_name
        if self.text_tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_name)

        # Special token layout shared with Kani-TTS
        self.tokeniser_length = self.conf.tokeniser_length
        self.start_of_text = self.conf.start_of_text
        self.end_of_text = self.conf.end_of_text
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

    def output_validation(self, out_ids: torch.Tensor) -> None:
        """Validate that output contains required speech markers."""
        start_of_speech_flag = self.start_of_speech in out_ids
        end_of_speech_flag = self.end_of_speech in out_ids
        if not (start_of_speech_flag and end_of_speech_flag):
            raise ValueError("Special speech tokens not exist!")

    def get_nano_codes(self, out_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and process audio codes from model output."""
        start_a_idx = (out_ids == self.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_a_idx = (out_ids == self.end_of_speech).nonzero(as_tuple=True)[0].item()
        if start_a_idx >= end_a_idx:
            raise ValueError("Invalid audio codes sequence!")

        audio_codes = out_ids[start_a_idx + 1 : end_a_idx]
        if len(audio_codes) % 4:
            raise ValueError("The length of the sequence must be a multiple of 4!")
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - torch.tensor([self.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError("Invalid audio tokens!")

        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]])
        return audio_codes, len_

    def get_text(self, out_ids: torch.Tensor) -> str:
        """Extract text from token sequence."""
        start_t_idx = (out_ids == self.start_of_text).nonzero(as_tuple=True)[0].item()
        end_t_idx = (out_ids == self.end_of_text).nonzero(as_tuple=True)[0].item()
        txt_tokens = out_ids[start_t_idx : end_t_idx + 1]
        text = self.tokenizer.decode(txt_tokens, skip_special_tokens=True)
        return text

    def get_waveform(self, out_ids: torch.Tensor) -> Tuple[np.ndarray, Optional[str]]:
        """Convert model output tokens to audio waveform."""
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes, len_ = audio_codes.to(self.device), len_.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        if self.text_tokenizer_name:
            text = self.get_text(out_ids)
            return output_audio, text
        else:
            return output_audio, None


class NaijaModel:
    """Text-to-speech model using a causal LM with NeMo NanoCodec.

    Differences vs KaniModel:
    - If `config.speaker_settings` is missing, we still allow arbitrary `speaker_id`
      strings and optionally expose a provided `default_speakers` list.
    """

    def __init__(
        self,
        config: TTSConfig,
        model_name: str,
        player: NemoAudioPlayer,
        default_speakers: Optional[List[str]] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.conf = config
        self.player = player
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token

        # Use bfloat16 (matches Kani-TTS defaults and your fine-tune); device_map controls placement.
        print(f"Downloading TTS model: {model_name} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.conf.device_map,
            token=self.hf_token,
        )
        print("TTS model loaded.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)

        # Speaker handling: read from model config if present; otherwise use provided defaults
        self.speaker_settings = getattr(self.model.config, "speaker_settings", None)
        self.status = "singlspeaker"
        self.speaker_list: List[str] = []

        if self.speaker_settings is not None:
            self.status = self.speaker_settings.get("status", "singlspeaker")
            self.speaker_list = self.speaker_settings.get("speaker_list", [])
        elif default_speakers:
            # Expose user-provided speakers for convenience; model will still accept arbitrary strings
            self.status = "multispeaker"
            self.speaker_list = list(default_speakers)

    def get_input_ids(self, text_prompt: str, speaker_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tokens with special markers and optional speaker prefix."""
        if speaker_id is not None and len(str(speaker_id).strip()) > 0:
            text_prompt = f"{speaker_id.strip()}: {text_prompt}"

        START_OF_HUMAN = self.player.start_of_human
        END_OF_TEXT = self.player.end_of_text
        END_OF_HUMAN = self.player.end_of_human

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        return modified_input_ids, attention_mask

    def model_request(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generate audio tokens from text tokens."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.conf.max_new_tokens,
                do_sample=True,
                temperature=self.conf.temperature,
                top_p=self.conf.top_p,
                repetition_penalty=self.conf.repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.player.end_of_speech,
            )
        return generated_ids.to("cpu")

    def run_model(self, text: str, speaker_id: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """Generate audio from text."""
        input_ids, attention_mask = self.get_input_ids(text, speaker_id)
        model_output = self.model_request(input_ids, attention_mask)
        audio, _ = self.player.get_waveform(model_output)
        return audio, text


