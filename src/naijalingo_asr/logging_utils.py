from __future__ import annotations
import logging
import os
from typing import Optional
import warnings

DEFAULT_LOG_LEVEL = os.getenv("NAIJALINGO_ASR_LOG", "WARNING").upper()


def configure_logging(level: Optional[str] = None) -> None:
    level_name = (level or DEFAULT_LOG_LEVEL).upper()
    numeric_level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def suppress_external_warnings() -> None:
    """Reduce verbosity of common upstream libs (Hugging Face, Transformers)."""
    # Disable Hugging Face Hub telemetry (keep progress bars visible)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        from huggingface_hub import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from transformers.utils import logging as tf_logging  # type: ignore

        tf_logging.set_verbosity_error()
    except Exception:
        pass

    # Librosa/audioread warnings are informational; lower to WARNING/ERROR globally
    logging.getLogger("librosa").setLevel(logging.ERROR)
    logging.getLogger("audioread").setLevel(logging.ERROR)
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)

    # Filter specific noisy warnings seen in practice
    warnings.filterwarnings(
        "ignore",
        message=r"PySoundFile failed.*Trying audioread instead.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"librosa\.core\.audio\.__audioread_load",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Invalid model-index\. Not loading eval results into CardData\.",
    )
