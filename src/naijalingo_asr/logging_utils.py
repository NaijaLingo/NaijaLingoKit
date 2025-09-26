from __future__ import annotations
import logging
import os
from typing import Optional

DEFAULT_LOG_LEVEL = os.getenv("NAIJALINGO_ASR_LOG", "WARNING").upper()


def configure_logging(level: Optional[str] = None) -> None:
    level_name = (level or DEFAULT_LOG_LEVEL).upper()
    numeric_level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
