import logging
import sys
from pathlib import Path
from typing import Optional

from core.config import get_settings


def setup_logging(
    name: str = "traffic_marl",
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    settings = get_settings()
    log_level = level or (logging.DEBUG if settings.debug else logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is None:
        log_dir = settings.project_root / settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "traffic_marl") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger
