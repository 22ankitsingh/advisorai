"""
utils/logger.py
───────────────
Application-wide logger factory.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys
from pathlib import Path

from utils.config import settings

# ── Ensure logs/ directory exists ─────────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_LOG_FILE = _LOG_DIR / "advisor_ai.log"

# ── Build a shared formatter ──────────────────────────────────────────────────
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── One-time root logger configuration ───────────────────────────────────────
def _configure_root_logger() -> None:
    root = logging.getLogger("advisor_ai")
    if root.handlers:
        return  # Already configured

    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Console handler (UTF-8 to handle all unicode characters on Windows)
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setFormatter(_FORMATTER)
    root.addHandler(console_handler)

    # File handler (rotating would be added in production)
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(_FORMATTER)
    root.addHandler(file_handler)

    root.propagate = False


_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'advisor_ai' namespace.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(f"advisor_ai.{name}")
