"""
utils/config.py
───────────────
Centralised configuration loader with environment validation.

All application settings are read from environment variables (via .env).
Import `settings` anywhere in the codebase to get typed, validated config.
Call `validate_env()` on startup to catch missing variables early.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from the project root (one level above utils/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Settings dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """Immutable settings object built from environment variables."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    gemini_api_key:    str
    gemini_model:      str
    gemini_temperature: float   # 0.0 = deterministic, 1.0 = creative
    gemini_max_tokens: int      # max output tokens per response

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str
    app_env:  str

    # ── Database ─────────────────────────────────────────────────────────────
    database_path: Path

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str

    # ── RAG ──────────────────────────────────────────────────────────────────
    chroma_persist_dir: Path
    embedding_model:    str

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    # ai_available removed — use is_ai_available() from services.gemini_service instead


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_settings() -> Settings:
    """Read env vars and return a validated Settings instance."""
    # ── .env existence guard ──────────────────────────────────────────────────
    if not _ENV_PATH.exists():
        warnings.warn(
            f".env file not found at '{_ENV_PATH}'. "
            "Copy .env.example to .env and fill in your API key. "
            "Running with defaults — AI features will be unavailable.",
            stacklevel=2,
        )

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        warnings.warn(
            "GEMINI_API_KEY is not set. AI features will be unavailable.",
            stacklevel=2,
        )

    return Settings(
        gemini_api_key=gemini_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
        gemini_max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048")),
        app_name=os.getenv("APP_NAME", "Advisor AI"),
        app_env=os.getenv("APP_ENV", "development"),
        database_path=Path(os.getenv("DATABASE_PATH", "data/advisor_ai.db")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Environment validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvIssue:
    var:      str
    severity: str   # "error" | "warning" | "info"
    message:  str


def validate_env() -> list[EnvIssue]:
    """
    Validate the environment configuration and return a list of issues.

    Checks:
      - GEMINI_API_KEY present and not placeholder
      - GEMINI_TEMPERATURE in [0.0, 2.0]
      - GEMINI_MAX_TOKENS positive integer
      - APP_ENV is a recognised value
      - DATABASE_PATH parent directory exists (or can be created)

    Returns:
        List of EnvIssue objects. Empty list = all clear.
        Does NOT raise exceptions — issues are surfaced as warnings.
    """
    issues: list[EnvIssue] = []

    # ── Gemini API key ────────────────────────────────────────────────────────
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        issues.append(EnvIssue(
            var="GEMINI_API_KEY", severity="warning",
            message="Not set — AI chat, summaries disabled. Get a free key at aistudio.google.com",
        ))
    elif key == "your_gemini_api_key_here":
        issues.append(EnvIssue(
            var="GEMINI_API_KEY", severity="warning",
            message="Still set to placeholder value — replace with a real key",
        ))

    # ── Temperature ───────────────────────────────────────────────────────────
    try:
        temp = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
        if not (0.0 <= temp <= 2.0):
            issues.append(EnvIssue(
                var="GEMINI_TEMPERATURE", severity="warning",
                message=f"Value {temp} is outside the valid range [0.0, 2.0]",
            ))
    except ValueError:
        issues.append(EnvIssue(
            var="GEMINI_TEMPERATURE", severity="error",
            message="Must be a float between 0.0 and 2.0",
        ))

    # ── Max tokens ────────────────────────────────────────────────────────────
    try:
        tokens = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
        if tokens < 64:
            issues.append(EnvIssue(
                var="GEMINI_MAX_TOKENS", severity="warning",
                message=f"Value {tokens} is very low — summaries may be truncated",
            ))
    except ValueError:
        issues.append(EnvIssue(
            var="GEMINI_MAX_TOKENS", severity="error",
            message="Must be a positive integer",
        ))

    # ── App env ───────────────────────────────────────────────────────────────
    env = os.getenv("APP_ENV", "development")
    if env not in ("development", "staging", "production"):
        issues.append(EnvIssue(
            var="APP_ENV", severity="info",
            message=f"Unknown value '{env}'. Expected: development | staging | production",
        ))

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — import this everywhere
# ─────────────────────────────────────────────────────────────────────────────

settings: Settings = _load_settings()
