"""
database/connection.py
───────────────────────
Single SQLite connection factory for the entire application.

This module is the SOLE place that opens a database connection.
All repositories, services, and views import `get_db_connection` from here
(or from the legacy re-export in services/database.py).

Features:
  - WAL journal mode for better concurrent read performance
  - Foreign-key enforcement ON by default
  - Row factory → columns accessible by name (row["client_name"])
  - Context-manager with automatic commit / rollback
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── DB path (relative to project root) ───────────────────────────────────────
DB_PATH: Path = Path(__file__).resolve().parent.parent / settings.database_path


def _ensure_db_dir() -> None:
    """Create the data/ directory if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a configured SQLite connection.

    Usage:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM clients").fetchall()

    The connection auto-commits on clean exit and rolls back on any exception.
    """
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")   # wait up to 5 s on lock

    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("DB error — rolled back: %s", exc)
        raise
    finally:
        conn.close()
