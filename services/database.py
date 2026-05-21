"""
services/database.py
─────────────────────
SQLite database service.

Responsibilities:
  - Create and manage the database connection
  - Run schema migrations (CREATE TABLE IF NOT EXISTS)
  - Expose a clean context-manager for queries

All other services import `get_db_connection` from here.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Resolve the DB path relative to the project root
_DB_PATH: Path = Path(__file__).resolve().parent.parent / settings.database_path


def _ensure_db_directory() -> None:
    """Create the data/ directory if it doesn't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a SQLite connection with row_factory set to sqlite3.Row
    so columns are accessible by name (e.g. row["client_name"]).

    Usage:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM clients").fetchall()
    """
    _ensure_db_directory()
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Better concurrent access
    conn.execute("PRAGMA foreign_keys=ON")    # Enforce FK constraints
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Database error — rolled back: %s", exc)
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- Clients table
CREATE TABLE IF NOT EXISTS clients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    email           TEXT    UNIQUE NOT NULL,
    phone           TEXT,
    risk_profile    TEXT    NOT NULL DEFAULT 'moderate',  -- conservative|moderate|aggressive
    aum             REAL    NOT NULL DEFAULT 0.0,         -- assets under management ($)
    advisor_notes   TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Portfolios table  (one portfolio per client for Phase 1)
CREATE TABLE IF NOT EXISTS portfolios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id       INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL DEFAULT 'Main Portfolio',
    total_value     REAL    NOT NULL DEFAULT 0.0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Holdings table  (assets within a portfolio)
CREATE TABLE IF NOT EXISTS holdings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    ticker          TEXT    NOT NULL,
    asset_name      TEXT    NOT NULL,
    asset_class     TEXT    NOT NULL,  -- equity|bond|etf|cash|alternative
    quantity        REAL    NOT NULL,
    avg_cost        REAL    NOT NULL,  -- average purchase price per unit
    current_price   REAL    NOT NULL,
    sector          TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Chat history table
CREATE TABLE IF NOT EXISTS chat_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    role            TEXT    NOT NULL,  -- user|assistant
    content         TEXT    NOT NULL,
    client_id       INTEGER REFERENCES clients(id),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Compliance alerts table
CREATE TABLE IF NOT EXISTS compliance_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id       INTEGER REFERENCES clients(id),
    alert_type      TEXT    NOT NULL,  -- concentration|risk_mismatch|rebalance|regulatory
    severity        TEXT    NOT NULL DEFAULT 'medium',  -- low|medium|high|critical
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    is_resolved     INTEGER NOT NULL DEFAULT 0,  -- 0=open, 1=resolved
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    resolved_at     TEXT
);

-- Audit log table (Phase 5)
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,  -- alert_generated|alert_resolved|rules_run|summary_generated|manual_note
    client_id   INTEGER REFERENCES clients(id),
    client_name TEXT,              -- denormalised for fast queries without JOIN
    rule_id     TEXT,              -- e.g. CONC-001
    severity    TEXT,
    summary     TEXT    NOT NULL,  -- one-line human-readable description
    detail      TEXT,              -- JSON or free-text extra info
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def init_database() -> None:
    """
    Run schema creation.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    logger.info("Initialising database at: %s", _DB_PATH)
    with get_db_connection() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("Database schema ready.")
