"""
utils/client_resolver.py
─────────────────────────
Bridge between SQLite integer client IDs and (optionally) mock_data string keys.

PHASE 2 UPDATE:
  - mock_key is now Optional[str] — new clients created via the UI will have
    mock_key = None and will be powered entirely by the DB.
  - get_all_client_refs() now returns ALL clients from the DB, not just the
    5 seeded mock clients.
  - AUM is read from the DB portfolios table for all clients.
  - Backward compatibility: the 5 demo clients still carry their mock_key so
    existing pages that call mock_data functions continue to work.

Usage in a page:
    from utils.client_resolver import get_selected_client, ClientRef

    ref = get_selected_client()
    if ref is None:
        st.info("Select a client from the sidebar.")
        return

    # Use DB id for SQL queries (works for ALL clients):
    conn.execute("SELECT * FROM portfolios WHERE client_id=?", (ref.db_id,))

    # Use mock_key ONLY for demo clients (may be None for new clients):
    if ref.mock_key:
        holdings = get_holdings_df(ref.mock_key)   # legacy path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

from portfolio.mock_data import CLIENTS
from services.database import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Name → mock_data key mapping (derived automatically from CLIENTS)
# This is used ONLY to maintain backward-compat for the 5 demo clients.
# ─────────────────────────────────────────────────────────────────────────────

# e.g. {"Sarah Mitchell": "sarah_mitchell", ...}
_NAME_TO_KEY: dict[str, str] = {
    info["name"]: key for key, info in CLIENTS.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# ClientRef — the single object all pages use
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClientRef:
    """
    Unified client reference.

    Attributes:
        db_id:        SQLite integer primary key (from clients table).
        name:         Display name (e.g. "Sarah Mitchell").
        risk_profile: "conservative" | "moderate" | "aggressive"
        aum:          Current AUM in dollars (from DB portfolio total_value).
        mock_key:     OPTIONAL — portfolio.mock_data string key.
                      Set only for the 5 original seeded clients.
                      New clients added via the UI will have mock_key = None.
                      Pages should check `if ref.mock_key` before using it.
        email:        Client email (for display and uniqueness).
        phone:        Client phone (optional display).
    """
    db_id:        int
    name:         str
    risk_profile: str
    aum:          float       = 0.0
    mock_key:     Optional[str] = None
    email:        str         = ""
    phone:        str         = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.risk_profile.title()})"

    @property
    def has_mock_data(self) -> bool:
        """True if this client has a mock_data entry (the 5 demo clients)."""
        return self.mock_key is not None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_client_refs() -> list[ClientRef]:
    """
    Return ALL clients from the DB as ClientRef objects.

    Phase 2 change: now includes clients with no mock_data entry.
    AUM is sourced from portfolios.total_value (DB truth), not mock_data.

    Returns:
        List of ClientRef, ordered by name.
    """
    refs: list[ClientRef] = []

    try:
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    c.id, c.name, c.email,
                    COALESCE(c.phone, '')        AS phone,
                    c.risk_profile,
                    COALESCE(SUM(p.total_value), c.aum, 0.0) AS aum
                FROM clients c
                LEFT JOIN portfolios p ON p.client_id = c.id
                GROUP BY c.id
                ORDER BY c.name
            """).fetchall()
    except Exception as exc:
        logger.error("Could not load clients from DB: %s", exc)
        return []

    for row in rows:
        name     = row["name"]
        mock_key = _NAME_TO_KEY.get(name)   # None for new clients — that's fine

        # Log non-demo clients at debug level (no longer a warning)
        if mock_key is None:
            logger.debug("Client '%s' (id=%d) has no mock_data entry — DB-only.", name, row["id"])

        refs.append(ClientRef(
            db_id=row["id"],
            name=name,
            email=row["email"] or "",
            phone=row["phone"] or "",
            risk_profile=row["risk_profile"],
            aum=float(row["aum"] or 0.0),
            mock_key=mock_key,
        ))

    return refs


def resolve_client(identifier) -> Optional[ClientRef]:
    """
    Resolve any client identifier to a ClientRef.

    Accepts:
        - int           → SQLite primary key
        - str (name)    → "Sarah Mitchell"
        - str (key)     → "sarah_mitchell" (mock_data key, demo clients only)
        - str (email)   → "sarah@example.com"
        - ClientRef     → returned as-is

    Returns:
        ClientRef, or None if not found.
    """
    if identifier is None:
        return None

    if isinstance(identifier, ClientRef):
        return identifier

    all_refs = get_all_client_refs()

    if isinstance(identifier, int):
        for ref in all_refs:
            if ref.db_id == identifier:
                return ref

    if isinstance(identifier, str):
        for ref in all_refs:
            if (ref.mock_key == identifier
                    or ref.name == identifier
                    or ref.email == identifier):
                return ref

    logger.warning("Could not resolve client identifier: %r", identifier)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit session state helpers
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_KEY = "selected_client_ref"


def get_selected_client() -> Optional[ClientRef]:
    """
    Return the currently selected ClientRef from Streamlit session state.
    Returns None if no client is selected.
    """
    return st.session_state.get(_SESSION_KEY)


def set_selected_client(ref: Optional[ClientRef]) -> None:
    """Store a ClientRef in session state (called by the sidebar)."""
    st.session_state[_SESSION_KEY] = ref
    # Keep backward-compat key for pages that haven't migrated yet
    st.session_state["selected_client_id"] = ref.db_id if ref else None
