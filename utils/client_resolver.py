"""
utils/client_resolver.py
─────────────────────────
Bridge between SQLite integer client IDs and mock_data string keys.

Problem solved:
  The sidebar loads clients from SQLite (integer PKs: 1, 2, 3...).
  Phase 3/4/5 modules (portfolio, summary, compliance) use mock_data string
  keys (e.g. "sarah_mitchell"). Without a bridge, each page needs its own
  ad-hoc mapping, leading to fragile hardcoded dicts.

Solution:
  - `ClientRef` dataclass holds both IDs simultaneously.
  - `resolve_client()` accepts either format and returns a ClientRef.
  - `get_all_client_refs()` returns all clients as ClientRefs.
  - The sidebar stores a ClientRef in session state so every page gets both.

Usage in a page:
    from utils.client_resolver import get_selected_client, ClientRef

    ref = get_selected_client()
    if ref is None:
        st.info("Select a client from the sidebar.")
        return

    # Use mock_data key for analytics:
    holdings = get_holdings_df(ref.mock_key)

    # Use DB id for SQL queries:
    conn.execute("SELECT * FROM portfolios WHERE client_id=?", (ref.db_id,))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from portfolio.mock_data import CLIENTS
from services.database import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Name → mock_data key mapping (derived automatically from CLIENTS)
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
    Unified client reference carrying both ID formats.

    Attributes:
        db_id:       SQLite integer primary key (from clients table).
        mock_key:    portfolio.mock_data string key (e.g. "sarah_mitchell").
        name:        Display name (e.g. "Sarah Mitchell").
        risk_profile: "conservative" | "moderate" | "aggressive"
        aum:         Current AUM in dollars (from mock_data portfolio value).
    """
    db_id:        int
    mock_key:     str
    name:         str
    risk_profile: str
    aum:          float = 0.0

    def __str__(self) -> str:
        return f"{self.name} ({self.risk_profile.title()})"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_client_refs() -> list[ClientRef]:
    """
    Return all clients as ClientRef objects.

    Loads from SQLite (for db_id) and cross-references with mock_data
    (for mock_key). Only clients present in both sources are returned.

    Returns:
        List of ClientRef, ordered by name.
    """
    refs: list[ClientRef] = []

    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT id, name, risk_profile FROM clients ORDER BY name"
            ).fetchall()
    except Exception as exc:
        logger.error("Could not load clients from DB: %s", exc)
        return []

    for row in rows:
        name     = row["name"]
        mock_key = _NAME_TO_KEY.get(name)

        if mock_key is None:
            logger.warning(
                "Client '%s' (DB id=%d) has no mock_data entry — skipping.",
                name, row["id"],
            )
            continue

        # Get current portfolio value from mock_data
        try:
            from portfolio.mock_data import get_holdings_df
            aum = get_holdings_df(mock_key)["market_value"].sum()
        except Exception:
            aum = 0.0

        refs.append(ClientRef(
            db_id=row["id"],
            mock_key=mock_key,
            name=name,
            risk_profile=row["risk_profile"],
            aum=aum,
        ))

    return refs


def resolve_client(identifier) -> Optional[ClientRef]:
    """
    Resolve any client identifier to a ClientRef.

    Accepts:
        - int           → SQLite primary key
        - str (name)    → "Sarah Mitchell"
        - str (key)     → "sarah_mitchell"
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
            if ref.mock_key == identifier or ref.name == identifier:
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
    # Also keep backward-compat key for pages that haven't migrated yet
    st.session_state["selected_client_id"] = ref.db_id if ref else None
