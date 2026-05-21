"""
database/repositories/portfolio_repository.py
───────────────────────────────────────────────
All database operations for the `portfolios` table.

Key feature:
  get_holdings_df(portfolio_id) — returns a pandas DataFrame compatible
  with the existing portfolio/analytics.py and portfolio/risk_engine.py
  functions, allowing those engines to work with DB data transparently.

CRUD:
  - get_for_client(client_id)   → list[dict]
  - get_by_id(portfolio_id)     → dict | None
  - get_primary(client_id)      → dict | None  (first/main portfolio)
  - get_holdings_df(portfolio_id) → pd.DataFrame
  - create(client_id, name)     → int
  - update_total_value(id)      → float  (recalculates from holdings)
  - delete(portfolio_id)        → bool
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)

# Asset class metadata needed to enrich the DataFrame the same way
# portfolio/mock_data.py does (so analytics.py works unchanged).
_ASSET_CLASS_META: dict[str, dict] = {
    "equity":      {"annual_vol": 0.20, "beta": 1.10, "expected_return": 0.10},
    "etf":         {"annual_vol": 0.15, "beta": 0.95, "expected_return": 0.08},
    "bond":        {"annual_vol": 0.06, "beta": 0.15, "expected_return": 0.04},
    "alternative": {"annual_vol": 0.55, "beta": 1.40, "expected_return": 0.18},
    "cash":        {"annual_vol": 0.00, "beta": 0.00, "expected_return": 0.05},
}


class PortfolioRepository:
    """Data-access layer for portfolios and their holdings."""

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_for_client(self, client_id: int) -> list[dict]:
        """
        Return all portfolios owned by a client.

        Args:
            client_id: SQLite FK to clients.id.

        Returns:
            List of portfolio dicts with total_value and holding_count.
        """
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    p.id, p.client_id, p.name,
                    p.total_value, p.created_at, p.updated_at,
                    COUNT(h.id) AS holding_count
                FROM portfolios p
                LEFT JOIN holdings h ON h.portfolio_id = p.id
                WHERE p.client_id = ?
                GROUP BY p.id
                ORDER BY p.created_at ASC
            """, (client_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, portfolio_id: int) -> Optional[dict]:
        """Return a single portfolio dict, or None if not found."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_for_client_by_mock_key(self, mock_key: str) -> list[dict]:
        """
        Return all portfolios for a client identified by their mock_data key.

        Resolves the mock_key → client name via portfolio.mock_data.CLIENTS
        and then looks up the DB client by name.
        Returns an empty list if no client with that name is found.
        """
        try:
            from portfolio.mock_data import CLIENTS
            name = CLIENTS[mock_key]["name"]
        except KeyError:
            return []

        with get_db_connection() as conn:
            client_row = conn.execute(
                "SELECT id FROM clients WHERE name = ?", (name,)
            ).fetchone()
            if not client_row:
                return []
            db_id = client_row["id"]
            rows = conn.execute("""
                SELECT p.id, p.client_id, p.name, p.total_value,
                       p.created_at, p.updated_at
                FROM portfolios p
                WHERE p.client_id = ?
                ORDER BY p.created_at ASC
            """, (db_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_primary(self, client_id: int) -> Optional[dict]:
        """
        Return the first (primary) portfolio for a client.

        This is a convenience method for the common case where each client
        has exactly one portfolio.

        Args:
            client_id: SQLite FK to clients.id.

        Returns:
            Portfolio dict, or None if the client has no portfolios.
        """
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT * FROM portfolios
                WHERE client_id = ?
                ORDER BY created_at ASC
                LIMIT 1
            """, (client_id,)).fetchone()
        return dict(row) if row else None

    def get_holdings_df(self, portfolio_id: int) -> pd.DataFrame:
        """
        Return all holdings for a portfolio as a DataFrame.

        The returned DataFrame is enriched with computed columns to be
        100% compatible with portfolio/analytics.py and risk_engine.py:
          - market_value, cost_basis, gain_loss, gain_pct, weight
          - annual_vol, beta, expected_return  (from asset class metadata)
          - name  (alias for asset_name, for legacy compatibility)
          - qty   (alias for quantity, for legacy compatibility)
          - price (alias for current_price)
          - avg_cost (kept as-is)

        Args:
            portfolio_id: The portfolio to load.

        Returns:
            DataFrame with one row per holding.  Empty DataFrame if none.
        """
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    id, portfolio_id, ticker, asset_name, asset_class,
                    quantity, avg_cost, current_price, sector,
                    COALESCE(target_allocation, 0) AS target_allocation,
                    created_at, updated_at
                FROM holdings
                WHERE portfolio_id = ?
                ORDER BY asset_class, ticker
            """, (portfolio_id,)).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])

        # ── Legacy column aliases ─────────────────────────────────────────
        df["name"]  = df["asset_name"]        # analytics uses "name"
        df["qty"]   = df["quantity"]           # analytics uses "qty"
        df["price"] = df["current_price"]      # analytics uses "price"

        # ── Computed columns ─────────────────────────────────────────────
        df["market_value"] = df["qty"] * df["price"]
        df["cost_basis"]   = df["qty"] * df["avg_cost"]
        df["gain_loss"]    = df["market_value"] - df["cost_basis"]

        # Avoid division by zero for zero-cost positions (e.g., transferred in)
        df["gain_pct"] = df.apply(
            lambda r: ((r["price"] - r["avg_cost"]) / r["avg_cost"]) * 100
            if r["avg_cost"] > 0 else 0.0,
            axis=1,
        )

        total_value = df["market_value"].sum()
        df["weight"] = (df["market_value"] / total_value * 100) if total_value > 0 else 0.0

        # ── Asset class metadata ─────────────────────────────────────────
        df["annual_vol"]      = df["asset_class"].map(
            lambda c: _ASSET_CLASS_META.get(c, _ASSET_CLASS_META["equity"])["annual_vol"]
        )
        df["beta"]            = df["asset_class"].map(
            lambda c: _ASSET_CLASS_META.get(c, _ASSET_CLASS_META["equity"])["beta"]
        )
        df["expected_return"] = df["asset_class"].map(
            lambda c: _ASSET_CLASS_META.get(c, _ASSET_CLASS_META["equity"])["expected_return"]
        )

        return df.reset_index(drop=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def create(self, client_id: int, name: str = "Main Portfolio") -> int:
        """
        Create a new portfolio for a client.

        Args:
            client_id: Owner's SQLite PK.
            name:      Portfolio display name.

        Returns:
            New portfolio_id.
        """
        with get_db_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO portfolios (client_id, name) VALUES (?, ?)",
                (client_id, name),
            )
            new_id = cursor.lastrowid
        logger.info("PortfolioRepository.create: id=%d client=%d", new_id, client_id)
        return new_id

    def update_total_value(self, portfolio_id: int) -> float:
        """
        Recalculate and persist the portfolio's total_value from its holdings.

        Call this after any holding insert/update/delete.

        Returns:
            The newly computed total value.
        """
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT COALESCE(SUM(quantity * current_price), 0.0) AS total
                FROM holdings
                WHERE portfolio_id = ?
            """, (portfolio_id,)).fetchone()
            total = row["total"]
            conn.execute(
                "UPDATE portfolios SET total_value=?, updated_at=datetime('now') WHERE id=?",
                (total, portfolio_id),
            )
        return total

    def rename(self, portfolio_id: int, new_name: str) -> bool:
        """Rename a portfolio."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                "UPDATE portfolios SET name=?, updated_at=datetime('now') WHERE id=?",
                (new_name, portfolio_id),
            )
        return cursor.rowcount > 0

    def delete(self, portfolio_id: int) -> bool:
        """Delete a portfolio and all its holdings (cascade)."""
        with get_db_connection() as conn:
            cursor = conn.execute("DELETE FROM portfolios WHERE id=?", (portfolio_id,))
            deleted = cursor.rowcount > 0
        logger.info("PortfolioRepository.delete: id=%d deleted=%s", portfolio_id, deleted)
        return deleted
