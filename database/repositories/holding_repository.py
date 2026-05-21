"""
database/repositories/holding_repository.py
─────────────────────────────────────────────
All database operations for the `holdings` table.

Each holding belongs to exactly one portfolio.  The portfolio's
total_value is NOT auto-updated here — call
portfolio_repository.update_total_value(portfolio_id) after changes.

CRUD:
  - get_for_portfolio(portfolio_id) → list[dict]
  - get_by_id(holding_id)           → dict | None
  - get_by_ticker(portfolio_id, ticker) → dict | None
  - create(portfolio_id, ...)        → int
  - update(holding_id, ...)          → bool
  - upsert(portfolio_id, ticker, ...) → int  (create or update)
  - delete(holding_id)               → bool
  - bulk_replace(portfolio_id, rows) → int  (for CSV import)
"""

from __future__ import annotations

from typing import Optional

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


class HoldingRepository:
    """Data-access layer for the holdings table."""

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_for_portfolio(self, portfolio_id: int) -> list[dict]:
        """
        Return all holdings for a portfolio, ordered by market value desc.

        Args:
            portfolio_id: FK to portfolios.id.

        Returns:
            List of holding dicts with computed market_value column.
        """
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    id, portfolio_id, ticker, asset_name, asset_class,
                    quantity, avg_cost, current_price, sector,
                    COALESCE(target_allocation, 0) AS target_allocation,
                    (quantity * current_price)      AS market_value,
                    created_at, updated_at
                FROM holdings
                WHERE portfolio_id = ?
                ORDER BY (quantity * current_price) DESC
            """, (portfolio_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, holding_id: int) -> Optional[dict]:
        """Return a single holding dict, or None."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM holdings WHERE id = ?", (holding_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_ticker(self, portfolio_id: int, ticker: str) -> Optional[dict]:
        """
        Return the holding for a specific ticker within a portfolio.

        Args:
            portfolio_id: Portfolio scope.
            ticker:       Ticker symbol (case-insensitive).

        Returns:
            Holding dict, or None if not found.
        """
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT * FROM holdings
                WHERE portfolio_id = ? AND UPPER(ticker) = UPPER(?)
            """, (portfolio_id, ticker)).fetchone()
        return dict(row) if row else None

    # ── Write ─────────────────────────────────────────────────────────────────

    def create(
        self,
        portfolio_id: int,
        ticker: str,
        asset_name: str,
        asset_class: str,
        quantity: float,
        avg_cost: float,
        current_price: float,
        sector: Optional[str] = None,
        target_allocation: Optional[float] = None,
    ) -> int:
        """
        Insert a new holding and return its id.

        Args:
            portfolio_id:      Parent portfolio.
            ticker:            Ticker symbol (e.g. "AAPL").
            asset_name:        Human-readable name.
            asset_class:       "equity" | "bond" | "etf" | "cash" | "alternative".
            quantity:          Number of units held.
            avg_cost:          Average purchase price per unit.
            current_price:     Current market price per unit.
            sector:            Sector string (optional).
            target_allocation: Target % weight in portfolio (optional).

        Returns:
            New holding_id.
        """
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO holdings
                    (portfolio_id, ticker, asset_name, asset_class,
                     quantity, avg_cost, current_price, sector, target_allocation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (portfolio_id, ticker.upper(), asset_name, asset_class,
                 quantity, avg_cost, current_price, sector, target_allocation),
            )
            new_id = cursor.lastrowid
        logger.debug("HoldingRepository.create: id=%d ticker=%s", new_id, ticker)
        return new_id

    def update(
        self,
        holding_id: int,
        *,
        ticker: Optional[str] = None,
        asset_name: Optional[str] = None,
        asset_class: Optional[str] = None,
        quantity: Optional[float] = None,
        avg_cost: Optional[float] = None,
        current_price: Optional[float] = None,
        sector: Optional[str] = None,
        target_allocation: Optional[float] = None,
    ) -> bool:
        """
        Partially update a holding (only supplied fields are changed).

        Returns:
            True if the row was found and updated.
        """
        fields: dict = {}
        if ticker             is not None: fields["ticker"]             = ticker.upper()
        if asset_name         is not None: fields["asset_name"]         = asset_name
        if asset_class        is not None: fields["asset_class"]        = asset_class
        if quantity           is not None: fields["quantity"]           = quantity
        if avg_cost           is not None: fields["avg_cost"]           = avg_cost
        if current_price      is not None: fields["current_price"]      = current_price
        if sector             is not None: fields["sector"]             = sector
        if target_allocation  is not None: fields["target_allocation"]  = target_allocation

        if not fields:
            return False

        set_clause = ", ".join(f"{col} = ?" for col in fields)
        set_clause += ", updated_at = datetime('now')"
        values = list(fields.values()) + [holding_id]

        with get_db_connection() as conn:
            cursor = conn.execute(
                f"UPDATE holdings SET {set_clause} WHERE id = ?", values
            )
        return cursor.rowcount > 0

    def upsert(
        self,
        portfolio_id: int,
        ticker: str,
        asset_name: str,
        asset_class: str,
        quantity: float,
        avg_cost: float,
        current_price: float,
        sector: Optional[str] = None,
        target_allocation: Optional[float] = None,
    ) -> int:
        """
        Insert or update a holding identified by (portfolio_id, ticker).

        If a holding with the same ticker already exists in the portfolio,
        it is updated in-place.  Otherwise a new row is inserted.

        Returns:
            The holding_id (existing or new).
        """
        existing = self.get_by_ticker(portfolio_id, ticker)
        if existing:
            self.update(
                existing["id"],
                ticker=ticker,
                asset_name=asset_name,
                asset_class=asset_class,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                sector=sector,
                target_allocation=target_allocation,
            )
            return existing["id"]
        return self.create(
            portfolio_id, ticker, asset_name, asset_class,
            quantity, avg_cost, current_price, sector, target_allocation,
        )

    def bulk_replace(self, portfolio_id: int, rows: list[dict]) -> int:
        """
        Replace ALL holdings in a portfolio with the supplied rows.

        Used by the CSV import feature.

        Args:
            portfolio_id: Portfolio to replace holdings in.
            rows: List of dicts, each with keys:
                  ticker, asset_name, asset_class, quantity,
                  avg_cost, current_price, sector (optional).

        Returns:
            Number of holdings inserted.
        """
        with get_db_connection() as conn:
            conn.execute("DELETE FROM holdings WHERE portfolio_id = ?", (portfolio_id,))
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO holdings
                        (portfolio_id, ticker, asset_name, asset_class,
                         quantity, avg_cost, current_price, sector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        portfolio_id,
                        row.get("ticker", "").upper(),
                        row.get("asset_name", row.get("name", "")),
                        row.get("asset_class", "equity"),
                        float(row.get("quantity", 0)),
                        float(row.get("average_cost", row.get("avg_cost", 0))),
                        float(row.get("current_price", 0)),
                        row.get("sector"),
                    ),
                )
        logger.info(
            "HoldingRepository.bulk_replace: portfolio=%d rows=%d",
            portfolio_id, len(rows),
        )
        return len(rows)

    def delete(self, holding_id: int) -> bool:
        """Delete a single holding by id."""
        with get_db_connection() as conn:
            cursor = conn.execute("DELETE FROM holdings WHERE id = ?", (holding_id,))
        return cursor.rowcount > 0
