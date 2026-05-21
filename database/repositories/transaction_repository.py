"""
database/repositories/transaction_repository.py
─────────────────────────────────────────────────
All database operations for the `transactions` table.

Transactions record every buy/sell event for audit and history purposes.
They do NOT automatically update holding quantities — that is the
caller's responsibility (portfolio_management view handles this).

CRUD:
  - get_for_portfolio(portfolio_id, limit) → list[dict]
  - get_for_client(client_id, limit)       → list[dict]
  - record(portfolio_id, ...)              → int
  - get_summary(portfolio_id)              → dict
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)

# Valid transaction types
TRANSACTION_TYPES = ("buy", "sell", "deposit", "withdrawal", "dividend")


class TransactionRepository:
    """Data-access layer for the transactions table."""

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_for_portfolio(
        self,
        portfolio_id: int,
        limit: int = 100,
        ticker: Optional[str] = None,
    ) -> list[dict]:
        """
        Return recent transactions for a portfolio.

        Args:
            portfolio_id: FK to portfolios.id.
            limit:        Maximum rows to return.
            ticker:       Filter to a specific ticker (optional).

        Returns:
            List of transaction dicts, most recent first.
        """
        params: list = [portfolio_id]
        where_extra = ""
        if ticker:
            where_extra = "AND UPPER(ticker) = UPPER(?)"
            params.append(ticker)

        params.append(limit)

        with get_db_connection() as conn:
            rows = conn.execute(f"""
                SELECT
                    t.id, t.portfolio_id, t.ticker,
                    t.transaction_type, t.quantity, t.price,
                    t.transaction_date, t.notes, t.created_at,
                    (t.quantity * t.price) AS total_value
                FROM transactions t
                WHERE t.portfolio_id = ? {where_extra}
                ORDER BY t.transaction_date DESC, t.created_at DESC
                LIMIT ?
            """, params).fetchall()
        return [dict(r) for r in rows]

    def get_for_client(self, client_id: int, limit: int = 50) -> list[dict]:
        """
        Return recent transactions across all portfolios owned by a client.

        Args:
            client_id: FK to clients.id.
            limit:     Maximum rows to return.

        Returns:
            List of transaction dicts with portfolio_name, most recent first.
        """
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    t.id, t.portfolio_id, t.ticker,
                    t.transaction_type, t.quantity, t.price,
                    t.transaction_date, t.notes, t.created_at,
                    (t.quantity * t.price) AS total_value,
                    p.name                 AS portfolio_name
                FROM transactions t
                JOIN portfolios p ON p.id = t.portfolio_id
                WHERE p.client_id = ?
                ORDER BY t.transaction_date DESC, t.created_at DESC
                LIMIT ?
            """, (client_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_as_df(self, portfolio_id: int) -> pd.DataFrame:
        """Return transactions as a pandas DataFrame (for display tables)."""
        rows = self.get_for_portfolio(portfolio_id, limit=500)
        if not rows:
            return pd.DataFrame(
                columns=["id", "ticker", "transaction_type", "quantity",
                         "price", "total_value", "transaction_date", "notes"]
            )
        return pd.DataFrame(rows)

    def get_summary(self, portfolio_id: int) -> dict:
        """
        Return aggregate statistics for a portfolio's transaction history.

        Returns:
            Dict with: total_trades, total_invested, total_realized, buy_count, sell_count.
        """
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                                AS total_trades,
                    SUM(CASE WHEN transaction_type='buy'  THEN quantity*price ELSE 0 END) AS total_invested,
                    SUM(CASE WHEN transaction_type='sell' THEN quantity*price ELSE 0 END) AS total_realized,
                    SUM(CASE WHEN transaction_type='buy'  THEN 1 ELSE 0 END)              AS buy_count,
                    SUM(CASE WHEN transaction_type='sell' THEN 1 ELSE 0 END)              AS sell_count
                FROM transactions
                WHERE portfolio_id = ?
            """, (portfolio_id,)).fetchone()
        return dict(row) if row else {
            "total_trades": 0, "total_invested": 0.0,
            "total_realized": 0.0, "buy_count": 0, "sell_count": 0,
        }

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        portfolio_id: int,
        ticker: str,
        transaction_type: str,
        quantity: float,
        price: float,
        transaction_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Insert a transaction record.

        Args:
            portfolio_id:      FK to portfolios.id.
            ticker:            Ticker symbol (normalised to UPPER).
            transaction_type:  "buy" | "sell" | "deposit" | "withdrawal" | "dividend".
            quantity:          Number of units.
            price:             Price per unit.
            transaction_date:  ISO date string "YYYY-MM-DD" (default: today).
            notes:             Optional free-text annotation.

        Returns:
            New transaction_id.

        Raises:
            ValueError if transaction_type is invalid.
        """
        if transaction_type not in TRANSACTION_TYPES:
            raise ValueError(
                f"Invalid transaction_type '{transaction_type}'. "
                f"Must be one of: {TRANSACTION_TYPES}"
            )

        date_expr = transaction_date or "date('now')"
        # If a literal date string is supplied, parameterise it.
        # If None (default), use the SQL expression directly.
        if transaction_date:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO transactions
                        (portfolio_id, ticker, transaction_type,
                         quantity, price, transaction_date, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (portfolio_id, ticker.upper(), transaction_type,
                     quantity, price, transaction_date, notes),
                )
                new_id = cursor.lastrowid
        else:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO transactions
                        (portfolio_id, ticker, transaction_type,
                         quantity, price, transaction_date, notes)
                    VALUES (?, ?, ?, ?, ?, date('now'), ?)
                    """,
                    (portfolio_id, ticker.upper(), transaction_type,
                     quantity, price, notes),
                )
                new_id = cursor.lastrowid

        logger.debug(
            "TransactionRepository.record: id=%d %s %s x%.2f @ $%.2f",
            new_id, transaction_type, ticker, quantity, price,
        )
        return new_id
