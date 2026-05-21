"""
database/repositories/client_repository.py
────────────────────────────────────────────
All database operations for the `clients` table.

Provides full CRUD:
  - get_all()          → list[dict]
  - get_by_id(id)      → dict | None
  - get_by_email(email)→ dict | None
  - create(...)        → int  (new client_id)
  - update(id, ...)    → bool
  - delete(id)         → bool
  - search(query)      → list[dict]

All returned dicts have the same keys as the `clients` table columns
plus a computed `portfolio_count` field.
"""

from __future__ import annotations

from typing import Optional

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


class ClientRepository:
    """Data-access layer for the clients table."""

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        """
        Return all clients ordered by AUM descending.

        Returns:
            List of dicts with client fields + open_alerts + portfolio_count.
        """
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    c.id, c.name, c.email, c.phone,
                    c.risk_profile, c.aum,
                    c.advisor_notes, c.created_at, c.updated_at,
                    COALESCE(c.age, 0)              AS age,
                    COALESCE(c.investment_goal, '') AS investment_goal,
                    COUNT(DISTINCT p.id)             AS portfolio_count,
                    COUNT(DISTINCT ca.id)            AS open_alerts
                FROM clients c
                LEFT JOIN portfolios p
                    ON p.client_id = c.id
                LEFT JOIN compliance_alerts ca
                    ON ca.client_id = c.id AND ca.is_resolved = 0
                GROUP BY c.id
                ORDER BY c.aum DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, client_id: int) -> Optional[dict]:
        """
        Return a single client by primary key, or None.

        Args:
            client_id: SQLite integer PK.
        """
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT
                    c.id, c.name, c.email, c.phone,
                    c.risk_profile, c.aum,
                    c.advisor_notes, c.created_at, c.updated_at,
                    COALESCE(c.age, 0)              AS age,
                    COALESCE(c.investment_goal, '') AS investment_goal
                FROM clients c
                WHERE c.id = ?
            """, (client_id,)).fetchone()
        return dict(row) if row else None

    def get_by_email(self, email: str) -> Optional[dict]:
        """Return a client dict matching the email address, or None."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM clients WHERE email = ?", (email,)
            ).fetchone()
        return dict(row) if row else None

    def search(self, query: str) -> list[dict]:
        """
        Case-insensitive search across name, email, and risk_profile.

        Args:
            query: Search string (partial match supported).

        Returns:
            List of matching client dicts.
        """
        like = f"%{query}%"
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    c.id, c.name, c.email, c.phone,
                    c.risk_profile, c.aum,
                    c.advisor_notes, c.created_at, c.updated_at,
                    COALESCE(c.age, 0)              AS age,
                    COALESCE(c.investment_goal, '') AS investment_goal,
                    COUNT(DISTINCT ca.id)            AS open_alerts
                FROM clients c
                LEFT JOIN compliance_alerts ca
                    ON ca.client_id = c.id AND ca.is_resolved = 0
                WHERE c.name LIKE ?
                   OR c.email LIKE ?
                   OR c.risk_profile LIKE ?
                GROUP BY c.id
                ORDER BY c.name
            """, (like, like, like)).fetchall()
        return [dict(r) for r in rows]

    def get_aum_summary(self) -> dict:
        """
        Return aggregate stats used by the dashboard.

        Returns:
            Dict with keys: total_aum, total_clients, avg_aum.
        """
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)    AS total_clients,
                    SUM(aum)    AS total_aum,
                    AVG(aum)    AS avg_aum
                FROM clients
            """).fetchone()
        return dict(row) if row else {"total_clients": 0, "total_aum": 0.0, "avg_aum": 0.0}

    # ── Write ─────────────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        email: str,
        phone: str = "",
        risk_profile: str = "moderate",
        aum: float = 0.0,
        advisor_notes: str = "",
        age: Optional[int] = None,
        investment_goal: Optional[str] = None,
    ) -> int:
        """
        Insert a new client and return the new client_id.

        Args:
            name:            Full display name.
            email:           Unique email address.
            phone:           Phone number (optional).
            risk_profile:    "conservative" | "moderate" | "aggressive".
            aum:             Initial assets under management (dollars).
            advisor_notes:   Free-text advisor notes.
            age:             Client age (optional).
            investment_goal: Short description of investment goal (optional).

        Returns:
            Integer primary key of the newly created client.

        Raises:
            sqlite3.IntegrityError if email is not unique.
        """
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO clients
                    (name, email, phone, risk_profile, aum, advisor_notes,
                     age, investment_goal, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (name, email, phone or "", risk_profile, aum,
                 advisor_notes or "", age, investment_goal),
            )
            new_id = cursor.lastrowid
        logger.info("ClientRepository.create: id=%d name=%r", new_id, name)
        return new_id

    def update(
        self,
        client_id: int,
        *,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        risk_profile: Optional[str] = None,
        aum: Optional[float] = None,
        advisor_notes: Optional[str] = None,
        age: Optional[int] = None,
        investment_goal: Optional[str] = None,
    ) -> bool:
        """
        Partially update a client row (only supplied fields are changed).

        Args:
            client_id: Client to update.
            **kwargs:  Fields to update (any subset).

        Returns:
            True if the row was found and updated, False otherwise.
        """
        fields: dict = {}
        if name            is not None: fields["name"]            = name
        if email           is not None: fields["email"]           = email
        if phone           is not None: fields["phone"]           = phone
        if risk_profile    is not None: fields["risk_profile"]    = risk_profile
        if aum             is not None: fields["aum"]             = aum
        if advisor_notes   is not None: fields["advisor_notes"]   = advisor_notes
        if age             is not None: fields["age"]             = age
        if investment_goal is not None: fields["investment_goal"] = investment_goal

        if not fields:
            return False  # Nothing to update

        fields["updated_at"] = "datetime('now')"

        # Build SET clause — values with datetime() use raw SQL, others use ?
        set_parts: list[str] = []
        values: list = []
        for col, val in fields.items():
            if val == "datetime('now')":
                set_parts.append(f"{col} = datetime('now')")
            else:
                set_parts.append(f"{col} = ?")
                values.append(val)

        values.append(client_id)
        sql = f"UPDATE clients SET {', '.join(set_parts)} WHERE id = ?"

        with get_db_connection() as conn:
            cursor = conn.execute(sql, values)
            updated = cursor.rowcount > 0

        logger.info("ClientRepository.update: id=%d updated=%s", client_id, updated)
        return updated

    def update_aum(self, client_id: int, aum: float) -> None:
        """Quick helper to sync AUM after portfolio changes."""
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE clients SET aum=?, updated_at=datetime('now') WHERE id=?",
                (aum, client_id),
            )

    def delete(self, client_id: int) -> bool:
        """
        Delete a client and all cascaded data (portfolios, holdings, alerts).

        Args:
            client_id: Client to delete.

        Returns:
            True if a row was deleted, False if not found.
        """
        with get_db_connection() as conn:
            cursor = conn.execute("DELETE FROM clients WHERE id = ?", (client_id,))
            deleted = cursor.rowcount > 0
        logger.info("ClientRepository.delete: id=%d deleted=%s", client_id, deleted)
        return deleted
