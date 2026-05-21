"""
database/repositories/audit_repository.py
───────────────────────────────────────────
Thin repository wrapper around the existing AuditLogger.

The compliance/audit_logger.py AuditLogger class already does everything
needed for audit log writes and reads.  This repository:
  1. Re-exports the existing AuditLogger as the canonical write interface.
  2. Adds a few extra read helpers that the new audit_logs.py view needs.
  3. Adds new event type constants for client/portfolio CRUD events.

Usage:
    from database.repositories.audit_repository import AuditRepository, get_audit_repo

    repo = get_audit_repo()
    repo.log_client_created(client_name="Sarah Mitchell", client_id=1)
    df = repo.get_recent(limit=50, event_type="client_created")
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from compliance.audit_logger import AuditLogger, get_audit_logger
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Extended event type constants ─────────────────────────────────────────────

# Inherited from audit_logger.py:
RULES_RUN         = "rules_run"
ALERT_GENERATED   = "alert_generated"
ALERT_RESOLVED    = "alert_resolved"
SUMMARY_GENERATED = "summary_generated"
MANUAL_NOTE       = "manual_note"

# New in Phase 5:
CLIENT_CREATED     = "client_created"
CLIENT_UPDATED     = "client_updated"
CLIENT_DELETED     = "client_deleted"
PORTFOLIO_CREATED  = "portfolio_created"
PORTFOLIO_UPDATED  = "portfolio_updated"
HOLDING_ADDED      = "holding_added"
HOLDING_REMOVED    = "holding_removed"
CSV_IMPORTED       = "csv_imported"
CHAT_SESSION_START = "chat_session_start"

ALL_EVENT_TYPES = [
    RULES_RUN, ALERT_GENERATED, ALERT_RESOLVED, SUMMARY_GENERATED, MANUAL_NOTE,
    CLIENT_CREATED, CLIENT_UPDATED, CLIENT_DELETED,
    PORTFOLIO_CREATED, PORTFOLIO_UPDATED,
    HOLDING_ADDED, HOLDING_REMOVED, CSV_IMPORTED,
    CHAT_SESSION_START,
]

EVENT_TYPE_LABELS = {
    RULES_RUN:          "Compliance Scan",
    ALERT_GENERATED:    "Alert Generated",
    ALERT_RESOLVED:     "Alert Resolved",
    SUMMARY_GENERATED:  "Summary Generated",
    MANUAL_NOTE:        "Manual Note",
    CLIENT_CREATED:     "Client Created",
    CLIENT_UPDATED:     "Client Updated",
    CLIENT_DELETED:     "Client Deleted",
    PORTFOLIO_CREATED:  "Portfolio Created",
    PORTFOLIO_UPDATED:  "Portfolio Updated",
    HOLDING_ADDED:      "Holding Added",
    HOLDING_REMOVED:    "Holding Removed",
    CSV_IMPORTED:       "CSV Import",
    CHAT_SESSION_START: "Chat Session Started",
}


class AuditRepository(AuditLogger):
    """
    Extended audit logger with additional write helpers for CRUD events.

    Inherits all existing methods from AuditLogger:
      - log_rules_run(...)
      - log_alert_generated(...)
      - log_alert_resolved(...)
      - log_summary_generated(...)
      - log_manual_note(...)
      - get_recent(...)
      - get_daily_counts(...)
      - get_stats(...)
    """

    # ── New write helpers ─────────────────────────────────────────────────────

    def log_client_created(
        self,
        client_name: str,
        client_id: Optional[int] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Log that a new client was created via the UI."""
        self._write(
            event_type=CLIENT_CREATED,
            client_id=client_id,
            client_name=client_name,
            summary=f"New client created: {client_name}",
            detail=json.dumps(details or {}),
        )

    def log_client_updated(
        self,
        client_name: str,
        client_id: Optional[int] = None,
        changed_fields: Optional[list[str]] = None,
    ) -> None:
        """Log a client profile edit."""
        fields_str = ", ".join(changed_fields or [])
        self._write(
            event_type=CLIENT_UPDATED,
            client_id=client_id,
            client_name=client_name,
            summary=f"Client updated: {client_name}" + (f" [{fields_str}]" if fields_str else ""),
            detail=json.dumps({"changed_fields": changed_fields or []}),
        )

    def log_client_deleted(
        self,
        client_name: str,
        client_id: Optional[int] = None,
    ) -> None:
        """Log client deletion."""
        self._write(
            event_type=CLIENT_DELETED,
            client_id=client_id,
            client_name=client_name,
            summary=f"Client deleted: {client_name}",
            detail=json.dumps({"client_id": client_id}),
        )

    def log_portfolio_created(
        self,
        client_name: str,
        portfolio_name: str,
        client_id: Optional[int] = None,
        portfolio_id: Optional[int] = None,
    ) -> None:
        """Log that a new portfolio was created."""
        self._write(
            event_type=PORTFOLIO_CREATED,
            client_id=client_id,
            client_name=client_name,
            summary=f"Portfolio created: '{portfolio_name}' for {client_name}",
            detail=json.dumps({"portfolio_id": portfolio_id, "portfolio_name": portfolio_name}),
        )

    def log_portfolio_updated(
        self,
        client_name: str,
        portfolio_id: int,
        action: str = "holdings_updated",
        client_id: Optional[int] = None,
    ) -> None:
        """Log a portfolio change (holdings added/edited/deleted)."""
        self._write(
            event_type=PORTFOLIO_UPDATED,
            client_id=client_id,
            client_name=client_name,
            summary=f"Portfolio #{portfolio_id} updated ({action}) for {client_name}",
            detail=json.dumps({"portfolio_id": portfolio_id, "action": action}),
        )

    def log_csv_import(
        self,
        client_name: str,
        portfolio_id: int,
        row_count: int,
        client_id: Optional[int] = None,
    ) -> None:
        """Log a CSV portfolio import."""
        self._write(
            event_type=CSV_IMPORTED,
            client_id=client_id,
            client_name=client_name,
            summary=f"CSV import: {row_count} holdings loaded into portfolio #{portfolio_id} for {client_name}",
            detail=json.dumps({"portfolio_id": portfolio_id, "row_count": row_count}),
        )

    def log_chat_session(
        self,
        session_uuid: str,
        client_name: Optional[str] = None,
        client_id: Optional[int] = None,
    ) -> None:
        """Log the start of a chat session."""
        self._write(
            event_type=CHAT_SESSION_START,
            client_id=client_id,
            client_name=client_name or "No client",
            summary=f"Chat session started" + (f" for {client_name}" if client_name else ""),
            detail=json.dumps({"session_uuid": session_uuid}),
        )

    # ── Extended read helpers ─────────────────────────────────────────────────

    def get_recent_for_view(
        self,
        limit: int = 200,
        event_types: Optional[list[str]] = None,
        client_name: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch audit log entries for the audit_logs.py view.

        Args:
            limit:        Maximum rows.
            event_types:  Filter to specific event types (OR logic).
            client_name:  Filter by exact client name.
            search_query: Partial text search on summary column.

        Returns:
            DataFrame with a 'label' column (human-readable event type).
        """
        filters: list[str] = []
        params: list = []

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            filters.append(f"event_type IN ({placeholders})")
            params.extend(event_types)

        if client_name:
            filters.append("client_name = ?")
            params.append(client_name)

        if search_query:
            filters.append("summary LIKE ?")
            params.append(f"%{search_query}%")

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.append(limit)

        from database.connection import get_db_connection
        with get_db_connection() as conn:
            rows = conn.execute(f"""
                SELECT id, event_type, client_name, rule_id, severity,
                       summary, detail, created_at
                FROM audit_log
                {where}
                ORDER BY created_at DESC
                LIMIT ?
            """, params).fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["id", "event_type", "label", "client_name",
                         "rule_id", "severity", "summary", "detail", "created_at"]
            )

        df = pd.DataFrame([dict(r) for r in rows])
        df["label"] = df["event_type"].map(
            lambda e: EVENT_TYPE_LABELS.get(e, e.replace("_", " ").title())
        )
        return df

    def get_client_names(self) -> list[str]:
        """Return all distinct client names present in the audit log."""
        from database.connection import get_db_connection
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT client_name
                FROM audit_log
                WHERE client_name IS NOT NULL
                ORDER BY client_name
            """).fetchall()
        return [r[0] for r in rows]


# ── Module-level singleton ────────────────────────────────────────────────────

_audit_repo: Optional[AuditRepository] = None


def get_audit_repo() -> AuditRepository:
    """Return the application-wide AuditRepository singleton."""
    global _audit_repo
    if _audit_repo is None:
        _audit_repo = AuditRepository()
    return _audit_repo
