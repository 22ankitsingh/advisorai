"""
compliance/audit_logger.py
───────────────────────────
Structured audit logging for all compliance events.

Every significant action — running rules, generating alerts, resolving alerts,
generating summaries — is written here as an immutable audit record.

This provides a chronological trail that regulators and senior advisors
can review to demonstrate due diligence.

Event types:
  rules_run           — compliance scan was executed for a client
  alert_generated     — a rule violation was detected and saved
  alert_resolved      — an alert was manually resolved
  summary_generated   — an AI/template summary was created
  manual_note         — a free-text note added by the advisor
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd

from services.database import get_db_connection, init_database
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Event type constants
# ─────────────────────────────────────────────────────────────────────────────

RULES_RUN          = "rules_run"
ALERT_GENERATED    = "alert_generated"
ALERT_RESOLVED     = "alert_resolved"
SUMMARY_GENERATED  = "summary_generated"
MANUAL_NOTE        = "manual_note"

ALL_EVENT_TYPES = [RULES_RUN, ALERT_GENERATED, ALERT_RESOLVED, SUMMARY_GENERATED, MANUAL_NOTE]

# Severity badge colours (reused in dashboard)
SEVERITY_COLOURS: dict[str, str] = {
    "critical": "#ff4757",
    "high":     "#ff6b35",
    "medium":   "#ffa502",
    "low":      "#2ed573",
    "info":     "#60a5fa",
}


# ─────────────────────────────────────────────────────────────────────────────
# AuditLogger
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Write and query structured audit log entries.

    Usage:
        log = AuditLogger()
        log.log_rules_run("sarah_mitchell", violations=3, total_rules=10)
        log.log_alert_resolved(alert_id=5, client_name="Sarah Mitchell")

        df = log.get_recent(limit=50)
    """

    def __init__(self) -> None:
        # Ensure the audit_log table exists (schema added in Phase 5)
        init_database()

    # ── Write methods ─────────────────────────────────────────────────────────

    def log_rules_run(
        self,
        client_id:    str,
        client_name:  str,
        violations:   int,
        total_rules:  int,
        db_client_id: Optional[int] = None,
    ) -> None:
        """Record that a compliance scan was run for a client."""
        self._write(
            event_type=RULES_RUN,
            client_id=db_client_id,
            client_name=client_name,
            summary=f"Compliance scan: {violations}/{total_rules} rules violated",
            detail=json.dumps({
                "violations":  violations,
                "total_rules": total_rules,
                "client_key":  client_id,
            }),
        )

    def log_alert_generated(
        self,
        client_name:  str,
        rule_id:      str,
        severity:     str,
        title:        str,
        db_client_id: Optional[int] = None,
    ) -> None:
        """Record that a new compliance alert was generated."""
        self._write(
            event_type=ALERT_GENERATED,
            client_id=db_client_id,
            client_name=client_name,
            rule_id=rule_id,
            severity=severity,
            summary=f"Alert generated [{rule_id}]: {title[:80]}",
            detail=json.dumps({"rule_id": rule_id, "title": title}),
        )

    def log_alert_resolved(
        self,
        alert_id:     int,
        client_name:  str,
        title:        str = "",
        db_client_id: Optional[int] = None,
    ) -> None:
        """Record that an alert was manually resolved."""
        self._write(
            event_type=ALERT_RESOLVED,
            client_id=db_client_id,
            client_name=client_name,
            summary=f"Alert #{alert_id} resolved" + (f": {title[:60]}" if title else ""),
            detail=json.dumps({"alert_id": alert_id}),
        )

    def log_summary_generated(
        self,
        client_name:  str,
        summary_type: str,
        is_ai:        bool,
        db_client_id: Optional[int] = None,
    ) -> None:
        """Record that an advisor summary was generated."""
        method = "AI (Gemini)" if is_ai else "Template"
        self._write(
            event_type=SUMMARY_GENERATED,
            client_id=db_client_id,
            client_name=client_name,
            summary=f"Summary generated ({summary_type}) via {method}",
            detail=json.dumps({
                "summary_type": summary_type,
                "method":       method,
            }),
        )

    def log_manual_note(
        self,
        client_name:  str,
        note:         str,
        db_client_id: Optional[int] = None,
    ) -> None:
        """Record an advisor-entered free-text audit note."""
        self._write(
            event_type=MANUAL_NOTE,
            client_id=db_client_id,
            client_name=client_name,
            summary=f"Manual note: {note[:100]}",
            detail=note,
        )

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_recent(
        self,
        limit:       int             = 100,
        event_type:  Optional[str]   = None,
        client_name: Optional[str]   = None,
    ) -> pd.DataFrame:
        """
        Fetch recent audit log entries.

        Args:
            limit:       Max rows to return.
            event_type:  Filter by event type (optional).
            client_name: Filter by client name (optional).

        Returns:
            DataFrame with audit log columns, most recent first.
        """
        filters = []
        params: list = []

        if event_type:
            filters.append("event_type = ?")
            params.append(event_type)
        if client_name:
            filters.append("client_name = ?")
            params.append(client_name)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.append(limit)

        sql = f"""
            SELECT id, event_type, client_name, rule_id, severity,
                   summary, detail, created_at
            FROM audit_log
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        with get_db_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["id", "event_type", "client_name", "rule_id",
                         "severity", "summary", "detail", "created_at"]
            )

        return pd.DataFrame([dict(r) for r in rows])

    def get_daily_counts(self, days: int = 30) -> pd.DataFrame:
        """
        Return daily event counts for the past N days.
        Useful for a trend chart in the dashboard.
        """
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT DATE(created_at) AS day,
                       event_type,
                       COUNT(*) AS count
                FROM audit_log
                WHERE created_at >= DATE('now', ?)
                GROUP BY day, event_type
                ORDER BY day ASC
                """,
                (f"-{days} days",),
            ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["day", "event_type", "count"])

        return pd.DataFrame([dict(r) for r in rows])

    def get_stats(self) -> dict:
        """Return summary statistics for the audit log."""
        with get_db_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            by_type = conn.execute(
                "SELECT event_type, COUNT(*) as cnt FROM audit_log GROUP BY event_type"
            ).fetchall()
            today = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE DATE(created_at)=DATE('now')"
            ).fetchone()[0]

        return {
            "total_entries": total,
            "today_entries": today,
            "by_type": {r["event_type"]: r["cnt"] for r in by_type},
        }

    def clear_old_entries(self, keep_days: int = 90) -> int:
        """Delete audit entries older than keep_days. Returns rows deleted."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM audit_log WHERE created_at < DATE('now', ?)",
                (f"-{keep_days} days",),
            )
        deleted = cursor.rowcount
        logger.info("Audit log pruned: %d entries older than %d days deleted.", deleted, keep_days)
        return deleted

    # ── Private ───────────────────────────────────────────────────────────────

    def _write(
        self,
        event_type:  str,
        summary:     str,
        client_id:   Optional[int] = None,
        client_name: Optional[str] = None,
        rule_id:     Optional[str] = None,
        severity:    Optional[str] = None,
        detail:      Optional[str] = None,
    ) -> None:
        """Insert one audit log entry."""
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """INSERT INTO audit_log
                       (event_type, client_id, client_name, rule_id, severity, summary, detail)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (event_type, client_id, client_name, rule_id, severity, summary, detail),
                )
            logger.debug("Audit log: [%s] %s", event_type, summary[:80])
        except Exception as exc:
            # Audit logging must never crash the main flow
            logger.error("Failed to write audit log entry: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (one instance per app run)
# ─────────────────────────────────────────────────────────────────────────────

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Return the application-wide AuditLogger singleton."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
