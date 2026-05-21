"""
compliance/alert_service.py
────────────────────────────
Alert orchestration service.

Responsibilities:
  - Build PortfolioContext from mock data + analytics
  - Run the RulesEngine for one or all clients
  - Persist new violations to the compliance_alerts SQLite table
  - Provide query methods for the dashboard
  - Deduplicate: same rule + client won't produce duplicate open alerts
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from compliance.rules_engine import (
    RulesEngine, RuleResult, PortfolioContext, get_all_rules
)
from portfolio.mock_data import (
    CLIENTS, get_holdings_df, get_nav_history
)
from portfolio.analytics import (
    portfolio_summary, performance_metrics, rolling_returns, drift_analysis
)
from services.database import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertSummary:
    """Aggregate counts returned after running alerts for a client."""
    client_id:    str
    client_name:  str
    total_rules:  int
    violations:   int
    passed:       int
    new_saved:    int           # Alerts newly written to DB this run
    by_severity:  dict          # {"critical": 2, "high": 1, ...}
    results:      list[RuleResult]


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_context(client_id: str, db_client_id: int | None = None) -> PortfolioContext:
    """
    Assemble PortfolioContext from DB holdings (with mock_data fallback).

    Args:
        client_id:    mock_data key string (e.g. 'sarah_mitchell').
        db_client_id: Integer DB id if known (avoids an extra lookup).
    """
    from database.repositories.portfolio_repository import PortfolioRepository
    pr = PortfolioRepository()

    # Try DB holdings first
    holdings = pd.DataFrame()
    if db_client_id:
        ports = pr.get_for_client(db_client_id)
        if ports:
            holdings = pr.get_holdings_df(ports[0]["id"])

    if holdings.empty and client_id in CLIENTS:
        # Fall back to mock_data for demo clients
        holdings = get_holdings_df(client_id)

    client = CLIENTS.get(client_id, {})
    nav_df = get_nav_history(client_id, weeks=52) if client_id in CLIENTS else pd.DataFrame()
    target = client.get("target_allocation", {})

    if not target and not holdings.empty:
        classes = holdings["asset_class"].unique()
        share   = round(100 / len(classes), 1)
        target  = {c: share for c in classes}

    return PortfolioContext(
        client_id=client_id,
        client_name=client.get("name", client_id),
        risk_profile=client.get("risk_profile", "moderate"),
        aum=holdings["market_value"].sum() if not holdings.empty else 0.0,
        advisor_notes=client.get("advisor_notes", ""),
        holdings=holdings,
        target_allocation=target,
        perf_metrics=performance_metrics(nav_df),
        rolling_returns=rolling_returns(nav_df),
        drift_df=drift_analysis(holdings, target) if not holdings.empty and target else pd.DataFrame(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication helpers
# ─────────────────────────────────────────────────────────────────────────────

# Map rule category → compliance_alerts.alert_type value
_CATEGORY_TO_TYPE: dict[str, str] = {
    "concentration":  "concentration",
    "suitability":    "risk_mismatch",
    "exposure":       "risk_mismatch",
    "diversification": "concentration",
    "rebalancing":    "rebalance",
    "regulatory":     "regulatory",
    "performance":    "rebalance",
}


def _open_alert_exists(conn, client_db_id: int, rule_id: str) -> bool:
    """Return True if there's already an open alert for this rule+client."""
    row = conn.execute(
        """SELECT id FROM compliance_alerts
           WHERE client_id=? AND is_resolved=0
           AND description LIKE ?""",
        (client_db_id, f"%[{rule_id}]%"),
    ).fetchone()
    return row is not None


def _get_client_db_id(conn, client_id: str) -> Optional[int]:
    """Look up the DB integer ID for a mock_data client_id string."""
    # Map by name since mock_data keys don't directly match DB rows
    name = CLIENTS[client_id]["name"]
    row  = conn.execute(
        "SELECT id FROM clients WHERE name=?", (name,)
    ).fetchone()
    return row["id"] if row else None


# ─────────────────────────────────────────────────────────────────────────────
# AlertService
# ─────────────────────────────────────────────────────────────────────────────

class AlertService:
    """
    Runs compliance rules and manages the alert lifecycle.

    Usage:
        svc     = AlertService()
        summary = svc.run_for_client("sarah_mitchell")
        all_    = svc.run_for_all_clients()
    """

    def __init__(self) -> None:
        self._engine = RulesEngine()
        logger.info(
            "AlertService initialised | %d rules loaded", self._engine.rule_count
        )

    # ── Public: run rules ─────────────────────────────────────────────────────

    def run_for_client(
        self,
        client_id:   str,
        persist:     bool = True,
        db_client_id: int | None = None,
    ) -> AlertSummary:
        """
        Run all rules for one client and optionally persist violations.

        Args:
            client_id:    mock_data key OR 'db:<int>' for DB-only clients.
            persist:      If True, save new violations to compliance_alerts table.
            db_client_id: Integer DB id (avoids a name lookup in persist step).

        Returns:
            AlertSummary with counts and full RuleResult list.
        """
        ctx     = _build_context(client_id, db_client_id=db_client_id)
        results = self._engine.run(ctx)

        violations = [r for r in results if not r.passed]
        new_saved  = 0

        if persist and violations:
            new_saved = self._persist_violations(ctx, violations)

        by_severity: dict[str, int] = {}
        for r in violations:
            by_severity[r.severity] = by_severity.get(r.severity, 0) + 1

        logger.info(
            "Compliance run | %s | rules=%d | violations=%d | saved=%d",
            ctx.client_name, len(results), len(violations), new_saved,
        )

        return AlertSummary(
            client_id=client_id,
            client_name=ctx.client_name,
            total_rules=len(results),
            violations=len(violations),
            passed=len(results) - len(violations),
            new_saved=new_saved,
            by_severity=by_severity,
            results=results,
        )

    def run_for_all_clients(self, persist: bool = True) -> list[AlertSummary]:
        """Run rules for every client in the DB and return summaries."""
        from services.database import get_db_connection as _gdb
        # Build a name→mock_key reverse map for legacy clients
        _name_to_mock_key = {v["name"]: k for k, v in CLIENTS.items()}

        with _gdb() as conn:
            rows = conn.execute(
                "SELECT id, name FROM clients ORDER BY id"
            ).fetchall()

        summaries = []
        for row in rows:
            db_id    = row["id"]
            name     = row["name"]
            mock_key = _name_to_mock_key.get(name)
            cid      = mock_key if mock_key else f"db:{db_id}"
            try:
                ctx     = _build_context(cid, db_client_id=db_id)
                results = self._engine.run(ctx)
                violations = [r for r in results if not r.passed]
                new_saved  = self._persist_violations(ctx, violations) if persist and violations else 0
                by_severity: dict[str, int] = {}
                for r in violations:
                    by_severity[r.severity] = by_severity.get(r.severity, 0) + 1
                summaries.append(AlertSummary(
                    client_id=cid,
                    client_name=ctx.client_name,
                    total_rules=len(results),
                    violations=len(violations),
                    passed=len(results) - len(violations),
                    new_saved=new_saved,
                    by_severity=by_severity,
                    results=results,
                ))
            except Exception as exc:
                logger.error("Error running alerts for client %s: %s", db_id, exc)
        return summaries

    # ── Public: query alerts from DB ─────────────────────────────────────────

    def get_open_alerts(
        self,
        client_id: Optional[str] = None,
        severity:  Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch open alerts from the database.

        Args:
            client_id: Filter to a specific mock_data client_id (optional).
            severity:  Filter by severity level (optional).

        Returns:
            DataFrame with columns matching compliance_alerts table + client_name.
        """
        filters = ["ca.is_resolved = 0"]
        params: list = []

        if client_id:
            name = CLIENTS[client_id]["name"]
            filters.append("c.name = ?")
            params.append(name)

        if severity:
            filters.append("ca.severity = ?")
            params.append(severity)

        where = " AND ".join(filters)
        sql = f"""
            SELECT ca.id, ca.alert_type, ca.severity, ca.title,
                   ca.description, ca.created_at,
                   COALESCE(c.name, 'N/A') AS client_name,
                   COALESCE(c.risk_profile, 'N/A') AS risk_profile
            FROM compliance_alerts ca
            LEFT JOIN clients c ON c.id = ca.client_id
            WHERE {where}
            ORDER BY
                CASE ca.severity
                    WHEN 'critical' THEN 1
                    WHEN 'high'     THEN 2
                    WHEN 'medium'   THEN 3
                    ELSE 4
                END,
                ca.created_at DESC
        """
        with get_db_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            [dict(r) for r in rows],
            columns=["id", "alert_type", "severity", "title",
                     "description", "created_at", "client_name", "risk_profile"],
        )

    def get_alert_counts(self) -> dict:
        """Return aggregate counts across all open alerts."""
        with get_db_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM compliance_alerts WHERE is_resolved=0"
            ).fetchone()[0]

            by_sev = conn.execute(
                """SELECT severity, COUNT(*) as cnt
                   FROM compliance_alerts WHERE is_resolved=0
                   GROUP BY severity"""
            ).fetchall()

            resolved = conn.execute(
                "SELECT COUNT(*) FROM compliance_alerts WHERE is_resolved=1"
            ).fetchone()[0]

        counts = {row["severity"]: row["cnt"] for row in by_sev}
        counts["total_open"]   = total
        counts["total_resolved"] = resolved
        return counts

    def resolve_alert(self, alert_id: int) -> bool:
        """Mark a specific alert as resolved. Returns True on success."""
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE compliance_alerts SET is_resolved=1, resolved_at=? WHERE id=?",
                    (datetime.now().isoformat(timespec="seconds"), alert_id),
                )
            logger.info("Alert %d resolved.", alert_id)
            return True
        except Exception as exc:
            logger.error("Failed to resolve alert %d: %s", alert_id, exc)
            return False

    def resolve_all_for_client(self, client_id: str) -> int:
        """Resolve all open alerts for a client. Returns count resolved."""
        name = CLIENTS[client_id]["name"]
        with get_db_connection() as conn:
            db_id = _get_client_db_id(conn, client_id)
            if not db_id:
                return 0
            cursor = conn.execute(
                """UPDATE compliance_alerts
                   SET is_resolved=1, resolved_at=?
                   WHERE client_id=? AND is_resolved=0""",
                (datetime.now().isoformat(timespec="seconds"), db_id),
            )
            count = cursor.rowcount
        logger.info("Resolved %d alerts for %s.", count, name)
        return count

    # ── Private: persistence ──────────────────────────────────────────────────

    def _persist_violations(
        self,
        ctx:        PortfolioContext,
        violations: list[RuleResult],
    ) -> int:
        """Write new violations to the DB, skipping already-open duplicates."""
        saved = 0
        with get_db_connection() as conn:
            db_id = _get_client_db_id(conn, ctx.client_id)
            if not db_id:
                logger.warning(
                    "Client %s not found in DB — alerts not persisted.", ctx.client_id
                )
                return 0

            for v in violations:
                if _open_alert_exists(conn, db_id, v.rule_id):
                    logger.debug("Duplicate open alert skipped: %s / %s", ctx.client_id, v.rule_id)
                    continue

                alert_type = _CATEGORY_TO_TYPE.get(v.category, "regulatory")
                description = f"[{v.rule_id}] {v.explanation}"

                conn.execute(
                    """INSERT INTO compliance_alerts
                       (client_id, alert_type, severity, title, description)
                       VALUES (?, ?, ?, ?, ?)""",
                    (db_id, alert_type, v.severity, v.title, description),
                )
                saved += 1

        return saved
