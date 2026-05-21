"""
views/audit_logs.py
─────────────────────
Audit Log Viewer — Phase 5.

Features:
  - KPI bar: total entries, today's events, compliance events, CRUD events
  - Activity trend chart: daily event counts over last 30 days (stacked bar)
  - Event type distribution donut chart
  - Filterable log table: event type, client, free-text search, date range
  - Per-row detail expander (shows JSON detail field)
  - Export filtered results to CSV
  - Prune old entries (retention management)
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.repositories.audit_repository import (
    AuditRepository, EVENT_TYPE_LABELS, ALL_EVENT_TYPES,
    CLIENT_CREATED, CLIENT_UPDATED, CLIENT_DELETED,
    PORTFOLIO_CREATED, PORTFOLIO_UPDATED, CSV_IMPORTED,
    CHAT_SESSION_START,
)
from compliance.audit_logger import AuditLogger, SEVERITY_COLOURS
from utils.helpers import apply_dark_theme, CHART_PALETTE
from utils.logger import get_logger

logger = get_logger(__name__)

_audit = AuditRepository()

# Severity → badge colour helper
_SEV_COLOURS = {
    "critical": ("#ff4757", "#fff"),
    "high":     ("#ff6b35", "#fff"),
    "medium":   ("#ffa502", "#000"),
    "low":      ("#2ed573", "#000"),
    "info":     ("#60a5fa", "#000"),
}

# Group event types for filter UX
_EVENT_GROUPS = {
    "All":        None,
    "Compliance": ["rules_run", "alert_generated", "alert_resolved"],
    "Client CRUD":["client_created", "client_updated", "client_deleted"],
    "Portfolio":  ["portfolio_created", "portfolio_updated", "csv_imported", "holding_added", "holding_removed"],
    "AI & Chat":  ["summary_generated", "chat_session_start"],
    "Other":      ["manual_note"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _severity_badge(severity: str | None) -> str:
    if not severity:
        return ""
    bg, fg = _SEV_COLOURS.get(severity.lower(), ("#888", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:10px;font-size:0.72rem;font-weight:700;">'
        f'{severity.upper()}</span>'
    )


def _event_label(event_type: str) -> str:
    return EVENT_TYPE_LABELS.get(event_type, event_type.replace("_", " ").title())


def _type_badge(event_type: str) -> str:
    colours = {
        "rules_run":        "#a78bfa",
        "alert_generated":  "#ff4757",
        "alert_resolved":   "#2ed573",
        "summary_generated":"#60a5fa",
        "client_created":   "#34d399",
        "client_updated":   "#fbbf24",
        "client_deleted":   "#f87171",
        "portfolio_created":"#60a5fa",
        "portfolio_updated":"#a78bfa",
        "csv_imported":     "#fb923c",
        "chat_session_start":"#c084fc",
        "manual_note":      "#94a3b8",
    }
    c = colours.get(event_type, "#94a3b8")
    label = _event_label(event_type)
    return (
        f'<span style="background:{c}20;color:{c};padding:2px 9px;'
        f'border-radius:10px;font-size:0.72rem;font-weight:600;'
        f'border:1px solid {c}40;">{label}</span>'
    )


# ── KPI bar ───────────────────────────────────────────────────────────────────

def _render_kpis(stats: dict) -> None:
    total     = stats.get("total_entries", 0)
    today     = stats.get("today_entries", 0)
    by_type   = stats.get("by_type", {})
    compliance_count = sum(by_type.get(t, 0) for t in ["rules_run","alert_generated","alert_resolved"])
    crud_count       = sum(by_type.get(t, 0) for t in ["client_created","client_updated","client_deleted",
                                                         "portfolio_created","portfolio_updated","csv_imported"])

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("📋 Total Entries", f"{total:,}")
    with k2: st.metric("📅 Today", today)
    with k3: st.metric("🛡️ Compliance Events", compliance_count)
    with k4: st.metric("✏️ CRUD Events", crud_count)
    with k5: st.metric("💬 AI Events", by_type.get("summary_generated", 0) + by_type.get("chat_session_start", 0))


# ── Trend chart ───────────────────────────────────────────────────────────────

def _render_trend(logger_instance: AuditLogger) -> None:
    df = logger_instance.get_daily_counts(days=30)
    if df.empty:
        st.info("No audit activity in the last 30 days.")
        return

    df["label"] = df["event_type"].map(_event_label)

    fig = px.bar(
        df, x="day", y="count", color="label",
        title="Daily Audit Activity (Last 30 Days)",
        labels={"day": "Date", "count": "Events", "label": "Event Type"},
        color_discrete_sequence=CHART_PALETTE,
        barmode="stack",
    )
    apply_dark_theme(fig, height=280)
    fig.update_layout(
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=50, b=40, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Distribution donut ────────────────────────────────────────────────────────

def _render_distribution(by_type: dict) -> None:
    if not by_type:
        return
    labels = [_event_label(k) for k in by_type]
    values = list(by_type.values())

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker_colors=CHART_PALETTE[:len(labels)],
        textposition="outside", textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value} events (%{percent})<extra></extra>",
    ))
    apply_dark_theme(fig)
    fig.update_layout(
        height=260, showlegend=False,
        margin=dict(t=20, b=20, l=10, r=10),
        annotations=[dict(text="Events", x=0.5, y=0.5, font_size=13, showarrow=False, font_color="#e8e8f0")],
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Filter controls ───────────────────────────────────────────────────────────

def _render_filters() -> tuple[list[str] | None, str | None, str | None, date | None, date | None]:
    """Return (event_types_filter, client_filter, search_query, date_from, date_to)."""
    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        group = st.selectbox("Event Category", list(_EVENT_GROUPS.keys()), key="al_group")
        event_types_filter = _EVENT_GROUPS[group]   # None means all
    with f2:
        client_names = ["All clients"] + _audit.get_client_names()
        client_sel   = st.selectbox("Client", client_names, key="al_client")
        client_filter = None if client_sel == "All clients" else client_sel
    with f3:
        search = st.text_input("🔍 Search summary", placeholder="e.g. 'portfolio updated'", label_visibility="collapsed", key="al_search")

    d1, d2, d3 = st.columns([2, 2, 2])
    with d1:
        date_from = st.date_input("From", value=date.today() - timedelta(days=30), key="al_from")
    with d2:
        date_to = st.date_input("To", value=date.today(), key="al_to")
    with d3:
        limit = st.selectbox("Max rows", [50, 100, 200, 500], index=1, key="al_limit")

    return event_types_filter, client_filter, search.strip() or None, date_from, date_to, limit


# ── Log table ─────────────────────────────────────────────────────────────────

def _render_log_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No audit entries match the current filters.")
        return

    st.markdown(
        f'<p style="font-size:0.8rem;color:rgba(255,255,255,0.4);">'
        f'Showing <b style="color:#a78bfa">{len(df)}</b> entries</p>',
        unsafe_allow_html=True,
    )

    # Display table (non-interactive columns)
    display = df[["created_at", "label", "client_name", "severity", "summary"]].copy()
    display.columns = ["Timestamp", "Event Type", "Client", "Severity", "Summary"]
    display["Timestamp"] = display["Timestamp"].str[:19].str.replace("T", " ")
    display["Client"]    = display["Client"].fillna("—")
    display["Severity"]  = display["Severity"].fillna("—")
    display["Summary"]   = display["Summary"].str[:80]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp":  st.column_config.TextColumn(width="small"),
            "Event Type": st.column_config.TextColumn(width="medium"),
            "Client":     st.column_config.TextColumn(width="small"),
            "Severity":   st.column_config.TextColumn(width="small"),
            "Summary":    st.column_config.TextColumn(width="large"),
        },
    )

    # Detail expander per entry
    if st.checkbox("Show entry details", key="al_show_detail"):
        st.markdown("##### Entry Details")
        for _, row in df.head(20).iterrows():
            with st.expander(
                f"**[{row['label']}]** {row['summary'][:60]}{'…' if len(row['summary']) > 60 else ''}  "
                f"*{row['created_at'][:19] if row['created_at'] else ''}*"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Event:** {row['label']}")
                    st.markdown(f"**Client:** {row.get('client_name') or '—'}")
                    st.markdown(f"**Severity:** {row.get('severity') or '—'}")
                    st.markdown(f"**Rule ID:** {row.get('rule_id') or '—'}")
                with col2:
                    st.markdown(f"**Summary:** {row['summary']}")
                    detail = row.get("detail")
                    if detail:
                        try:
                            parsed = json.loads(detail)
                            st.json(parsed)
                        except Exception:
                            st.text(detail)


# ── CSV export ────────────────────────────────────────────────────────────────

def _render_export(df: pd.DataFrame) -> None:
    if df.empty:
        return
    csv = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Export to CSV",
        data=csv,
        file_name=f"audit_log_{date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Prune panel ───────────────────────────────────────────────────────────────

def _render_prune() -> None:
    with st.expander("🗂️ Log Retention Management"):
        st.warning("Deleting old entries is irreversible. Use with caution.")
        keep_days = st.number_input(
            "Keep entries from the last N days",
            min_value=7, max_value=3650, value=90,
            key="al_keep_days",
        )
        if st.button("🗑️ Prune Old Entries", type="secondary", key="al_prune"):
            deleted = _audit.clear_old_entries(keep_days=int(keep_days))
            if deleted:
                st.success(f"✅ Deleted {deleted} entries older than {keep_days} days.")
            else:
                st.info(f"No entries older than {keep_days} days found.")
            st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## 📋 Audit Log")
    st.markdown("*Complete chronological record of all system events*")
    st.divider()

    # KPI bar
    stats = _audit.get_stats()
    _render_kpis(stats)
    st.divider()

    # Charts row
    if stats["total_entries"] > 0:
        ch1, ch2 = st.columns([3, 2])
        with ch1:
            _render_trend(_audit)
        with ch2:
            st.markdown("##### Event Distribution")
            _render_distribution(stats.get("by_type", {}))
        st.divider()

    # Filters
    st.markdown("##### 🔍 Filter Entries")
    event_types_filter, client_filter, search_query, date_from, date_to, limit = _render_filters()
    st.divider()

    # Build date-range where clause by injecting into search
    # AuditRepository.get_recent_for_view handles event_types, client_name, search_query
    df = _audit.get_recent_for_view(
        limit=int(limit),
        event_types=event_types_filter,
        client_name=client_filter,
        search_query=search_query,
    )

    # Client-side date filter (simpler than adding to SQL)
    if not df.empty and "created_at" in df.columns:
        df["_date"] = pd.to_datetime(df["created_at"]).dt.date
        df = df[(df["_date"] >= date_from) & (df["_date"] <= date_to)]
        df = df.drop(columns=["_date"])

    # Table + export
    col_table, col_export = st.columns([5, 1])
    with col_export:
        _render_export(df)

    _render_log_table(df)
    st.divider()

    # Retention management
    _render_prune()
