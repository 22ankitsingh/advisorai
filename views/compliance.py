"""
pages/compliance.py
────────────────────
Compliance alerts page.

Shows:
  - Alert summary KPIs (by severity)
  - Full alert list with resolve action
  - Alert type breakdown chart
"""

import streamlit as st
import plotly.express as px
import pandas as pd

from services.database import get_db_connection
from compliance.alert_service import AlertService
from utils.helpers import apply_dark_theme, SEVERITY_COLOURS
from utils.client_resolver import get_selected_client
from utils.logger import get_logger

logger = get_logger(__name__)

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@st.cache_resource
def _get_alert_service() -> AlertService:
    """Cached AlertService singleton — created once per Streamlit process."""
    return AlertService()


def _load_alerts(show_resolved: bool = False) -> pd.DataFrame:
    """Fetch compliance alerts, optionally including resolved ones."""
    where = "" if show_resolved else "WHERE ca.is_resolved = 0"
    with get_db_connection() as conn:
        rows = conn.execute(f"""
            SELECT ca.id, ca.title, ca.description, ca.alert_type, ca.severity,
                   ca.is_resolved, ca.created_at, ca.resolved_at,
                   COALESCE(c.name, 'N/A') AS client_name
            FROM compliance_alerts ca
            LEFT JOIN clients c ON c.id = ca.client_id
            {where}
            ORDER BY
                CASE ca.severity
                    WHEN 'critical' THEN 1
                    WHEN 'high'     THEN 2
                    WHEN 'medium'   THEN 3
                    ELSE 4
                END,
                ca.created_at DESC
        """).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=[
        "id", "title", "description", "alert_type", "severity",
        "is_resolved", "created_at", "resolved_at", "client_name",
    ])


def _resolve_alert(alert_id: int) -> bool:
    """Resolve an alert via AlertService (handles DB update + audit logging)."""
    return _get_alert_service().resolve_alert(alert_id)


def _render_scan_section(ref) -> None:
    """Compliance scan controls — runs AlertService rules and persists violations."""
    svc = _get_alert_service()
    with st.expander("🔍 Run Compliance Scan", expanded=False):
        st.markdown(
            "Runs all **10 compliance rules** against live portfolio data and persists "
            "new violations to the database. Already-open duplicate alerts are skipped."
        )
        scan_col, all_col = st.columns(2)

        with scan_col:
            label = f"Scan: {ref.name}" if ref else "Select a client first"
            if st.button(
                f"⚡ {label}", type="primary",
                disabled=(ref is None),
                use_container_width=True, key="btn_scan_client",
            ):
                with st.spinner(f"Scanning {ref.name}..."):
                    summary = svc.run_for_client(ref.mock_key, persist=True)
                if summary.violations:
                    sev_text = ", ".join(f"{v} {k}" for k, v in summary.by_severity.items())
                    st.warning(
                        f"**{summary.violations}** violation(s) ({sev_text}) · "
                        f"{summary.new_saved} new alert(s) saved."
                    )
                else:
                    st.success(f"All {summary.total_rules} rules passed for {summary.client_name}.")
                st.rerun()

        with all_col:
            if st.button("🌐 Scan All Clients", use_container_width=True, key="btn_scan_all"):
                with st.spinner("Scanning all clients..."):
                    summaries = svc.run_for_all_clients(persist=True)
                total_v = sum(s.violations for s in summaries)
                total_n = sum(s.new_saved for s in summaries)
                if total_v:
                    st.warning(
                        f"**{total_v}** violation(s) across {len(summaries)} clients · "
                        f"{total_n} new alert(s) saved."
                    )
                else:
                    st.success(f"All clients passed. ({len(summaries)} scanned)")
                st.rerun()


def render() -> None:
    """Render the Compliance Centre page."""
    st.markdown("## ⚠️ Compliance Centre")
    st.markdown("*Rule-based compliance monitoring — powered by the Phase 5 engine*")
    st.divider()

    ref = get_selected_client()
    _render_scan_section(ref)
    st.divider()

    show_resolved = st.toggle("Show resolved alerts", value=False)
    alerts_df = _load_alerts(show_resolved=show_resolved)

    # ── KPI row ───────────────────────────────────────────────────────────────
    open_df = alerts_df[alerts_df["is_resolved"] == 0] if not alerts_df.empty else pd.DataFrame()

    total_open  = len(open_df)
    critical    = len(open_df[open_df["severity"] == "critical"]) if not open_df.empty else 0
    high        = len(open_df[open_df["severity"] == "high"]) if not open_df.empty else 0
    resolved    = len(alerts_df[alerts_df["is_resolved"] == 1]) if not alerts_df.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("⚠️ Open Alerts",  total_open,  delta_color="off")
    with k2:
        st.metric("🔴 Critical",     critical,    delta_color="inverse" if critical else "off")
    with k3:
        st.metric("🟠 High",         high,         delta_color="off")
    with k4:
        st.metric("✅ Resolved",     resolved)

    st.divider()

    if alerts_df.empty:
        st.success("🎉 No compliance alerts to display.")
        return

    # ── Chart + Alert list ────────────────────────────────────────────────────
    chart_col, list_col = st.columns([1, 2])

    with chart_col:
        st.markdown("#### 📊 Alert Type Breakdown")
        type_counts = open_df["alert_type"].value_counts().reset_index() if not open_df.empty else pd.DataFrame()
        if not type_counts.empty:
            type_counts.columns = ["Type", "Count"]
            type_counts["Type"] = type_counts["Type"].str.replace("_", " ").str.title()
            fig = px.bar(
                type_counts,
                x="Count",
                y="Type",
                orientation="h",
                color="Count",
                color_continuous_scale=["#4f46e5", "#ff4757"],
                text="Count",
            )
            apply_dark_theme(fig)
            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open alerts to chart.")

    with list_col:
        st.markdown("#### 📋 Alert Details")

        # Filter by client if one is selected (ref already set at top of render)
        display_df = alerts_df.copy()
        if ref:
            filtered = display_df[display_df["client_name"] == ref.name]
            if not filtered.empty:
                st.caption(f"📌 Filtered to: **{ref.name}**")
                display_df = filtered

        for _, row in display_df.iterrows():
            sev    = row["severity"]
            colour = SEVERITY_COLOURS.get(sev, ("#888", "#fff"))[0]
            resolved_tag = "✅ Resolved" if row["is_resolved"] else ""

            with st.expander(
                f"{'~~' if row['is_resolved'] else ''}{row['title']}{'~~' if row['is_resolved'] else ''}"
                f"  {resolved_tag}",
                expanded=False,
            ):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f"<span style='background:{colour}30; color:{colour}; "
                        f"padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.8rem;'>"
                        f"{'🔴' if sev=='critical' else '🟠' if sev=='high' else '🟡' if sev=='medium' else '🟢'} "
                        f"{sev.upper()}</span>  "
                        f"&nbsp; 🏷️ **{row['alert_type'].replace('_', ' ').title()}**  "
                        f"&nbsp; 👤 **{row['client_name']}**",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"> {row['description']}")
                    st.caption(f"Created: {row['created_at'][:10]}")

                with col_b:
                    if not row["is_resolved"]:
                        if st.button(
                            "✔ Resolve",
                            key=f"resolve_{row['id']}",
                            type="primary",
                        ):
                            _resolve_alert(row["id"])
                            st.success("Alert resolved!")
                            st.rerun()
                    else:
                        st.markdown(
                            f"<span style='color:#2ed573; font-size:0.85rem;'>✅ Resolved<br>"
                            f"<small>{(row['resolved_at'] or '')[:10]}</small></span>",
                            unsafe_allow_html=True,
                        )
