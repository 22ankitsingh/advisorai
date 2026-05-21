"""
pages/dashboard.py
───────────────────
Dashboard page — the home screen of Advisor AI.

Shows:
  - KPI summary cards (total AUM, clients, alerts, avg return)
  - Client list table
  - AUM distribution pie chart
  - Risk profile breakdown
  - Recent compliance alerts
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from services.database import get_db_connection
from utils.helpers import severity_badge, apply_dark_theme, RISK_PROFILE_COLOURS, fmt_usd
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_dashboard_data() -> dict:
    """Fetch all data needed for the dashboard in a single DB pass."""
    with get_db_connection() as conn:
        clients = pd.DataFrame(
            conn.execute("""
                SELECT c.id, c.name, c.email, c.risk_profile, c.aum,
                       COUNT(DISTINCT p.id) AS portfolios,
                       COUNT(DISTINCT ca.id) AS open_alerts
                FROM clients c
                LEFT JOIN portfolios p ON p.client_id = c.id
                LEFT JOIN compliance_alerts ca ON ca.client_id = c.id AND ca.is_resolved = 0
                GROUP BY c.id
                ORDER BY c.aum DESC
            """).fetchall(),
            columns=["id", "name", "email", "risk_profile", "aum", "portfolios", "open_alerts"],
        )

        alerts = pd.DataFrame(
            conn.execute("""
                SELECT ca.title, ca.severity, ca.alert_type, ca.created_at, c.name AS client_name
                FROM compliance_alerts ca
                JOIN clients c ON c.id = ca.client_id
                WHERE ca.is_resolved = 0
                ORDER BY
                    CASE ca.severity
                        WHEN 'critical' THEN 1
                        WHEN 'high'     THEN 2
                        WHEN 'medium'   THEN 3
                        ELSE 4
                    END,
                    ca.created_at DESC
                LIMIT 5
            """).fetchall(),
            columns=["title", "severity", "alert_type", "created_at", "client_name"],
        )

    return {"clients": clients, "alerts": alerts}



def render() -> None:
    """Render the Dashboard page."""
    st.markdown("## 🏠 Dashboard")
    st.markdown("*Your financial advisory command centre*")
    st.divider()

    data = _load_dashboard_data()
    clients_df = data["clients"]
    alerts_df  = data["alerts"]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_aum    = clients_df["aum"].sum()
    total_clients = len(clients_df)
    total_alerts = clients_df["open_alerts"].sum()
    # Avg gain from DB portfolio analytics (no mock_data dependency)
    try:
        from database.repositories.portfolio_repository import PortfolioRepository
        from portfolio.analytics import portfolio_summary
        pr = PortfolioRepository()
        gains = []
        for _, row in clients_df.iterrows():
            ports = pr.get_for_client(int(row["id"]))
            for p in ports:
                df = pr.get_holdings_df(p["id"])
                if not df.empty:
                    s = portfolio_summary(df)
                    gains.append(s["total_gain_pct"])
        avg_gain = round(sum(gains) / len(gains), 1) if gains else 0.0
    except Exception:
        avg_gain = 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="💰 Total AUM",
            value=f"${total_aum:,.0f}",
            delta=f"+{avg_gain}% YTD",
        )
    with kpi2:
        st.metric(label="👥 Total Clients", value=total_clients)
    with kpi3:
        st.metric(
            label="⚠️ Open Alerts",
            value=int(total_alerts),
            delta="Needs attention" if total_alerts > 0 else "All clear",
            delta_color="inverse",
        )
    with kpi4:
        st.metric(label="📈 Avg Portfolio Return", value=f"+{avg_gain}%", delta="YTD")

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("#### 📊 Client AUM Distribution")
        fig_pie = px.pie(
            clients_df,
            names="name",
            values="aum",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_pie.update_traces(
            textposition="outside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>AUM: $%{value:,.0f}<extra></extra>",
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8f0", family="Inter"),
            legend=dict(
                orientation="v",
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("#### 🎯 Risk Profile Breakdown")
        risk_counts = clients_df["risk_profile"].value_counts().reset_index()
        risk_counts.columns = ["Risk Profile", "Count"]

        colour_map = {
            "conservative": "#2ed573",
            "moderate":     "#ffa502",
            "aggressive":   "#ff4757",
        }
        fig_bar = px.bar(
            risk_counts,
            x="Risk Profile",
            y="Count",
            color="Risk Profile",
            color_discrete_map=colour_map,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8f0", family="Inter"),
            showlegend=False,
            xaxis=dict(showgrid=False, color="#e8e8f0"),
            yaxis=dict(showgrid=False, color="#e8e8f0", dtick=1),
            margin=dict(t=30, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Client table + Alerts ─────────────────────────────────────────────────
    tbl_col, alert_col = st.columns([1.6, 1])

    with tbl_col:
        st.markdown("#### 👥 Client Overview")
        display_df = clients_df[["name", "risk_profile", "aum", "open_alerts"]].copy()
        display_df.columns = ["Client", "Risk Profile", "AUM ($)", "Open Alerts"]
        display_df["AUM ($)"] = display_df["AUM ($)"].apply(lambda x: f"${x:,.0f}")
        display_df["Risk Profile"] = display_df["Risk Profile"].str.title()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Open Alerts": st.column_config.NumberColumn(format="%d ⚠️"),
            },
        )

    with alert_col:
        st.markdown("#### ⚠️ Recent Alerts")
        if alerts_df.empty:
            st.success("✅ No open compliance alerts.")
        else:
            for _, row in alerts_df.iterrows():
                with st.container():
                    st.markdown(
                        f"{severity_badge(row['severity'])} &nbsp;"
                        f"**{row['title'][:45]}{'...' if len(row['title'])>45 else ''}**",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"👤 {row['client_name']}")
                    st.markdown("---")
