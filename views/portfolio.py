"""
pages/portfolio.py
───────────────────
Portfolio analysis page — Phase 6 upgrade: fully DB-driven.

Data source: DB via PortfolioRepository (primary) + mock_data NAV fallback
             for legacy demo clients that still have mock_key set.
Analytics:   portfolio/analytics.py  — summary, allocation, performance, drift
Risk:        portfolio/risk_engine.py — composite risk score, flags
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from portfolio.analytics import (
    portfolio_summary,
    allocation_by_class,
    allocation_by_sector,
    performance_metrics,
    rolling_returns,
    top_holdings,
    drift_analysis,
)
from portfolio.risk_engine import compute_risk_report
from database.repositories.portfolio_repository import PortfolioRepository
from utils.client_resolver import get_selected_client
from utils.helpers import (
    fmt_usd, fmt_pct, apply_dark_theme,
    severity_badge, RISK_PROFILE_COLOURS, ASSET_CLASS_COLOURS,
)
from utils.logger import get_logger

logger = get_logger(__name__)
_port_repo = PortfolioRepository()


# ─────────────────────────────────────────────────────────────────────────────
# Data loader (cached per client to avoid recomputing on every rerun)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _load_portfolio_data(db_id: int, mock_key: str | None, risk_profile: str) -> dict:
    """
    Load and compute all portfolio data for a client.

    Holdings come from the DB (PortfolioRepository).  NAV history and drift
    targets come from mock_data only when a mock_key is available.

    Args:
        db_id:        Client's integer ID in the DB.
        mock_key:     Optional legacy mock_data key (None for new DB-only clients).
        risk_profile: Client's risk profile string.

    Returns:
        Dict with keys: holdings, summary, alloc_class, alloc_sector,
        perf, rolling, drift, risk_report, top5.
    """
    import pandas as pd

    # ── Holdings from DB ──────────────────────────────────────────────────────
    portfolios = _port_repo.get_for_client(db_id)
    if portfolios:
        holdings = _port_repo.get_holdings_df(portfolios[0]["id"])
    else:
        holdings = pd.DataFrame()

    # Fall back to mock_data holdings if DB is empty and client has a mock_key
    if holdings.empty and mock_key:
        from portfolio.mock_data import get_holdings_df as _mock_holdings
        holdings = _mock_holdings(mock_key)

    # ── NAV history (mock_data or empty) ─────────────────────────────────────
    nav_df = pd.DataFrame()
    if mock_key:
        from portfolio.mock_data import get_nav_history
        nav_df = get_nav_history(mock_key, weeks=52)

    # ── Target allocation (mock_data or even-split fallback) ─────────────────
    target: dict = {}
    if mock_key:
        from portfolio.mock_data import CLIENTS
        target = CLIENTS.get(mock_key, {}).get("target_allocation", {})
    if not target and not holdings.empty:
        # Even split across all asset classes present
        classes = holdings["asset_class"].unique()
        share   = round(100 / len(classes), 1)
        target  = {c: share for c in classes}

    if holdings.empty:
        empty = pd.DataFrame()
        return {
            "holdings":    empty,
            "summary":     {"total_value":0,"total_cost":0,"total_gain_loss":0,
                            "total_gain_pct":0,"num_positions":0,"num_asset_classes":0,
                            "largest_position_weight":0,"cash_weight":0},
            "alloc_class": empty,
            "alloc_sector":empty,
            "perf":        {},
            "rolling":     {},
            "drift":       empty,
            "risk_report": compute_risk_report(pd.DataFrame(
                columns=["ticker","asset_class","market_value","cost_basis",
                         "gain_loss","gain_pct","weight","sector"]
            ), "db_client", risk_profile=risk_profile),
            "top5":        empty,
        }

    rr = compute_risk_report(holdings, mock_key or "db_client",
                             risk_profile=risk_profile if not mock_key else None)

    return {
        "holdings":    holdings,
        "summary":     portfolio_summary(holdings),
        "alloc_class": allocation_by_class(holdings),
        "alloc_sector":allocation_by_sector(holdings),
        "perf":        performance_metrics(nav_df),
        "rolling":     rolling_returns(nav_df),
        "drift":       drift_analysis(holdings, target),
        "risk_report": rr,
        "top5":        top_holdings(holdings, n=5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sub-renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_kpis(data: dict) -> None:
    """KPI row: value, gain/loss, risk score, Sharpe, drawdown."""
    s   = data["summary"]
    pm  = data["perf"]
    rr  = data["risk_report"]

    k1, k2, k3, k4, k5 = st.columns(5)

    delta_pct   = s["total_gain_pct"]
    delta_colour = "normal" if delta_pct >= 0 else "inverse"

    with k1:
        st.metric("💼 Portfolio Value", fmt_usd(s["total_value"]))
    with k2:
        st.metric(
            "📈 Total Return",
            fmt_pct(delta_pct, show_sign=True),
            delta=fmt_usd(s["total_gain_loss"], show_sign=True),
            delta_color=delta_colour,
        )
    with k3:
        risk_colour = {
            "Very Low": "normal", "Low": "normal",
            "Moderate": "off", "High": "inverse", "Very High": "inverse",
        }.get(rr.risk_band, "off")
        st.metric(
            "⚠️ Risk Score",
            f"{rr.overall_score:.0f}/100",
            delta=rr.risk_band,
            delta_color=risk_colour,
        )
    with k4:
        sharpe = pm.get("sharpe_ratio", 0)
        st.metric(
            "📊 Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="Good" if sharpe > 1 else "Low" if sharpe < 0 else "Fair",
            delta_color="normal" if sharpe > 1 else "inverse" if sharpe < 0 else "off",
        )
    with k5:
        dd = pm.get("max_drawdown_pct", 0)
        st.metric(
            "📉 Max Drawdown",
            f"{dd:.1f}%",
            delta_color="off",
        )


def _render_allocation_charts(data: dict) -> None:
    """Asset allocation donut + sector exposure bar chart."""
    alloc_class  = data["alloc_class"]
    alloc_sector = data["alloc_sector"]

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### 🥧 Asset Allocation")

        colour_list = [
            ASSET_CLASS_COLOURS.get(ac, "#888")
            for ac in alloc_class["asset_class"]
        ]
        fig = px.pie(
            alloc_class,
            names="asset_class",
            values="market_value",
            hole=0.52,
            color="asset_class",
            color_discrete_map=ASSET_CLASS_COLOURS,
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>",
        )
        apply_dark_theme(fig)
        fig.update_layout(
            showlegend=True,
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_right:
        st.markdown("#### 🏭 Sector Exposure")

        if alloc_sector.empty:
            st.info("No sector data available.")
        else:
            sector_df = alloc_sector.sort_values("market_value", ascending=True)
            fig = px.bar(
                sector_df,
                x="market_value",
                y="sector",
                orientation="h",
                color="market_value",
                color_continuous_scale=["#4f46e5", "#7c3aed", "#a78bfa"],
                labels={"market_value": "Value ($)", "sector": ""},
            )
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>"
            )
            apply_dark_theme(fig)
            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_performance_section(data: dict) -> None:
    """Rolling returns + allocation drift side by side."""
    pm      = data["perf"]
    rolling = data["rolling"]
    drift   = data["drift"]

    perf_col, drift_col = st.columns(2)

    with perf_col:
        st.markdown("#### 📅 Rolling Returns")
        periods = ["4W", "13W", "26W", "52W"]
        labels  = ["4 Week", "13 Week", "26 Week", "52 Week"]
        values  = [rolling.get(p) for p in periods]

        roll_data = [
            {"Period": l, "Return (%)": v}
            for l, v in zip(labels, values) if v is not None
        ]
        if roll_data:
            roll_df = pd.DataFrame(roll_data)
            colours = ["#2ed573" if r >= 0 else "#ff4757" for r in roll_df["Return (%)"]]
            fig = go.Figure(go.Bar(
                x=roll_df["Period"],
                y=roll_df["Return (%)"],
                marker_color=colours,
                text=[f"{v:+.1f}%" for v in roll_df["Return (%)"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>%{y:+.2f}%<extra></extra>",
            ))
            apply_dark_theme(fig, height=280)
            fig.update_layout(
                showlegend=False,
                margin=dict(t=30, b=10, l=10, r=10),
                yaxis_title="Return (%)",
            )
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for rolling returns.")

    with drift_col:
        st.markdown("#### 🎯 Allocation vs. Target")
        if drift.empty:
            st.info("No target allocation defined.")
        else:
            drift_display = drift.copy()
            drift_display["actual_weight"]  = drift_display["actual_weight"].apply(lambda x: f"{x:.1f}%")
            drift_display["target_weight"]  = drift_display["target_weight"].apply(lambda x: f"{x:.1f}%")
            drift_display["drift"]          = drift_display["drift"].apply(lambda x: f"{x:+.1f}%")
            drift_display.columns = ["Asset Class", "Actual", "Target", "Drift", "Status"]
            st.dataframe(drift_display, hide_index=True, use_container_width=True)

            # Drift bar chart
            fig = px.bar(
                drift,
                x="asset_class",
                y="drift",
                color="drift",
                color_continuous_scale=["#ff4757", "#ffa502", "#2ed573"],
                labels={"asset_class": "", "drift": "Drift (%)"},
                text=drift["drift"].apply(lambda x: f"{x:+.1f}%"),
            )
            fig.update_traces(textposition="outside")
            apply_dark_theme(fig, height=200)
            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            st.plotly_chart(fig, use_container_width=True)


def _render_holdings_table(data: dict) -> None:
    """Full holdings table with all computed columns."""
    holdings = data["holdings"]

    display = holdings[[
        "ticker", "name", "asset_class", "qty",
        "avg_cost", "price", "market_value", "gain_loss", "gain_pct", "weight",
    ]].copy()

    display.columns = [
        "Ticker", "Name", "Class", "Qty",
        "Avg Cost", "Price", "Market Value", "Gain/Loss ($)", "Gain/Loss (%)", "Weight (%)",
    ]

    display["Avg Cost"]      = display["Avg Cost"].apply(lambda x: f"${x:,.2f}")
    display["Price"]         = display["Price"].apply(lambda x: f"${x:,.2f}")
    display["Market Value"]  = display["Market Value"].apply(lambda x: f"${x:,.0f}")
    display["Gain/Loss ($)"] = display["Gain/Loss ($)"].apply(lambda x: f"${x:+,.0f}")
    display["Gain/Loss (%)"] = display["Gain/Loss (%)"].apply(lambda x: f"{x:+.1f}%")
    display["Weight (%)"]    = display["Weight (%)"].apply(lambda x: f"{x:.1f}%")
    display["Qty"]           = display["Qty"].apply(lambda x: f"{x:,.0f}")
    display["Class"]         = display["Class"].str.title()

    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_performers(data: dict) -> None:
    """Top 3 and bottom 3 positions by gain %."""
    holdings  = data["holdings"]
    non_cash  = holdings[holdings["ticker"] != "CASH"].copy()

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### 🏆 Top Performers")
        top = non_cash.nlargest(3, "gain_pct")[["ticker", "name", "gain_pct"]]
        for _, row in top.iterrows():
            st.markdown(
                f"**{row['ticker']}** — {row['name'][:30]} "
                f"<span style='color:#2ed573; font-weight:600;'>{row['gain_pct']:+.1f}%</span>",
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("#### 📉 Underperformers")
        bottom = non_cash.nsmallest(3, "gain_pct")[["ticker", "name", "gain_pct"]]
        for _, row in bottom.iterrows():
            colour = "#ff4757" if row["gain_pct"] < 0 else "#ffa502"
            st.markdown(
                f"**{row['ticker']}** — {row['name'][:30]} "
                f"<span style='color:{colour}; font-weight:600;'>{row['gain_pct']:+.1f}%</span>",
                unsafe_allow_html=True,
            )


def _render_risk_flags(data: dict) -> None:
    """Display active risk flags from the risk engine."""
    flags = data["risk_report"].flags
    if not flags:
        st.success("✅ No active risk flags for this portfolio.")
        return

    st.markdown(f"**{len(flags)} active risk flag(s):**")
    for flag in flags:
        severity_colours = {
            "critical": "#ff4757", "high": "#ff6b35",
            "medium": "#ffa502",   "low":  "#2ed573",
        }
        c = severity_colours.get(flag.severity, "#aaa")
        st.markdown(
            f'<div style="border-left:3px solid {c}; padding:8px 14px; '
            f'margin:4px 0; background:rgba(255,255,255,0.03); border-radius:0 8px 8px 0;">'
            f'<span style="color:{c}; font-weight:700; font-size:0.8rem;">'
            f'{getattr(flag, "emoji", "⚠️")} {flag.severity.upper()} · {flag.category}</span><br>'
            f'<span style="font-size:0.88rem; color:#e8e8f0;">{flag.title}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Portfolio Analysis page."""
    st.markdown("## 📊 Portfolio Analysis")
    st.markdown("*Powered by Phase 6 — DB-driven holdings, live analytics*")
    st.divider()

    ref = get_selected_client()
    if ref is None:
        st.info("👈 Please select a client from the sidebar to view their portfolio.")
        return

    with st.spinner("Loading portfolio analytics..."):
        try:
            data = _load_portfolio_data(ref.db_id, ref.mock_key, ref.risk_profile)
        except Exception as exc:
            st.error(f"Failed to load portfolio data: {exc}")
            logger.error("Portfolio load failed for %s: %s", ref.db_id, exc)
            return

    rr = data["risk_report"]
    pm = data["perf"]

    # ── Client header ─────────────────────────────────────────────────────────
    profile_colour = RISK_PROFILE_COLOURS.get(ref.risk_profile, "#a78bfa")
    st.markdown(
        f"### 👤 {ref.name} "
        f"<span style='background:{profile_colour}20; color:{profile_colour}; "
        f"padding:3px 12px; border-radius:20px; font-size:0.85rem; "
        f"font-weight:600; border:1px solid {profile_colour}50;'>"
        f"{ref.risk_profile.title()} Risk</span>",
        unsafe_allow_html=True,
    )
    if data["holdings"].empty:
        st.info("📂 No holdings found. Add holdings via **Portfolio Mgmt** to see analytics.")
        return

    # ── KPI cards ─────────────────────────────────────────────────────────────
    _render_kpis(data)
    st.divider()

    # ── Allocation charts ─────────────────────────────────────────────────────
    _render_allocation_charts(data)
    st.divider()

    # ── Performance + drift ───────────────────────────────────────────────────
    _render_performance_section(data)
    st.divider()

    # ── Holdings detail table ─────────────────────────────────────────────────
    st.markdown("#### 📋 Holdings Detail")
    _render_holdings_table(data)
    st.divider()

    # ── Top / bottom performers ───────────────────────────────────────────────
    _render_performers(data)
    st.divider()

    # ── Risk flags ────────────────────────────────────────────────────────────
    st.markdown("#### 🚨 Risk Flags")
    _render_risk_flags(data)

    # ── Performance summary footer ────────────────────────────────────────────
    ann = pm.get("annualised_return", 0)
    vol = pm.get("volatility_annual", 0)
    st.markdown(
        f"<p style='font-size:0.72rem; color:rgba(255,255,255,0.3); margin-top:1rem;'>"
        f"52-week annualised return: {ann:+.1f}% · Volatility: {vol:.1f}% · "
        f"Risk score: {rr.overall_score:.0f}/100 ({rr.risk_band})"
        f"</p>",
        unsafe_allow_html=True,
    )
