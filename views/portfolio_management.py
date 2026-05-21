"""
views/portfolio_management.py
───────────────────────────────
Portfolio & Holdings CRUD — Phase 3.

Features:
  - Create / rename portfolios
  - Add, edit, delete holdings
  - Record transactions (buy/sell)
  - CSV import (bulk replace holdings)
  - Live analytics preview (allocation chart + risk score)
  - Auto-recalculates portfolio total and client AUM on every change
"""

from __future__ import annotations
import io
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px

from database.repositories.client_repository import ClientRepository
from database.repositories.portfolio_repository import PortfolioRepository
from database.repositories.holding_repository import HoldingRepository
from database.repositories.transaction_repository import TransactionRepository, TRANSACTION_TYPES
from database.repositories.audit_repository import get_audit_repo
from utils.client_resolver import get_selected_client
from utils.helpers import fmt_usd, fmt_large, apply_dark_theme, ASSET_CLASS_COLOURS, RISK_PROFILE_COLOURS
from utils.logger import get_logger

logger = get_logger(__name__)

_client_repo  = ClientRepository()
_port_repo    = PortfolioRepository()
_hold_repo    = HoldingRepository()
_txn_repo     = TransactionRepository()
_audit        = get_audit_repo()

ASSET_CLASSES = ["equity", "etf", "bond", "alternative", "cash"]

# ── Session state ─────────────────────────────────────────────────────────────
_MODE       = "pm_mode"        # list | add_holding | edit_holding | add_txn | csv_import | add_portfolio
_PORT_ID    = "pm_portfolio_id"
_HOLD_ID    = "pm_edit_hold_id"
_MSG        = "pm_message"

def _init():
    for k, v in {_MODE:"list", _PORT_ID:None, _HOLD_ID:None, _MSG:None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _msg(kind, text): st.session_state[_MSG] = (kind, text)

def _show_msg():
    m = st.session_state.get(_MSG)
    if m:
        kind, text = m
        getattr(st, kind)(text)
        st.session_state[_MSG] = None

def _sync(portfolio_id: int, client_id: int):
    """Recalculate portfolio total and sync client AUM."""
    total = _port_repo.update_total_value(portfolio_id)
    _client_repo.update_aum(client_id, total)


# ── Inline analytics preview ──────────────────────────────────────────────────

def _render_analytics(portfolio_id: int, risk_profile: str):
    holdings_df = _port_repo.get_holdings_df(portfolio_id)
    if holdings_df.empty:
        st.info("No holdings yet. Add holdings to see analytics.")
        return

    from portfolio.analytics import portfolio_summary, allocation_by_class
    from portfolio.risk_engine import compute_risk_report

    s = portfolio_summary(holdings_df)
    alloc = allocation_by_class(holdings_df)

    try:
        rr = compute_risk_report(holdings_df, "db_client", risk_profile=risk_profile)
        risk_score = f"{rr.overall_score:.0f}/100 ({rr.risk_band})"
        risk_colour = rr.risk_colour
    except Exception:
        risk_score = "—"
        risk_colour = "#a78bfa"

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("💼 Value", fmt_large(s["total_value"]))
    with k2: st.metric("📈 Return", f"{s['total_gain_pct']:+.1f}%", delta=fmt_usd(s["total_gain_loss"], show_sign=True))
    with k3: st.metric("🔢 Positions", s["num_positions"])
    with k4: st.metric("⚠️ Risk Score", risk_score)

    if not alloc.empty:
        fig = px.pie(alloc, names="asset_class", values="market_value",
                     hole=0.5, color="asset_class",
                     color_discrete_map=ASSET_CLASS_COLOURS)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>")
        apply_dark_theme(fig)
        fig.update_layout(margin=dict(t=20,b=20,l=10,r=10), showlegend=True, height=260)
        st.plotly_chart(fig, use_container_width=True)


# ── Holdings table ────────────────────────────────────────────────────────────

def _render_holdings(portfolio_id: int, client_id: int):
    holdings = _hold_repo.get_for_portfolio(portfolio_id)
    if not holdings:
        st.info("No holdings in this portfolio. Use **Add Holding** or **Import CSV** below.")
        return

    df = pd.DataFrame(holdings)
    disp = df[["ticker","asset_name","asset_class","quantity","avg_cost","current_price","market_value","sector"]].copy()
    disp.columns = ["Ticker","Name","Class","Qty","Avg Cost","Price","Market Value","Sector"]
    disp["Avg Cost"]     = disp["Avg Cost"].apply(lambda x: f"${x:,.2f}")
    disp["Price"]        = disp["Price"].apply(lambda x: f"${x:,.2f}")
    disp["Market Value"] = disp["Market Value"].apply(lambda x: f"${x:,.0f}")
    disp["Sector"]       = disp["Sector"].fillna("—")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Per-row actions
    st.markdown("**Edit / Delete:**")
    n_cols = min(len(holdings), 4)
    cols = st.columns(n_cols)
    for i, h in enumerate(holdings):
        with cols[i % n_cols]:
            if st.button(f"✏️ {h['ticker']}", key=f"he_{h['id']}", use_container_width=True):
                st.session_state[_MODE]    = "edit_holding"
                st.session_state[_HOLD_ID] = h["id"]
                st.rerun()
            if st.button(f"🗑️ {h['ticker']}", key=f"hd_{h['id']}", use_container_width=True):
                _hold_repo.delete(h["id"])
                _sync(portfolio_id, client_id)
                _audit.log_portfolio_updated(
                    "client", portfolio_id, action=f"holding_deleted:{h['ticker']}",
                    client_id=client_id,
                )
                _msg("success", f"Deleted holding **{h['ticker']}**.")
                st.rerun()


# ── Add / Edit holding form ───────────────────────────────────────────────────

def _render_holding_form(portfolio_id: int, client_id: int, existing: Optional[dict] = None):
    is_edit = existing is not None
    st.markdown(f"### {'✏️ Edit' if is_edit else '➕ Add'} Holding")

    with st.form("holding_form"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker *", value=existing["ticker"] if is_edit else "", placeholder="AAPL")
        with c2:
            asset_name = st.text_input("Name *", value=existing["asset_name"] if is_edit else "", placeholder="Apple Inc.")

        c3, c4 = st.columns(2)
        with c3:
            ac_default = ASSET_CLASSES.index(existing["asset_class"]) if is_edit and existing["asset_class"] in ASSET_CLASSES else 0
            asset_class = st.selectbox("Asset Class *", ASSET_CLASSES, index=ac_default, format_func=str.title)
        with c4:
            sector = st.text_input("Sector", value=existing.get("sector") or "" if is_edit else "", placeholder="Technology")

        c5, c6, c7 = st.columns(3)
        with c5:
            quantity = st.number_input("Quantity *", min_value=0.0, value=float(existing["quantity"]) if is_edit else 0.0, step=1.0)
        with c6:
            avg_cost = st.number_input("Avg Cost ($) *", min_value=0.0, value=float(existing["avg_cost"]) if is_edit else 0.0, step=0.01, format="%.2f")
        with c7:
            current_price = st.number_input("Current Price ($) *", min_value=0.0, value=float(existing["current_price"]) if is_edit else 0.0, step=0.01, format="%.2f")

        mv = quantity * current_price
        st.markdown(f"*Market value: **{fmt_usd(mv)}** · Gain/Loss: **{fmt_usd(mv - quantity * avg_cost, show_sign=True)}***")

        sub, cancel = st.columns([2,1])
        with sub:
            submitted = st.form_submit_button("💾 Save Holding", type="primary", use_container_width=True)
        with cancel:
            cancelled = st.form_submit_button("✖ Cancel", use_container_width=True)

        if cancelled:
            st.session_state[_MODE] = "list"; st.rerun()

        if submitted:
            if not ticker.strip() or not asset_name.strip():
                st.error("Ticker and Name are required."); st.stop()
            if quantity <= 0 or current_price <= 0:
                st.error("Quantity and Current Price must be > 0."); st.stop()

            try:
                if is_edit:
                    _hold_repo.update(
                        existing["id"],
                        ticker=ticker.strip().upper(),
                        asset_name=asset_name.strip(),
                        asset_class=asset_class,
                        quantity=quantity, avg_cost=avg_cost,
                        current_price=current_price,
                        sector=sector.strip() or None,
                    )
                    action = "holding_updated"
                else:
                    _hold_repo.upsert(
                        portfolio_id, ticker.strip().upper(), asset_name.strip(),
                        asset_class, quantity, avg_cost, current_price, sector.strip() or None,
                    )
                    action = f"holding_added:{ticker.upper()}"

                _sync(portfolio_id, client_id)
                _audit.log_portfolio_updated("client", portfolio_id, action=action, client_id=client_id)
                _msg("success", f"✅ Holding **{ticker.upper()}** {'updated' if is_edit else 'added'}.")
                st.session_state[_MODE] = "list"
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to save: {exc}")


# ── Transaction form ──────────────────────────────────────────────────────────

def _render_txn_form(portfolio_id: int, client_id: int):
    st.markdown("### 📝 Record Transaction")
    holdings = _hold_repo.get_for_portfolio(portfolio_id)
    tickers = [h["ticker"] for h in holdings] if holdings else []

    with st.form("txn_form"):
        c1, c2 = st.columns(2)
        with c1:
            txn_type = st.selectbox("Type *", list(TRANSACTION_TYPES), format_func=str.title)
        with c2:
            ticker_input = st.text_input("Ticker *", placeholder="AAPL")

        c3, c4, c5 = st.columns(3)
        with c3:
            qty = st.number_input("Quantity *", min_value=0.0, step=1.0)
        with c4:
            price = st.number_input("Price ($) *", min_value=0.0, step=0.01, format="%.2f")
        with c5:
            txn_date = st.date_input("Date")

        notes = st.text_input("Notes (optional)")

        sub, cancel = st.columns([2,1])
        with sub:
            submitted = st.form_submit_button("📝 Record", type="primary", use_container_width=True)
        with cancel:
            cancelled = st.form_submit_button("✖ Cancel", use_container_width=True)

        if cancelled:
            st.session_state[_MODE] = "list"; st.rerun()

        if submitted:
            if not ticker_input.strip() or qty <= 0 or price <= 0:
                st.error("Ticker, Quantity, and Price are required."); st.stop()
            try:
                _txn_repo.record(
                    portfolio_id, ticker_input.strip().upper(), txn_type,
                    qty, price, str(txn_date), notes or None,
                )
                _msg("success", f"✅ Recorded {txn_type} of {qty:.0f}x **{ticker_input.upper()}** @ ${price:.2f}")
                st.session_state[_MODE] = "list"
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to record: {exc}")


# ── CSV Import ────────────────────────────────────────────────────────────────

def _render_csv_import(portfolio_id: int, client_id: int, client_name: str):
    st.markdown("### 📂 Import Holdings from CSV")
    st.markdown("""
    Upload a CSV file with these columns *(order doesn't matter)*:
    `ticker`, `asset_name`, `asset_class`, `quantity`, `average_cost`, `current_price`, `sector` *(optional)*
    """)

    sample = pd.DataFrame([
        {"ticker":"AAPL","asset_name":"Apple Inc.","asset_class":"equity","quantity":100,"average_cost":155.00,"current_price":189.40,"sector":"Technology"},
        {"ticker":"BND","asset_name":"Vanguard Bond ETF","asset_class":"bond","quantity":200,"average_cost":72.00,"current_price":74.20,"sector":""},
    ])
    with st.expander("📋 Show sample CSV format"):
        st.dataframe(sample, hide_index=True, use_container_width=True)
        csv_bytes = sample.to_csv(index=False).encode()
        st.download_button("⬇️ Download Sample CSV", csv_bytes, "sample_holdings.csv", "text/csv")

    uploaded = st.file_uploader("Upload Holdings CSV", type=["csv"], key="csv_upload")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip().str.lower()

            required = {"ticker","asset_name","asset_class","quantity","current_price"}
            missing  = required - set(df.columns)
            if missing:
                st.error(f"Missing required columns: {missing}"); return

            # Normalise avg_cost / average_cost alias
            if "average_cost" in df.columns and "avg_cost" not in df.columns:
                df.rename(columns={"average_cost":"avg_cost"}, inplace=True)
            if "avg_cost" not in df.columns:
                df["avg_cost"] = df["current_price"]

            df = df.dropna(subset=["ticker","asset_class","quantity","current_price"])
            df["sector"] = df.get("sector", pd.Series(dtype=str)).fillna("").replace("", None)

            st.markdown(f"**Preview** — {len(df)} holding(s) found:")
            st.dataframe(df[["ticker","asset_name","asset_class","quantity","avg_cost","current_price","sector"]].head(20),
                         hide_index=True, use_container_width=True)

            warn_col, conf_col = st.columns([2,1])
            with warn_col:
                st.warning("⚠️ This will **replace all current holdings** in the portfolio.")
            with conf_col:
                if st.button("✅ Confirm Import", type="primary", use_container_width=True, key="csv_confirm"):
                    rows = df.to_dict("records")
                    _hold_repo.bulk_replace(portfolio_id, rows)
                    _sync(portfolio_id, client_id)
                    _audit.log_csv_import(client_name, portfolio_id, len(rows), client_id=client_id)
                    _msg("success", f"✅ Imported **{len(rows)}** holdings successfully.")
                    st.session_state[_MODE] = "list"
                    st.rerun()

        except Exception as exc:
            st.error(f"CSV parse error: {exc}")

    if st.button("✖ Cancel", key="csv_cancel"):
        st.session_state[_MODE] = "list"; st.rerun()


# ── Create portfolio form ─────────────────────────────────────────────────────

def _render_create_portfolio(client_id: int, client_name: str):
    st.markdown("### 🆕 Create Portfolio")
    with st.form("create_port"):
        name = st.text_input("Portfolio Name *", value="Main Portfolio", placeholder="e.g. Growth Portfolio")
        sub, cancel = st.columns([2,1])
        with sub:
            submitted = st.form_submit_button("✅ Create", type="primary", use_container_width=True)
        with cancel:
            cancelled = st.form_submit_button("✖ Cancel", use_container_width=True)
        if cancelled:
            st.session_state[_MODE] = "list"; st.rerun()
        if submitted:
            if not name.strip():
                st.error("Portfolio name is required."); st.stop()
            pid = _port_repo.create(client_id, name.strip())
            _audit.log_portfolio_created(client_name, name.strip(), client_id=client_id, portfolio_id=pid)
            _msg("success", f"✅ Portfolio **{name}** created.")
            st.session_state[_PORT_ID] = pid
            st.session_state[_MODE]    = "list"
            st.rerun()


# ── Transaction history ───────────────────────────────────────────────────────

def _render_txn_history(portfolio_id: int):
    df = _txn_repo.get_as_df(portfolio_id)
    if df.empty:
        st.caption("No transactions recorded yet.")
        return
    disp = df[["transaction_date","ticker","transaction_type","quantity","price","total_value","notes"]].copy()
    disp.columns = ["Date","Ticker","Type","Qty","Price","Total","Notes"]
    disp["Price"] = disp["Price"].apply(lambda x: f"${x:,.2f}")
    disp["Total"] = disp["Total"].apply(lambda x: f"${x:,.0f}")
    disp["Notes"] = disp["Notes"].fillna("—")
    st.dataframe(disp, hide_index=True, use_container_width=True)


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    _init()
    st.markdown("## 📁 Portfolio Management")
    st.markdown("*Manage holdings, record transactions, and import data*")
    st.divider()
    _show_msg()

    ref = get_selected_client()
    if ref is None:
        st.info("👈 Select a client from the sidebar to manage their portfolio.")
        return

    client    = _client_repo.get_by_id(ref.db_id)
    if not client:
        st.error("Client not found."); return

    client_id   = ref.db_id
    client_name = ref.name
    colour      = RISK_PROFILE_COLOURS.get(ref.risk_profile, "#a78bfa")

    # Client header
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid {colour}30;'
        f'border-radius:12px;padding:14px 18px;margin-bottom:16px;">'
        f'<span style="font-size:1rem;font-weight:700;color:#e8e8f0;">👤 {client_name}</span>'
        f'&nbsp;&nbsp;<span style="background:{colour}20;color:{colour};padding:2px 10px;'
        f'border-radius:20px;font-size:0.75rem;font-weight:600;border:1px solid {colour}50;">'
        f'{ref.risk_profile.title()} Risk</span>'
        f'<span style="float:right;font-size:0.85rem;color:#a78bfa;font-weight:600;">'
        f'AUM: {fmt_large(ref.aum)}</span></div>',
        unsafe_allow_html=True,
    )

    # Load portfolios
    portfolios = _port_repo.get_for_client(client_id)

    # ── Mode: CREATE PORTFOLIO ─────────────────────────────────────────────────
    if st.session_state[_MODE] == "add_portfolio":
        _render_create_portfolio(client_id, client_name)
        return

    # ── Portfolio selector ─────────────────────────────────────────────────────
    if not portfolios:
        st.warning("This client has no portfolios yet.")
        if st.button("🆕 Create First Portfolio", type="primary"):
            st.session_state[_MODE] = "add_portfolio"; st.rerun()
        return

    port_map = {f"{p['name']} ({fmt_large(p['total_value'])})": p["id"] for p in portfolios}
    # Default to previously selected or first
    current_pid = st.session_state.get(_PORT_ID)
    port_labels = list(port_map.keys())
    default_idx = 0
    if current_pid:
        for i, pid in enumerate(port_map.values()):
            if pid == current_pid:
                default_idx = i; break

    hdr1, hdr2 = st.columns([3,1])
    with hdr1:
        selected_label = st.selectbox("📁 Select Portfolio", port_labels, index=default_idx, label_visibility="collapsed")
    with hdr2:
        if st.button("🆕 New Portfolio", use_container_width=True):
            st.session_state[_MODE] = "add_portfolio"; st.rerun()

    portfolio_id = port_map[selected_label]
    st.session_state[_PORT_ID] = portfolio_id

    # ── Modes requiring portfolio context ─────────────────────────────────────
    mode = st.session_state[_MODE]

    if mode == "add_holding":
        _render_holding_form(portfolio_id, client_id)
        return
    if mode == "edit_holding":
        h = _hold_repo.get_by_id(st.session_state[_HOLD_ID])
        if h:
            _render_holding_form(portfolio_id, client_id, existing=h)
        else:
            st.error("Holding not found.")
            st.session_state[_MODE] = "list"
        return
    if mode == "add_txn":
        _render_txn_form(portfolio_id, client_id)
        return
    if mode == "csv_import":
        _render_csv_import(portfolio_id, client_id, client_name)
        return

    # ── LIST MODE ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📋 Holdings", "🔄 Transactions"])

    with tab1:
        _render_analytics(portfolio_id, ref.risk_profile)

    with tab2:
        # Action buttons
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("➕ Add Holding", use_container_width=True, type="primary"):
                st.session_state[_MODE] = "add_holding"; st.rerun()
        with a2:
            if st.button("📂 Import CSV", use_container_width=True):
                st.session_state[_MODE] = "csv_import"; st.rerun()
        with a3:
            if st.button("🔄 Recalculate Value", use_container_width=True):
                _sync(portfolio_id, client_id)
                _msg("success", "✅ Portfolio value recalculated.")
                st.rerun()
        st.divider()
        _render_holdings(portfolio_id, client_id)

    with tab3:
        if st.button("📝 Record Transaction", use_container_width=True, type="primary"):
            st.session_state[_MODE] = "add_txn"; st.rerun()
        st.divider()
        _render_txn_history(portfolio_id)
