"""
views/client_management.py
────────────────────────────
Full Client CRUD page — Phase 2.

Features:
  - KPI cards (total clients, AUM, risk breakdown)
  - Search + risk-profile filter
  - Client card grid with Edit / Delete / Select actions
  - Add New Client form
  - Edit Client form (inline panel)
  - Audit logging on all mutations
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from database.repositories.client_repository import ClientRepository
from database.repositories.portfolio_repository import PortfolioRepository
from database.repositories.audit_repository import get_audit_repo
from utils.client_resolver import get_all_client_refs, set_selected_client, resolve_client
from utils.helpers import (
    fmt_usd, fmt_large, risk_profile_badge, apply_dark_theme, RISK_PROFILE_COLOURS
)
from utils.logger import get_logger

logger = get_logger(__name__)

_client_repo    = ClientRepository()
_portfolio_repo = PortfolioRepository()
_audit          = get_audit_repo()

# ── Session state keys ────────────────────────────────────────────────────────
_MODE_KEY   = "cm_mode"       # "list" | "add" | "edit"
_EDIT_KEY   = "cm_edit_id"    # client_id being edited
_SEARCH_KEY = "cm_search"
_RISK_KEY   = "cm_risk_filter"
_MSG_KEY    = "cm_message"    # (type, text) tuple for success/error


def _init_state() -> None:
    defaults = {
        _MODE_KEY:   "list",
        _EDIT_KEY:   None,
        _SEARCH_KEY: "",
        _RISK_KEY:   "All",
        _MSG_KEY:    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── CSS helpers ───────────────────────────────────────────────────────────────

def _card(content: str, accent: str = "#a78bfa") -> None:
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid {accent}30;'
        f'border-radius:12px;padding:16px 18px;margin-bottom:12px;">'
        f'{content}</div>',
        unsafe_allow_html=True,
    )


def _form_section(title: str) -> None:
    st.markdown(
        f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);'
        f'text-transform:uppercase;letter-spacing:.08em;margin:16px 0 6px;">'
        f'{title}</div>',
        unsafe_allow_html=True,
    )


# ── Flash message ─────────────────────────────────────────────────────────────

def _show_message() -> None:
    msg = st.session_state.get(_MSG_KEY)
    if msg:
        kind, text = msg
        if kind == "success":
            st.success(text)
        elif kind == "error":
            st.error(text)
        elif kind == "warning":
            st.warning(text)
        st.session_state[_MSG_KEY] = None


def _set_msg(kind: str, text: str) -> None:
    st.session_state[_MSG_KEY] = (kind, text)


# ── KPI cards ─────────────────────────────────────────────────────────────────

def _render_kpis(clients: list[dict]) -> None:
    if not clients:
        return

    df = pd.DataFrame(clients)
    total_aum     = df["aum"].sum()
    total_clients = len(df)
    risk_counts   = df["risk_profile"].value_counts().to_dict()
    open_alerts   = int(df["open_alerts"].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("👥 Total Clients", total_clients)
    with k2:
        st.metric("💰 Total AUM", fmt_large(total_aum))
    with k3:
        st.metric("🟢 Conservative", risk_counts.get("conservative", 0))
    with k4:
        st.metric("🟡 Moderate", risk_counts.get("moderate", 0))
    with k5:
        st.metric("🔴 Aggressive", risk_counts.get("aggressive", 0))

    if open_alerts > 0:
        st.markdown(
            f'<div style="background:#ff475715;border:1px solid #ff475740;'
            f'border-radius:8px;padding:8px 14px;font-size:0.85rem;color:#ff4757;">'
            f'⚠️ <b>{open_alerts}</b> open compliance alert(s) across all clients</div>',
            unsafe_allow_html=True,
        )


# ── Client card ───────────────────────────────────────────────────────────────

def _render_client_card(c: dict, idx: int) -> None:
    colour   = RISK_PROFILE_COLOURS.get(c["risk_profile"], "#a78bfa")
    alerts   = c.get("open_alerts", 0)
    ports    = c.get("portfolio_count", 0)
    goal     = c.get("investment_goal") or "—"
    age_str  = f"Age {c.get('age')}" if c.get("age") else ""

    badge_bg   = f"{colour}20"
    badge_bdr  = f"{colour}50"

    card_html = (
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid {colour}25;'
        f'border-radius:14px;padding:18px;height:100%;transition:border-color .2s;">'
        # Header row
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
        f'<div style="width:40px;height:40px;border-radius:50%;'
        f'background:linear-gradient(135deg,{colour}60,{colour}20);'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:1.1rem;flex-shrink:0;">👤</div>'
        f'<div>'
        f'<div style="font-size:0.95rem;font-weight:700;color:#e8e8f0;">{c["name"]}</div>'
        f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.45);">{c["email"]}</div>'
        f'</div></div>'
        # Risk badge
        f'<div style="margin-bottom:10px;">'
        f'<span style="background:{badge_bg};color:{colour};padding:2px 10px;'
        f'border-radius:20px;font-size:0.72rem;font-weight:600;'
        f'border:1px solid {badge_bdr};">{c["risk_profile"].title()} Risk</span>'
        + (f'<span style="background:#ff475720;color:#ff4757;padding:2px 8px;'
           f'border-radius:20px;font-size:0.7rem;font-weight:600;'
           f'border:1px solid #ff475740;margin-left:6px;">⚠️ {alerts}</span>'
           if alerts else "")
        + f'</div>'
        # Stats
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px;">'
        f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.45);">AUM</div>'
        f'<div style="font-size:0.72rem;color:#a78bfa;font-weight:600;">{fmt_large(c["aum"])}</div>'
        f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.45);">Portfolios</div>'
        f'<div style="font-size:0.72rem;color:#e8e8f0;">{ports}</div>'
        + (f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.45);">Age</div>'
           f'<div style="font-size:0.72rem;color:#e8e8f0;">{age_str}</div>'
           if age_str else "")
        + (f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.45);">Goal</div>'
           f'<div style="font-size:0.72rem;color:#e8e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{goal[:22]}</div>'
           if goal != "—" else "")
        + f'</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # Action buttons under the card
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("✏️ Edit", key=f"edit_{idx}", use_container_width=True):
            st.session_state[_MODE_KEY] = "edit"
            st.session_state[_EDIT_KEY] = c["id"]
            st.rerun()
    with b2:
        if st.button("📊 Select", key=f"sel_{idx}", use_container_width=True):
            ref = resolve_client(c["id"])
            if ref:
                set_selected_client(ref)
                _set_msg("success", f"✅ Active client set to **{c['name']}**")
            st.rerun()
    with b3:
        if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True):
            st.session_state[f"confirm_del_{c['id']}"] = True
            st.rerun()

    # Inline delete confirmation
    if st.session_state.get(f"confirm_del_{c['id']}"):
        st.warning(f"⚠️ Delete **{c['name']}**? This removes all portfolios and data.")
        ca, cb = st.columns(2)
        with ca:
            if st.button("✅ Confirm Delete", key=f"conf_{idx}", type="primary"):
                _audit.log_client_deleted(c["name"], client_id=c["id"])
                _client_repo.delete(c["id"])
                st.session_state.pop(f"confirm_del_{c['id']}", None)
                _set_msg("success", f"🗑️ Client **{c['name']}** deleted.")
                st.rerun()
        with cb:
            if st.button("✖ Cancel", key=f"canc_{idx}"):
                st.session_state.pop(f"confirm_del_{c['id']}", None)
                st.rerun()


# ── Add / Edit form ───────────────────────────────────────────────────────────

def _render_form(existing: dict | None = None) -> None:
    """Render the Add New Client or Edit Client form."""
    is_edit = existing is not None
    title   = f"✏️ Edit Client — {existing['name']}" if is_edit else "➕ Add New Client"

    st.markdown(f"### {title}")
    st.divider()

    with st.form("client_form", clear_on_submit=not is_edit):
        # ── Basic info ────────────────────────────────────────────────────
        _form_section("Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(
                "Full Name *",
                value=existing["name"] if is_edit else "",
                placeholder="e.g. Jane Smith",
            )
        with col2:
            email = st.text_input(
                "Email Address *",
                value=existing["email"] if is_edit else "",
                placeholder="jane@example.com",
            )

        col3, col4 = st.columns(2)
        with col3:
            phone = st.text_input(
                "Phone",
                value=existing.get("phone", "") if is_edit else "",
                placeholder="+1-555-0100",
            )
        with col4:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=int(existing.get("age") or 40) if is_edit else 40,
            )

        # ── Risk & goals ──────────────────────────────────────────────────
        _form_section("Risk Profile & Goals")
        col5, col6 = st.columns(2)
        with col5:
            risk_opts = ["conservative", "moderate", "aggressive"]
            current_risk = existing.get("risk_profile", "moderate") if is_edit else "moderate"
            risk_profile = st.selectbox(
                "Risk Profile *",
                options=risk_opts,
                index=risk_opts.index(current_risk) if current_risk in risk_opts else 1,
                format_func=str.title,
            )
        with col6:
            investment_goal = st.text_input(
                "Investment Goal",
                value=existing.get("investment_goal", "") if is_edit else "",
                placeholder="e.g. Retirement at 65",
            )

        # ── AUM ───────────────────────────────────────────────────────────
        _form_section("Assets Under Management")
        aum = st.number_input(
            "Initial AUM ($)",
            min_value=0.0,
            value=float(existing.get("aum", 0.0)) if is_edit else 0.0,
            step=1000.0,
            format="%.2f",
            help="This is updated automatically when you add portfolio holdings.",
        )

        # ── Notes ─────────────────────────────────────────────────────────
        _form_section("Advisor Notes")
        advisor_notes = st.text_area(
            "Notes",
            value=existing.get("advisor_notes", "") if is_edit else "",
            placeholder="Investment preferences, goals, important context...",
            height=100,
        )

        # ── Submit row ────────────────────────────────────────────────────
        st.markdown("")
        sub_col, cancel_col = st.columns([2, 1])
        with sub_col:
            submit = st.form_submit_button(
                "💾 Save Client" if is_edit else "➕ Create Client",
                type="primary",
                use_container_width=True,
            )
        with cancel_col:
            cancel = st.form_submit_button("✖ Cancel", use_container_width=True)

        if cancel:
            st.session_state[_MODE_KEY] = "list"
            st.session_state[_EDIT_KEY] = None
            st.rerun()

        if submit:
            # ── Validation ────────────────────────────────────────────────
            errors = []
            if not name.strip():
                errors.append("Full Name is required.")
            if not email.strip() or "@" not in email:
                errors.append("A valid email address is required.")

            if errors:
                for e in errors:
                    st.error(e)
                st.stop()

            try:
                if is_edit:
                    _client_repo.update(
                        existing["id"],
                        name=name.strip(),
                        email=email.strip(),
                        phone=phone.strip(),
                        age=int(age),
                        risk_profile=risk_profile,
                        aum=float(aum),
                        advisor_notes=advisor_notes.strip(),
                        investment_goal=investment_goal.strip() or None,
                    )
                    _audit.log_client_updated(
                        name.strip(),
                        client_id=existing["id"],
                        changed_fields=["name","email","phone","age","risk_profile","aum","notes"],
                    )
                    _set_msg("success", f"✅ **{name}** updated successfully.")
                else:
                    new_id = _client_repo.create(
                        name=name.strip(),
                        email=email.strip(),
                        phone=phone.strip(),
                        risk_profile=risk_profile,
                        aum=float(aum),
                        advisor_notes=advisor_notes.strip(),
                        age=int(age),
                        investment_goal=investment_goal.strip() or None,
                    )
                    # Auto-create a default portfolio
                    _portfolio_repo.create(new_id, name="Main Portfolio")
                    _audit.log_client_created(
                        name.strip(),
                        client_id=new_id,
                        details={"email": email, "risk_profile": risk_profile},
                    )
                    _set_msg("success", f"✅ Client **{name}** created (ID #{new_id}).")

                st.session_state[_MODE_KEY] = "list"
                st.session_state[_EDIT_KEY] = None
                st.rerun()

            except Exception as exc:
                if "UNIQUE constraint" in str(exc):
                    st.error(f"❌ Email **{email}** is already registered to another client.")
                else:
                    st.error(f"❌ Failed to save client: {exc}")
                    logger.error("Client save failed: %s", exc)


# ── Client table view (compact) ───────────────────────────────────────────────

def _render_table(clients: list[dict]) -> None:
    df = pd.DataFrame(clients)
    disp = df[["name", "email", "risk_profile", "aum", "portfolio_count", "open_alerts"]].copy()
    disp.columns = ["Name", "Email", "Risk Profile", "AUM ($)", "Portfolios", "Open Alerts"]
    disp["AUM ($)"]      = disp["AUM ($)"].apply(lambda x: f"${x:,.0f}")
    disp["Risk Profile"] = disp["Risk Profile"].str.title()
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Open Alerts": st.column_config.NumberColumn(format="%d ⚠️"),
        },
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Client Management page."""
    _init_state()

    st.markdown("## 👥 Client Management")
    st.markdown("*Add, edit, and manage your advisory clients*")
    st.divider()

    _show_message()

    mode = st.session_state[_MODE_KEY]

    # ── ADD / EDIT FORM MODE ──────────────────────────────────────────────────
    if mode in ("add", "edit"):
        existing = None
        if mode == "edit":
            edit_id  = st.session_state.get(_EDIT_KEY)
            existing = _client_repo.get_by_id(edit_id) if edit_id else None
            if existing is None:
                st.error("Client not found.")
                st.session_state[_MODE_KEY] = "list"
                st.rerun()

        _render_form(existing=existing)
        return

    # ── LIST MODE ─────────────────────────────────────────────────────────────
    # Header row: Add button + view toggle
    hdr_col, btn_col = st.columns([4, 1])
    with hdr_col:
        search = st.text_input(
            "🔍 Search clients",
            value=st.session_state[_SEARCH_KEY],
            placeholder="Name, email, risk profile...",
            label_visibility="collapsed",
            key="cm_search_input",
        )
        st.session_state[_SEARCH_KEY] = search
    with btn_col:
        if st.button("➕ Add Client", type="primary", use_container_width=True, key="btn_add"):
            st.session_state[_MODE_KEY] = "add"
            st.rerun()

    # Risk filter + view toggle
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        risk_filter = st.selectbox(
            "Risk Profile",
            options=["All", "Conservative", "Moderate", "Aggressive"],
            index=["All","Conservative","Moderate","Aggressive"].index(
                st.session_state[_RISK_KEY]
            ),
            key="cm_risk_select",
        )
        st.session_state[_RISK_KEY] = risk_filter
    with f2:
        view_mode = st.radio(
            "View",
            options=["Cards", "Table"],
            horizontal=True,
            label_visibility="collapsed",
            key="cm_view_toggle",
        )
    with f3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Name", "AUM ↓", "AUM ↑", "Alerts"],
            key="cm_sort",
        )

    st.divider()

    # Load data
    all_clients = _client_repo.get_all()

    # Apply search
    if search:
        q = search.lower()
        all_clients = [
            c for c in all_clients
            if q in c["name"].lower()
            or q in c["email"].lower()
            or q in c["risk_profile"].lower()
        ]

    # Apply risk filter
    if risk_filter != "All":
        all_clients = [c for c in all_clients if c["risk_profile"] == risk_filter.lower()]

    # Apply sort
    if sort_by == "Name":
        all_clients.sort(key=lambda c: c["name"])
    elif sort_by == "AUM ↓":
        all_clients.sort(key=lambda c: c["aum"], reverse=True)
    elif sort_by == "AUM ↑":
        all_clients.sort(key=lambda c: c["aum"])
    elif sort_by == "Alerts":
        all_clients.sort(key=lambda c: c["open_alerts"], reverse=True)

    # KPI cards
    _render_kpis(all_clients)
    st.divider()

    if not all_clients:
        st.info("No clients match your filter. Try adjusting your search or add a new client.")
        return

    # Client count banner
    st.markdown(
        f'<p style="font-size:0.8rem;color:rgba(255,255,255,0.4);">'
        f'Showing <b style="color:#a78bfa">{len(all_clients)}</b> client(s)</p>',
        unsafe_allow_html=True,
    )

    # ── Table view ────────────────────────────────────────────────────────────
    if view_mode == "Table":
        _render_table(all_clients)

        st.markdown("##### Actions")
        act_cols = st.columns(min(len(all_clients), 5))
        for i, c in enumerate(all_clients):
            col = act_cols[i % len(act_cols)]
            with col:
                if st.button(f"✏️ {c['name'][:12]}", key=f"tedit_{i}", use_container_width=True):
                    st.session_state[_MODE_KEY] = "edit"
                    st.session_state[_EDIT_KEY] = c["id"]
                    st.rerun()
        return

    # ── Cards view ────────────────────────────────────────────────────────────
    cols_per_row = 3
    for row_start in range(0, len(all_clients), cols_per_row):
        row_clients = all_clients[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, c in enumerate(row_clients):
            with cols[i]:
                _render_client_card(c, row_start + i)
