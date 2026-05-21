"""
pages/client_summary.py
────────────────────────
Client Summary page — AI-powered advisor briefing generator.

Features:
  - Client selector (synced with sidebar)
  - Four summary type tabs with icons and descriptions
  - Context preview panel (key metrics used as AI input)
  - Generate button with streaming-style spinner
  - Editable output text area
  - Copy to clipboard button
  - Download as Markdown and plain-text
  - AI vs. template badge indicator
  - Session caching so summaries survive tab switches
"""

from __future__ import annotations

import re
from datetime import date

import streamlit as st

from services.summary_service import (
    SummaryService, SummaryResult, SUMMARY_TYPES, build_client_context
)
from portfolio.mock_data import CLIENTS
from services.gemini_service import is_ai_available
from utils.client_resolver import get_selected_client
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Session state keys
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_CACHE_KEY = "summary_cache"   # dict[client_id + summary_type → SummaryResult]
_ACTIVE_TAB_KEY    = "summary_active_tab"


def _init_session() -> None:
    if _SUMMARY_CACHE_KEY not in st.session_state:
        st.session_state[_SUMMARY_CACHE_KEY] = {}
    if _ACTIVE_TAB_KEY not in st.session_state:
        st.session_state[_ACTIVE_TAB_KEY] = "meeting_prep"


def _cache_key(client_id: str, summary_type: str) -> str:
    return f"{client_id}::{summary_type}"


def _get_cached(client_id: str, summary_type: str) -> SummaryResult | None:
    return st.session_state[_SUMMARY_CACHE_KEY].get(_cache_key(client_id, summary_type))


def _set_cached(result: SummaryResult, client_id: str, summary_type: str) -> None:
    st.session_state[_SUMMARY_CACHE_KEY][_cache_key(client_id, summary_type)] = result


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ai_badge(is_ai: bool) -> str:
    if is_ai:
        return (
            '<span style="background:#7c3aed20; color:#a78bfa; padding:3px 10px; '
            'border-radius:20px; font-size:0.75rem; font-weight:600; border:1px solid #7c3aed50;">'
            '✨ AI Generated</span>'
        )
    return (
        '<span style="background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.5); '
        'padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; '
        'border:1px solid rgba(255,255,255,0.1);">📄 Template</span>'
    )


def _metric_card(label: str, value: str, colour: str = "#a78bfa") -> str:
    return (
        f'<div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); '
        f'border-radius:10px; padding:12px 16px; text-align:center;">'
        f'<div style="font-size:0.7rem; color:rgba(255,255,255,0.45); margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:1.05rem; font-weight:700; color:{colour};">{value}</div>'
        f'</div>'
    )


def _render_context_panel(client_id: str) -> None:
    """Show a compact data snapshot so the advisor can see what the AI was given."""
    ctx = build_client_context(client_id)
    rr  = ctx.risk_report
    s   = ctx.summary
    pm  = ctx.perf_metrics
    colour_map = {
        "Very Low": "#2ed573", "Low": "#7bed9f",
        "Moderate": "#ffa502", "High": "#ff6b35", "Very High": "#ff4757",
    }
    risk_colour = colour_map.get(rr.risk_band, "#a78bfa")

    with st.expander("📊 Data used as AI context — click to expand", expanded=False):
        st.markdown("**Portfolio snapshot fed into the AI prompt:**")

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(
                _metric_card("Portfolio Value", f"${s['total_value']:,.0f}"),
                unsafe_allow_html=True,
            )
        with c2:
            delta_colour = "#2ed573" if s["total_gain_pct"] >= 0 else "#ff4757"
            st.markdown(
                _metric_card("Total Gain/Loss", f"{s['total_gain_pct']:+.1f}%", delta_colour),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                _metric_card("Risk Score", f"{rr.overall_score:.0f}/100", risk_colour),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                _metric_card("Sharpe Ratio", f"{pm.get('sharpe_ratio', 0):.2f}"),
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                _metric_card("Active Flags", str(len(rr.flags)),
                             "#ff4757" if rr.flags else "#2ed573"),
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Allocation + top holdings side by side
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Asset Allocation**")
            alloc = ctx.alloc_class[["asset_class", "weight", "gain_loss"]].copy()
            alloc.columns = ["Class", "Weight %", "Gain/Loss ($)"]
            alloc["Weight %"]    = alloc["Weight %"].apply(lambda x: f"{x:.1f}%")
            alloc["Gain/Loss ($)"] = alloc["Gain/Loss ($)"].apply(lambda x: f"${x:+,.0f}")
            st.dataframe(alloc, hide_index=True, use_container_width=True)

        with col_r:
            st.markdown("**Top 5 Holdings**")
            top = ctx.top5[["ticker", "weight", "gain_pct"]].copy()
            top.columns = ["Ticker", "Weight %", "Gain %"]
            top["Weight %"] = top["Weight %"].apply(lambda x: f"{x:.1f}%")
            top["Gain %"]   = top["Gain %"].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(top, hide_index=True, use_container_width=True)

        # Risk flags
        if rr.flags:
            st.markdown("**Risk Flags**")
            for flag in rr.flags:
                severity_colours = {
                    "critical": "#ff4757", "high": "#ff6b35",
                    "medium": "#ffa502",   "low": "#2ed573",
                }
                c = severity_colours.get(flag.severity, "#aaa")
                st.markdown(
                    f'<div style="border-left:3px solid {c}; padding:6px 12px; '
                    f'margin:4px 0; background:rgba(255,255,255,0.03); border-radius:0 6px 6px 0;">'
                    f'<span style="color:{c}; font-weight:600; font-size:0.8rem;">'
                    f'{flag.emoji} {flag.severity.upper()} · {flag.category}</span><br>'
                    f'<span style="font-size:0.85rem;">{flag.title}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _render_download_buttons(result: SummaryResult) -> None:
    """Render export buttons for the generated summary."""
    filename_base = (
        f"advisor_ai_{result.summary_type}_{result.client_name.lower().replace(' ', '_')}"
        f"_{date.today().isoformat()}"
    )

    col_md, col_txt = st.columns(2)

    with col_md:
        st.download_button(
            label="⬇️ Download as Markdown (.md)",
            data=result.content.encode("utf-8"),
            file_name=f"{filename_base}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_txt:
        # Strip markdown for plain-text export
        import re
        plain = re.sub(r"[#*`_~>|]", "", result.content)
        plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
        st.download_button(
            label="⬇️ Download as Plain Text (.txt)",
            data=plain.encode("utf-8"),
            file_name=f"{filename_base}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _render_summary_tab(
    summary_type: str,
    client_id:    str,
    service:      SummaryService,
) -> None:
    """Render the content for a single summary type tab."""
    type_info = SUMMARY_TYPES[summary_type]

    st.markdown(
        f"<p style='color:rgba(255,255,255,0.5); font-size:0.9rem; margin-bottom:1rem;'>"
        f"{type_info['description']}</p>",
        unsafe_allow_html=True,
    )

    # Check cache
    cached = _get_cached(client_id, summary_type)

    # Generate button row
    gen_col, badge_col = st.columns([2, 1])
    with gen_col:
        generate_label = (
            "✨ Generate with AI" if is_ai_available() else "📄 Generate Template Brief"
        )
        regen_label = (
            "🔄 Regenerate with AI" if is_ai_available() else "🔄 Refresh Template"
        )
        btn_label = regen_label if cached else generate_label
        do_generate = st.button(
            btn_label,
            key=f"gen_{summary_type}",
            type="primary",
            use_container_width=True,
        )
    with badge_col:
        if cached:
            st.markdown(
                f"<div style='padding-top:8px;'>{_ai_badge(cached.is_ai_generated)}</div>",
                unsafe_allow_html=True,
            )

    # Handle generation
    if do_generate:
        with st.spinner(
            f"{'Asking Gemini' if is_ai_available() else 'Building template'} "
            f"for {CLIENTS[client_id]['name']}..."
        ):
            result = service.generate(client_id, summary_type)
            _set_cached(result, client_id, summary_type)

        if result.error:
            st.warning(f"⚠️ AI generation failed — showing template: {result.error}")
        else:
            st.success(
                f"{'AI summary' if result.is_ai_generated else 'Template'} generated successfully!"
            )
        cached = result
        st.rerun()

    # Display the result
    if cached:
        st.divider()

        # Editable text area
        st.markdown("**Summary output** *(editable — make any adjustments before downloading)*")
        edited_content = st.text_area(
            label="Summary content",
            value=cached.content,
            height=520,
            key=f"edit_{summary_type}_{client_id}",
            label_visibility="collapsed",
        )

        # Update cache if edited (so download reflects edits)
        if edited_content != cached.content:
            cached.content = edited_content
            _set_cached(cached, client_id, summary_type)

        st.divider()

        # Preview in rendered Markdown below the editor
        with st.expander("👁️ Rendered preview", expanded=False):
            st.markdown(edited_content)

        st.divider()
        st.markdown("**Export**")
        _render_download_buttons(cached)

        # Metadata footer
        st.markdown(
            f"<p style='color:rgba(255,255,255,0.25); font-size:0.7rem; margin-top:0.5rem;'>"
            f"Generated: {cached.generated_on} · "
            f"{'Gemini AI' if cached.is_ai_generated else 'Template'} · "
            f"Model: {st.session_state.get('gemini_model', 'gemini-1.5-flash')}"
            f"</p>",
            unsafe_allow_html=True,
        )
    else:
        # Empty state
        st.markdown(
            f"""
            <div style="
                text-align:center; padding:3rem 1rem;
                background:rgba(255,255,255,0.02);
                border:1px dashed rgba(255,255,255,0.1);
                border-radius:12px; margin-top:1rem;
            ">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">
                    {type_info['icon']}
                </div>
                <p style="color:rgba(255,255,255,0.4); font-size:0.95rem; margin:0;">
                    Click <strong>Generate</strong> to create this summary
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Client Summary page."""
    _init_session()

    st.markdown("## 📋 Client Summaries")
    st.markdown("*AI-powered advisor briefings, risk explanations, and client profiles*")
    st.divider()

    # ── Client guard via client_resolver ─────────────────────────────────────
    ref = get_selected_client()
    if ref is None:
        st.info("👈 Please select a client from the sidebar to generate summaries.")
        return

    client_id = ref.mock_key
    client    = CLIENTS[client_id]

    # ── Client header ──────────────────────────────────────────────────────────
    col_name, col_ai_status = st.columns([3, 1])
    with col_name:
        profile_colour = {
            "conservative": "#2ed573",
            "moderate":     "#ffa502",
            "aggressive":   "#ff4757",
        }.get(client["risk_profile"], "#a78bfa")

        st.markdown(
            f"### 👤 {client['name']} "
            f"<span style='background:{profile_colour}20; color:{profile_colour}; "
            f"padding:3px 12px; border-radius:20px; font-size:0.85rem; "
            f"font-weight:600; border:1px solid {profile_colour}50;'>"
            f"{client['risk_profile'].title()} Risk</span>",
            unsafe_allow_html=True,
        )
    with col_ai_status:
        if is_ai_available():
            st.markdown(
                '<div style="text-align:right; padding-top:8px;">'
                '<span style="background:#7c3aed20; color:#a78bfa; padding:4px 12px; '
                'border-radius:20px; font-size:0.8rem; font-weight:600; '
                'border:1px solid #7c3aed50;">✨ Gemini Active</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:right; padding-top:8px;">'
                '<span style="background:rgba(255,165,0,0.1); color:#ffa502; padding:4px 12px; '
                'border-radius:20px; font-size:0.8rem; font-weight:600; '
                'border:1px solid rgba(255,165,0,0.3);">📄 Template Mode</span></div>',
                unsafe_allow_html=True,
            )

    # ── Context data panel ────────────────────────────────────────────────────
    _render_context_panel(client_id)

    st.divider()

    # ── Summary type tabs ─────────────────────────────────────────────────────
    tab_labels = [
        f"{info['icon']} {info['label']}"
        for info in SUMMARY_TYPES.values()
    ]
    tabs = st.tabs(tab_labels)

    # Lazy-init the service — guard against None (app.py seeds key with None on startup)
    if not isinstance(st.session_state.get("summary_service"), SummaryService):
        st.session_state.summary_service = SummaryService()
    service: SummaryService = st.session_state.summary_service

    for tab, (summary_type, _) in zip(tabs, SUMMARY_TYPES.items()):
        with tab:
            _render_summary_tab(summary_type, client_id, service)

    # ── Generate all button ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🚀 Batch Generate")
    st.caption("Generate all four summaries for this client in one click.")

    if st.button("Generate All Summaries", use_container_width=True):
        with st.spinner("Generating all summaries... this may take a moment."):
            for stype in SUMMARY_TYPES:
                result = service.generate(client_id, stype)
                _set_cached(result, client_id, stype)
        st.success(f"All 4 summaries generated for {client['name']}!")
        st.rerun()

    # ── Clear cache button ────────────────────────────────────────────────────
    if st.button("🗑️ Clear All Cached Summaries", use_container_width=True):
        st.session_state[_SUMMARY_CACHE_KEY] = {}
        st.success("Summary cache cleared.")
        st.rerun()
