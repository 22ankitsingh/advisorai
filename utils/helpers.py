"""
utils/helpers.py
─────────────────
Shared formatting, UI, and Plotly utilities used across all pages.

Importing these instead of repeating logic in each page:
  - Eliminates copy-paste bugs
  - Ensures consistent formatting across the whole app
  - Makes chart styling changes a one-line edit

Usage:
    from utils.helpers import fmt_usd, fmt_pct, apply_dark_theme, severity_badge
"""

from __future__ import annotations

from typing import Optional
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Number formatters
# ─────────────────────────────────────────────────────────────────────────────

def fmt_usd(value: float, show_sign: bool = False) -> str:
    """Format a number as USD currency. e.g. 1234567.8 → '$1,234,568'"""
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}${value:,.0f}"


def fmt_pct(value: float, show_sign: bool = True, decimals: int = 1) -> str:
    """Format a number as a percentage. e.g. 12.4 → '+12.4%'"""
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def fmt_large(value: float) -> str:
    """Compact format for large numbers: 1.2M, 450K, etc."""
    abs_v = abs(value)
    sign  = "-" if value < 0 else ""
    if abs_v >= 1_000_000:
        return f"{sign}${abs_v/1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{sign}${abs_v/1_000:.0f}K"
    return fmt_usd(value)


def fmt_ratio(value: float, decimals: int = 2) -> str:
    """Format a ratio/multiplier. e.g. 1.234 → '1.23x'"""
    return f"{value:.{decimals}f}x"


# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_COLOURS: dict[str, tuple[str, str]] = {
    # (background, text)
    "critical": ("#ff4757", "#fff"),
    "high":     ("#ff6b35", "#fff"),
    "medium":   ("#ffa502", "#000"),
    "low":      ("#2ed573", "#000"),
    "info":     ("#60a5fa", "#000"),
}

RISK_PROFILE_COLOURS: dict[str, str] = {
    "conservative": "#2ed573",
    "moderate":     "#ffa502",
    "aggressive":   "#ff4757",
}

ASSET_CLASS_COLOURS: dict[str, str] = {
    "equity":      "#a78bfa",
    "etf":         "#60a5fa",
    "bond":        "#34d399",
    "alternative": "#f97316",
    "cash":        "#94a3b8",
}

CHART_PALETTE = [
    "#a78bfa", "#60a5fa", "#34d399", "#f97316",
    "#fb923c", "#f472b6", "#facc15", "#38bdf8",
]


# ─────────────────────────────────────────────────────────────────────────────
# HTML badge helpers
# ─────────────────────────────────────────────────────────────────────────────

def severity_badge(severity: str) -> str:
    """
    Return a styled HTML badge for an alert/risk severity level.

    Usage (in Streamlit):
        st.markdown(severity_badge("high"), unsafe_allow_html=True)
    """
    bg, fg = SEVERITY_COLOURS.get(severity.lower(), ("#888", "#fff"))
    label  = severity.upper()
    return (
        f'<span style="background:{bg}; color:{fg}; padding:2px 9px; '
        f'border-radius:12px; font-size:0.74rem; font-weight:700; '
        f'letter-spacing:0.03em;">{label}</span>'
    )


def risk_profile_badge(profile: str) -> str:
    """Return a styled HTML badge for a risk profile label."""
    colour = RISK_PROFILE_COLOURS.get(profile.lower(), "#a78bfa")
    return (
        f'<span style="background:{colour}20; color:{colour}; '
        f'padding:2px 10px; border-radius:20px; font-size:0.8rem; '
        f'font-weight:600; border:1px solid {colour}50;">'
        f'{profile.title()}</span>'
    )


def metric_card_html(label: str, value: str, colour: str = "#a78bfa") -> str:
    """
    Return a styled HTML metric card for use in st.markdown(..., unsafe_allow_html=True).

    Typically placed inside a st.columns() cell.
    """
    return (
        f'<div style="background:rgba(255,255,255,0.04); '
        f'border:1px solid rgba(255,255,255,0.08); '
        f'border-radius:10px; padding:12px 16px; text-align:center;">'
        f'<div style="font-size:0.7rem; color:rgba(255,255,255,0.45); '
        f'margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em;">'
        f'{label}</div>'
        f'<div style="font-size:1.05rem; font-weight:700; color:{colour};">'
        f'{value}</div>'
        f'</div>'
    )


def info_banner(message: str, icon: str = "ℹ️", colour: str = "#60a5fa") -> str:
    """Return a styled HTML info/warning banner."""
    return (
        f'<div style="background:{colour}15; border:1px solid {colour}40; '
        f'border-radius:8px; padding:10px 16px; margin:8px 0; '
        f'font-size:0.9rem; color:{colour};">'
        f'{icon} {message}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotly dark theme
# ─────────────────────────────────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8e8f0", family="Inter, sans-serif", size=12),
    margin=dict(t=30, b=30, l=20, r=20),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        color="#e8e8f0",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        color="#e8e8f0",
        tickfont=dict(size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    hoverlabel=dict(
        bgcolor="rgba(15,12,41,0.95)",
        bordercolor="rgba(167,139,250,0.4)",
        font=dict(color="#e8e8f0", family="Inter"),
    ),
)


def apply_dark_theme(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    """
    Apply the Advisor AI dark theme to any Plotly figure.

    Args:
        fig:    Any Plotly figure object.
        height: Optional height in pixels.

    Returns:
        The same figure with theme applied (mutates in place).

    Usage:
        fig = px.bar(...)
        apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    """
    layout = dict(_DARK_LAYOUT)
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig
