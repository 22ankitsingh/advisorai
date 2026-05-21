"""
app.py
───────
Advisor AI — Main Streamlit entry point.

Responsibilities:
  - Configure global Streamlit page settings
  - Initialise the database (schema + seed) on first run
  - Validate environment configuration
  - Set up persistent session state
  - Render the sidebar (navigation + client selector with ClientRef)
  - Route to the correct page module
"""

import streamlit as st

# ── Must be the FIRST Streamlit call ─────────────────────────────────────────
st.set_page_config(
    page_title="Advisor AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**Advisor AI** — AI-powered financial advisory assistant.",
    },
)

# ── Imports (after set_page_config) ──────────────────────────────────────────
from services.database import init_database
from data.seed import seed_all
from utils.logger import get_logger
from utils.config import settings, validate_env
from utils.client_resolver import (
    get_all_client_refs, set_selected_client, get_selected_client, ClientRef
)
from utils.helpers import (
    fmt_usd, fmt_large, severity_badge, risk_profile_badge,
    RISK_PROFILE_COLOURS
)
from services.gemini_service import is_ai_available

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# One-time startup tasks (cached — runs once per server process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _startup() -> None:
    """Initialise DB, seed mock data, and validate environment once per start."""
    logger.info("Running startup tasks...")
    init_database()
    seed_all()   # No-op if already seeded

    issues = validate_env()
    for issue in issues:
        logger.warning("ENV [%s] %s: %s", issue.severity.upper(), issue.var, issue.message)

    logger.info("Startup complete. AI available: %s", is_ai_available())


_startup()


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(167,139,250,0.15);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: opacity 0.2s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
    }

    /* ── Select boxes & inputs ── */
    .stSelectbox > div > div,
    .stTextInput > div > div {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        color: #e8e8f0;
    }

    /* ── Chat message bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: white;
    }
    .assistant-bubble {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 85%;
        color: #e8e8f0;
        backdrop-filter: blur(8px);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        color: rgba(255,255,255,0.6);
    }
    .stTabs [aria-selected="true"] {
        color: #a78bfa;
        border-bottom: 2px solid #a78bfa;
    }

    /* ── Dividers ── */
    hr { border-color: rgba(255,255,255,0.08); }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(167,139,250,0.4);
        border-radius: 3px;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Set default values for all session state keys used across pages."""
    defaults: dict = {
        "selected_client_ref": None,   # ClientRef (replaces selected_client_id int)
        "selected_client_id":  None,   # Kept for backward compat with older pages
        "chat_messages":       [],
        "chat_session_id":     None,
        "page":                "Dashboard",
        "summary_service":     None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Navigation config
# ─────────────────────────────────────────────────────────────────────────────

NAV_ITEMS: list[tuple[str, str]] = [
    ("🏠  Dashboard",       "Dashboard"),
    ("💬  AI Chat",         "Chat"),
    ("📊  Portfolio",       "Portfolio"),
    ("📋  Client Summary",  "ClientSummary"),
    ("🔍  Research",        "Research"),
    ("⚠️  Compliance",      "Compliance"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:

    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size: 2.5rem;">💼</div>
        <h1 style="font-size:1.4rem; font-weight:700; margin:4px 0 0;
                   background: linear-gradient(135deg, #a78bfa, #60a5fa);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Advisor AI
        </h1>
        <p style="font-size:0.72rem; color: rgba(255,255,255,0.45); margin:0;">
            Financial Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # AI status indicator
    if is_ai_available():
        st.markdown(
            '<div style="text-align:center; margin-bottom:0.5rem;">'
            '<span style="background:#7c3aed20; color:#a78bfa; padding:2px 10px; '
            'border-radius:20px; font-size:0.72rem; font-weight:600; '
            'border:1px solid #7c3aed50;">✨ Gemini Active</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center; margin-bottom:0.5rem;">'
            '<span style="background:rgba(255,165,0,0.1); color:#ffa502; padding:2px 10px; '
            'border-radius:20px; font-size:0.72rem; font-weight:600; '
            'border:1px solid rgba(255,165,0,0.3);">📄 Template Mode</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.72rem; color:rgba(255,255,255,0.4); "
        "text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>"
        "Navigation</p>",
        unsafe_allow_html=True,
    )

    for label, page_name in NAV_ITEMS:
        is_active = st.session_state.page == page_name
        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = page_name
            st.rerun()

    st.divider()

    # ── Client selector ────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.72rem; color:rgba(255,255,255,0.4); "
        "text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>"
        "Active Client</p>",
        unsafe_allow_html=True,
    )

    all_refs  = get_all_client_refs()
    ref_map   = {"— Select a client —": None}
    ref_map.update({str(ref): ref for ref in all_refs})

    current_ref = get_selected_client()
    current_label = str(current_ref) if current_ref else "— Select a client —"

    selected_label = st.selectbox(
        "Client selector",
        options=list(ref_map.keys()),
        index=list(ref_map.keys()).index(current_label)
              if current_label in ref_map else 0,
        label_visibility="collapsed",
        key="sidebar_client_select",
    )

    new_ref = ref_map[selected_label]
    set_selected_client(new_ref)

    # ── Client info card ───────────────────────────────────────────────────────
    if new_ref:
        profile_colour = RISK_PROFILE_COLOURS.get(new_ref.risk_profile, "#a78bfa")
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04); '
            f'border:1px solid rgba(255,255,255,0.08); '
            f'border-radius:10px; padding:10px 14px; margin-top:8px;">'
            f'<div style="font-size:0.8rem; font-weight:600; color:#e8e8f0;">'
            f'👤 {new_ref.name}</div>'
            f'<div style="font-size:0.72rem; color:rgba(255,255,255,0.5); margin:3px 0;">'
            f'AUM: <span style="color:#a78bfa; font-weight:600;">'
            f'{fmt_large(new_ref.aum)}</span></div>'
            f'<div style="margin-top:4px;">'
            f'<span style="background:{profile_colour}20; color:{profile_colour}; '
            f'padding:1px 8px; border-radius:12px; font-size:0.7rem; font-weight:600; '
            f'border:1px solid {profile_colour}40;">'
            f'{new_ref.risk_profile.title()} Risk</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.67rem; color:rgba(255,255,255,0.25); "
        f"text-align:center; margin:0;'>"
        f"v1.0.0 · {settings.app_env.title()}</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page routing
# ─────────────────────────────────────────────────────────────────────────────

page = st.session_state.page

if page == "Dashboard":
    from views.dashboard import render
    render()
elif page == "Chat":
    from views.chat import render
    render()
elif page == "Portfolio":
    from views.portfolio import render
    render()
elif page == "ClientSummary":
    from views.client_summary import render
    render()
elif page == "Research":
    from views.research import render
    render()
elif page == "Compliance":
    from views.compliance import render
    render()
else:
    st.error(f"Unknown page: {page!r}. Please use the sidebar to navigate.")
    st.session_state.page = "Dashboard"
    st.rerun()
