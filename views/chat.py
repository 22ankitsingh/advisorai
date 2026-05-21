"""
pages/chat.py
──────────────
AI Chat page — Gemini-powered conversational assistant.

Architecture:
  - GeminiService is initialised once per session and stored in session state.
  - When the active client changes, the service is reinitialised with the new
    client context so the model is aware of who is being discussed.
  - Responses stream in real time via a st.empty() placeholder updated
    chunk-by-chunk, then saved to session state history on completion.
  - When GEMINI_API_KEY is missing, the page shows a clear setup guide
    and disables input — no crashes, no fake responses.
"""

from __future__ import annotations

import uuid
from typing import Optional

import streamlit as st

from services.database import get_db_connection
from services.gemini_service import GeminiService, GeminiServiceError, is_ai_available
from utils.client_resolver import get_selected_client, ClientRef
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Session state keys
# ─────────────────────────────────────────────────────────────────────────────

_SVC_KEY        = "gemini_service"          # GeminiService instance
_CLIENT_KEY_KEY = "gemini_client_key"       # mock_key of the client the service was built for
_MESSAGES_KEY   = "chat_messages"           # list[dict] — shadow history for display
_SESSION_KEY    = "chat_session_id"         # UUID persisted for DB logging


# ─────────────────────────────────────────────────────────────────────────────
# GeminiService lifecycle helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_init_service(ref: Optional[ClientRef]) -> Optional[GeminiService]:
    """
    Return a ready GeminiService, creating or reinitialising as needed.

    - First call: creates a new service.
    - Client changed: reinitialises with new client context and clears history.
    - No API key: returns None (caller shows setup guide).
    """
    if not is_ai_available():
        return None

    current_key = ref.mock_key if ref else None
    stored_key  = st.session_state.get(_CLIENT_KEY_KEY)

    # Reinit if client changed or service not yet created
    if st.session_state.get(_SVC_KEY) is None or current_key != stored_key:
        client_ctx = None
        if ref:
            client_ctx = {
                "name":          ref.name,
                "risk_profile":  ref.risk_profile,
                "aum":           ref.aum,
                "advisor_notes": "",
            }
        try:
            svc = GeminiService(client_context=client_ctx)
            st.session_state[_SVC_KEY]        = svc
            st.session_state[_CLIENT_KEY_KEY] = current_key
            logger.info("GeminiService (re)initialised | client=%s", current_key or "none")
        except GeminiServiceError as exc:
            logger.error("Failed to init GeminiService: %s", exc)
            return None

    return st.session_state[_SVC_KEY]


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_session_id() -> str:
    """Create a new chat session UUID if one doesn't exist."""
    if not st.session_state.get(_SESSION_KEY):
        st.session_state[_SESSION_KEY] = str(uuid.uuid4())
    return st.session_state[_SESSION_KEY]


def _save_message(session_id: str, role: str, content: str, db_id: Optional[int]) -> None:
    """Persist a chat message to SQLite for audit/history purposes."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content, client_id) VALUES (?,?,?,?)",
                (session_id, role, content, db_id),
            )
    except Exception as exc:
        logger.error("Failed to save chat message: %s", exc)


def _render_message(role: str, content: str) -> None:
    """Render a single chat bubble using the app's CSS classes."""
    if role == "user":
        st.markdown(
            f'<div class="user-bubble">👤 {content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">🤖 {content}</div>',
            unsafe_allow_html=True,
        )


def _stream_response(service: GeminiService, user_input: str) -> str:
    """
    Stream the Gemini response into a live-updating placeholder bubble.

    Returns the complete response string once streaming is done.
    Raises GeminiServiceError on API failure.
    """
    placeholder = st.empty()
    chunks: list[str] = []

    for chunk in service.stream_chat(user_input):
        chunks.append(chunk)
        # Show partial response with blinking cursor
        placeholder.markdown(
            f'<div class="assistant-bubble">🤖 {"".join(chunks)}▋</div>',
            unsafe_allow_html=True,
        )

    full_response = "".join(chunks)
    # Final render without cursor
    placeholder.markdown(
        f'<div class="assistant-bubble">🤖 {full_response}</div>',
        unsafe_allow_html=True,
    )
    return full_response


def _send_message(user_input: str, service: GeminiService, session_id: str, db_id: Optional[int]) -> None:
    """
    Handle a user message end-to-end:
      1. Append to display history
      2. Render user bubble
      3. Stream assistant response
      4. Save both to history + DB
      5. Rerun to stabilise UI
    """
    user_input = user_input.strip()
    if not user_input:
        return

    # 1. Save and render user message immediately
    st.session_state[_MESSAGES_KEY].append({"role": "user", "content": user_input})
    _save_message(session_id, "user", user_input, db_id)
    _render_message("user", user_input)

    # 2. Stream the AI response
    try:
        response = _stream_response(service, user_input)
        st.session_state[_MESSAGES_KEY].append({"role": "assistant", "content": response})
        _save_message(session_id, "assistant", response, db_id)

    except GeminiServiceError as exc:
        error_msg = str(exc)
        logger.error("Gemini error during chat: %s", error_msg)
        st.session_state[_MESSAGES_KEY].append({"role": "assistant", "content": f"⚠️ {error_msg}"})
        st.error(f"**AI Error:** {error_msg}")

    # 3. Rerun to clean up cursor and reset input widget
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Render the AI Chat page."""
    st.markdown("## 💬 AI Chat")
    st.markdown("*Ask anything about your clients, portfolios, or the market*")
    st.divider()

    session_id = _ensure_session_id()
    ref        = get_selected_client()
    db_id      = ref.db_id if ref else None

    # Ensure messages list exists
    if _MESSAGES_KEY not in st.session_state:
        st.session_state[_MESSAGES_KEY] = []

    # ── AI availability gate ──────────────────────────────────────────────────
    if not is_ai_available():
        st.warning(
            "**AI Chat requires a Gemini API key.**\n\n"
            "1. Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
            "2. Add it to your `.env` file:  `GEMINI_API_KEY=your_key_here`\n"
            "3. Restart the app\n\n"
            "_All other pages (Portfolio, Summaries, Compliance) work without an API key._"
        )
        st.divider()
        # Still show existing history (read-only)
        for msg in st.session_state[_MESSAGES_KEY]:
            _render_message(msg["role"], msg["content"])
        return

    # ── Initialise / update GeminiService ────────────────────────────────────
    service = _get_or_init_service(ref)
    if service is None:
        st.error("Failed to initialise AI service. Check your GEMINI_API_KEY.")
        return

    # ── Context banner ────────────────────────────────────────────────────────
    if ref:
        st.info(
            f"📌 **Context:** Chatting about **{ref.name}** "
            f"({ref.risk_profile.title()} risk · AUM ${ref.aum:,.0f})"
        )
    else:
        st.caption("💡 Select a client from the sidebar for context-aware responses.")

    # ── Chat history display ──────────────────────────────────────────────────
    messages = st.session_state[_MESSAGES_KEY]

    if not messages:
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 1rem; opacity:0.5;">
                <div style="font-size:3rem;">🤖</div>
                <p style="font-size:1rem; margin-top:0.5rem;">
                    Hello! I'm your AI financial advisor assistant.<br>
                    Ask me about portfolios, risk, rebalancing, or compliance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in messages:
            _render_message(msg["role"], msg["content"])

    st.divider()

    # ── Quick-action chips ────────────────────────────────────────────────────
    st.markdown("**Quick prompts:**")
    quick_prompts = [
        "Summarise this client's portfolio",
        "What are the main risks?",
        "Suggest rebalancing actions",
        "Generate a compliance summary",
    ]
    chip_cols = st.columns(len(quick_prompts))
    for i, prompt in enumerate(quick_prompts):
        with chip_cols[i]:
            if st.button(prompt, key=f"chip_{i}", use_container_width=True):
                st.session_state["_pending_prompt"] = prompt
                st.rerun()

    # ── Handle chip-triggered prompt ──────────────────────────────────────────
    if "_pending_prompt" in st.session_state:
        pending = st.session_state.pop("_pending_prompt")
        _send_message(pending, service, session_id, db_id)

    # ── Message input ─────────────────────────────────────────────────────────
    col_input, col_btn, col_clear = st.columns([6, 1, 1])

    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about portfolios, risk, compliance...",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        send_clicked = st.button("Send ➤", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state[_MESSAGES_KEY] = []
            st.session_state[_SESSION_KEY]  = str(uuid.uuid4())
            # Reinit service to clear Gemini's native history too
            if service:
                service.clear()
            st.rerun()

    # ── Send on button click ──────────────────────────────────────────────────
    if send_clicked and user_input.strip():
        _send_message(user_input, service, session_id, db_id)

    # ── Conversation stats footer ─────────────────────────────────────────────
    if messages:
        turns = len(messages) // 2
        model = st.session_state.get("gemini_model", "gemini-2.5-flash")
        st.markdown(
            f"<p style='font-size:0.7rem; color:rgba(255,255,255,0.25); "
            f"text-align:right; margin-top:0.5rem;'>"
            f"{turns} exchange(s) · {model}</p>",
            unsafe_allow_html=True,
        )
