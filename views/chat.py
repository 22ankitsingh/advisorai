"""
views/chat.py
──────────────
AI Chat page — Phase 4: Persistent Chat History.

New in Phase 4:
  - Full session persistence via ChatHistoryService + ChatRepository
  - Collapsible "Chat History" panel in the left gutter listing previous sessions
  - Click any session to restore its full message history
  - "New Chat" button starts a fresh session
  - Delete individual sessions from the history panel
  - Session auto-titled from first user message
  - Sessions are scoped to the active client when one is selected
  - Gemini service context is rebuilt when switching clients

Preserved from original:
  - Streaming response with live cursor
  - Quick-prompt chips
  - Clear button
  - Template mode (no API key) fallback
"""

from __future__ import annotations

import uuid
from typing import Optional

import streamlit as st

from services.chat_history_service import (
    get_chat_history_service, MESSAGES_KEY, SESSION_UUID_KEY
)
from services.gemini_service import GeminiService, GeminiServiceError, is_ai_available
from utils.client_resolver import get_selected_client, ClientRef
from utils.helpers import fmt_large
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Session-state keys ────────────────────────────────────────────────────────
_SVC_KEY        = "gemini_service"
_CLIENT_ID_KEY  = "gemini_active_client_id"   # int | None — which client service was built for
_HISTORY_OPEN   = "chat_history_panel_open"


# ── GeminiService lifecycle ───────────────────────────────────────────────────

def _get_or_init_service(ref: Optional[ClientRef]) -> Optional[GeminiService]:
    """
    Return a ready GeminiService, creating or reinitialising when client changes.
    Returns None if no API key is configured.
    """
    if not is_ai_available():
        return None

    current_id = ref.db_id if ref else None
    stored_id  = st.session_state.get(_CLIENT_ID_KEY)

    if st.session_state.get(_SVC_KEY) is None or current_id != stored_id:
        client_ctx = None
        if ref:
            client_ctx = {
                "name":         ref.name,
                "risk_profile": ref.risk_profile,
                "aum":          ref.aum,
                "advisor_notes": "",
            }
        try:
            svc = GeminiService(client_context=client_ctx)
            st.session_state[_SVC_KEY]       = svc
            st.session_state[_CLIENT_ID_KEY] = current_id
            logger.info("GeminiService (re)initialised for client_id=%s", current_id)
        except GeminiServiceError as exc:
            logger.error("Failed to init GeminiService: %s", exc)
            return None

    return st.session_state[_SVC_KEY]


# ── Message rendering ─────────────────────────────────────────────────────────

def _render_message(role: str, content: str) -> None:
    if role == "user":
        st.markdown(f'<div class="user-bubble">👤 {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">🤖 {content}</div>', unsafe_allow_html=True)


def _stream_response(service: GeminiService, user_input: str) -> str:
    placeholder = st.empty()
    chunks: list[str] = []
    for chunk in service.stream_chat(user_input):
        chunks.append(chunk)
        placeholder.markdown(
            f'<div class="assistant-bubble">🤖 {"".join(chunks)}▋</div>',
            unsafe_allow_html=True,
        )
    full = "".join(chunks)
    placeholder.markdown(f'<div class="assistant-bubble">🤖 {full}</div>', unsafe_allow_html=True)
    return full


def _send_message(
    user_input: str,
    service: GeminiService,
    history_svc,
    session_uuid: str,
    db_id: Optional[int],
    is_first_message: bool,
) -> None:
    user_input = user_input.strip()
    if not user_input:
        return

    # Add to in-memory state and render immediately
    st.session_state[MESSAGES_KEY].append({"role": "user", "content": user_input})
    history_svc.save_message("user", user_input, session_uuid, client_id=db_id)
    _render_message("user", user_input)

    # Auto-title the session from the first message
    if is_first_message:
        history_svc.auto_title_session(session_uuid, user_input)

    # Stream and persist AI response
    try:
        response = _stream_response(service, user_input)
        st.session_state[MESSAGES_KEY].append({"role": "assistant", "content": response})
        history_svc.save_message("assistant", response, session_uuid, client_id=db_id)
    except GeminiServiceError as exc:
        err = str(exc)
        logger.error("Gemini error: %s", err)
        err_msg = f"⚠️ {err}"
        st.session_state[MESSAGES_KEY].append({"role": "assistant", "content": err_msg})
        st.error(f"**AI Error:** {err}")

    st.rerun()


# ── History panel ─────────────────────────────────────────────────────────────

def _render_history_panel(history_svc, client_id: Optional[int], active_uuid: str) -> None:
    """Render the collapsible session history panel."""
    sessions = history_svc.get_recent_sessions(client_id=client_id, limit=20)

    with st.expander(f"🕐 Chat History ({len(sessions)} session{'s' if len(sessions) != 1 else ''})", expanded=st.session_state.get(_HISTORY_OPEN, False)):
        st.session_state[_HISTORY_OPEN] = True

        if not sessions:
            st.caption("No previous conversations yet.")
            return

        for s in sessions:
            suuid    = s["session_uuid"]
            title    = s.get("title") or "Untitled conversation"
            count    = s.get("message_count", 0)
            created  = s.get("created_at", "")[:10]
            is_active = (suuid == active_uuid)

            col_title, col_del = st.columns([5, 1])
            with col_title:
                label = f"**{title[:35]}{'…' if len(title) > 35 else ''}**  \n{created} · {count} msg(s)"
                if is_active:
                    st.markdown(
                        f'<div style="background:#a78bfa20;border:1px solid #a78bfa40;'
                        f'border-radius:8px;padding:6px 10px;font-size:0.8rem;color:#a78bfa;">'
                        f'▶ {title[:35]}{"…" if len(title) > 35 else ""}'
                        f'<br><span style="color:rgba(255,255,255,0.4);font-size:0.7rem;">'
                        f'{created} · {count} msg(s)</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(label, key=f"sess_{suuid[:8]}", use_container_width=True):
                        history_svc.switch_to_session(suuid)
                        # Reset Gemini service so it gets fresh context
                        st.session_state[_SVC_KEY] = None
                        st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{suuid[:8]}", help="Delete session"):
                    history_svc.delete_session(suuid)
                    st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## 💬 AI Chat")
    st.markdown("*Ask anything about your clients, portfolios, or the market*")
    st.divider()

    ref          = get_selected_client()
    db_id        = ref.db_id if ref else None
    history_svc  = get_chat_history_service()

    # Ensure active session exists (creates or loads from DB)
    session_uuid = history_svc.ensure_active_session(client_id=db_id)

    # ── Top toolbar: new chat + history toggle ────────────────────────────────
    tool1, tool2, tool3 = st.columns([3, 1, 1])
    with tool1:
        if ref:
            st.info(
                f"📌 **Context:** Chatting about **{ref.name}** "
                f"({ref.risk_profile.title()} risk · {fmt_large(ref.aum)})"
            )
        else:
            st.caption("💡 Select a client from the sidebar for context-aware responses.")
    with tool2:
        if st.button("🆕 New Chat", use_container_width=True):
            new_uuid = history_svc.new_session(client_id=db_id)
            st.session_state[_SVC_KEY] = None  # reinit AI context
            st.rerun()
    with tool3:
        if st.button("🕐 History", use_container_width=True):
            st.session_state[_HISTORY_OPEN] = not st.session_state.get(_HISTORY_OPEN, False)
            st.rerun()

    # ── History panel ─────────────────────────────────────────────────────────
    if st.session_state.get(_HISTORY_OPEN, False):
        _render_history_panel(history_svc, client_id=db_id, active_uuid=session_uuid)
        st.divider()

    # ── AI availability gate ──────────────────────────────────────────────────
    if not is_ai_available():
        st.warning(
            "**AI Chat requires a Gemini API key.**\n\n"
            "1. Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
            "2. Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`\n"
            "3. Restart the app\n\n"
            "_All other pages work without an API key._"
        )
        st.divider()
        for msg in st.session_state.get(MESSAGES_KEY, []):
            _render_message(msg["role"], msg["content"])
        return

    # ── Initialise GeminiService ──────────────────────────────────────────────
    service = _get_or_init_service(ref)
    if service is None:
        st.error("Failed to initialise AI service. Check your GEMINI_API_KEY.")
        return

    # ── Chat history display ──────────────────────────────────────────────────
    messages = st.session_state.get(MESSAGES_KEY, [])

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

    # ── Quick-prompt chips ────────────────────────────────────────────────────
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

    # Handle chip-triggered prompt
    if "_pending_prompt" in st.session_state:
        pending = st.session_state.pop("_pending_prompt")
        _send_message(
            pending, service, history_svc, session_uuid, db_id,
            is_first_message=(len(messages) == 0),
        )

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
            # Clear memory + start a fresh session, but don't delete the old one
            history_svc.new_session(client_id=db_id)
            st.session_state[_SVC_KEY] = None
            st.rerun()

    # Send on button click
    if send_clicked and user_input.strip():
        _send_message(
            user_input, service, history_svc, session_uuid, db_id,
            is_first_message=(len(messages) == 0),
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    if messages:
        turns = len(messages) // 2
        model = "gemini-2.5-flash"
        msg_count = history_svc._repo.get_message_count(session_uuid)
        st.markdown(
            f"<p style='font-size:0.7rem; color:rgba(255,255,255,0.25); text-align:right;'>"
            f"{turns} exchange(s) · {msg_count} msg(s) stored · {model}</p>",
            unsafe_allow_html=True,
        )
