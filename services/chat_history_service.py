"""
services/chat_history_service.py
──────────────────────────────────
High-level service wrapping ChatRepository for the chat UI.

Responsibilities:
  - Manage the active session lifecycle (create, switch, title)
  - Load existing session messages back into Streamlit session state
  - Save messages to the DB as they arrive
  - Provide a list of recent sessions for the history sidebar panel

This is intentionally kept thin — all DB work is delegated to
database/repositories/chat_repository.py.
"""

from __future__ import annotations

import uuid
from typing import Optional

import streamlit as st

from database.repositories.chat_repository import ChatRepository
from database.repositories.audit_repository import get_audit_repo
from utils.logger import get_logger

logger = get_logger(__name__)

_repo  = ChatRepository()
_audit = get_audit_repo()

# Streamlit session-state keys owned by this service
SESSION_UUID_KEY = "chat_session_id"     # str: active UUID
MESSAGES_KEY     = "chat_messages"       # list[dict]: in-memory display history
LOADED_UUID_KEY  = "chat_loaded_uuid"    # str: which UUID was last loaded into state


class ChatHistoryService:
    """
    Manages persistent chat sessions for a single Streamlit user session.

    Usage:
        svc = ChatHistoryService()
        svc.ensure_active_session(client_id=ref.db_id)
        svc.save_message("user", user_input)
        svc.save_message("assistant", response)
        sessions = svc.get_recent_sessions(client_id=ref.db_id)
    """

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def ensure_active_session(self, client_id: Optional[int] = None) -> str:
        """
        Ensure a valid active session UUID is in Streamlit state.

        - If no session exists → create one.
        - If session exists but messages haven't been loaded from DB → load them.

        Args:
            client_id: Current client (used if creating a new session row).

        Returns:
            Active session UUID string.
        """
        if MESSAGES_KEY not in st.session_state:
            st.session_state[MESSAGES_KEY] = []

        # Create UUID if missing
        if not st.session_state.get(SESSION_UUID_KEY):
            session_uuid = str(uuid.uuid4())
            st.session_state[SESSION_UUID_KEY] = session_uuid
            _repo.create_session(session_uuid, client_id=client_id)
            logger.debug("ChatHistoryService: new session %s", session_uuid)
        else:
            session_uuid = st.session_state[SESSION_UUID_KEY]
            # Ensure the row exists (idempotent)
            _repo.ensure_session_exists(session_uuid, client_id=client_id)

        # Load messages from DB if we haven't loaded this session yet
        loaded_uuid = st.session_state.get(LOADED_UUID_KEY)
        if loaded_uuid != session_uuid:
            self._load_messages_from_db(session_uuid)
            st.session_state[LOADED_UUID_KEY] = session_uuid

        return session_uuid

    def _load_messages_from_db(self, session_uuid: str) -> None:
        """Replace in-memory history with messages from the DB for this session."""
        db_messages = _repo.get_messages(session_uuid)
        st.session_state[MESSAGES_KEY] = [
            {"role": m["role"], "content": m["content"]}
            for m in db_messages
        ]
        logger.debug(
            "ChatHistoryService: loaded %d messages for session %s",
            len(db_messages), session_uuid,
        )

    def switch_to_session(self, session_uuid: str) -> None:
        """
        Switch the active session to a different one (from session history panel).

        Loads that session's messages into state and updates the active UUID.
        The Gemini service history will be out of sync — caller must call
        service.clear() and replay if needed (we just reinitialise context).

        Args:
            session_uuid: UUID of the session to activate.
        """
        st.session_state[SESSION_UUID_KEY]  = session_uuid
        st.session_state[LOADED_UUID_KEY]   = None  # force re-load
        st.session_state[MESSAGES_KEY]      = []
        self._load_messages_from_db(session_uuid)
        logger.info("ChatHistoryService: switched to session %s", session_uuid)

    def new_session(self, client_id: Optional[int] = None) -> str:
        """
        Start a fresh blank session and update session state.

        Returns:
            New session UUID.
        """
        session_uuid = str(uuid.uuid4())
        _repo.create_session(session_uuid, client_id=client_id)
        st.session_state[SESSION_UUID_KEY] = session_uuid
        st.session_state[LOADED_UUID_KEY]  = session_uuid
        st.session_state[MESSAGES_KEY]     = []
        logger.info("ChatHistoryService: new blank session %s", session_uuid)
        return session_uuid

    def auto_title_session(self, session_uuid: str, first_user_message: str) -> None:
        """
        Set a session title from the first user message (truncated to 50 chars).

        Only updates if the session has no title yet.
        """
        existing = _repo.get_session(session_uuid)
        if existing and not existing.get("title"):
            title = first_user_message[:50].strip()
            if len(first_user_message) > 50:
                title += "…"
            _repo.update_session_title(session_uuid, title)

    # ── Message persistence ────────────────────────────────────────────────────

    def save_message(
        self,
        role: str,
        content: str,
        session_uuid: Optional[str] = None,
        client_id: Optional[int] = None,
    ) -> None:
        """
        Persist a message to the DB.

        Args:
            role:         "user" | "assistant".
            content:      Message text.
            session_uuid: Session to save to (defaults to active session).
            client_id:    Client FK (optional, for fast queries).
        """
        uuid = session_uuid or st.session_state.get(SESSION_UUID_KEY)
        if not uuid:
            logger.warning("ChatHistoryService.save_message: no active session UUID")
            return
        try:
            _repo.save_message(uuid, role, content, client_id=client_id)
        except Exception as exc:
            logger.error("ChatHistoryService.save_message failed: %s", exc)

    # ── History reads ──────────────────────────────────────────────────────────

    def get_recent_sessions(
        self,
        client_id: Optional[int] = None,
        limit: int = 15,
    ) -> list[dict]:
        """
        Return recent sessions for display in the history panel.

        If client_id is given, only that client's sessions are returned.
        Otherwise all sessions are returned (useful for the general chat case).

        Each dict has: session_uuid, title, created_at, message_count.
        """
        if client_id:
            return _repo.get_sessions_for_client(client_id, limit=limit)
        return _repo.get_all_sessions(limit=limit)

    def delete_session(self, session_uuid: str) -> None:
        """Delete a session and its messages. Clears active state if it's the active session."""
        _repo.delete_session(session_uuid)
        if st.session_state.get(SESSION_UUID_KEY) == session_uuid:
            st.session_state[SESSION_UUID_KEY] = None
            st.session_state[LOADED_UUID_KEY]  = None
            st.session_state[MESSAGES_KEY]     = []

    def get_active_uuid(self) -> Optional[str]:
        """Return the currently active session UUID from state."""
        return st.session_state.get(SESSION_UUID_KEY)


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[ChatHistoryService] = None

def get_chat_history_service() -> ChatHistoryService:
    """Return the app-wide ChatHistoryService singleton."""
    global _instance
    if _instance is None:
        _instance = ChatHistoryService()
    return _instance
