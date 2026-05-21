"""
services/gemini_service.py
───────────────────────────
Gemini AI service — all LLM logic lives here, zero UI coupling.

Responsibilities:
  - Initialise the Gemini client
  - Maintain multi-turn conversation history (native Gemini chat session)
  - Build system prompts with optional client financial context
  - Send messages and stream responses with full error handling
  - Expose a clean interface the UI calls without knowing any LLM internals

Architecture:
  We use the google-generativeai SDK directly (genai.ChatSession) for multi-turn
  conversation — it natively maintains the history as a list of Content objects.

  We also keep a simple Python list as a portable shadow copy of the history so
  Streamlit can render it without touching the Gemini internals.
"""

from __future__ import annotations

import textwrap
from typing import Generator, Optional

import google.generativeai as genai

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = textwrap.dedent("""
    You are Advisor AI, an intelligent financial advisor assistant designed to
    help human financial advisors serve their clients better.

    Your capabilities include:
    - Analysing client portfolio data and explaining it in plain English
    - Identifying concentration risk, rebalancing opportunities, and performance outliers
    - Answering questions about investment strategies, asset classes, and market concepts
    - Generating concise client summaries and compliance observations
    - Providing educational explanations about financial instruments

    Tone guidelines:
    - Professional but approachable — avoid jargon unless the advisor uses it first
    - Be concise; use bullet points when listing multiple items
    - Always note that your output is for informational purposes only and does
      not constitute regulated financial advice
    - When uncertain, say so rather than fabricating data

    {client_context}
""").strip()

_CLIENT_CONTEXT_TEMPLATE = textwrap.dedent("""
    ─── ACTIVE CLIENT CONTEXT ────────────────────────────
    Name:         {name}
    Risk Profile: {risk_profile}
    AUM:          ${aum:,.0f}
    Advisor Notes: {notes}
    ──────────────────────────────────────────────────────
    Use this context to give relevant, personalised responses.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class GeminiServiceError(Exception):
    """Raised when the Gemini service cannot complete a request."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# GeminiService
# ─────────────────────────────────────────────────────────────────────────────

class GeminiService:
    """
    Manages a single Gemini chat session with full multi-turn memory.

    Example usage:
        service = GeminiService(client_context={"name": "Sarah", ...})
        response = service.chat("What's this client's biggest risk?")

        # Or with streaming (Streamlit):
        with st.chat_message("assistant"):
            full = st.write_stream(service.stream_chat(user_input))
    """

    def __init__(self, client_context: Optional[dict] = None) -> None:
        """
        Initialise the Gemini service.

        Args:
            client_context: Optional dict with client data to inject into the
                            system prompt. Keys: name, risk_profile, aum, advisor_notes.
        """
        _validate_api_key()

        # Configure the SDK with our API key
        genai.configure(api_key=settings.gemini_api_key)

        # Build system instruction (injected once at session start)
        self._system_instruction = _build_system_prompt(client_context)

        # Create the GenerativeModel with generation config from settings
        self._model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=self._system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_tokens,
                candidate_count=1,
            ),
        )

        # Native Gemini chat session — handles multi-turn history internally
        self._chat_session = self._model.start_chat(history=[])

        # Shadow copy of messages as plain dicts for the Streamlit UI to render.
        # Format: [{"role": "user"|"assistant", "content": "..."}]
        self._messages: list[dict] = []

        logger.info(
            "GeminiService ready | model=%s | temp=%.1f | client=%s",
            settings.gemini_model,
            settings.gemini_temperature,
            client_context.get("name", "none") if client_context else "none",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Send a message and return the complete AI response as a string.

        Args:
            user_message: The text the user typed.

        Returns:
            The assistant's full response text.

        Raises:
            GeminiServiceError: On API failure or missing key.
        """
        if not user_message.strip():
            return ""

        logger.debug("Sending to Gemini: %s", user_message[:100])

        try:
            response = self._chat_session.send_message(user_message)
            reply = response.text

            # Save to shadow history for UI rendering
            self._messages.append({"role": "user",      "content": user_message})
            self._messages.append({"role": "assistant", "content": reply})

            logger.debug("Gemini replied (%d chars)", len(reply))
            return reply

        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise GeminiServiceError(_friendly_error(exc)) from exc

    def stream_chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Send a message and yield response text chunks as they stream in.

        Designed for use with Streamlit's st.write_stream():

            with st.chat_message("assistant"):
                full_response = st.write_stream(service.stream_chat(prompt))

        Yields:
            str: Text chunks from the Gemini streaming response.

        Raises:
            GeminiServiceError: On API failure.
        """
        if not user_message.strip():
            return

        logger.debug("Streaming to Gemini: %s", user_message[:100])

        try:
            collected: list[str] = []

            response = self._chat_session.send_message(
                user_message, stream=True
            )

            for chunk in response:
                piece = chunk.text
                collected.append(piece)
                yield piece

            # After stream completes, save full exchange to shadow history
            full_reply = "".join(collected)
            self._messages.append({"role": "user",      "content": user_message})
            self._messages.append({"role": "assistant", "content": full_reply})

            logger.debug("Stream complete (%d chars)", len(full_reply))

        except Exception as exc:
            logger.error("Gemini streaming error: %s", exc)
            raise GeminiServiceError(_friendly_error(exc)) from exc

    def get_messages(self) -> list[dict]:
        """
        Return the full conversation history as a list of dicts.

        Returns:
            [{"role": "user"|"assistant", "content": "..."}, ...]
        """
        return list(self._messages)

    def clear(self) -> None:
        """Reset the conversation — fresh Gemini session, empty history."""
        self._chat_session = self._model.start_chat(history=[])
        self._messages = []
        logger.info("GeminiService: conversation cleared.")

    @property
    def message_count(self) -> int:
        """Number of complete user↔assistant exchanges in this session."""
        # Each exchange = 2 messages (user + assistant)
        return len(self._messages) // 2


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_ai_available() -> bool:
    """
    Return True if a valid Gemini API key is configured.
    Use this in the UI to gate AI-dependent features.
    """
    key = settings.gemini_api_key
    return bool(key and key != "your_gemini_api_key_here")


def _validate_api_key() -> None:
    """Raise GeminiServiceError immediately if the API key is missing."""
    key = settings.gemini_api_key
    if not key or key == "your_gemini_api_key_here":
        raise GeminiServiceError(
            "GEMINI_API_KEY is not set.\n\n"
            "To fix this:\n"
            "  1. Get a free key at https://aistudio.google.com/app/apikey\n"
            "  2. Add it to your .env file:  GEMINI_API_KEY=your_key_here\n"
            "  3. Restart the app."
        )


def _build_system_prompt(client_context: Optional[dict]) -> str:
    """
    Assemble the full system prompt string.

    Args:
        client_context: Client data dict, or None for general mode.

    Returns:
        Complete system instruction for the Gemini model.
    """
    if client_context:
        client_block = _CLIENT_CONTEXT_TEMPLATE.format(
            name=client_context.get("name", "Unknown"),
            risk_profile=client_context.get("risk_profile", "N/A").title(),
            aum=float(client_context.get("aum", 0)),
            notes=client_context.get("advisor_notes") or "No notes on file.",
        )
    else:
        client_block = (
            "No client is currently selected. "
            "Respond in general financial advisory mode."
        )

    return _BASE_SYSTEM_PROMPT.format(client_context=client_block)


def _friendly_error(exc: Exception) -> str:
    """
    Convert a raw Gemini/network exception into a user-readable message.
    """
    msg = str(exc).lower()
    if "api_key" in msg or "api key" in msg or "permission" in msg:
        return "Invalid API key. Check your GEMINI_API_KEY in .env."
    if "quota" in msg or "rate" in msg or "429" in msg:
        return "Gemini rate limit reached. Please wait a moment and try again."
    if "timeout" in msg or "deadline" in msg:
        return "Request timed out. Check your internet connection."
    if "safety" in msg or "blocked" in msg:
        return "This message was blocked by Gemini's safety filters. Try rephrasing."
    return f"AI service error: {exc}"
