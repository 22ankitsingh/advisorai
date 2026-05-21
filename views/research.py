"""
pages/research.py
──────────────────
Research Hub — RAG-powered PDF search and Q&A.

Pipeline (all backend work in rag/):
  Upload: PDF → document_loader → text_chunker → embedding_service → ChromaDB
  Search: query → embedding_service → ChromaDB → Gemini synthesis → answer

This page is the sole UI entry point for the RAG system.
It imports only from rag.retriever — all pipeline complexity is hidden there.
"""

from __future__ import annotations

import streamlit as st

from rag.retriever import get_retriever
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Severity-to-colour mapping for source cards
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_TO_COLOUR = {
    0.20: "#2ed573",   # Excellent match (cosine dist < 0.20)
    0.40: "#7c3aed",   # Good
    0.60: "#ffa502",   # Fair
    1.00: "#ff4757",   # Weak
}


def _score_colour(score: float) -> str:
    for threshold, colour in _SCORE_TO_COLOUR.items():
        if score <= threshold:
            return colour
    return "#888"


def _score_label(score: float) -> str:
    if score <= 0.20:   return "Excellent"
    if score <= 0.35:   return "Strong"
    if score <= 0.50:   return "Good"
    if score <= 0.65:   return "Fair"
    return "Weak"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_upload_section(retriever) -> bool:
    """
    Upload panel — returns True if new files were just indexed.
    Uses an expander so it doesn't dominate the page after first use.
    """
    stats = retriever.stats()
    label = (
        f"📤 Upload Documents  ·  {stats['total_chunks']} chunks · "
        f"{stats['total_sources']} source(s) indexed"
        if stats["total_chunks"] > 0
        else "📤 Upload Documents  ·  No documents indexed yet"
    )

    with st.expander(label, expanded=(stats["total_chunks"] == 0)):
        st.markdown(
            "Upload PDF research documents. They will be chunked, embedded, "
            "and stored in ChromaDB for semantic search."
        )

        uploaded = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_uploader",
            label_visibility="collapsed",
        )

        col_index, col_clear = st.columns([3, 1])
        with col_index:
            index_clicked = st.button(
                "⚡ Index Documents",
                type="primary",
                disabled=not uploaded,
                use_container_width=True,
                key="btn_index",
            )
        with col_clear:
            clear_clicked = st.button(
                "🗑️ Clear All",
                use_container_width=True,
                key="btn_clear_all",
                disabled=(stats["total_chunks"] == 0),
            )

        # ── Clear all indexed docs ─────────────────────────────────────────
        if clear_clicked:
            for src in retriever.list_sources():
                retriever.delete_source(src)
            st.success("All indexed documents cleared.")
            st.rerun()

        # ── Index uploaded files ───────────────────────────────────────────
        if index_clicked and uploaded:
            with st.spinner(f"Indexing {len(uploaded)} file(s)..."):
                result = retriever.ingest_files(uploaded)

            if result.success:
                st.success(
                    f"✅ Indexed **{result.chunks_indexed}** chunks from "
                    f"**{result.files_processed}** file(s). "
                    f"Total in store: **{result.total_chunks_stored}** chunks."
                )
                if result.failed_files:
                    st.warning(f"⚠️ Failed: {', '.join(result.failed_files)}")
            else:
                st.error(f"❌ Indexing failed: {result.error}")
                if result.failed_files:
                    st.warning(f"Failed files: {', '.join(result.failed_files)}")

            st.rerun()

        # ── Indexed sources list ───────────────────────────────────────────
        sources = retriever.list_sources()
        if sources:
            st.markdown("**Indexed documents:**")
            for src in sources:
                del_col, name_col = st.columns([1, 8])
                with name_col:
                    st.markdown(
                        f"<span style='color:#a78bfa;'>📄</span> {src}",
                        unsafe_allow_html=True,
                    )
                with del_col:
                    if st.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                        n = retriever.delete_source(src)
                        st.toast(f"Removed {n} chunks for '{src}'")
                        st.rerun()

    return False


def _render_search_results(result) -> None:
    """Display search answer + source chunk cards."""
    if result.error:
        if "No documents" in result.error:
            st.info("📂 " + result.error)
        else:
            st.error(result.error)
        return

    if not result.has_results:
        st.warning("No relevant content found for your query. Try different keywords.")
        return

    # ── AI answer box ──────────────────────────────────────────────────────
    badge = (
        '<span style="background:#7c3aed20; color:#a78bfa; padding:2px 10px; '
        'border-radius:12px; font-size:0.75rem; font-weight:600; '
        'border:1px solid #7c3aed50;">🤖 AI Answer</span>'
        if result.is_ai
        else '<span style="background:#06b6d420; color:#06b6d4; padding:2px 10px; '
        'border-radius:12px; font-size:0.75rem; font-weight:600; '
        'border:1px solid #06b6d450;">📄 Excerpt</span>'
    )
    st.markdown(
        f'<div style="background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.25); '
        f'border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1.2rem;">'
        f'<div style="margin-bottom:0.6rem;">{badge}</div>'
        f'<div style="color:#e8e8f0; line-height:1.65;">{result.answer}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Source chunks ──────────────────────────────────────────────────────
    st.markdown(f"**{len(result.chunks)} source chunk(s) retrieved:**")

    for i, chunk in enumerate(result.chunks, 1):
        colour = _score_colour(chunk.score)
        label  = _score_label(chunk.score)

        with st.expander(
            f"[{i}] {chunk.source}  ·  page {chunk.page}  ·  {label} match",
            expanded=(i == 1),
        ):
            st.markdown(
                f'<div style="border-left:3px solid {colour}; '
                f'padding:8px 14px; border-radius:0 8px 8px 0; '
                f'background:rgba(255,255,255,0.03);">'
                f'<p style="color:rgba(255,255,255,0.85); '
                f'font-size:0.88rem; line-height:1.6; margin:0;">'
                f'{chunk.text}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Source: **{chunk.source}** · Page {chunk.page} · "
                f"Similarity distance: {chunk.score:.4f} ({label})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Research Hub page."""
    st.markdown("## 🔍 Research Hub")
    st.markdown("*Semantic search across indexed financial research documents*")
    st.divider()

    # Retriever is a singleton — ChromaDB stays connected across reruns
    retriever = get_retriever()

    # ── Upload section ────────────────────────────────────────────────────
    _render_upload_section(retriever)
    st.divider()

    # ── Stats row ────────────────────────────────────────────────────────
    stats = retriever.stats()
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("📦 Chunks Indexed", stats["total_chunks"])
    with s2:
        st.metric("📄 Documents", stats["total_sources"])
    with s3:
        from services.gemini_service import is_ai_available
        ai_status = "✅ Active" if is_ai_available() else "⚠️ No API key"
        st.metric("🤖 AI Answers", ai_status)

    st.divider()

    # ── Search section ────────────────────────────────────────────────────
    st.markdown("### 🔎 Search")

    search_col, filter_col = st.columns([4, 1])
    with search_col:
        query = st.text_input(
            "Query",
            placeholder="e.g. 'Federal Reserve rate outlook' · 'ESG emerging markets' · 'bond duration risk'",
            label_visibility="collapsed",
            key="rag_query",
        )
    with filter_col:
        sources = retriever.list_sources()
        source_options = ["All documents"] + sources
        selected_source = st.selectbox(
            "Source filter",
            options=source_options,
            key="rag_source_filter",
            label_visibility="collapsed",
        )

    n_results = st.slider(
        "Max results", min_value=1, max_value=10, value=5, key="rag_n_results"
    )

    search_clicked = st.button(
        "🔍 Search",
        type="primary",
        disabled=(not query.strip()),
        key="btn_search",
    )

    # ── Run search ────────────────────────────────────────────────────────
    if search_clicked and query.strip():
        src_filter = None if selected_source == "All documents" else selected_source

        # Update retriever n_results on the fly
        retriever._n_results = n_results

        with st.spinner("Searching and generating answer..."):
            result = retriever.search(query, source_filter=src_filter)

        st.session_state["rag_last_result"] = result
        st.session_state["rag_last_query"]  = query

    # ── Display last result (persists across reruns) ──────────────────────
    if "rag_last_result" in st.session_state:
        last_result = st.session_state["rag_last_result"]
        last_query  = st.session_state.get("rag_last_query", "")

        st.markdown(f"**Results for:** *{last_query}*")
        _render_search_results(last_result)

    elif stats["total_chunks"] == 0:
        st.info(
            "👆 Upload at least one PDF document to start searching.\n\n"
            "Supported: financial reports, market outlooks, research notes, strategy papers."
        )
    else:
        st.info(
            f"✅ {stats['total_chunks']} chunks ready across {stats['total_sources']} document(s). "
            "Enter a query above to search."
        )
