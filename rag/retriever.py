"""
rag/retriever.py
─────────────────
High-level RAG orchestration — the public interface for the research page.

Responsibilities:
  - Ingest uploaded PDFs end-to-end (load → chunk → embed → store)
  - Execute semantic search (embed query → vector lookup)
  - Generate Gemini answers grounded in retrieved context
  - Return typed results ready for the UI to render

This module is the ONLY thing pages/research.py needs to import.
It hides all pipeline complexity behind two clean functions:
  - ingest_files(files)  → IndexResult
  - search(query)        → SearchResult

Usage:
    from rag.retriever import get_retriever

    retriever = get_retriever()
    idx = retriever.ingest_files(uploaded_files)   # st.file_uploader result
    res = retriever.search("What is the outlook for tech stocks?")
    print(res.answer)
    for chunk in res.chunks:
        print(chunk.source, chunk.page, chunk.text[:200])
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Optional

from rag.document_loader import load_pdfs, Document
from rag.text_chunker import chunk_documents, Chunk
from rag.embedding_service import get_embedding_service
from rag.vector_store import get_vector_store, QueryResult
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_CONTEXT_CHARS = 6_000   # Limit context injected into Gemini prompt


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexResult:
    """Outcome of ingesting one or more PDF files."""
    files_processed:   int
    files_failed:      int
    chunks_indexed:    int
    total_chunks_stored: int
    failed_files:      list[str] = field(default_factory=list)
    error:             str = ""

    @property
    def success(self) -> bool:
        return self.files_processed > 0 and not self.error


@dataclass
class SearchResult:
    """Outcome of a semantic search query."""
    query:     str
    chunks:    list[QueryResult]      # Retrieved context chunks
    answer:    str                    # Gemini-generated answer (or excerpt if no AI)
    is_ai:     bool                   # True if Gemini was used
    error:     str = ""

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    End-to-end RAG pipeline for research document Q&A.

    Pipeline:
      Ingest:  PDF files → Documents → Chunks → Embeddings → ChromaDB
      Search:  Query → Embedding → ChromaDB lookup → Gemini synthesis → Answer
    """

    def __init__(
        self,
        n_results:  int = 5,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
    ) -> None:
        """
        Args:
            n_results:     Number of chunks to retrieve per query.
            chunk_size:    Target character count per chunk.
            chunk_overlap: Characters of overlap between consecutive chunks.
        """
        self._n_results    = n_results
        self._chunk_size   = chunk_size
        self._chunk_overlap= chunk_overlap
        self._embedder     = get_embedding_service()
        self._store        = get_vector_store()

        logger.info(
            "Retriever ready | chunks_stored=%d | n_results=%d",
            self._store.count(), n_results,
        )

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest_files(self, files: list) -> IndexResult:
        """
        Full ingestion pipeline: load PDFs → chunk → embed → store.

        Args:
            files: List of Streamlit UploadedFile objects (or bytes / Path).

        Returns:
            IndexResult with counts and any error info.
        """
        if not files:
            return IndexResult(
                files_processed=0, files_failed=0,
                chunks_indexed=0, total_chunks_stored=self._store.count(),
                error="No files provided.",
            )

        # 1. Load PDFs → Documents
        logger.info("Ingesting %d file(s)...", len(files))
        all_docs, load_results = load_pdfs(files)

        failed = [r.filename for r in load_results if not r.success]
        if not all_docs:
            return IndexResult(
                files_processed=0,
                files_failed=len(failed),
                chunks_indexed=0,
                total_chunks_stored=self._store.count(),
                failed_files=failed,
                error="No text could be extracted from the uploaded file(s).",
            )

        # 2. Chunk documents
        chunks = chunk_documents(
            all_docs,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        logger.info("Created %d chunks from %d pages", len(chunks), len(all_docs))

        if not chunks:
            return IndexResult(
                files_processed=len(all_docs),
                files_failed=len(failed),
                chunks_indexed=0,
                total_chunks_stored=self._store.count(),
                error="Documents were loaded but produced no chunks.",
            )

        # 3. Generate embeddings
        try:
            texts      = [c.text for c in chunks]
            embeddings = self._embedder.encode(texts, show_progress=False)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return IndexResult(
                files_processed=len(all_docs),
                files_failed=len(failed),
                chunks_indexed=0,
                total_chunks_stored=self._store.count(),
                error=f"Embedding generation failed: {exc}",
            )

        # 4. Upsert into ChromaDB
        try:
            indexed = self._store.upsert(chunks, embeddings)
        except Exception as exc:
            logger.error("Vector store upsert failed: %s", exc)
            return IndexResult(
                files_processed=len(all_docs),
                files_failed=len(failed),
                chunks_indexed=0,
                total_chunks_stored=self._store.count(),
                error=f"Failed to store embeddings: {exc}",
            )

        return IndexResult(
            files_processed=len(load_results) - len(failed),
            files_failed=len(failed),
            chunks_indexed=indexed,
            total_chunks_stored=self._store.count(),
            failed_files=failed,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:         str,
        source_filter: Optional[str] = None,
    ) -> SearchResult:
        """
        Semantic search + optional Gemini-grounded answer generation.

        Pipeline:
          1. Embed the query
          2. Retrieve top-N similar chunks from ChromaDB
          3. If AI available: pass chunks as context to Gemini, return answer
          4. If AI unavailable: return the top chunk excerpt as a plain answer

        Args:
            query:         Natural language search query.
            source_filter: Restrict results to a specific PDF filename.

        Returns:
            SearchResult with retrieved chunks and synthesised answer.
        """
        query = query.strip()
        if not query:
            return SearchResult(query=query, chunks=[], answer="", is_ai=False,
                                error="Empty query.")

        if self._store.count() == 0:
            return SearchResult(
                query=query, chunks=[], answer="", is_ai=False,
                error="No documents have been indexed yet. Please upload a PDF first.",
            )

        # 1. Embed the query
        try:
            query_vec = self._embedder.encode_single(query)
        except Exception as exc:
            return SearchResult(
                query=query, chunks=[], answer="", is_ai=False,
                error=f"Embedding error: {exc}",
            )

        # 2. Retrieve chunks
        chunks = self._store.query(
            query_embedding=query_vec,
            n_results=self._n_results,
            source_filter=source_filter,
        )

        if not chunks:
            return SearchResult(
                query=query, chunks=[], answer="No relevant content found.", is_ai=False,
            )

        # 3. Generate answer
        answer, is_ai = _synthesise_answer(query, chunks)

        return SearchResult(
            query=query,
            chunks=chunks,
            answer=answer,
            is_ai=is_ai,
        )

    # ── Store passthrough ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return vector store statistics (chunk count, sources, etc.)."""
        return self._store.stats()

    def list_sources(self) -> list[str]:
        """Return all indexed source filenames."""
        return self._store.list_sources()

    def delete_source(self, filename: str) -> int:
        """Remove all chunks for a specific source file."""
        return self._store.delete_source(filename)


# ─────────────────────────────────────────────────────────────────────────────
# Answer synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[QueryResult]) -> str:
    """
    Assemble retrieved chunks into a numbered context block for the prompt.
    Truncates at _MAX_CONTEXT_CHARS to stay within token limits.
    """
    lines: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        entry = (
            f"[{i}] Source: {chunk.source} (page {chunk.page})\n"
            f"{chunk.text.strip()}\n"
        )
        if total_chars + len(entry) > _MAX_CONTEXT_CHARS:
            lines.append(f"[{i}] ... (context truncated)")
            break
        lines.append(entry)
        total_chars += len(entry)

    return "\n---\n".join(lines)


def _synthesise_answer(query: str, chunks: list[QueryResult]) -> tuple[str, bool]:
    """
    Generate an answer grounded in retrieved chunks.

    Returns:
        (answer_text, is_ai_generated)
    """
    from services.gemini_service import is_ai_available, GeminiService, GeminiServiceError

    context_block = _build_context_block(chunks)

    if not is_ai_available():
        # Fallback: return the best chunk excerpt with a note
        best = chunks[0]
        excerpt = textwrap.shorten(best.text, width=500, placeholder="...")
        return (
            f"**Top result** from *{best.source}* (page {best.page}):\n\n"
            f"> {excerpt}\n\n"
            f"*Add a GEMINI_API_KEY to .env to enable AI-synthesised answers.*",
            False,
        )

    prompt = textwrap.dedent(f"""
        You are a financial research assistant. Answer the question below using
        ONLY the provided document excerpts. Cite your sources using [1], [2], etc.
        If the answer is not in the documents, say so clearly — do not invent facts.

        Question: {query}

        Document excerpts:
        {context_block}

        Provide a concise, accurate answer with citations:
    """).strip()

    try:
        gemini  = GeminiService()
        answer  = gemini.chat(prompt)
        return answer, True
    except GeminiServiceError as exc:
        logger.error("Gemini synthesis failed: %s", exc)
        best    = chunks[0]
        excerpt = textwrap.shorten(best.text, width=500, placeholder="...")
        return (
            f"*AI answer unavailable ({exc})*\n\n"
            f"**Top result** from *{best.source}* (page {best.page}):\n\n"
            f"> {excerpt}",
            False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """
    Return the shared Retriever instance (lazy singleton).

    Usage:
        from rag.retriever import get_retriever
        r = get_retriever()
        r.ingest_files(uploaded_files)
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
