"""
rag/vector_store.py
────────────────────
ChromaDB persistence layer for the RAG pipeline.

Responsibilities:
  - Create and persist a ChromaDB collection to disk
  - Upsert document chunks with their embeddings and metadata
  - Delete all chunks belonging to a specific source file
  - Query the collection by embedding vector
  - Report collection statistics

Design:
  - One shared collection named "advisor_research"
  - Persisted to settings.chroma_persist_dir (default: data/chroma/)
  - Uses deterministic IDs (chunk.chunk_id) so re-uploading the same file
    produces upserts, not duplicates
  - Embeddings are provided externally (EmbeddingService) — this module
    only handles storage and retrieval

Usage:
    store = VectorStore()
    store.upsert(chunks, embeddings)
    results = store.query(query_embedding, n_results=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rag.text_chunker import Chunk
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ChromaDB collection name (single collection for all research docs)
_COLLECTION_NAME = "advisor_research"


# ─────────────────────────────────────────────────────────────────────────────
# QueryResult
# ─────────────────────────────────────────────────────────────────────────────

class QueryResult:
    """
    A single search result returned by VectorStore.query().

    Attributes:
        chunk_id:   Unique chunk identifier.
        text:       The chunk text content.
        source:     Source filename (e.g. "Q4_Outlook.pdf").
        page:       Page number within the source document.
        score:      Similarity distance (lower = more similar for L2; for
                    cosine it's 1 - similarity, so lower = better match).
        metadata:   Full metadata dict stored with the chunk.
    """
    def __init__(self, chunk_id: str, text: str, source: str,
                 page: int, score: float, metadata: dict) -> None:
        self.chunk_id = chunk_id
        self.text     = text
        self.source   = source
        self.page     = page
        self.score    = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"QueryResult(source={self.source!r}, page={self.page}, "
            f"score={self.score:.4f}, text={self.text[:60]!r}...)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    ChromaDB-backed vector store for research document chunks.

    Example:
        store = VectorStore()
        store.upsert(chunks, embeddings)

        results = store.query(query_vec, n_results=5)
        for r in results:
            print(r.source, r.page, r.text[:100])
    """

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        """
        Args:
            persist_dir: Override default ChromaDB storage directory.
                         Defaults to settings.chroma_persist_dir.
        """
        self._persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client     = None
        self._collection = None
        self._init_collection()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_collection(self) -> None:
        """Connect to ChromaDB and get-or-create the collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # get_or_create is idempotent — safe to call on every startup
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},   # cosine similarity
            )
            logger.info(
                "VectorStore ready | collection=%s | docs=%d | path=%s",
                _COLLECTION_NAME, self._collection.count(), self._persist_dir,
            )
        except ImportError:
            raise RuntimeError(
                "chromadb is not installed. Run: pip install chromadb"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise ChromaDB: {exc}") from exc

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunks:     list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Insert or update chunks in the collection.

        Uses ChromaDB's upsert — if a chunk with the same ID already exists
        it is overwritten, so re-uploading the same PDF is safe.

        Args:
            chunks:     List of Chunk objects from text_chunker.
            embeddings: Parallel list of embedding vectors (one per chunk).
                        Must be the same length as chunks.

        Returns:
            Number of chunks upserted.

        Raises:
            ValueError: If chunks and embeddings lengths don't match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length."
            )
        if not chunks:
            logger.warning("upsert() called with empty chunks list — no-op.")
            return 0

        ids       = [c.chunk_id for c in chunks]
        texts     = [c.text for c in chunks]
        metadatas = [
            {
                "source":      c.source,
                "page":        str(c.page),          # ChromaDB requires str values
                "chunk_index": str(c.chunk_index),
                "total_chunks":str(c.total_chunks),
                "char_count":  str(c.char_end - c.char_start),
            }
            for c in chunks
        ]

        try:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(
                "Upserted %d chunks | sources: %s",
                len(chunks),
                list({c.source for c in chunks}),
            )
            return len(chunks)
        except Exception as exc:
            logger.error("ChromaDB upsert failed: %s", exc)
            raise

    def delete_source(self, source_filename: str) -> int:
        """
        Remove all chunks belonging to a specific source file.

        Args:
            source_filename: Filename as stored in chunk metadata
                             (e.g. "Q4_Market_Outlook.pdf").

        Returns:
            Number of chunks deleted (0 if none found).
        """
        try:
            results = self._collection.get(
                where={"source": source_filename},
                include=[],   # IDs only
            )
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                logger.info(
                    "Deleted %d chunks for source: %s", len(ids), source_filename
                )
            return len(ids)
        except Exception as exc:
            logger.error("Failed to delete source %s: %s", source_filename, exc)
            return 0

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        n_results:        int = 5,
        source_filter:    Optional[str] = None,
    ) -> list[QueryResult]:
        """
        Retrieve the most semantically similar chunks.

        Args:
            query_embedding: Embedding vector of the search query.
            n_results:       Maximum number of results to return.
            source_filter:   Optionally restrict results to one source file.

        Returns:
            List of QueryResult objects ordered by similarity (best first).
        """
        if not query_embedding:
            return []

        n_available = self._collection.count()
        if n_available == 0:
            logger.debug("Collection is empty — no results.")
            return []

        # Can't ask for more results than exist
        n_results = min(n_results, n_available)

        where = {"source": source_filter} if source_filter else None

        try:
            raw = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        results: list[QueryResult] = []
        for i, chunk_id in enumerate(raw["ids"][0]):
            text     = raw["documents"][0][i]
            meta     = raw["metadatas"][0][i]
            distance = raw["distances"][0][i]

            results.append(QueryResult(
                chunk_id=chunk_id,
                text=text,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=distance,
                metadata=meta,
            ))

        logger.debug(
            "Query returned %d results (top score=%.4f)",
            len(results), results[0].score if results else 0,
        )
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks stored in the collection."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """
        Return a deduplicated list of all source filenames in the collection.
        Returns an empty list if the collection is empty.
        """
        if self._collection.count() == 0:
            return []
        try:
            all_meta = self._collection.get(include=["metadatas"])["metadatas"]
            sources  = sorted({m.get("source", "unknown") for m in all_meta})
            return sources
        except Exception as exc:
            logger.error("Failed to list sources: %s", exc)
            return []

    def stats(self) -> dict:
        """Return a summary dict for UI display."""
        sources = self.list_sources()
        return {
            "total_chunks": self.count(),
            "total_sources": len(sources),
            "sources": sources,
            "persist_dir": str(self._persist_dir),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor (Streamlit-friendly)
# ─────────────────────────────────────────────────────────────────────────────

_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Return the shared VectorStore instance (created once per process).

    Usage:
        from rag.vector_store import get_vector_store
        store = get_vector_store()
        results = store.query(vec)
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance
