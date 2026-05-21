"""
rag/embedding_service.py
─────────────────────────
Embedding generation using SentenceTransformers.

Responsibilities:
  - Load the embedding model once (cached as singleton)
  - Encode lists of text strings into float vectors
  - Return consistent embedding dimensionality for ChromaDB

Model: all-MiniLM-L6-v2 (configurable via settings.embedding_model)
  - 384-dimensional embeddings
  - Fast CPU inference (~50ms per batch of 32 chunks)
  - Downloaded automatically on first use (~90MB, cached in ~/.cache)

Usage:
    svc = EmbeddingService()
    vecs = svc.encode(["text one", "text two"])  # list[list[float]]
"""

from __future__ import annotations

import threading
from typing import Optional

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton model loader (thread-safe, one load per process)
# ─────────────────────────────────────────────────────────────────────────────

_model      = None
_model_lock = threading.Lock()


def _get_model():
    """
    Lazily load and cache the SentenceTransformer model.
    Thread-safe — multiple Streamlit threads won't double-load.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:   # Double-checked locking
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", settings.embedding_model)
            _model = SentenceTransformer(settings.embedding_model)
            # API changed in sentence-transformers 3.x
            try:
                dim = _model.get_embedding_dimension()
            except AttributeError:
                dim = _model.get_sentence_embedding_dimension()
            logger.info("Embedding model ready | dim=%d", dim)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding model: {exc}") from exc

    return _model


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingService
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Thin wrapper around SentenceTransformer that the rest of the RAG
    pipeline can depend on without importing sentence_transformers directly.

    Example:
        svc  = EmbeddingService()
        vecs = svc.encode(["inflation hedge", "bond duration"])
        # vecs: list[list[float]], shape (2, 384)
    """

    def __init__(self, batch_size: int = 32) -> None:
        """
        Args:
            batch_size: Chunk encoding batch size. Larger = faster but more RAM.
        """
        self._batch_size = batch_size
        self._model      = None   # Lazy — don't load until first encode()

    def encode(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Encode a list of text strings into embedding vectors.

        Args:
            texts:         List of strings to embed. Empty strings are accepted
                           but will produce noisy vectors — filter beforehand.
            show_progress: Show a tqdm progress bar (useful for large batches).

        Returns:
            List of float lists, one per input string.
            Dimensionality is determined by the configured model (384 for MiniLM).

        Raises:
            RuntimeError: If the model fails to load.
        """
        if not texts:
            return []

        model = _get_model()

        logger.debug("Encoding %d texts (batch_size=%d)", len(texts), self._batch_size)
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """
        Encode a single string. Convenience wrapper for query encoding.

        Args:
            text: The query or string to embed.

        Returns:
            A single embedding vector as a list of floats.
        """
        result = self.encode([text])
        return result[0] if result else []

    @property
    def embedding_dim(self) -> int:
        """Return the output dimensionality of the current model."""
        model = _get_model()
        try:
            return model.get_embedding_dimension()
        except AttributeError:
            return model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return settings.embedding_model


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (recommended usage pattern)
# ─────────────────────────────────────────────────────────────────────────────

_svc_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Return the shared EmbeddingService instance.
    Creates it on first call (lazy + singleton pattern).

    Usage:
        from rag.embedding_service import get_embedding_service
        svc  = get_embedding_service()
        vecs = svc.encode(chunks)
    """
    global _svc_instance
    if _svc_instance is None:
        _svc_instance = EmbeddingService()
    return _svc_instance
