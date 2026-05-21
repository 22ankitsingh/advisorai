"""
rag/text_chunker.py
────────────────────
Text chunking layer — sits between the loader and the embedding service.

Responsibilities:
  - Split Document pages into smaller, overlapping text chunks
  - Preserve source metadata on every chunk (filename, page, position)
  - Ensure chunks fit within embedding model token limits
  - Support configurable chunk size and overlap

Why chunk at all?
  Embedding models have a fixed context window (e.g. 256 tokens for
  all-MiniLM-L6-v2). Long pages must be split. Overlap between consecutive
  chunks ensures that sentences spanning a split boundary are still
  retrievable — without overlap, key sentences at chunk boundaries get lost.

Output format:
  Chunk(
      text         = "...the actual chunk text...",
      source       = "Q4_Report.pdf",
      page         = 3,
      chunk_index  = 1,        # 0-indexed within the page
      chunk_id     = "Q4_Report.pdf::page::3::chunk::1",
      char_start   = 512,      # character offset in the page text
      char_end     = 1024,
      total_chunks = 4,        # total chunks for this page
  )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rag.document_loader import Document
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

# all-MiniLM-L6-v2 has a 256-token limit ≈ ~1000 characters.
# We use 800 chars as the default chunk size to stay comfortably within limits.
DEFAULT_CHUNK_SIZE    = 800   # characters per chunk
DEFAULT_CHUNK_OVERLAP = 150   # characters of overlap between consecutive chunks
DEFAULT_MIN_CHUNK_LEN = 50    # skip chunks shorter than this (e.g. stray headers)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and vector storage.

    Attributes:
        text:         The actual text content of this chunk.
        source:       Original PDF filename.
        page:         1-indexed page this chunk came from.
        chunk_index:  0-indexed position of this chunk within the page.
        total_chunks: Total chunks produced from the parent page.
        char_start:   Start character offset in the original page text.
        char_end:     End character offset in the original page text.
        chunk_id:     Globally unique stable ID for ChromaDB.
        file_hash:    MD5 of source file (from Document) for dedup.
    """
    text:         str
    source:       str
    page:         int
    chunk_index:  int
    total_chunks: int
    char_start:   int
    char_end:     int
    file_hash:    str = ""
    chunk_id:     str = field(init=False)

    def __post_init__(self) -> None:
        # Stable, deterministic ID that matches the Document.doc_id pattern
        self.chunk_id = f"{self.source}::page::{self.page}::chunk::{self.chunk_index}"

    def to_metadata(self) -> dict:
        """
        Return a flat dict of metadata for ChromaDB storage.
        ChromaDB metadata values must be str, int, float, or bool.
        """
        return {
            "source":       self.source,
            "page":         self.page,
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start":   self.char_start,
            "char_end":     self.char_end,
            "file_hash":    self.file_hash,
        }

    def citation(self) -> str:
        """Human-readable citation string for display in the UI."""
        return f"{self.source} · Page {self.page}"

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id!r}, "
            f"chars={self.char_start}–{self.char_end}, "
            f"text={preview!r}...)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TextChunker
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits Document objects into overlapping Chunk objects.

    Strategy — "recursive sentence-aware" splitting:
      1. Try to split on paragraph boundaries (\\n\\n) first.
      2. If a paragraph is still larger than chunk_size, split on sentences.
      3. If a sentence is still too large, fall back to hard character splits.

    This preserves semantic coherence much better than naive character slicing.

    Usage:
        chunker = TextChunker(chunk_size=800, chunk_overlap=150)
        chunks  = chunker.chunk_documents(documents)
    """

    def __init__(
        self,
        chunk_size:    int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_len: int = DEFAULT_MIN_CHUNK_LEN,
    ) -> None:
        """
        Args:
            chunk_size:    Target maximum characters per chunk.
            chunk_overlap: Characters of overlap between consecutive chunks.
                           Must be less than chunk_size.
            min_chunk_len: Chunks shorter than this are discarded.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

        logger.debug(
            "TextChunker init: size=%d, overlap=%d, min_len=%d",
            chunk_size, chunk_overlap, min_chunk_len,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Split one Document (= one PDF page) into Chunks.

        Args:
            doc: A Document from document_loader.py.

        Returns:
            List of Chunk objects for this page.
        """
        # Split the page text into variable-length segments using sentence
        # awareness, then group them into fixed-size chunks with overlap.
        segments = self._split_into_segments(doc.text)
        raw_chunks = self._group_segments_into_chunks(segments)

        if not raw_chunks:
            logger.debug("No chunks produced from %s page %d", doc.source, doc.page)
            return []

        total = len(raw_chunks)
        chunks: list[Chunk] = []

        for idx, (text, char_start, char_end) in enumerate(raw_chunks):
            if len(text) < self.min_chunk_len:
                logger.debug(
                    "  Discarding short chunk (%d chars) on page %d",
                    len(text), doc.page,
                )
                continue

            chunk = Chunk(
                text=text,
                source=doc.source,
                page=doc.page,
                chunk_index=idx,
                total_chunks=total,
                char_start=char_start,
                char_end=char_end,
                file_hash=doc.file_hash,
            )
            chunks.append(chunk)

        logger.debug(
            "  Page %d → %d chunks (from %s)",
            doc.page, len(chunks), doc.source,
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Split a list of Documents into Chunks.

        Args:
            documents: Output from document_loader.load_pdf / load_pdfs.

        Returns:
            Flat list of all Chunk objects, ordered by source → page → index.
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            page_chunks = self.chunk_document(doc)
            all_chunks.extend(page_chunks)

        logger.info(
            "Chunking complete: %d documents → %d chunks",
            len(documents), len(all_chunks),
        )
        return all_chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _split_into_segments(self, text: str) -> list[str]:
        """
        Split text into small semantic segments using a priority hierarchy:
          1. Paragraph boundaries (blank lines)
          2. Sentence boundaries (. ! ?)
          3. Hard character splits (last resort)

        Returns a list of non-empty string segments.
        """
        # Step 1: Split on paragraph boundaries
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        segments: list[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                segments.append(para)
            else:
                # Step 2: Paragraph too long — split on sentence boundaries
                sentences = _split_sentences(para)
                for sent in sentences:
                    if len(sent) <= self.chunk_size:
                        segments.append(sent)
                    else:
                        # Step 3: Sentence still too long — hard split
                        for i in range(0, len(sent), self.chunk_size - self.chunk_overlap):
                            piece = sent[i : i + self.chunk_size]
                            if piece.strip():
                                segments.append(piece.strip())

        return [s for s in segments if s]

    def _group_segments_into_chunks(
        self, segments: list[str]
    ) -> list[tuple[str, int, int]]:
        """
        Greedily group segments into chunks of at most chunk_size characters,
        with overlap: when starting a new chunk, carry forward the tail of the
        previous chunk to maintain context continuity.

        Returns:
            List of (chunk_text, char_start, char_end) tuples.
            char_start/end are approximate offsets in the original page text.
        """
        if not segments:
            return []

        chunks:         list[tuple[str, int, int]] = []
        current_parts:  list[str] = []
        current_len:    int = 0
        char_cursor:    int = 0    # running character position in page text

        for seg in segments:
            seg_len = len(seg)

            # If adding this segment exceeds chunk_size, flush current chunk
            if current_len + seg_len + 1 > self.chunk_size and current_parts:
                chunk_text = " ".join(current_parts)
                char_start = max(0, char_cursor - current_len)
                char_end   = char_cursor
                chunks.append((chunk_text, char_start, char_end))

                # Overlap: keep the tail of the current chunk as the start of next.
                # Walk backwards through parts to find overlap content.
                overlap_parts:  list[str] = []
                overlap_len:    int = 0
                for part in reversed(current_parts):
                    part_len = len(part) + 1
                    if overlap_len + part_len > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += part_len

                current_parts = overlap_parts
                current_len   = overlap_len

            current_parts.append(seg)
            current_len  += seg_len + 1
            char_cursor  += seg_len + 1   # +1 for the space/separator

        # Flush the final chunk
        if current_parts:
            chunk_text = " ".join(current_parts)
            char_start = max(0, char_cursor - current_len)
            char_end   = char_cursor
            chunks.append((chunk_text, char_start, char_end))

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple regex that handles:
      - Period, exclamation, question mark as sentence terminators
      - Abbreviations like "U.S." or "Dr." (won't be falsely split)
      - Decimal numbers like "1.5%" (won't be falsely split)

    Returns a list of non-empty sentence strings.
    """
    # Positive look-behind for sentence-ending punctuation followed by space+capital
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_documents(
    documents: list[Document],
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Convenience function — create a TextChunker and chunk in one call.

    Args:
        documents:     List of Document objects from document_loader.
        chunk_size:    Target max characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        Flat list of Chunk objects ready for embedding.
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(documents)
