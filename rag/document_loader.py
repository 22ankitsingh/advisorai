"""
rag/document_loader.py
───────────────────────
PDF ingestion layer — the entry point of the RAG pipeline.

Responsibilities:
  - Accept one or more uploaded Streamlit file objects (or local file paths)
  - Extract raw text page-by-page using pypdf
  - Return structured Document objects carrying text + metadata
  - Handle corrupt / encrypted / empty PDFs gracefully without crashing

Design principle:
  This module only extracts text. It knows nothing about chunking, embeddings,
  or storage — those are handled downstream in the pipeline.

Output format (a Document is a plain dataclass):
  Document(
      text        = "raw extracted text...",
      source      = "Q4_Report.pdf",
      page        = 1,              # 1-indexed
      total_pages = 12,
      doc_id      = "Q4_Report.pdf::page::1"
  )
"""

from __future__ import annotations

import io
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pypdf

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """
    A single page of extracted text from a source document.

    Attributes:
        text:        Raw extracted text from this page.
        source:      Original filename (e.g. "Q4_Market_Outlook.pdf").
        page:        1-indexed page number within the source file.
        total_pages: Total number of pages in the source file.
        doc_id:      Unique stable identifier for this page across the pipeline.
        file_hash:   MD5 of the source file — used to detect duplicate uploads.
    """
    text:        str
    source:      str
    page:        int
    total_pages: int
    doc_id:      str = field(init=False)
    file_hash:   str = ""

    def __post_init__(self) -> None:
        # Deterministic ID: "<filename>::page::<n>"
        # Makes ChromaDB deduplication straightforward — same file re-uploaded
        # produces the same IDs, so ChromaDB upsert won't create duplicates.
        self.doc_id = f"{self.source}::page::{self.page}"

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, page={self.page}, text={preview!r}...)"


# ─────────────────────────────────────────────────────────────────────────────
# Loader result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadResult:
    """
    Outcome of loading one PDF file.

    Attributes:
        filename:    Name of the source file.
        documents:   List of Document objects (one per page with extractable text).
        page_count:  Total pages in the file (including blank / unextractable).
        success:     True if at least one page was extracted.
        error:       Human-readable error message if loading failed.
        skipped_pages: Pages that yielded no text (e.g. scanned images).
    """
    filename:      str
    documents:     list[Document]
    page_count:    int = 0
    success:       bool = True
    error:         str = ""
    skipped_pages: list[int] = field(default_factory=list)

    @property
    def extracted_pages(self) -> int:
        return len(self.documents)

    @property
    def total_chars(self) -> int:
        return sum(len(d.text) for d in self.documents)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(
    file: Union[bytes, io.BytesIO, Path, str],
    filename: str = "unknown.pdf",
    min_page_chars: int = 20,
) -> LoadResult:
    """
    Extract text from a single PDF and return a LoadResult.

    Args:
        file:           PDF content as bytes, BytesIO, or a file path.
        filename:       Display name for the source (shown in citations).
        min_page_chars: Pages with fewer characters than this are skipped.
                        Prevents storing meaningless whitespace chunks.

    Returns:
        LoadResult with extracted Document objects and metadata.
    """
    logger.info("Loading PDF: %s", filename)

    # ── Normalise input to BytesIO ─────────────────────────────────────────
    try:
        if isinstance(file, (str, Path)):
            with open(file, "rb") as fh:
                raw_bytes = fh.read()
        elif isinstance(file, bytes):
            raw_bytes = file
        elif isinstance(file, io.BytesIO):
            file.seek(0)
            raw_bytes = file.read()
        else:
            # Streamlit UploadedFile — has .read() and .name
            raw_bytes = file.read()
            if not filename or filename == "unknown.pdf":
                filename = getattr(file, "name", "unknown.pdf")

        file_hash = hashlib.md5(raw_bytes).hexdigest()
        buffer = io.BytesIO(raw_bytes)

    except Exception as exc:
        logger.error("Could not read file %s: %s", filename, exc)
        return LoadResult(
            filename=filename,
            documents=[],
            success=False,
            error=f"Could not read file: {exc}",
        )

    # ── Parse with pypdf ───────────────────────────────────────────────────
    try:
        reader = pypdf.PdfReader(buffer)

        # Guard: encrypted PDFs
        if reader.is_encrypted:
            logger.warning("PDF is encrypted: %s", filename)
            return LoadResult(
                filename=filename,
                documents=[],
                success=False,
                error=(
                    "This PDF is password-protected. "
                    "Please provide an unlocked version."
                ),
            )

        total_pages = len(reader.pages)
        logger.info("  %d pages found in %s", total_pages, filename)

    except pypdf.errors.PdfReadError as exc:
        logger.error("Invalid PDF %s: %s", filename, exc)
        return LoadResult(
            filename=filename,
            documents=[],
            success=False,
            error=f"Invalid or corrupted PDF: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected error reading %s: %s", filename, exc)
        return LoadResult(
            filename=filename,
            documents=[],
            success=False,
            error=f"Unexpected error: {exc}",
        )

    # ── Extract text page by page ──────────────────────────────────────────
    documents: list[Document] = []
    skipped: list[int] = []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # Convert to 1-indexed

        try:
            raw_text = page.extract_text() or ""
            # Clean up excessive whitespace while preserving paragraph breaks
            cleaned_text = _clean_text(raw_text)

        except Exception as exc:
            logger.warning("  Could not extract page %d of %s: %s", page_num, filename, exc)
            skipped.append(page_num)
            continue

        # Skip pages with too little content (likely scanned images or blanks)
        if len(cleaned_text) < min_page_chars:
            logger.debug("  Skipping page %d (only %d chars)", page_num, len(cleaned_text))
            skipped.append(page_num)
            continue

        doc = Document(
            text=cleaned_text,
            source=filename,
            page=page_num,
            total_pages=total_pages,
            file_hash=file_hash,
        )
        documents.append(doc)

    # ── Build result ──────────────────────────────────────────────────────
    if not documents:
        return LoadResult(
            filename=filename,
            documents=[],
            page_count=total_pages,
            success=False,
            skipped_pages=skipped,
            error=(
                "No readable text found. The PDF may consist entirely of "
                "scanned images. Consider using an OCR tool first."
            ),
        )

    logger.info(
        "  Loaded %d/%d pages from %s (skipped: %s)",
        len(documents), total_pages, filename, skipped or "none",
    )

    return LoadResult(
        filename=filename,
        documents=documents,
        page_count=total_pages,
        success=True,
        skipped_pages=skipped,
    )


def load_pdfs(
    files: list,
    min_page_chars: int = 20,
) -> tuple[list[Document], list[LoadResult]]:
    """
    Load multiple PDF files and aggregate results.

    Args:
        files:          List of file-like objects (Streamlit UploadedFile,
                        bytes, BytesIO, or Path).
        min_page_chars: Minimum characters per page to include.

    Returns:
        Tuple of:
          - all_documents: Flat list of all successfully extracted Documents.
          - results:       Per-file LoadResult objects (for UI status display).
    """
    all_documents: list[Document] = []
    results: list[LoadResult] = []

    for file in files:
        filename = getattr(file, "name", "unknown.pdf")
        result = load_pdf(file, filename=filename, min_page_chars=min_page_chars)
        results.append(result)
        if result.success:
            all_documents.extend(result.documents)

    total_docs = len(all_documents)
    total_files = len(files)
    ok_files = sum(1 for r in results if r.success)

    logger.info(
        "Batch load complete: %d/%d files OK, %d pages extracted",
        ok_files, total_files, total_docs,
    )

    return all_documents, results


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalise extracted PDF text.

    - Collapse runs of 3+ newlines to 2 (preserve paragraph breaks)
    - Collapse runs of spaces/tabs to a single space
    - Strip leading/trailing whitespace
    """
    import re
    # Collapse horizontal whitespace (spaces, tabs) but NOT newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than 2 consecutive newlines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
