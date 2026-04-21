"""
converter.py
Converts documents (PDF, DOCX, PPTX, HTML, web URLs, Markdown) into a
normalised ConvertedDocument using Docling, and extracts Citation metadata
so every downstream chunk can be traced back to its source.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Citation — source attribution carried through the whole pipeline
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """All the information needed to point a reader back to the source."""
    source_path: str            # original file path or URL
    source_type: str            # pdf | docx | pptx | html | url | markdown | text
    title: str                  # document title (best guess)
    page_count: int | None = None
    page_number: int | None = None   # filled at chunk level for PDFs
    author: str | None = None
    created_date: str | None = None
    url: str | None = None           # canonical URL for web sources

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# ConvertedDocument — output of the conversion step
# ---------------------------------------------------------------------------

@dataclass
class ConvertedDocument:
    citation: Citation
    docling_doc: Any            # DoclingDocument (lazy import, may be None for .md)
    markdown: str               # Docling markdown export — always populated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_source_type(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return "url"
    return {
        ".pdf":  "pdf",
        ".docx": "docx",
        ".doc":  "docx",
        ".pptx": "pptx",
        ".ppt":  "pptx",
        ".html": "html",
        ".htm":  "html",
        ".md":   "markdown",
        ".txt":  "text",
        ".xlsx": "xlsx",
        ".xls":  "xlsx",
    }.get(Path(source).suffix.lower(), "unknown")


def _title_from_source(source: str, source_type: str) -> str:
    if source_type == "url":
        path = urlparse(source).path.rstrip("/")
        return path.split("/")[-1] or source
    return Path(source).stem


# ---------------------------------------------------------------------------
# Markdown passthrough (no Docling needed)
# ---------------------------------------------------------------------------

def _convert_markdown(source: str) -> ConvertedDocument:
    path = Path(source)
    text = path.read_text(encoding="utf-8")

    # Pull title from first H1 or filename
    title = path.stem
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    citation = Citation(
        source_path=source,
        source_type="markdown",
        title=title,
    )
    return ConvertedDocument(citation=citation, docling_doc=None, markdown=text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_document(source: str | Path) -> ConvertedDocument:
    """
    Convert a document to a :class:`ConvertedDocument` using Docling.

    Markdown files are passed through directly without Docling (they are
    already structured text). Everything else goes through Docling's
    DocumentConverter which handles PDF, DOCX, PPTX, HTML, and URLs.

    Parameters
    ----------
    source:
        File path (str or Path) or HTTP/HTTPS URL.

    Returns
    -------
    ConvertedDocument
        citation metadata + DoclingDocument (or None for Markdown)
        + markdown export string.

    Raises
    ------
    ImportError
        If docling is not installed and the source is not a Markdown file.
    """
    source = str(source)
    source_type = _detect_source_type(source)

    # Markdown: no Docling needed
    if source_type == "markdown":
        logger.info("Markdown passthrough: %s", source)
        return _convert_markdown(source)

    # All other formats — use Docling
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "docling is required for non-Markdown document conversion. "
            "Install it with: uv add docling"
        ) from exc

    logger.info("Converting %s (%s) …", source, source_type)
    converter = DocumentConverter()
    result = converter.convert(source)
    doc = result.document

    # ── title ──────────────────────────────────────────────────────────────
    title = _title_from_source(source, source_type)
    # Prefer docling's extracted title when available
    try:
        if doc.description and getattr(doc.description, "title_text", None):
            title = doc.description.title_text
    except Exception:
        pass
    # Fall back to doc.name (usually the filename stem)
    if not title:
        title = getattr(doc, "name", None) or _title_from_source(source, source_type)

    # ── page count ─────────────────────────────────────────────────────────
    page_count: int | None = None
    try:
        pages = getattr(doc, "pages", None)
        if pages:
            page_count = len(pages)
    except Exception:
        pass

    # ── author / created_date ──────────────────────────────────────────────
    author: str | None = None
    created_date: str | None = None
    try:
        desc = getattr(doc, "description", None)
        if desc:
            raw_authors = getattr(desc, "authors", None) or []
            if raw_authors:
                author = ", ".join(
                    getattr(a, "name", str(a)) for a in raw_authors
                )
            raw_date = getattr(desc, "creation_date", None) or getattr(desc, "created_date", None)
            if raw_date:
                created_date = str(raw_date)
    except Exception:
        pass

    citation = Citation(
        source_path=source,
        source_type=source_type,
        title=title,
        page_count=page_count,
        author=author,
        created_date=created_date,
        url=source if source_type == "url" else None,
    )

    markdown = doc.export_to_markdown()
    logger.info(
        "Converted '%s' — %s page(s), %d chars",
        title,
        page_count if page_count is not None else "?",
        len(markdown),
    )
    return ConvertedDocument(citation=citation, docling_doc=doc, markdown=markdown)
