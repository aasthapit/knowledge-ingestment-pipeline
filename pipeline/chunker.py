"""
chunker.py
Splits documents into semantically meaningful chunks.

Two strategies are available:
  - chunk_markdown / chunk_markdown_file  — heading-based splitting for .md
  - chunk_docling                          — Docling HybridChunker for PDFs,
                                             DOCX, HTML, and other rich formats

Both strategies produce the same :class:`Chunk` dataclass.
Citation metadata is stored in ``Chunk.metadata["citation"]``.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pipeline.converter import Citation, ConvertedDocument

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One logical piece of a document, ready to be embedded and stored."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""           # file path or URL
    title: str = ""            # top-level document title (H1 or filename)
    section: str = ""          # heading path, e.g. "Intro > Setup > Step 1"
    content: str = ""          # plain text of this chunk
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "content": self.content,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Front-matter parser
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter dict, remaining markdown body)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    body = text[match.end():]
    return fm, body


# ---------------------------------------------------------------------------
# Heading-based splitter
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


def _split_by_headings(body: str) -> list[tuple[int, str, str]]:
    """
    Return a list of (level, heading_text, section_body) tuples.
    Content before the first heading is yielded with level=0 and heading=''.
    """
    segments: list[tuple[int, str, str]] = []
    matches = list(_HEADING_RE.finditer(body))

    if not matches:
        return [(0, "", body.strip())]

    # Content before first heading (preamble)
    preamble = body[: matches[0].start()].strip()
    if preamble:
        segments.append((0, "", preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        segments.append((level, heading, content))

    return segments


def _build_section_path(stack: list[str]) -> str:
    return " > ".join(s for s in stack if s)


# ---------------------------------------------------------------------------
# Chunk size enforcement (character-based sliding window)
# ---------------------------------------------------------------------------

def _split_large_chunk(
    text: str, max_chars: int, overlap: int
) -> list[str]:
    """Further split text that exceeds max_chars with overlap."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        parts.append(text[start:end])
        start += max_chars - overlap
    return parts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_markdown(
    text: str,
    source: str,
    extra_tags: list[str] | None = None,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Parse a markdown string into a list of :class:`Chunk` objects.

    Parameters
    ----------
    text:       Raw markdown content.
    source:     File path or URL used to populate ``Chunk.source``.
    extra_tags: Additional tags to merge with any front-matter tags.
    max_chars:  Maximum character length per chunk before further splitting.
    overlap:    Character overlap when a section exceeds max_chars.
    """
    extra_tags = extra_tags or []
    fm, body = _parse_frontmatter(text)

    # Resolve title — prefer front-matter, fall back to first H1, then filename
    fm_title: str = fm.get("title", "")
    fm_tags: list[str] = [str(t) for t in fm.get("tags", [])]
    all_tags = list(dict.fromkeys(fm_tags + extra_tags))  # deduplicate, keep order

    # Build metadata from remaining front-matter keys
    reserved = {"title", "tags"}
    base_metadata: dict[str, Any] = {k: v for k, v in fm.items() if k not in reserved}

    segments = _split_by_headings(body)

    # Derive document title from first H1 if not in front-matter
    doc_title = fm_title
    if not doc_title:
        for level, heading, _ in segments:
            if level == 1:
                doc_title = heading
                break
    if not doc_title:
        doc_title = Path(source).stem

    chunks: list[Chunk] = []
    # heading stack for building breadcrumb paths
    heading_stack: list[str] = []

    for level, heading, content in segments:
        # Update breadcrumb stack
        if level == 0:
            section_path = doc_title
        else:
            # Trim stack to current level
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            section_path = _build_section_path([doc_title] + heading_stack)

        if not content:
            continue

        # Split oversized sections
        sub_texts = _split_large_chunk(content, max_chars, overlap)
        for idx, sub in enumerate(sub_texts):
            sub = sub.strip()
            if not sub:
                continue
            sec = section_path if len(sub_texts) == 1 else f"{section_path} [{idx + 1}/{len(sub_texts)}]"
            chunks.append(
                Chunk(
                    source=source,
                    title=doc_title,
                    section=sec,
                    content=sub,
                    tags=list(all_tags),
                    metadata=dict(base_metadata),
                )
            )

    return chunks


def chunk_character(
    text: str,
    source: str,
    extra_tags: list[str] | None = None,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Pure character sliding-window chunker — no heading detection.

    Splits *text* into overlapping windows of *max_chars* characters.
    Use when documents lack heading structure or when explicit size control
    is preferred over semantic splitting.
    """
    extra_tags = extra_tags or []
    title = Path(source).stem
    parts = _split_large_chunk(text.strip(), max_chars, overlap)
    n = len(parts)
    chunks: list[Chunk] = []
    for idx, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        section = f"{title} [{idx + 1}/{n}]" if n > 1 else title
        chunks.append(
            Chunk(
                source=source,
                title=title,
                section=section,
                content=part,
                tags=list(extra_tags),
            )
        )
    return chunks


def chunk_markdown_file(
    path: str | Path,
    extra_tags: list[str] | None = None,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Convenience wrapper that reads a file before chunking."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return chunk_markdown(
        text,
        source=str(path),
        extra_tags=extra_tags,
        max_chars=max_chars,
        overlap=overlap,
    )


# ---------------------------------------------------------------------------
# Docling-based chunker (PDF, DOCX, PPTX, HTML, URLs)
# ---------------------------------------------------------------------------

def chunk_docling(
    converted_doc: "ConvertedDocument",
    tags: list[str] | None = None,
    max_tokens: int = 512,
) -> list[Chunk]:
    """
    Chunk a :class:`~pipeline.converter.ConvertedDocument` using Docling's
    HybridChunker, which respects document structure (headings, pages, tables).

    Each chunk receives:
    - A ``section`` breadcrumb built from the heading hierarchy
    - The full :class:`~pipeline.converter.Citation` stored in ``metadata["citation"]``
    - A ``page_number`` in ``metadata["citation"]["page_number"]`` for PDFs

    Falls back to :func:`chunk_markdown` when Docling is unavailable or when
    the source document was already Markdown (no ``docling_doc``).

    Parameters
    ----------
    converted_doc:
        Output of :func:`~pipeline.converter.convert_document`.
    tags:
        Extra tags to attach to every chunk (e.g. from quality auto-tagger).
    max_tokens:
        Maximum tokens per chunk for HybridChunker (default 512).
    """
    from pipeline.converter import Citation  # runtime import (no circular dep at class def time)

    citation: Citation = converted_doc.citation
    all_tags: list[str] = list(tags or [])

    # ── Markdown / no docling_doc → fall back to heading-based splitter ───
    if converted_doc.docling_doc is None:
        base_chunks = chunk_markdown(
            converted_doc.markdown,
            source=citation.source_path,
            extra_tags=all_tags,
        )
        for c in base_chunks:
            c.metadata["citation"] = citation.to_dict()
        return base_chunks

    # ── Docling HybridChunker ─────────────────────────────────────────────
    try:
        from docling.chunking import HybridChunker
    except ImportError as exc:
        raise ImportError(
            "docling is required for chunk_docling(). "
            "Install it with: uv add docling"
        ) from exc

    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    docling_chunks = list(chunker.chunk(converted_doc.docling_doc))

    chunks: list[Chunk] = []
    for dc in docling_chunks:
        # Serialise to plain text (includes table markdown, list items, etc.)
        try:
            text = chunker.serialize(chunk=dc)
        except Exception:
            text = getattr(dc, "text", "") or ""
        if not text.strip():
            continue

        # ── Section breadcrumb ────────────────────────────────────────────
        headings: list[str] = []
        try:
            raw_headings = dc.meta.headings or []
            headings = [h for h in raw_headings if h and h.strip()]
        except Exception:
            pass
        section = " > ".join([citation.title] + headings) if headings else citation.title

        # ── Page number (PDF provenance) ──────────────────────────────────
        page_no: int | None = None
        try:
            for item in dc.meta.doc_items:
                provs = getattr(item, "prov", None) or []
                if provs:
                    page_no = provs[0].page_no
                    break
        except Exception:
            pass

        chunk_citation = Citation(
            source_path=citation.source_path,
            source_type=citation.source_type,
            title=citation.title,
            page_count=citation.page_count,
            page_number=page_no,
            author=citation.author,
            created_date=citation.created_date,
            url=citation.url,
        )

        chunks.append(
            Chunk(
                source=citation.source_path,
                title=citation.title,
                section=section,
                content=text,
                tags=list(all_tags),
                metadata={"citation": chunk_citation.to_dict()},
            )
        )

    return chunks
