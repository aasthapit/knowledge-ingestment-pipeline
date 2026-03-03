"""
chunker.py
Splits a markdown document into semantically meaningful chunks by heading.
Also parses YAML front-matter for tags and metadata.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
