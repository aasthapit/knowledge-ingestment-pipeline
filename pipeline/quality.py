"""
quality.py
Assesses document and chunk quality based on content signals and recency.

Replaces the old structural heading-count scorer. The new model evaluates:
  - Per-chunk size (too short / too long)
  - Per-chunk boilerplate detection (nav menus, TOC, login prompts, etc.)
  - Document-level recency (from creation date or file mtime)

Quality score = fraction of chunks that have no flags (0.0 – 1.0).
A document passes (auto-approves) only when it has zero flagged chunks
and is not stale. Any flag routes it to human review.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.chunker import Chunk
    from pipeline.converter import Citation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (can be overridden via config if needed later)
# ---------------------------------------------------------------------------

MIN_CHUNK_CHARS    = 100    # fewer chars → "too_short"
MAX_CHUNK_CHARS    = 2000   # more chars  → "too_long" (embedding model risk)
STALE_THRESHOLD_DAYS = 180  # 6 months

# ---------------------------------------------------------------------------
# Boilerplate patterns
# ---------------------------------------------------------------------------

_BOILERPLATE_PHRASES: frozenset[str] = frozenset({
    "log in", "sign in", "sign up", "login", "sign out", "log out",
    "table of contents", "on this page", "in this section", "contents",
    "copyright", "all rights reserved", "privacy policy",
    "terms of service", "terms and conditions", "cookie policy",
    "back to top", "return to top", "top of page",
    "click here", "read more", "learn more", "see also",
    "page restrictions", "this page is restricted",
    "no content found", "this space has no content",
    "skip to main content", "skip to content",
    "breadcrumb", "you are here",
})

_BOILERPLATE_RE = re.compile(
    r"(?i)(?:"
    r"^\s*(log\s*in|sign\s*in|sign\s*up|register)\s*$"
    r"|copyright\s+\(?(?:19|20)\d{2}"
    r"|all\s+rights\s+reserved"
    r"|powered\s+by\s+atlassian"
    r"|atlassian\s+confluence"
    r")",
    re.MULTILINE,
)


def _is_boilerplate(text: str) -> bool:
    """
    Return True if the chunk content appears to be navigation, TOC, or
    template boilerplate rather than substantive knowledge content.

    Conservative — only flags obvious cases to avoid false positives
    on legitimate technical content.
    """
    stripped = text.strip()

    # Skip code blocks — many short lines are normal in code
    if "```" in stripped or stripped.count("    ") > stripped.count("\n") * 0.5:
        return False

    lower = stripped.lower()

    # Regex patterns (copyright notices, Atlassian footers, login prompts)
    if _BOILERPLATE_RE.search(stripped):
        return True

    # Known boilerplate phrases that dominate short content
    for phrase in _BOILERPLATE_PHRASES:
        if phrase in lower and len(stripped) < 300:
            return True

    # Navigation-style content: ≥8 non-empty lines all very short (avg < 30 chars)
    # Catches exported sidebar menus, breadcrumb lists, TOC entries
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    if len(lines) >= 8:
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 30:
            return True

    return False


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


def _compute_age_days(
    date_str: str | None,
    source_path: str | None = None,
) -> int | None:
    """
    Return age in whole days from a date string (ISO 8601) or file mtime.
    Returns None if neither is available or parseable.
    """
    now = datetime.now(timezone.utc)

    if date_str:
        # Strip trailing Z / timezone for strptime, then re-attach UTC
        clean = date_str.strip().rstrip("Z").split("+")[0][:26]
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
                return max(0, (now - dt).days)
            except (ValueError, TypeError):
                continue

    if source_path:
        try:
            p = Path(source_path)
            if p.exists():
                mtime = p.stat().st_mtime
                dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
                return max(0, (now - dt).days)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    """Result of a document quality assessment under the new model."""
    score: float                               # fraction of clean chunks (0–1)
    passed: bool                               # True when score == 1.0 and not stale
    flags: list[str] = field(default_factory=list)        # document-level human-readable issues
    chunk_flags: dict[int, list[str]] = field(default_factory=dict)  # {chunk_idx: [flag_names]}
    age_days: int | None = None                # None if date unknown
    is_stale: bool = False                     # age_days > STALE_THRESHOLD_DAYS
    chunks_too_short: int = 0
    chunks_too_long: int = 0
    chunks_boilerplate: int = 0
    suggested_tags: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tag extraction (kept — still useful for auto-tagging)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with by from is are was were be "
    "been being have has had do does did will would could should may might must "
    "can this that these those it its we our you your he she they their what "
    "which how when where who why all each every some any one two three four "
    "five six seven eight nine ten about into over after before between".split()
)


def extract_tags(
    markdown: str,
    title: str = "",
    extra: list[str] | None = None,
) -> list[str]:
    """
    Derive up to 10 lowercase keyword tags from headings and title.
    No external service calls.
    """
    heading_re = re.compile(r"^#{1,3}\s+(.*)", re.MULTILINE)
    heading_texts = heading_re.findall(markdown)[:8]

    seen: set[str] = set()
    tags: list[str] = []

    for text in ([title] if title else []) + heading_texts:
        for token in re.split(r"[\s\-_/\\,.;:()\[\]{}|]+", text):
            word = token.lower().strip("'\"`")
            if (
                len(word) >= 3
                and word not in _STOP_WORDS
                and re.fullmatch(r"[a-z0-9]+", word)
            ):
                if word not in seen:
                    seen.add(word)
                    tags.append(word)
                if len(tags) >= 10:
                    break
        if len(tags) >= 10:
            break

    for t in extra or []:
        t = t.lower().strip()
        if t and t not in seen:
            seen.add(t)
            tags.append(t)

    return tags[:10]


# ---------------------------------------------------------------------------
# Core assessment
# ---------------------------------------------------------------------------

def assess_document(
    chunks: "list[Chunk]",
    citation: "Citation",
) -> QualityResult:
    """
    Assess document quality from its chunks and citation metadata.

    Called AFTER chunking so signals are based on the actual content
    units that will be embedded and retrieved, not on document structure.

    Parameters
    ----------
    chunks:
        List of Chunk objects produced by the chunker.
    citation:
        Source metadata (title, created_date, source_path, etc.).

    Returns
    -------
    QualityResult
        score = fraction of clean chunks.
        passed = True only when all chunks are clean AND content is not stale.
    """
    flags: list[str] = []
    chunk_flags: dict[int, list[str]] = {}
    n_too_short = n_too_long = n_boilerplate = 0

    for i, chunk in enumerate(chunks):
        content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
        issues: list[str] = []

        char_count = len(content.strip())

        if char_count < MIN_CHUNK_CHARS:
            issues.append("too_short")
            n_too_short += 1
        elif char_count > MAX_CHUNK_CHARS:
            issues.append("too_long")
            n_too_long += 1

        if _is_boilerplate(content):
            issues.append("boilerplate")
            n_boilerplate += 1

        if issues:
            chunk_flags[i] = issues

    total = len(chunks)

    if n_too_short:
        flags.append(
            f"{n_too_short} chunk(s) are too short (< {MIN_CHUNK_CHARS} chars) "
            "— may be section stubs, nav elements, or empty headings."
        )
    if n_too_long:
        flags.append(
            f"{n_too_long} chunk(s) exceed {MAX_CHUNK_CHARS} chars "
            "— risk truncation by the embedding model."
        )
    if n_boilerplate:
        flags.append(
            f"{n_boilerplate} chunk(s) appear to be boilerplate "
            "— navigation menus, table of contents, or template text."
        )

    # Recency
    date_str    = getattr(citation, "created_date", None)
    source_path = getattr(citation, "source_path", None)
    age_days    = _compute_age_days(date_str, source_path)
    is_stale    = age_days is not None and age_days > STALE_THRESHOLD_DAYS

    if is_stale:
        age_str = f"{age_days} days" if age_days is not None else "unknown age"
        flags.append(
            f"Content is {age_str} old — consider refreshing from the source."
        )

    # Score and pass/fail
    n_flagged = len(chunk_flags)
    score = round(1.0 - (n_flagged / max(total, 1)), 4)
    passed = (len(flags) == 0)

    # Auto-tags from headings + title
    suggested_tags = extract_tags("", title=getattr(citation, "title", ""))

    return QualityResult(
        score=score,
        passed=passed,
        flags=flags,
        chunk_flags=chunk_flags,
        age_days=age_days,
        is_stale=is_stale,
        chunks_too_short=n_too_short,
        chunks_too_long=n_too_long,
        chunks_boilerplate=n_boilerplate,
        suggested_tags=suggested_tags,
        signals={
            "total_chunks":   total,
            "flagged_chunks": n_flagged,
            "clean_chunks":   total - n_flagged,
            "age_days":       age_days,
        },
    )
