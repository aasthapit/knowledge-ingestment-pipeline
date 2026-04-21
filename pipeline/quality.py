"""
quality.py
Assesses whether an ingested document is well-structured enough for
automatic chunking and auto-tagging, or should be queued for human review.

Scoring is derived entirely from the markdown export produced by Docling,
so it works for every supported format without extra ML calls.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.converter import ConvertedDocument

from pipeline.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-word list for tag extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with by from is are was were be "
    "been being have has had do does did will would could should may might must "
    "can this that these those it its we our you your he she they their what "
    "which how when where who why all each every some any one two three four "
    "five six seven eight nine ten about into over after before between".split()
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    score: float                              # 0.0 – 1.0
    passed: bool                              # score >= settings.quality_threshold
    signals: dict[str, float]                 # individual weighted component scores
    flags: list[str] = field(default_factory=list)         # human-readable issues
    suggested_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal extraction (from markdown — always available after Docling conversion)
# ---------------------------------------------------------------------------

def _extract_signals(markdown: str) -> dict:
    """Extract raw structural signals from a markdown string."""
    heading_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    headings = heading_re.findall(markdown)

    heading_count = len(headings)
    heading_levels = {len(h) for h, _ in headings}
    has_title = any(len(h) == 1 for h, _ in headings)

    # Body text: everything that isn't a heading line
    body = heading_re.sub("", markdown)
    # Split into non-empty paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    total_body_chars = sum(len(p) for p in paragraphs)
    avg_section_len = total_body_chars / max(heading_count, 1)

    # YAML / TOML front-matter → metadata present
    has_frontmatter = bool(re.match(r"^---\s*\n.*?\n---\s*\n", markdown, re.DOTALL))

    # Short empty-looking sections (< 60 chars) as a fraction of heading count
    short_sections = sum(1 for p in paragraphs if len(p) < 60)
    short_fraction = short_sections / max(len(paragraphs), 1)

    return {
        "has_title": has_title,
        "heading_count": heading_count,
        "heading_levels": heading_levels,
        "avg_section_len": avg_section_len,
        "total_body_chars": total_body_chars,
        "short_fraction": short_fraction,
        "has_frontmatter": has_frontmatter,
        "paragraphs": paragraphs,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(signals: dict) -> tuple[float, list[str], dict[str, float]]:
    """Map raw signals to a weighted quality score + flag list."""
    flags: list[str] = []
    components: dict[str, float] = {}

    # 1. Has a title (H1) — weight 0.25
    if signals["has_title"]:
        components["has_title"] = 1.0
    else:
        components["has_title"] = 0.0
        flags.append("No H1 title found — document identity is unclear.")

    # 2. Has multiple headings — weight 0.25
    hc = signals["heading_count"]
    if hc >= 4:
        components["has_headings"] = 1.0
    elif hc >= 2:
        components["has_headings"] = 0.6
    elif hc == 1:
        components["has_headings"] = 0.3
        flags.append("Only 1 heading — document may be a flat text block.")
    else:
        components["has_headings"] = 0.0
        flags.append("No headings found — document has no navigable structure.")

    # 3. Heading hierarchy depth — weight 0.15
    levels = signals["heading_levels"]
    if len(levels) >= 2:
        components["heading_depth"] = 1.0
    elif len(levels) == 1:
        components["heading_depth"] = 0.5
        flags.append("Only one heading level used — shallow hierarchy.")
    else:
        components["heading_depth"] = 0.0

    # 4. Section richness (average body chars per heading) — weight 0.20
    avg = signals["avg_section_len"]
    sf = signals["short_fraction"]
    if avg >= 300 and sf < 0.3:
        components["section_richness"] = 1.0
    elif avg >= 100:
        components["section_richness"] = 0.65
    elif avg >= 40:
        components["section_richness"] = 0.35
        flags.append(
            f"Sections are sparse (avg {avg:.0f} chars) — "
            "may be a table of contents or fragmented content."
        )
    else:
        components["section_richness"] = 0.0
        flags.append("Almost no body text found — document may be image-only or empty.")

    # 5. Metadata / front-matter present — weight 0.15
    if signals["has_frontmatter"]:
        components["has_metadata"] = 1.0
    else:
        components["has_metadata"] = 0.2
        flags.append("No YAML front-matter found — tags and metadata will be auto-suggested.")

    weights = {
        "has_title":       0.25,
        "has_headings":    0.25,
        "heading_depth":   0.15,
        "section_richness": 0.20,
        "has_metadata":    0.15,
    }
    score = sum(weights[k] * components[k] for k in weights)
    return score, flags, components


# ---------------------------------------------------------------------------
# Auto-tag extraction
# ---------------------------------------------------------------------------

def extract_tags(markdown: str, title: str = "", extra: list[str] | None = None) -> list[str]:
    """
    Derive up to 10 lowercase keyword tags from the document's headings and title.
    Does not call any external service.
    """
    heading_re = re.compile(r"^#{1,3}\s+(.*)", re.MULTILINE)
    heading_texts = heading_re.findall(markdown)[:8]  # top 8 headings

    seen: set[str] = set()
    tags: list[str] = []

    for text in ([title] if title else []) + heading_texts:
        for token in re.split(r"[\s\-_/\\,.;:()\[\]{}|]+", text):
            word = token.lower().strip("'\"`")
            if len(word) >= 3 and word not in _STOP_WORDS and re.fullmatch(r"[a-z0-9]+", word):
                if word not in seen:
                    seen.add(word)
                    tags.append(word)
                if len(tags) >= 10:
                    break
        if len(tags) >= 10:
            break

    # Merge explicit extras (deduplicated)
    for t in (extra or []):
        t = t.lower().strip()
        if t and t not in seen:
            seen.add(t)
            tags.append(t)

    return tags[:10]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_quality(converted_doc: "ConvertedDocument") -> QualityResult:
    """
    Assess the structural quality of a converted document.

    Operates on the markdown export (always available after Docling
    conversion) so it requires no additional ML inference.

    Returns a :class:`QualityResult` with score, pass/fail flag,
    per-component scores, human-readable flags, and suggested tags.
    """
    markdown = converted_doc.markdown
    title = converted_doc.citation.title

    signals = _extract_signals(markdown)
    score, flags, components = _score(signals)

    threshold = settings.quality_threshold
    passed = score >= threshold

    if passed:
        logger.info("Quality PASS (%.2f >= %.2f): %s", score, threshold, title)
    else:
        logger.warning(
            "Quality FAIL (%.2f < %.2f): %s\n  Flags: %s",
            score, threshold, title, " | ".join(flags),
        )

    suggested_tags = extract_tags(markdown, title=title)

    return QualityResult(
        score=round(score, 4),
        passed=passed,
        signals=components,
        flags=flags,
        suggested_tags=suggested_tags,
    )
