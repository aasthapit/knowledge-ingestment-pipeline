#!/usr/bin/env python3
"""
crawl_ocp_docs.py
─────────────────
Crawl docs.redhat.com/en/documentation/openshift_container_platform/<version>/ →
parse HTML → chunk by heading section → emit JSONL for vector DB ingestion.

Note: docs.openshift.com redirected to docs.redhat.com. The sitemap is a
sitemap index at docs.redhat.com/sitemaps/docs/docs-sitemap-index.xml.

Flow
────
1. Fetch sitemap index → fetch each sub-sitemap → filter OCP URLs for the version
2. For each URL: GET page, extract article content, split on headings
3. Sub-split oversized sections on paragraph breaks
4. Write JSONL records with semantic context prefix

Usage
─────
    python crawl_ocp_docs.py --version 4.18 --output ocp_4.18.jsonl
    python crawl_ocp_docs.py --version 4.18 --output ocp_4.18.jsonl --resume
    python crawl_ocp_docs.py --version 4.18 --output ocp_4.18.jsonl --max-section-chars 1500 --overlap 100

Requirements
────────────
    pip install requests beautifulsoup4 lxml
"""

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SITEMAP_NS      = "http://www.sitemaps.org/schemas/sitemap/0.9"
USECASE_ID      = "GENAI1597_SSOP"
AGENT_FILTER    = "ssop_cloud_operations_knowledge_agent"
DATA_CLASS      = "Internal"
UNIT            = "chars"

DEFAULT_VERSION  = "4.18"
DEFAULT_OUTPUT   = "ocp_docs.jsonl"
DEFAULT_PROGRESS = "crawl_progress.json"
DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP   = 100
DEFAULT_DELAY     = 0.3   # seconds between requests (~3 req/s)

HEADING_TAGS = {"h1", "h2", "h3", "h4"}


# ─────────────────────────────────────────────────────────────
# HTTP session
# ─────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "ocp-docs-ingest/1.0 (internal)"
    return s


# ─────────────────────────────────────────────────────────────
# Sitemap → URL list
# ─────────────────────────────────────────────────────────────

SITEMAP_INDEX = "https://docs.redhat.com/sitemaps/docs/docs-sitemap-index.xml"
DOCS_BASE     = "https://docs.redhat.com/en/documentation/openshift_container_platform"


def get_page_urls(version: str, session: requests.Session) -> list[str]:
    """
    Fetch the sitemap index, then each sub-sitemap, and collect all page URLs
    for the requested OCP version. Filters to HTML doc pages only (skipping
    top-level version index pages that have no article content).
    """
    print(f"[sitemap] Fetching sitemap index: {SITEMAP_INDEX}")
    idx_r = session.get(SITEMAP_INDEX, timeout=30)
    idx_r.raise_for_status()
    idx_root = ET.fromstring(idx_r.text)
    sub_sitemaps = [loc.text for loc in idx_root.findall(f".//{{{SITEMAP_NS}}}loc") if loc.text]
    print(f"[sitemap] {len(sub_sitemaps)} sub-sitemaps found, scanning for version {version}...")

    url_prefix = f"{DOCS_BASE}/{version}/"
    urls: list[str] = []

    for i, sm_url in enumerate(sub_sitemaps, 1):
        r = session.get(sm_url, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        found = [
            loc.text for loc in root.findall(f".//{{{SITEMAP_NS}}}loc")
            if loc.text and loc.text.startswith(url_prefix)
        ]
        urls.extend(found)
        print(f"[sitemap] ({i}/{len(sub_sitemaps)}) {sm_url.split('/')[-1]} — {len(found)} OCP {version} URLs")
        time.sleep(0.1)

    # Drop bare version index, html-single (whole-book pages that duplicate
    # paginated /html/ content), and deduplicate while preserving order
    seen: set[str] = set()
    filtered: list[str] = []
    bare = url_prefix.rstrip("/")
    for u in urls:
        if u == bare or u in seen:
            continue
        if "/html-single/" in u:   # single-page books — skip, same content
            continue
        if "/pdf/" in u or u.endswith(".pdf"):   # PDF downloads — skip
            continue
        seen.add(u)
        filtered.append(u)

    print(f"[sitemap] {len(filtered)} page URLs for OCP {version}")
    return filtered


# ─────────────────────────────────────────────────────────────
# HTML parsing helpers
# ─────────────────────────────────────────────────────────────

def element_to_text(el, _depth: int = 0) -> str:
    """Recursively convert a BS4 element to plain text."""
    if _depth > 50:
        return el.get_text(" ", strip=True) if isinstance(el, Tag) else ""
    if isinstance(el, NavigableString):
        return str(el).strip()
    if not isinstance(el, Tag):
        return ""
    # Skip purely structural / UI noise elements
    if el.name in ("colgroup", "col", "script", "style", "rh-tooltip",
                   "rh-icon", "rh-button", "button"):
        return ""
    # Code blocks — preserve verbatim (whitespace matters)
    if el.name in ("pre", "code"):
        return el.get_text()
    # List items — prefix with dash
    if el.name == "li":
        return "- " + el.get_text(" ", strip=True)
    # Table rows → tab-separated cells (only td/th, skip headers-only rows)
    if el.name == "tr":
        cells = el.find_all(["td", "th"])
        return "\t".join(c.get_text(" ", strip=True) for c in cells)
    # Admonition blocks — extract only content cell, skip icon cell
    if "admonitionblock" in (el.get("class") or []):
        content = el.find("td", class_="content")
        if content:
            return content.get_text(" ", strip=True)
        return ""
    # rh-table → treat like a regular table wrapper
    if el.name == "rh-table":
        inner = el.find("table")
        return element_to_text(inner, _depth + 1) if inner else ""
    # Default: recurse into children
    parts = []
    for child in el.children:
        t = element_to_text(child, _depth + 1)
        if t.strip():
            parts.append(t.strip())
    return "\n".join(parts)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Chapter/section numbering prefix: "Chapter 1. ", "2.3. ", "1.2.3.4. "
_SECTION_NUM_RE = re.compile(r"^(?:Chapter\s+)?\d+(?:\.\d+)*\.\s*")


def _clean_heading(text: str) -> str:
    text = _SECTION_NUM_RE.sub("", text)
    # Normalize non-breaking spaces
    return text.replace("\xa0", " ").strip()


def _section_heading_text(section: Tag) -> str:
    """
    Extract clean heading text from a <section> element.

    The heading is inside <div class="titlepage"> (direct child only — do NOT
    recurse into child sections). The readable text lives in
    <a class="anchor-heading"> (the rh-tooltip copy-link widget is a sibling
    inside the <h*> and is excluded by targeting the <a> directly).
    """
    # recursive=False: only look at direct children of section
    titlepage = section.find("div", class_="titlepage", recursive=False)
    if titlepage:
        anchor_a = titlepage.find("a", class_="anchor-heading")
        if anchor_a:
            return _clean_heading(anchor_a.get_text(" ", strip=True))
        h = titlepage.find(["h1", "h2", "h3", "h4"])
        if h:
            for tooltip in h.find_all("rh-tooltip"):
                tooltip.decompose()
            return _clean_heading(h.get_text(" ", strip=True))
    return ""


def _direct_content_text(section: Tag) -> str:
    """
    Extract plain text from direct content children of a <section>, skipping:
      - <div class="titlepage">  (contains the heading)
      - child <section> elements  (handled separately as sub-sections)
      - script/style
    """
    parts: list[str] = []
    for child in section.children:
        if not isinstance(child, Tag):
            t = str(child).strip()
            if t:
                parts.append(t)
            continue
        if child.name in ("section", "script", "style"):
            continue
        if "titlepage" in (child.get("class") or []):
            continue
        parts.append(element_to_text(child))
    return clean_text("\n".join(parts))


def _walk_sections(
    section: Tag,
    depth: int,
    breadcrumb_stack: list[str],
    page_title: str,
    page_url: str,
    out: list[dict],
):
    """Recursively walk <section> elements and emit one dict per section."""
    anchor = section.get("id", "")
    heading = _section_heading_text(section)

    # Push heading onto breadcrumb stack for this level
    stack_at_entry = list(breadcrumb_stack)
    if heading:
        breadcrumb_stack.append(heading)

    body = _direct_content_text(section)
    if heading and body:
        # level: chapter = 1, first nest = 2, etc.
        level = depth
        crumbs = stack_at_entry  # parent headings only (not self)
        out.append({
            "heading":     heading,
            "level":       level,
            "anchor":      anchor,
            "breadcrumbs": crumbs,
            "page_title":  page_title,
            "body":        body,
            "page_url":    f"{page_url}#{anchor}" if anchor else page_url,
        })

    # Recurse into child sections
    for child in section.children:
        if isinstance(child, Tag) and child.name == "section":
            _walk_sections(child, depth + 1, breadcrumb_stack, page_title, page_url, out)

    # Restore breadcrumb stack (pop what we pushed)
    if heading:
        breadcrumb_stack.pop()


def extract_sections(html: str, page_url: str) -> list[dict]:
    """
    Parse a docs.redhat.com OCP page into a list of section dicts.

    Site structure (as of 2026):
      <article>
        <div class="docs-content-container">
          <h1 class="chapter-title">Chapter N. Page title</h1>
          <section class="rhdocs">
            <section class="chapter" id="page-slug">
              <p>preamble</p>
              <section class="section" id="section-id">
                <div class="titlepage"><div><div>
                  <h2><a class="anchor-heading">Section title</a><rh-tooltip>…</rh-tooltip></h2>
                </div></div></div>
                <p>content</p>
                <section class="section" id="sub-id">…</section>

    Anchors are on the <section id>, not the <h*> tag.
    Heading text is cleanly extracted from <a class="anchor-heading">.
    """
    soup = BeautifulSoup(html, "lxml")

    article = soup.find("article")
    if not article:
        article = soup.find("main") or soup.find("body")
    if not article:
        return []

    # Strip noise
    for sel in ["nav", "script", "style", ".pagination", ".feedback-container",
                ".edit-this-page", ".toolbar", ".toc-container",
                ".page-content-options", ".upper-nav-container"]:
        for el in article.select(sel):
            el.decompose()

    # Page title: <h1 class="chapter-title"> or first <h1>
    h1_el = article.find("h1", class_="chapter-title") or article.find("h1")
    if h1_el:
        page_title = _clean_heading(h1_el.get_text(" ", strip=True))
    else:
        page_title = urlparse(page_url).path.rstrip("/").split("/")[-1]

    # Find the outermost content section(s) (direct children of article / docs-content-container)
    content_div = article.find("div", class_="docs-content-container") or article
    top_sections = [c for c in content_div.children
                    if isinstance(c, Tag) and c.name == "section"]

    if not top_sections:
        # Fallback: no sections at all — treat whole article as one chunk
        body = clean_text(element_to_text(content_div))
        if body:
            return [{
                "heading":     page_title,
                "level":       1,
                "anchor":      "",
                "breadcrumbs": [],
                "page_title":  page_title,
                "body":        body,
                "page_url":    page_url,
            }]
        return []

    sections: list[dict] = []
    for top in top_sections:
        _walk_sections(top, depth=1, breadcrumb_stack=[], page_title=page_title,
                       page_url=page_url, out=sections)

    return sections


# ─────────────────────────────────────────────────────────────
# Context prefix
# ─────────────────────────────────────────────────────────────

def build_prefix(page_title: str, breadcrumbs: list, heading: str) -> str:
    trail = [h for h in breadcrumbs if h != page_title] + [heading]
    lines = [f"Page: {page_title}"]
    if trail:
        lines.append("Section: " + " > ".join(trail))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Sub-splitting
# ─────────────────────────────────────────────────────────────

def subsplit(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split a long body on paragraph boundaries with optional overlap."""
    if len(text) <= max_chars:
        return [text]

    paras = re.split(r"\n{2,}", text)
    chunks, current = [], ""

    for para in paras:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para).strip() if current else para
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            tail = current[-overlap:].lstrip() if overlap else ""
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


# ─────────────────────────────────────────────────────────────
# ID helpers
# ─────────────────────────────────────────────────────────────

def stable_uuid(*parts) -> str:
    key = "|".join(str(p) for p in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────
# Record assembly
# ─────────────────────────────────────────────────────────────

def make_record(
    page_url: str,
    page_title: str,
    section: dict,
    sub_body: str,
    chunk_index: int,
    document_id: str,
    max_chars: int,
    overlap: int,
    sor_last_modified: str,
    version: str,
) -> dict:
    prefix = build_prefix(page_title, section["breadcrumbs"], section["heading"])
    full_text = f"{prefix}\n\n{sub_body}"
    chunk_id = stable_uuid(
        document_id, chunk_index, content_hash(full_text)
    )

    parsed = urlparse(page_url)
    path_parts = parsed.path.strip("/").split("/")
    file_name  = path_parts[-1] if path_parts else ""
    source_dir = "/".join(path_parts[:-1])

    return {
        "id":                  chunk_id,
        "source_file":         file_name,
        "source_dir":          source_dir,
        "relative_path":       parsed.path.lstrip("/"),
        "document_id":         document_id,
        "chunk_id":            chunk_id,
        "chunk_index":         chunk_index,
        "unit":                UNIT,
        "chunk_size_setting":  max_chars,
        "overlap_setting":     overlap,
        "usecase_id":          USECASE_ID,
        "file_name":           file_name,
        "data_classification": DATA_CLASS,
        "sor_last_modified":   sor_last_modified,
        "page_url":            section["page_url"],
        "page_name":           page_title,
        "agent_filter":        AGENT_FILTER,
        "char_count":          len(full_text),
        "word_count":          len(full_text.split()),
        "section_heading":     section["heading"],
        "section_breadcrumbs": section["breadcrumbs"],
        "section_anchor":      section["anchor"],
        "text":                full_text,
    }


# ─────────────────────────────────────────────────────────────
# Resume / progress tracking
# ─────────────────────────────────────────────────────────────

def load_seen(path: str) -> set:
    try:
        with open(path) as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_seen(seen: set, path: str):
    with open(path, "w") as f:
        json.dump(sorted(seen), f)


# ─────────────────────────────────────────────────────────────
# Main crawl loop
# ─────────────────────────────────────────────────────────────

def run(version: str, output_path: str, max_chars: int, overlap: int,
        progress_file: str, delay: float):
    session = make_session()
    urls    = get_page_urls(version, session)
    seen    = load_seen(progress_file)

    pending = [u for u in urls if u not in seen]
    print(f"[crawl] {len(seen)} already done, {len(pending)} remaining")

    total_chunks  = 0
    total_pages   = 0
    skipped_pages = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, url in enumerate(pending, 1):
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                print(f"[skip]  ({i}/{len(pending)}) {url} — {e}")
                skipped_pages += 1
                seen.add(url)
                save_seen(seen, progress_file)
                time.sleep(delay)
                continue

            # Skip non-HTML responses (PDFs, binaries that slipped through)
            ct = resp.headers.get("Content-Type", "")
            if "text/html" not in ct and "application/xhtml" not in ct:
                print(f"[skip]  ({i}/{len(pending)}) {url} — Content-Type: {ct}")
                skipped_pages += 1
                seen.add(url)
                save_seen(seen, progress_file)
                time.sleep(delay)
                continue

            # Best-effort last-modified: HTTP header, then meta tag
            sor_last_modified = resp.headers.get("Last-Modified", "")
            if not sor_last_modified:
                soup_meta = BeautifulSoup(resp.text, "lxml")
                meta = soup_meta.find("meta", attrs={"name": "dcterms.modified"})
                if meta:
                    sor_last_modified = meta.get("content", "")

            sections = extract_sections(resp.text, url)
            if not sections:
                print(f"[empty] ({i}/{len(pending)}) {url}")
                skipped_pages += 1
                seen.add(url)
                save_seen(seen, progress_file)
                time.sleep(delay)
                continue

            page_title  = sections[0]["page_title"]
            document_id = stable_uuid(version, url)
            chunk_index = 0

            for section in sections:
                for sub_body in subsplit(section["body"], max_chars, overlap):
                    record = make_record(
                        page_url=url,
                        page_title=page_title,
                        section=section,
                        sub_body=sub_body,
                        chunk_index=chunk_index,
                        document_id=document_id,
                        max_chars=max_chars,
                        overlap=overlap,
                        sor_last_modified=sor_last_modified,
                        version=version,
                    )
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_index  += 1
                    total_chunks += 1

            total_pages += 1
            seen.add(url)
            save_seen(seen, progress_file)

            if i % 50 == 0 or i == len(pending):
                print(f"[crawl] ({i}/{len(pending)}) pages done — {total_chunks} chunks so far")

            time.sleep(delay)

    print(f"\n[done] Pages crawled   : {total_pages}")
    print(f"[done] Pages skipped   : {skipped_pages}")
    print(f"[done] Chunks written  : {total_chunks}")
    print(f"[done] Output          : {output_path}")
    print(f"[done] Progress file   : {progress_file}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Crawl docs.openshift.com and emit a JSONL file for vector DB ingestion."
    )
    p.add_argument("--version", default=DEFAULT_VERSION,
                   help=f"OCP version to crawl (default: {DEFAULT_VERSION})")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--resume", action="store_true",
                   help="Resume from crawl_progress.json (skip already-seen URLs)")
    p.add_argument("--progress-file", default=DEFAULT_PROGRESS,
                   help=f"Progress tracking file (default: {DEFAULT_PROGRESS})")
    p.add_argument("--max-section-chars", type=int, default=DEFAULT_MAX_CHARS,
                   help=f"Max chars per chunk before sub-splitting (default: {DEFAULT_MAX_CHARS})")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                   help=f"Overlap chars when sub-splitting (default: {DEFAULT_OVERLAP})")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help=f"Seconds between requests (default: {DEFAULT_DELAY})")
    args = p.parse_args()

    if not args.resume:
        # Warn if progress file already exists and --resume was not passed
        import os
        if os.path.exists(args.progress_file):
            print(f"[warn]  Progress file '{args.progress_file}' exists but --resume not set.")
            print(f"[warn]  To resume from where you left off, pass --resume.")
            print(f"[warn]  Continuing will re-crawl all URLs (appending to output).")

    try:
        run(
            version=args.version,
            output_path=args.output,
            max_chars=args.max_section_chars,
            overlap=args.overlap,
            progress_file=args.progress_file,
            delay=args.delay,
        )
    except KeyboardInterrupt:
        print("\n[interrupt] Crawl stopped. Run with --resume to continue.")
        sys.exit(0)


if __name__ == "__main__":
    main()
