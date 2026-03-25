#!/usr/bin/env python3
"""
openshift_docs_to_jsonl.py
──────────────────────────
Converts openshift/openshift-docs into a chunked JSONL file for vector DB
ingestion.

Chunking strategy
─────────────────
Rather than splitting on character count, this script splits on AsciiDoc
heading boundaries (==, ===, ====).  Each heading section becomes one chunk.
Very long sections are sub-split on paragraphs while still keeping the
heading context.

Every chunk's "text" field gets a context prefix prepended so the embedding
model has full semantic context:

    Page: AI Workloads / Red Hat Build of Kueue
    Section: Release notes > Compatible environments

    <body text of the section>

This dramatically improves retrieval quality vs embedding a bare paragraph.

The "page_url" is a deep link including the section anchor, e.g.:
    https://docs.openshift.com/.../ai_workloads/...html#compatible-environments-arch

Usage
─────
    # Auto-clone and process enterprise-4.17
    python openshift_docs_to_jsonl.py

    # Custom branch / output
    python openshift_docs_to_jsonl.py --branch enterprise-4.16 --output ocp_4.16.jsonl

    # Point at an already-cloned repo
    python openshift_docs_to_jsonl.py --repo-dir ./openshift-docs --branch enterprise-4.17

    # Tune max section size before sub-splitting kicks in
    python openshift_docs_to_jsonl.py --max-section-chars 2000 --overlap 150

Requirements
────────────
    pip install pyyaml
    pandoc on PATH  (brew install pandoc)  — strongly recommended for clean text
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

REPO_URL        = "https://github.com/openshift/openshift-docs.git"
SITE_BASE       = "https://docs.openshift.com"
DISTRO_MAP_FILE = "_distro_map.yml"
TOPIC_MAP_FILE  = "_topic_map.yml"
TARGET_DISTRO   = "openshift-enterprise"

USECASE_ID          = "GENAI1597_SSOP"
AGENT_FILTER        = "ssop_cloud_operations_knowledge_agent"
DATA_CLASSIFICATION = "Internal"
UNIT                = "chars"

DEFAULT_BRANCH      = "enterprise-4.17"
DEFAULT_OUTPUT      = "openshift_docs.jsonl"
DEFAULT_MAX_SECTION = 1500   # chars — above this, sub-split on paragraphs
DEFAULT_OVERLAP     = 100    # chars — overlap when sub-splitting


# ─────────────────────────────────────────────────────────────
# Repo helpers
# ─────────────────────────────────────────────────────────────

def clone_repo(branch: str, target_dir: str):
    print(f"[clone] {REPO_URL} @ {branch} -> {target_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch,
         "--single-branch", REPO_URL, target_dir],
        check=True,
    )


def git_mtime(repo_dir: str, rel_path: str) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", repo_dir, "log", "-1", "--format=%cI", "--", rel_path],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# Distro map -> base URL for this branch
# ─────────────────────────────────────────────────────────────

def load_distro_map(repo_dir: str, branch: str) -> str:
    path = os.path.join(repo_dir, DISTRO_MAP_FILE)
    with open(path) as f:
        raw = yaml.safe_load(f)

    site_url = raw.get("site_url", SITE_BASE).rstrip("/")

    for key, val in raw.items():
        if key == "site_url" or not isinstance(val, dict):
            continue
        branches = val.get("branches", {})
        if branch in branches:
            dir_path = branches[branch].get("dir", "").strip("/")
            return f"{site_url}/{dir_path}"

    # Fallback: enterprise-4.17 -> container-platform/4.17
    version = branch.replace("enterprise-", "")
    return f"{site_url}/container-platform/{version}"


# ─────────────────────────────────────────────────────────────
# Topic map -> {stem: {page_url, page_name, dir}}
# ─────────────────────────────────────────────────────────────

def load_topic_map(repo_dir: str, base_url: str) -> dict:
    path = os.path.join(repo_dir, TOPIC_MAP_FILE)
    with open(path) as f:
        docs = list(yaml.safe_load_all(f))

    url_map = {}

    def walk(topics, current_dir):
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            distros = topic.get("Distros", "")
            if distros and TARGET_DISTRO not in distros:
                continue
            dir_part = topic.get("Dir", current_dir)
            if "File" in topic:
                stem = topic["File"]
                page_name = topic.get("Name", stem)
                path_part = f"{dir_part}/{stem}" if dir_part else stem
                url_map[stem] = {
                    "page_url":  f"{base_url}/{path_part}.html",
                    "page_name": page_name,
                    "dir":       dir_part,
                }
            if "Topics" in topic:
                walk(topic["Topics"], dir_part)

    for section in docs:
        if isinstance(section, dict) and "Topics" in section:
            walk(section["Topics"], section.get("Dir", ""))

    return url_map


# ─────────────────────────────────────────────────────────────
# AsciiDoc parsing
# ─────────────────────────────────────────────────────────────

_ANCHOR_RE  = re.compile(r'(?:\[id=["\']([^"\']+)["\']\]|\[\[([^\]]+)\]\])')
_HEADING_RE = re.compile(r'^(={1,4})\s+(.+)$')
_INCLUDE_RE = re.compile(r'^include::([^\[]+)\[.*?\]$', re.MULTILINE)
_ATTR_RE    = re.compile(r'^:[a-zA-Z0-9_\-]+:.*', re.MULTILINE)
_COND_RE    = re.compile(r'^(ifdef|ifndef|endif|ifeval)::[^\n]*', re.MULTILINE)
_DELIM_RE   = re.compile(r'^[-=.+_*]{4,}$', re.MULTILINE)
_ROLE_RE    = re.compile(r'^\[.*?\]$', re.MULTILINE)
_IMG_RE     = re.compile(r'(image|icon)::[^\[]*\[[^\]]*\]')
_XREF_RE    = re.compile(r'xref:[^\[]+\[([^\]]*)\]')
_LINK_RE    = re.compile(r'link:[^\[]+\[([^\]]*)\]')
_SUBST_RE   = re.compile(r'\{[a-zA-Z0-9_\-]+\}')

# List bullets: strip before inline markup so "* *bold*" doesn't confuse bold regex.
# Matches leading *+ or -+ bullets (AsciiDoc unordered lists, any depth).
_LIST_BULLET = re.compile(r'^(\*+|-+)\s+', re.MULTILINE)

# Inline markup — unconstrained (double-marker) first, then constrained (single-marker).
# Constrained bold/italic require a non-word boundary on both sides so list
# bullets and mid-word underscores are never mistaken for markup delimiters.
_BOLD_UNCON = re.compile(r'\*\*(.+?)\*\*', re.DOTALL)
_BOLD_CON   = re.compile(r'(?<!\w|\*)\*(\S.*?\S|\S)\*(?!\w|\*)')
_ITAL_UNCON = re.compile(r'__(.+?)__',     re.DOTALL)
_ITAL_CON   = re.compile(r'(?<!\w|_)_(\S.*?\S|\S)_(?!\w|_)')
_MONO_RE    = re.compile(r'`([^`\n]+)`')


def _strip_adoc_markup(text: str) -> str:
    """
    Remove AsciiDoc syntax, returning readable plain text.

    Ordering is intentional:
      1. Block-level directives (conditionals, includes, attributes, delimiters)
      2. Macros (images, xrefs, links)
      3. Block role annotations
      4. List bullets — must come before inline bold/italic so that the
         bullet character '*' is gone before the bold regex runs
      5. Inline markup — unconstrained (double markers) before constrained
         (single markers) to avoid partial matches
      6. Attribute substitutions and whitespace normalisation
    """
    # 1. Block directives
    text = _COND_RE.sub("", text)
    text = _INCLUDE_RE.sub("", text)
    text = _ATTR_RE.sub("", text)
    text = _DELIM_RE.sub("", text)
    # 2. Macros
    text = _IMG_RE.sub("", text)
    text = _XREF_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    # 3. Role annotations
    text = _ROLE_RE.sub("", text)
    # 4. List bullets → plain dash so list structure is still readable
    text = _LIST_BULLET.sub(r"- ", text)
    # 5. Inline markup
    text = _BOLD_UNCON.sub(r"\1", text)
    text = _BOLD_CON.sub(r"\1", text)
    text = _ITAL_UNCON.sub(r"\1", text)
    text = _ITAL_CON.sub(r"\1", text)
    text = _MONO_RE.sub(r"\1", text)
    # 6. Attribute substitutions + whitespace
    text = _SUBST_RE.sub("", text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _pandoc_plain(adoc_path: str) -> str:
    """Convert an .adoc file to plain text via pandoc."""
    try:
        r = subprocess.run(
            ["pandoc", "-f", "asciidoc", "-t", "plain", "--wrap=none", adoc_path],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout if r.stdout.strip() else ""
    except FileNotFoundError:
        return ""


def resolve_includes(adoc_path: str, repo_dir: str, depth: int = 0) -> str:
    """
    Read an .adoc file and inline all include:: directives recursively.
    Assembly files are largely pointers to modules via include:: — without
    resolving these the assembly body is nearly empty.
    """
    if depth > 8:
        return ""
    try:
        with open(adoc_path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return ""

    def replace_include(m):
        rel = m.group(1).strip()
        base = Path(adoc_path).parent
        candidate = (base / rel).resolve()
        if not candidate.exists():
            candidate = (Path(repo_dir) / rel).resolve()
        if candidate.exists():
            return resolve_includes(str(candidate), repo_dir, depth + 1)
        return ""

    return _INCLUDE_RE.sub(replace_include, raw)


def parse_sections(adoc_path: str, repo_dir: str, page_name: str) -> list:
    """
    Parse an assembly .adoc (with includes resolved) into a list of sections.

    Each section dict:
        heading_text  : str   — e.g. "Compatible environments"
        heading_level : int   — 1 = page title, 2 = ==, 3 = ===, 4 = ====
        anchor        : str   — e.g. "compatible-environments-arch_release-notes"
        breadcrumbs   : list  — parent heading texts above this one
        body          : str   — plain text of just this section's content
    """
    raw = resolve_includes(adoc_path, repo_dir)
    lines = raw.splitlines()

    sections = []
    current_anchor = ""
    current_heading = page_name
    current_level   = 1
    current_body_lines = []
    # Breadcrumb stack: index = heading level - 1, value = heading text
    heading_stack = [page_name]

    def flush(anchor, heading, level, body_lines, stack):
        body = _strip_adoc_markup("\n".join(body_lines))
        if body.strip():
            sections.append({
                "heading_text":  heading,
                "heading_level": level,
                "anchor":        anchor,
                "breadcrumbs":   list(stack[:-1]),  # exclude self
                "body":          body,
            })

    pending_anchor = ""
    for line in lines:
        am = _ANCHOR_RE.match(line.strip())
        if am:
            pending_anchor = am.group(1) or am.group(2) or ""
            continue

        hm = _HEADING_RE.match(line)
        if hm:
            flush(current_anchor, current_heading,
                  current_level, current_body_lines, heading_stack)
            current_body_lines = []

            level = len(hm.group(1))
            heading_text = _SUBST_RE.sub("", hm.group(2).strip()).strip()

            # Trim breadcrumb stack to this level
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(heading_text)

            current_anchor  = pending_anchor
            pending_anchor  = ""
            current_heading = heading_text
            current_level   = level
        else:
            current_body_lines.append(line)

    flush(current_anchor, current_heading,
          current_level, current_body_lines, heading_stack)

    # If pandoc produces better plain text and there is only one flat section,
    # use it for the body (avoids hand-rolled markup stripping artifacts).
    if len(sections) == 1:
        try:
            with tempfile.NamedTemporaryFile(
                    suffix=".adoc", mode="w", encoding="utf-8", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            pandoc_out = _pandoc_plain(tmp_path)
            os.unlink(tmp_path)
            if pandoc_out.strip():
                sections[0]["body"] = pandoc_out.strip()
        except Exception:
            pass

    return sections


# ─────────────────────────────────────────────────────────────
# Context prefix
# ─────────────────────────────────────────────────────────────

def build_prefix(page_name: str, breadcrumbs: list, heading_text: str) -> str:
    """
    Builds the semantic context header prepended to every chunk.

    Example:
        Page: AI Workloads / Red Hat Build of Kueue
        Section: Release notes > Compatible environments
    """
    parts = breadcrumbs + [heading_text]
    section_path = " > ".join(p for p in parts if p and p != page_name)
    lines = [f"Page: {page_name}"]
    if section_path:
        lines.append(f"Section: {section_path}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Sub-splitting for oversized sections
# ─────────────────────────────────────────────────────────────

def subsplit(text: str, max_chars: int, overlap: int) -> list:
    """Split a long body on paragraph boundaries, with optional overlap."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para) if current else para
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            # Carry tail for overlap
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
# Main pipeline
# ─────────────────────────────────────────────────────────────

def process_repo(repo_dir, branch, output_path, max_section, overlap):
    print("[parse] Loading distro map...")
    base_url = load_distro_map(repo_dir, branch)
    print(f"[parse] Base URL: {base_url}")

    print("[parse] Loading topic map...")
    topic_map = load_topic_map(repo_dir, base_url)
    print(f"[parse] {len(topic_map)} assembly entries found")

    adoc_files     = list(Path(repo_dir).rglob("*.adoc"))
    assembly_files = [f for f in adoc_files if f.stem in topic_map]
    print(f"[scan]  {len(assembly_files)} mapped assemblies / {len(adoc_files)} total .adoc files")

    total_chunks = 0
    skipped      = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for adoc_path in assembly_files:
            stem       = adoc_path.stem
            meta       = topic_map[stem]
            rel_path   = str(adoc_path.relative_to(repo_dir))
            source_dir = str(adoc_path.parent.relative_to(repo_dir))
            file_name  = adoc_path.name
            page_name  = meta["page_name"]
            page_url   = meta["page_url"]

            sor_last_modified = git_mtime(repo_dir, rel_path)
            document_id = stable_uuid(branch, rel_path)

            sections = parse_sections(str(adoc_path), repo_dir, page_name)
            if not sections:
                skipped += 1
                continue

            chunk_index = 0

            for section in sections:
                heading = section["heading_text"]
                anchor  = section["anchor"]
                crumbs  = section["breadcrumbs"]
                body    = section["body"]

                if not body.strip():
                    continue

                # Deep-link: page URL + section anchor
                section_url = f"{page_url}#{anchor}" if anchor else page_url

                # Context prefix for semantic embedding
                prefix = build_prefix(page_name, crumbs, heading)

                # Sub-split long sections
                sub_chunks = subsplit(body, max_section, overlap)

                for sub_body in sub_chunks:
                    # Full text = what the embedding model encodes
                    full_text = f"{prefix}\n\n{sub_body}"

                    char_count = len(full_text)
                    word_count = len(full_text.split())
                    chunk_id   = stable_uuid(document_id, chunk_index,
                                             content_hash(full_text))

                    record = {
                        # ── Required fields ──────────────────────────────────
                        "id":                  chunk_id,
                        "source_file":         file_name,
                        "source_dir":          source_dir,
                        "relative_path":       rel_path,
                        "document_id":         document_id,
                        "chunk_id":            chunk_id,
                        "chunk_index":         chunk_index,
                        "unit":                UNIT,
                        "chunk_size_setting":  max_section,
                        "overlap_setting":     overlap,
                        "usecase_id":          USECASE_ID,
                        "file_name":           file_name,
                        "data_classification": DATA_CLASSIFICATION,
                        "sor_last_modified":   sor_last_modified,
                        "page_url":            section_url,
                        "page_name":           page_name,
                        "agent_filter":        AGENT_FILTER,
                        "char_count":          char_count,
                        "word_count":          word_count,
                        # ── Extended fields (remove if schema is strict) ─────
                        # Useful for filtering, prompt construction, & debugging
                        "section_heading":     heading,
                        "section_breadcrumbs": crumbs,
                        "section_anchor":      anchor,
                        "text":                full_text,
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_index  += 1
                    total_chunks += 1

            if chunk_index == 0:
                skipped += 1

    print(f"\n[done] Assemblies processed : {len(assembly_files) - skipped}")
    print(f"[done] Assemblies skipped   : {skipped}")
    print(f"[done] Total chunks written : {total_chunks}")
    print(f"[done] Output               : {output_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert openshift/openshift-docs to a JSONL file for vector DB ingestion."
    )
    p.add_argument("--branch", default=DEFAULT_BRANCH,
                   help=f"Git branch to clone (default: {DEFAULT_BRANCH})")
    p.add_argument("--repo-dir", default=None,
                   help="Already-cloned repo path (skips auto-clone)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--max-section-chars", type=int, default=DEFAULT_MAX_SECTION,
                   help=f"Max chars per section before sub-splitting (default: {DEFAULT_MAX_SECTION})")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                   help=f"Overlap chars when sub-splitting (default: {DEFAULT_OVERLAP})")
    p.add_argument("--keep-repo", action="store_true",
                   help="Keep the auto-cloned repo after processing")
    args = p.parse_args()

    tmp_dir  = None
    repo_dir = args.repo_dir

    if repo_dir:
        if not os.path.isdir(repo_dir):
            print(f"[error] --repo-dir '{repo_dir}' not found.", file=sys.stderr)
            sys.exit(1)
        print(f"[repo]  Using existing repo: {repo_dir}")
    else:
        tmp_dir  = tempfile.mkdtemp(prefix="openshift-docs-")
        repo_dir = os.path.join(tmp_dir, "openshift-docs")
        try:
            clone_repo(args.branch, repo_dir)
        except subprocess.CalledProcessError as e:
            print(f"[error] Git clone failed: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        process_repo(
            repo_dir    = repo_dir,
            branch      = args.branch,
            output_path = args.output,
            max_section = args.max_section_chars,
            overlap     = args.overlap,
        )
    finally:
        if tmp_dir and not args.keep_repo:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
