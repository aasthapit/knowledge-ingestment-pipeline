"""
confluence.py
Crawls a Confluence page and all of its descendants, converting each page
to a pipeline-schema JSONL record.

Supports both Confluence Cloud (email + API token) and Confluence Server /
Data Center (Personal Access Token).

Usage
-----
    from pipeline.confluence import ConfluenceCrawler

    crawler = ConfluenceCrawler(
        base_url="https://mycompany.atlassian.net",
        auth_type="cloud",      # or "server"
        email="me@example.com", # Cloud only
        api_token="ATATT3...",  # Cloud API token OR Server PAT
    )
    pages = crawler.crawl(page_url="https://.../.../pages/12345678/My-Page")
    crawler.export_jsonl(pages, "output/confluence_export.jsonl")
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from atlassian import Confluence
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConfluencePage:
    page_id: str
    title: str
    space_key: str
    url: str
    content_text: str
    ancestors: list[str] = field(default_factory=list)   # titles of parent pages
    labels: list[str]   = field(default_factory=list)
    version: int = 1
    author: str = ""
    last_modified: str = ""


# ---------------------------------------------------------------------------
# HTML → text helpers
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """
    Convert Confluence storage-format HTML to clean plain text.

    Preserves heading structure as text lines so the content is useful
    for embedding without markup noise.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Replace block-level elements with newlines so text doesn't run together
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    for tag in soup.find_all(["p", "li", "tr", "br", "div"]):
        tag.insert_after("\n")

    # Expand code blocks so code is readable
    for tag in soup.find_all("ac:plain-text-body"):
        tag.replace_with(f"\n```\n{tag.get_text()}\n```\n")

    text = soup.get_text(separator=" ")
    # Collapse excessive whitespace while keeping paragraph breaks
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Page-ID extraction from URL
# ---------------------------------------------------------------------------

def _extract_page_id(url: str) -> str | None:
    """
    Try to pull a numeric page ID from a Confluence URL.

    Handles:
    - Cloud:  .../pages/123456789/Page-Title
    - Server: .../viewpage.action?pageId=123456789
    - Bare ID string passed directly
    """
    if re.fullmatch(r"\d+", url.strip()):
        return url.strip()

    # Cloud URL pattern
    m = re.search(r"/pages/(\d+)", url)
    if m:
        return m.group(1)

    # Server/DC viewpage.action
    m = re.search(r"[?&]pageId=(\d+)", url)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# ConfluenceCrawler
# ---------------------------------------------------------------------------

class ConfluenceCrawler:
    """
    Recursively crawl a Confluence page tree and convert to plain-text records.

    Parameters
    ----------
    base_url:
        Root URL of your Confluence instance, e.g.
        ``https://mycompany.atlassian.net`` (Cloud) or
        ``https://confluence.mycompany.com`` (Server).
    auth_type:
        ``"cloud"`` — uses HTTP Basic Auth with *email* + *api_token*.
        ``"server"`` — uses Bearer token auth with *api_token* (PAT).
    email:
        Required for Cloud auth only.
    api_token:
        Atlassian API token (Cloud) or Personal Access Token (Server/DC).
    timeout:
        HTTP request timeout in seconds (default 30).
    verify_ssl:
        Set to False to skip SSL certificate verification.  Useful for
        on-premise Confluence instances with self-signed certificates.
        Defaults to True.
    """

    def __init__(
        self,
        base_url: str,
        auth_type: str = "cloud",
        email: str = "",
        api_token: str = "",
        timeout: int = 30,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url  = base_url.rstrip("/")
        self.auth_type = auth_type.lower()

        if self.auth_type == "cloud":
            if not email or not api_token:
                raise ValueError("Cloud auth requires both email and api_token.")
            self._confluence = Confluence(
                url=self.base_url,
                username=email,
                password=api_token,
                cloud=True,
                verify_ssl=verify_ssl,
                timeout=timeout,
            )
        else:
            if not api_token:
                raise ValueError("Server auth requires an api_token (Personal Access Token).")
            self._confluence = Confluence(
                url=self.base_url,
                token=api_token,
                verify_ssl=verify_ssl,
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _get_page(self, page_id: str) -> dict:
        return self._confluence.get_page_by_id(
            page_id,
            expand="body.storage,ancestors,metadata.labels,version,space",
        )

    def _get_children(self, page_id: str) -> list[dict]:
        return self._confluence.get_child_pages(page_id)

    def _page_url(self, page_id: str, space_key: str, title: str) -> str:
        """Build the canonical web URL for a page."""
        slug = title.replace(" ", "+")
        return f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}/{slug}"

    # ------------------------------------------------------------------
    # Record parsing
    # ------------------------------------------------------------------

    def _parse_page(self, raw: dict) -> ConfluencePage:
        page_id   = str(raw["id"])
        title     = raw.get("title", "")
        space_key = raw.get("space", {}).get("key", "")
        version   = raw.get("version", {}).get("number", 1)
        author    = (
            raw.get("version", {})
               .get("by", {})
               .get("displayName", "")
        )
        last_mod  = raw.get("version", {}).get("when", "")

        # Ancestor breadcrumb (titles only, root → immediate parent)
        ancestors = [a.get("title", "") for a in raw.get("ancestors", [])]

        # Labels
        label_items = (
            raw.get("metadata", {})
               .get("labels", {})
               .get("results", [])
        )
        labels = [l.get("name", "") for l in label_items if l.get("name")]

        # Body
        html = raw.get("body", {}).get("storage", {}).get("value", "")
        text = _html_to_text(html)

        url = self._page_url(page_id, space_key, title)

        return ConfluencePage(
            page_id=page_id,
            title=title,
            space_key=space_key,
            url=url,
            content_text=text,
            ancestors=ancestors,
            labels=labels,
            version=version,
            author=author,
            last_modified=last_mod,
        )

    # ------------------------------------------------------------------
    # Recursive crawl
    # ------------------------------------------------------------------

    def _iter_descendants(
        self,
        page_id: str,
        depth: int,
        max_depth: int,
        visited: set[str],
    ) -> Iterator[str]:
        """Yield all descendant page IDs via DFS."""
        if page_id in visited:
            return
        visited.add(page_id)

        if max_depth >= 0 and depth > max_depth:
            return

        for child in self._get_children(page_id):
            child_id = str(child["id"])
            if child_id not in visited:
                yield child_id
                yield from self._iter_descendants(
                    child_id, depth + 1, max_depth, visited
                )

    def crawl(
        self,
        page_url: str,
        max_depth: int = -1,
        progress_cb: Callable[[int, int], None] | None = None,
        extra_tags: list[str] | None = None,
    ) -> list[ConfluencePage]:
        """
        Fetch a page and all its descendants.

        Parameters
        ----------
        page_url:
            URL of the parent page, or a bare page ID string.
        max_depth:
            Maximum recursion depth (-1 = unlimited).
        progress_cb:
            ``progress_cb(fetched, total_discovered)`` — total is -1 until
            the full tree has been discovered.
        extra_tags:
            Additional labels applied to every page.

        Returns
        -------
        list[ConfluencePage]
        """
        page_id = _extract_page_id(page_url)
        if not page_id:
            raise ValueError(
                f"Could not extract a page ID from: {page_url!r}\n"
                "Pass a URL containing /pages/12345678/… or a bare numeric ID."
            )

        extra_tags = extra_tags or []

        # Collect all IDs first so we can show accurate progress
        logger.info("Discovering page tree from root %s …", page_id)
        visited: set[str] = set()
        all_ids = [page_id] + list(
            self._iter_descendants(page_id, depth=1, max_depth=max_depth, visited=visited)
        )
        total = len(all_ids)
        logger.info("Found %d pages to fetch.", total)

        pages: list[ConfluencePage] = []
        for i, pid in enumerate(all_ids, 1):
            try:
                raw  = self._get_page(pid)
                page = self._parse_page(raw)
                page.labels.extend(t for t in extra_tags if t not in page.labels)
                if page.content_text.strip():
                    pages.append(page)
                else:
                    logger.debug("Skipping empty page %s (%s)", pid, page.title)
            except Exception as exc:
                logger.warning("Could not fetch page %s: %s", pid, exc)
            if progress_cb:
                progress_cb(i, total)

        logger.info("Crawled %d pages successfully.", len(pages))
        return pages

    # ------------------------------------------------------------------
    # JSONL export
    # ------------------------------------------------------------------

    @staticmethod
    def to_record(page: ConfluencePage) -> dict:
        """
        Convert a ConfluencePage to a pipeline-schema JSONL record.

        The record is in the "pipeline" schema (content + source) so it
        can be imported directly via the JSONL import tab.
        """
        # Build section breadcrumb:  Space > Parent > … > Page Title
        section_parts = page.ancestors + [page.title]
        section = " > ".join(section_parts)

        return {
            "chunk_id":    str(uuid.uuid5(uuid.NAMESPACE_URL, page.url)),
            "source":      page.url,
            "title":       page.title,
            "section":     section,
            "content":     page.content_text,
            "tags":        page.labels,
            "metadata": {
                "citation": {
                    "source_path": page.url,
                    "source_type": "confluence",
                    "title":       page.title,
                    "url":         page.url,
                    "author":      page.author,
                    "created_date": page.last_modified,
                },
                "confluence": {
                    "page_id":      page.page_id,
                    "space_key":    page.space_key,
                    "version":      page.version,
                    "ancestors":    page.ancestors,
                    "last_modified": page.last_modified,
                },
            },
        }

    def export_jsonl(
        self,
        pages: list[ConfluencePage],
        output_path: str | Path,
    ) -> Path:
        """Write pages to a JSONL file and return the path."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for page in pages:
                fh.write(json.dumps(self.to_record(page), ensure_ascii=False) + "\n")
        logger.info("Wrote %d records to %s", len(pages), out)
        return out
