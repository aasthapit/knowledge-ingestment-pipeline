"""Confluence Import — crawl one or more page trees and stage them under a Knowledge Base."""
from __future__ import annotations

import io
import json
import re
import time

import streamlit as st

from pipeline.config import settings

st.title("🔗 Confluence Import")
st.caption(
    "Connect to Confluence, pick a Knowledge Base, and pull all its registered page trees "
    "into staging in one go. You can also crawl a one-off URL without registering it."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_confluence_kbs() -> list[dict]:
    from pipeline.mongo_store import get_kb_store
    return get_kb_store().list_all(source_type="confluence")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ── Knowledge Base selector ───────────────────────────────────────────────────

st.subheader("Knowledge Base")

try:
    conf_kbs = _load_confluence_kbs()
except Exception as _exc:
    conf_kbs = []
    st.warning(f"Could not load Knowledge Bases: {_exc}")

if conf_kbs:
    kb_options = {kb["name"]: kb["kb_id"] for kb in conf_kbs}
    kb_options["＋ Create new Confluence KB…"] = "__new__"
else:
    kb_options = {"＋ Create new Confluence KB…": "__new__"}

selected_kb_label = st.selectbox(
    "Target Knowledge Base",
    options=list(kb_options.keys()),
    help="Crawled pages are staged under this Knowledge Base.",
)
selected_kb_id = kb_options.get(selected_kb_label)

if selected_kb_id == "__new__":
    st.info("Create a Confluence Knowledge Base on the **Knowledge Bases** page first, then return here.")
    new_kb_name = st.text_input("New KB name *", placeholder="e.g. team-confluence")
    if st.button("Create KB", type="primary") and new_kb_name.strip():
        try:
            from pipeline.mongo_store import get_kb_store
            new_kb_id = get_kb_store().create(
                name=new_kb_name.strip(),
                source_type="confluence",
                description="",
                confluence_sources=[],
            )
            st.success(f"Created KB **{new_kb_name.strip()}**. Select it above and add page URLs on the Knowledge Bases page.")
            _load_confluence_kbs.clear()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.stop()

if not selected_kb_id:
    st.stop()

selected_kb   = next((kb for kb in conf_kbs if kb["kb_id"] == selected_kb_id), None)
kb_name       = selected_kb["name"] if selected_kb else "kb"
kb_sources    = (selected_kb.get("confluence_sources") or []) if selected_kb else []
kb_max_depth  = (selected_kb.get("max_depth") or -1) if selected_kb else -1

# ── Connection ────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Connection")

_default_auth_type = (
    "Cloud (API token)"
    if settings.confluence_auth_type.lower() == "cloud"
    else "Server / DC (PAT)"
)

col_url, col_type = st.columns([3, 1])
with col_url:
    base_url = st.text_input(
        "Confluence base URL",
        value=settings.confluence_base_url,
        placeholder="https://mycompany.atlassian.net",
        help="Root URL of your Confluence instance — no trailing slash.",
    )
with col_type:
    auth_type = st.selectbox(
        "Auth type",
        ["Cloud (API token)", "Server / DC (PAT)"],
        index=0 if _default_auth_type == "Cloud (API token)" else 1,
    )

is_cloud = auth_type.startswith("Cloud")

if is_cloud:
    email = st.text_input(
        "Atlassian account email",
        value=settings.confluence_email,
        placeholder="you@example.com",
    )
else:
    email = ""

api_token = st.text_input(
    "API token / Personal Access Token",
    value=settings.confluence_api_token,
    type="password",
)

col_ssl, col_wiki = st.columns(2)
with col_ssl:
    verify_ssl = not st.checkbox(
        "Disable SSL certificate verification  *(self-signed / internal CA)*",
        value=not settings.confluence_verify_ssl,
    )
with col_wiki:
    strip_wiki = st.checkbox(
        "Strip `/wiki` from source URLs",
        value=True,
        help="Remove the /wiki path prefix from source URLs in the JSONL output. "
             "Enable if your Confluence links don't include /wiki.",
    )

# ── Page selection ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Pages to crawl")

col_depth, col_tags = st.columns(2)
with col_depth:
    max_depth = st.number_input(
        "Max depth (-1 = all)",
        min_value=-1,
        value=kb_max_depth,
        step=1,
        help="How many levels of child pages to follow. -1 fetches the entire tree.",
    )
with col_tags:
    extra_tags_raw = st.text_input(
        "Extra tags  *(optional, applied to all pages)*",
        placeholder="confluence, internal",
        help="Comma-separated tags added to every crawled page.",
    )
extra_tags = [t.strip() for t in extra_tags_raw.split(",") if t.strip()]

# Show registered KB sources with checkboxes
if kb_sources:
    st.markdown(f"**Registered sources** for *{kb_name}* — select which to crawl:")
    selected_sources: list[dict] = []
    for i, src in enumerate(kb_sources):
        url  = src.get("url", "")
        desc = src.get("description", "")
        src_tags = src.get("tags") or []
        label = url + (f" — {desc}" if desc else "")
        checked = st.checkbox(label, value=True, key=f"src_chk_{i}")
        if checked:
            selected_sources.append(src)
else:
    selected_sources = []
    st.info("No sources registered to this KB yet. Use the one-off URL below, or add sources on the **Knowledge Bases** page.")

# One-off URL expander
with st.expander("＋ Crawl additional (one-off) URL"):
    oneoff_url = st.text_input(
        "Page URL",
        placeholder="https://mycompany.atlassian.net/wiki/spaces/TEAM/pages/123456789/My-Page",
        key="oneoff_url",
    )
    oneoff_desc = st.text_input("Description (optional)", key="oneoff_desc")
    oneoff_tags_raw = st.text_input("Tags (comma-separated)", key="oneoff_tags")
    oneoff_tags = [t.strip() for t in oneoff_tags_raw.split(",") if t.strip()]
    save_oneoff = st.checkbox("Save this URL to the KB for future use", value=True, key="save_oneoff")

    if oneoff_url.strip():
        selected_sources.append({
            "url": oneoff_url.strip(),
            "description": oneoff_desc.strip(),
            "tags": oneoff_tags,
            "_oneoff": True,
            "_save": save_oneoff,
        })

# ── Output format ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("Output")

output_mode = st.radio(
    "After crawling",
    [
        "Stage directly in Review Queue",
        "Download as JSONL file",
        "Stage + download",
    ],
    horizontal=True,
    help=(
        "Stage = push into MongoDB staging so you can review and export from the "
        "Knowledge Bases page.  Download = saves a .jsonl file to your machine."
    ),
)

# ── Crawl ─────────────────────────────────────────────────────────────────────

st.divider()

conn_ready = bool(base_url and api_token and selected_sources)
if is_cloud and not email:
    conn_ready = False

if not selected_sources:
    st.caption("Select at least one source to crawl.")
elif not conn_ready:
    st.caption("Fill in connection details to get started.")

if st.button(
    "🚀  Start crawl",
    type="primary",
    width="stretch",
    disabled=not conn_ready,
):
    try:
        from pipeline.confluence import ConfluenceCrawler
        crawler = ConfluenceCrawler(
            base_url=base_url.strip(),
            auth_type="cloud" if is_cloud else "server",
            email=email.strip(),
            api_token=api_token.strip(),
            verify_ssl=verify_ssl,
            strip_wiki_prefix=strip_wiki,
        )
    except Exception as exc:
        st.error(f"Could not initialise connector: {exc}")
        st.stop()

    all_pages: list = []
    all_jsonl_lines: list[str] = []

    progress_bar = st.progress(0.0, text="Starting…")
    total_sources = len(selected_sources)

    for src_idx, src in enumerate(selected_sources):
        url      = src.get("url", "")
        src_tags = list(set(extra_tags + (src.get("tags") or [])))
        progress_bar.progress(src_idx / total_sources, text=f"Crawling {url}…")

        def _cb(done: int, total: int, _url: str = url) -> None:
            if total > 0:
                frac = (src_idx + done / total) / total_sources
                progress_bar.progress(frac, text=f"Fetching {done:,}/{total:,} pages from {_url}…")

        try:
            pages = crawler.crawl(
                page_url=url,
                max_depth=int(max_depth),
                progress_cb=_cb,
                extra_tags=src_tags,
            )
        except Exception as exc:
            st.warning(f"Crawl failed for {url}: {exc}")
            continue

        all_pages.extend(pages)
        all_jsonl_lines.extend(
            json.dumps(crawler.to_record(p), ensure_ascii=False) for p in pages
        )

        # Save one-off URL to KB if requested
        if src.get("_oneoff") and src.get("_save"):
            try:
                from pipeline.mongo_store import get_kb_store
                kb_store = get_kb_store()
                current_kb = kb_store.get(selected_kb_id)
                existing = current_kb.get("confluence_sources") or [] if current_kb else []
                if not any(s.get("url") == url for s in existing):
                    new_src = {"url": url, "description": src.get("description", ""), "tags": src.get("tags") or []}
                    kb_store.update(selected_kb_id, confluence_sources=existing + [new_src])
                    _load_confluence_kbs.clear()
            except Exception:
                pass

    progress_bar.progress(1.0, text="Crawl complete!")

    if not all_pages:
        st.warning("No pages with content were found. Check the page URLs and permissions.")
        st.stop()

    st.success(f"Fetched **{len(all_pages):,}** pages across {total_sources} source(s).")

    with st.expander(f"Preview — first 5 of {len(all_pages):,} pages"):
        for pg in all_pages[:5]:
            st.markdown(f"**{pg.title}**")
            st.caption(" > ".join(pg.ancestors + [pg.title]))
            st.caption(f"🔗 {pg.url}")
            st.text(pg.content_text[:400] + ("…" if len(pg.content_text) > 400 else ""))
            st.divider()

    ts       = int(time.time())
    filename = f"{_slug(kb_name)}_{ts}.jsonl"

    # Build bytes for staging (without export-only fields — staging re-reads the raw records)
    stage_bytes = ("\n".join(all_jsonl_lines) + "\n").encode("utf-8")

    # Build bytes for download with injected export fields
    import uuid as _uuid
    doc_id = str(_uuid.uuid4())
    dl_lines = []
    for raw in all_jsonl_lines:
        rec = json.loads(raw)
        rec["document_id"] = doc_id
        rec["usecase_id"]  = ""
        rec["agent_filter"] = ""
        dl_lines.append(json.dumps(rec, ensure_ascii=False))
    dl_bytes = ("\n".join(dl_lines) + "\n").encode("utf-8")

    if output_mode in ("Stage directly in Review Queue", "Stage + download"):
        with st.spinner("Staging pages in MongoDB…"):
            try:
                from pipeline.ingest import ingest_jsonl
                buf      = io.BytesIO(stage_bytes)
                buf.name = filename
                result   = ingest_jsonl(
                    source=buf,
                    batch_name=filename,
                    extra_tags=extra_tags,
                    kb_id=selected_kb_id,
                )
                st.session_state["confluence_import_result"] = result
            except Exception as exc:
                st.error(f"Staging failed: {exc}")

    if output_mode in ("Download as JSONL file", "Stage + download"):
        st.download_button(
            label=f"⬇️  Download {filename}",
            data=dl_bytes,
            file_name=filename,
            mime="application/x-ndjson",
            width="stretch",
        )

# ── Staging result card ───────────────────────────────────────────────────────

if "confluence_import_result" in st.session_state:
    r = st.session_state["confluence_import_result"]
    st.divider()
    with st.container(border=True):
        st.success(
            f"**{r['batch_name']}** staged — "
            f"**{r['total_chunks']:,}** pages from **{r['unique_sources']:,}** URLs."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Pages staged",  f"{r['total_chunks']:,}")
        m2.metric("Unique URLs",   f"{r['unique_sources']:,}")
        m3.metric("Pre-embedded",  "Yes" if r.get("has_embeddings") else "No")
        st.info(
            "Go to **Review Queue** to approve these pages, then download JSONL from the "
            "**Knowledge Bases** page.\n\n"
            f"Batch ID: `{r['doc_id']}`"
        )
        if st.button("Clear", key="clear_confluence"):
            del st.session_state["confluence_import_result"]
            st.rerun()

# ── Help ──────────────────────────────────────────────────────────────────────

with st.expander("How to get an API token"):
    st.markdown("""
**Confluence Cloud**
1. Go to [id.atlassian.com](https://id.atlassian.com/manage-profile/security/api-tokens)
2. Click **Create API token** → give it a name → copy the token.
3. Enter your Atlassian account email in the *Email* field above.

**Confluence Server / Data Center**
1. Log in to Confluence → click your avatar → **Profile**.
2. Go to **Personal Access Tokens** → click **Create token**.
3. Copy the token — leave the *Email* field empty above.

**Required permissions**
The account or token only needs **read (view) access** to the spaces and pages
you want to import. No write permissions are required.
""")
