"""Confluence Import — crawl a page tree and stage it under a Knowledge Base."""
from __future__ import annotations

import io
import json

import streamlit as st

from pipeline.config import settings

st.title("🔗 Confluence Import")
st.caption(
    "Connect to Confluence, pick a parent page, and pull the entire page tree "
    "into a Knowledge Base in one go."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_confluence_kbs() -> list[dict]:
    from pipeline.mongo_store import get_kb_store
    return get_kb_store().list_all(source_type="confluence")


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
                confluence_urls=[],
            )
            st.success(f"Created KB **{new_kb_name.strip()}**. Select it above and fill in a page URL.")
            _load_confluence_kbs.clear()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.stop()

if not selected_kb_id:
    st.stop()

# Pre-populate page URLs from KB config
selected_kb = next((kb for kb in conf_kbs if kb["kb_id"] == selected_kb_id), None)
default_page_url = (selected_kb.get("confluence_urls") or [""])[0] if selected_kb else ""

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

verify_ssl = not st.checkbox(
    "Disable SSL certificate verification  *(self-signed / internal CA)*",
    value=not settings.confluence_verify_ssl,
)

# ── Page selection ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Page")

page_url = st.text_input(
    "Parent page URL or numeric page ID",
    value=default_page_url,
    placeholder="https://mycompany.atlassian.net/wiki/spaces/TEAM/pages/123456789/My-Page",
    help="The crawler will fetch this page and every sub-page beneath it.",
)

col_depth, col_tags = st.columns(2)
with col_depth:
    max_depth = st.number_input(
        "Max depth (-1 = all)",
        min_value=-1,
        value=-1,
        step=1,
        help="How many levels of child pages to follow. -1 fetches the entire tree.",
    )
with col_tags:
    extra_tags_raw = st.text_input(
        "Extra tags  *(optional)*",
        placeholder="confluence, internal, team-docs",
        help="Comma-separated tags added to every page.",
    )
extra_tags = [t.strip() for t in extra_tags_raw.split(",") if t.strip()]

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
        "Stage = push into MongoDB staging so you can review and push from the "
        "Review Queue.  Download = saves a .jsonl file to your machine."
    ),
)

# ── Crawl ─────────────────────────────────────────────────────────────────────

st.divider()

conn_ready = bool(base_url and api_token and page_url)
if is_cloud and not email:
    conn_ready = False

if not conn_ready:
    st.caption("Fill in connection details and a page URL to get started.")

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
        )
    except Exception as exc:
        st.error(f"Could not initialise connector: {exc}")
        st.stop()

    progress_bar = st.progress(0.0, text="Discovering pages…")

    def _cb(done: int, total: int) -> None:
        if total > 0:
            progress_bar.progress(done / total, text=f"Fetching {done:,} of {total:,} pages…")
        else:
            progress_bar.progress(min(done / 100, 0.99), text=f"Fetching page {done:,}…")

    try:
        pages = crawler.crawl(
            page_url=page_url.strip(),
            max_depth=int(max_depth),
            progress_cb=_cb,
            extra_tags=extra_tags,
        )
        progress_bar.progress(1.0, text="Crawl complete!")
    except Exception as exc:
        st.error(f"Crawl failed: {exc}")
        st.stop()

    if not pages:
        st.warning("No pages with content were found. Check the page URL and permissions.")
        st.stop()

    st.success(f"Fetched **{len(pages):,}** pages.")

    with st.expander(f"Preview — first 5 of {len(pages):,} pages"):
        for pg in pages[:5]:
            st.markdown(f"**{pg.title}**")
            breadcrumb = " > ".join(pg.ancestors + [pg.title])
            st.caption(breadcrumb)
            st.caption(f"🔗 {pg.url}")
            preview = pg.content_text[:400]
            st.text(preview + ("…" if len(pg.content_text) > 400 else ""))
            st.divider()

    jsonl_lines = [json.dumps(crawler.to_record(p), ensure_ascii=False) for p in pages]
    jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
    filename    = f"confluence_{len(pages)}_pages.jsonl"

    if output_mode in ("Stage directly in Review Queue", "Stage + download"):
        with st.spinner("Staging pages in MongoDB…"):
            try:
                from pipeline.ingest import ingest_jsonl
                buf = io.BytesIO(jsonl_bytes)
                buf.name = filename
                result = ingest_jsonl(
                    source=buf,
                    batch_name=filename,
                    extra_tags=extra_tags,
                    kb_id=selected_kb_id,
                )
                # Update KB with the crawled page URL
                try:
                    from pipeline.mongo_store import get_kb_store
                    kb_store = get_kb_store()
                    kb = kb_store.get(selected_kb_id)
                    existing_urls = kb.get("confluence_urls") or [] if kb else []
                    if page_url.strip() not in existing_urls:
                        existing_urls.append(page_url.strip())
                        kb_store.update(selected_kb_id, confluence_urls=existing_urls)
                except Exception:
                    pass
                st.session_state["confluence_import_result"] = result
            except Exception as exc:
                st.error(f"Staging failed: {exc}")

    if output_mode in ("Download as JSONL file", "Stage + download"):
        st.download_button(
            label=f"⬇️  Download {filename}",
            data=jsonl_bytes,
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
            "Go to **Review Queue** to approve these pages, then push them via a corpus.\n\n"
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
