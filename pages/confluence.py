"""Confluence Import — crawl a page tree and stage it for the knowledge base."""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import streamlit as st

from pipeline.config import settings

st.title("🔗 Confluence Import")
st.caption(
    "Connect to Confluence, pick a parent page, and pull the entire page tree "
    "into the knowledge base in one go."
)

# ── Connection ────────────────────────────────────────────────────────────────

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
        help=(
            "Cloud: email + API token from id.atlassian.com/manage-profile/security.\n"
            "Server/DC: Personal Access Token from your profile."
        ),
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
    help=(
        "Cloud: generate at id.atlassian.com → Security → API tokens.\n"
        "Server/DC: generate at Your Profile → Personal Access Tokens."
    ),
)

verify_ssl = not st.checkbox(
    "Disable SSL certificate verification  *(self-signed / internal CA)*",
    value=not settings.confluence_verify_ssl,
    help=(
        "Tick this for on-premise Confluence instances that use a self-signed "
        "or internally-signed certificate.  Do not use on public/cloud instances."
    ),
)


# ── Page selection ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Page")

page_url = st.text_input(
    "Parent page URL or numeric page ID",
    placeholder="https://mycompany.atlassian.net/wiki/spaces/TEAM/pages/123456789/My-Page",
    help="The crawler will fetch this page and every sub-page beneath it.",
)

col_depth, col_kb = st.columns(2)
with col_depth:
    max_depth = st.number_input(
        "Max depth (-1 = all)",
        min_value=-1,
        value=-1,
        step=1,
        help="How many levels of child pages to follow. -1 fetches the entire tree.",
    )
with col_kb:
    kb_name = st.text_input(
        "Knowledge base name",
        value="default",
        help="Logical name for ledger grouping and drift tracking.",
    )

extra_tags_raw = st.text_input(
    "Extra tags  *(optional)*",
    placeholder="confluence, internal, team-docs",
    help="Comma-separated tags added to every page.",
)
extra_tags = [t.strip() for t in extra_tags_raw.split(",") if t.strip()]

# Use case tracking — optional for Confluence imports; enables Use Case Ledger entry
uc_col1, uc_col2 = st.columns(2)
with uc_col1:
    conf_usecase_id = st.text_input(
        "Use case ID  *(optional)*",
        placeholder="GENAI1597_SSOP",
        help="When set, staged pages are tracked in the Use Case Ledger under this identifier.",
    )
with uc_col2:
    conf_agent_filter = st.text_input(
        "Agent filter  *(optional)*",
        placeholder="ssop_cloud_operations_knowledge_agent",
        help="When set along with Use case ID, enables ledger tracking and refresh scheduling.",
    )

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
        "Stage = pushed into MongoDB staging so you can review and push to the "
        "vector DB from the Review Queue page.  "
        "Download = saves a .jsonl file to your machine."
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
    use_container_width=True,
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
    status_text  = st.empty()

    def _cb(done: int, total: int) -> None:
        if total > 0:
            frac = done / total
            progress_bar.progress(frac, text=f"Fetching {done:,} of {total:,} pages…")
        else:
            progress_bar.progress(
                min(done / 100, 0.99),
                text=f"Fetching page {done:,}…",
            )

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

    # ── Preview ───────────────────────────────────────────────────────────
    with st.expander(f"Preview — first 5 of {len(pages):,} pages"):
        for pg in pages[:5]:
            st.markdown(f"**{pg.title}**")
            breadcrumb = " > ".join(pg.ancestors + [pg.title])
            st.caption(breadcrumb)
            st.caption(f"🔗 {pg.url}")
            preview = pg.content_text[:400]
            st.text(preview + ("…" if len(pg.content_text) > 400 else ""))
            st.divider()

    # ── Build JSONL bytes ─────────────────────────────────────────────────
    jsonl_lines = [
        json.dumps(crawler.to_record(p), ensure_ascii=False) for p in pages
    ]
    jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
    filename    = f"confluence_{len(pages)}_pages.jsonl"

    # ── Stage ─────────────────────────────────────────────────────────────
    if output_mode in ("Stage directly in Review Queue", "Stage + download"):
        with st.spinner("Staging pages in MongoDB…"):
            try:
                from pipeline.ingest import ingest_jsonl
                buf = io.BytesIO(jsonl_bytes)
                buf.name = filename
                _uc_id = conf_usecase_id.strip() or None
                _ag_flt = conf_agent_filter.strip() or None
                result = ingest_jsonl(
                    source=buf,
                    batch_name=filename,
                    extra_tags=extra_tags,
                    kb_name=kb_name.strip() or "default",
                    usecase_id=_uc_id,
                    agent_filter=_ag_flt,
                )
                st.session_state["confluence_import_result"] = result

                # Register this crawl in the Use Case Ledger so it can be
                # scheduled for periodic refresh
                if _uc_id and _ag_flt:
                    try:
                        from pipeline.mongo_store import get_usecase_ledger
                        get_usecase_ledger().upsert_confluence_source(
                            usecase_id=_uc_id,
                            agent_filter=_ag_flt,
                            kb_name=kb_name.strip() or "default",
                            page_urls=[page_url.strip()],
                            max_depth=int(max_depth),
                            extra_tags=extra_tags,
                        )
                    except Exception as uc_exc:
                        st.warning(f"Could not register Use Case Ledger source: {uc_exc}")
            except Exception as exc:
                st.error(f"Staging failed: {exc}")

    # ── Download ──────────────────────────────────────────────────────────
    if output_mode in ("Download as JSONL file", "Stage + download"):
        st.download_button(
            label=f"⬇️  Download {filename}",
            data=jsonl_bytes,
            file_name=filename,
            mime="application/x-ndjson",
            use_container_width=True,
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
        m1.metric("Pages staged",    f"{r['total_chunks']:,}")
        m2.metric("Unique URLs",     f"{r['unique_sources']:,}")
        m3.metric("Pre-embedded",    "Yes" if r.get("has_embeddings") else "No")
        st.info(
            "Go to **Review Queue** → **Push to Knowledge Base** to embed "
            "and make these pages searchable.\n\n"
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
