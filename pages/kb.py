"""Knowledge Bases — named document sources (Confluence, JSONL, web, or file)."""
from __future__ import annotations

import io
import re
import time

import streamlit as st

st.title("📂 Knowledge Bases")
st.caption(
    "A Knowledge Base is a named source of documents — a Confluence page tree, a JSONL file, "
    "a web crawler output, or uploaded files. KBs are shared across corpora."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_kbs(source_type: str | None = None) -> list[dict]:
    from pipeline.mongo_store import get_kb_store
    return get_kb_store().list_all(source_type=source_type)


@st.cache_data(ttl=30)
def _load_kb(kb_id: str) -> dict | None:
    from pipeline.mongo_store import get_kb_store
    return get_kb_store().get(kb_id)


def _invalidate() -> None:
    _load_kbs.clear()
    _load_kb.clear()


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16] if iso else "—"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


STATUS_COLOR = {"empty": "gray", "staging": "orange", "ready": "green"}
SOURCE_ICON  = {"confluence": "🔗", "jsonl": "📦", "web": "🌐", "file": "📄"}

# ── Connection guard ──────────────────────────────────────────────────────────

try:
    all_kbs = _load_kbs()
except Exception as _exc:
    st.error(f"Could not connect to MongoDB: {_exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────

if "kb_selected" not in st.session_state:
    st.session_state.kb_selected = None
if "kb_create_open" not in st.session_state:
    st.session_state.kb_create_open = False
if "kb_edit_open" not in st.session_state:
    st.session_state.kb_edit_open = False

# ── Layout ────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 3], gap="large")

# =============================================================================
# Left — KB list + create button
# =============================================================================

with left:
    col_hd, col_btn = st.columns([2, 1])
    with col_hd:
        st.subheader("Knowledge Bases")
    with col_btn:
        if st.button("＋ New", width="stretch"):
            st.session_state.kb_create_open = not st.session_state.kb_create_open

    # ── Create form ───────────────────────────────────────────────────────────
    if st.session_state.kb_create_open:
        with st.form("create_kb_form", border=True):
            st.markdown("**Create Knowledge Base**")
            new_name = st.text_input("Name *", placeholder="e.g. openshift-docs")
            new_desc = st.text_area("Description", height=60)
            new_type = st.radio(
                "Source type",
                ["confluence", "jsonl", "web", "file"],
                horizontal=True,
                key="create_kb_type",
            )

            if new_type == "confluence":
                st.markdown("**Confluence sources** *(add more after creation)*")
                first_url  = st.text_input(
                    "First page URL *",
                    placeholder="https://mycompany.atlassian.net/wiki/spaces/OPS/pages/123",
                )
                first_desc = st.text_input("Description (optional)", placeholder="e.g. API reference docs")
                first_tags = st.text_input("Tags (comma-separated)", placeholder="api, internal")
                new_depth  = st.number_input("Max depth (-1 = all)", min_value=-1, value=-1)
                new_cron   = st.text_input(
                    "Refresh schedule (cron expression, optional)",
                    placeholder="0 2 * * 1  (every Monday at 2 AM)",
                )
            elif new_type == "web":
                first_url  = st.text_area(
                    "Seed URLs *(optional, for reference)*",
                    placeholder="https://docs.example.com/",
                    height=60,
                    help="Stored for reference only — run your external web crawler and import JSONL via Add Document.",
                )
                first_desc = ""
                first_tags = ""
                new_depth  = -1
                new_cron   = ""
                st.info(
                    "🌐 **Web KB** — run your own web crawler against these URLs, "
                    "export as JSONL, then import on the **Add Document** page."
                )
            else:
                first_url  = ""
                first_desc = ""
                first_tags = ""
                new_depth  = -1
                new_cron   = ""
                if new_type == "file":
                    st.info("📄 Upload documents on the **Add Document** page and assign them to this KB.")

            submitted = st.form_submit_button("Create", type="primary")
            if submitted:
                if not new_name.strip():
                    st.error("Name is required.")
                elif new_type == "confluence" and not first_url.strip():
                    st.error("At least one Confluence URL is required.")
                else:
                    try:
                        from pipeline.mongo_store import get_kb_store
                        sources = []
                        if first_url.strip():
                            tags = [t.strip() for t in first_tags.split(",") if t.strip()]
                            sources = [{"url": first_url.strip(), "description": first_desc.strip(), "tags": tags}]
                        kb_id = get_kb_store().create(
                            name=new_name.strip(),
                            source_type=new_type,
                            description=new_desc.strip(),
                            confluence_sources=sources,
                            max_depth=int(new_depth),
                            refresh_cron=new_cron.strip() or None,
                        )
                        _invalidate()
                        st.session_state.kb_selected = kb_id
                        st.session_state.kb_create_open = False
                        st.success(f"Created KB **{new_name.strip()}**")
                        st.rerun()
                    except Exception as exc:
                        if "duplicate" in str(exc).lower() or "E11000" in str(exc):
                            st.error(f"Name **{new_name.strip()}** already exists.")
                        else:
                            st.error(str(exc))

    # ── Type filter ───────────────────────────────────────────────────────────
    type_filter = st.segmented_control(
        "Filter",
        ["All", "Confluence", "JSONL", "Web", "File"],
        default="All",
        label_visibility="collapsed",
    )
    filter_map = {"All": None, "Confluence": "confluence", "JSONL": "jsonl", "Web": "web", "File": "file"}
    try:
        filtered_kbs = _load_kbs(source_type=filter_map.get(type_filter or "All"))
    except Exception:
        filtered_kbs = all_kbs

    # ── KB list ───────────────────────────────────────────────────────────────
    if not filtered_kbs:
        st.info("No Knowledge Bases yet. Click **＋ New** to create one.")
    else:
        for kb in filtered_kbs:
            kid    = kb["kb_id"]
            icon   = SOURCE_ICON.get(kb.get("source_type", ""), "📂")
            status = kb.get("status", "empty")
            color  = STATUS_COLOR.get(status, "gray")
            label  = f"{icon} **{kb['name']}**  \n:{color}[{status}] · {len(kb.get('doc_ids') or [])} docs"
            btn_type = "primary" if st.session_state.kb_selected == kid else "secondary"
            if st.button(label, key=f"sel_kb_{kid}", width="stretch", type=btn_type):
                st.session_state.kb_selected = kid
                st.session_state.kb_edit_open = False
                st.rerun()

# =============================================================================
# Right — KB detail
# =============================================================================

with right:
    sel_id = st.session_state.kb_selected

    if sel_id is None:
        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:240px;border:2px dashed #ccc;border-radius:8px;"
            "color:#888;font-size:0.9rem'>Select a Knowledge Base to view details</div>",
            unsafe_allow_html=True,
        )
    else:
        kb = _load_kb(sel_id)
        if kb is None:
            st.warning("Knowledge Base not found — it may have been deleted.")
            st.session_state.kb_selected = None
        else:
            source_type = kb.get("source_type", "")
            kb_name     = kb.get("name", "kb")

            # ── Header ────────────────────────────────────────────────────────
            hd1, hd2, hd3, hd4 = st.columns([3, 1, 1, 1])
            with hd1:
                icon = SOURCE_ICON.get(source_type, "📂")
                st.subheader(f"{icon} {kb_name}")
                if kb.get("description"):
                    st.caption(kb["description"])
            with hd2:
                if source_type == "confluence" and st.button("🔄 Refresh", width="stretch", help="Re-crawl all sources and replace staged docs"):
                    st.session_state[f"confirm_refresh_{sel_id}"] = True
            with hd3:
                if st.button("Edit", width="stretch"):
                    st.session_state.kb_edit_open = not st.session_state.kb_edit_open
            with hd4:
                if st.button("Delete", width="stretch", type="secondary"):
                    st.session_state[f"confirm_del_kb_{sel_id}"] = True

            # ── Refresh confirm ───────────────────────────────────────────────
            if st.session_state.get(f"confirm_refresh_{sel_id}"):
                st.warning(
                    f"Refresh **{kb_name}**? This will clear all staged docs and re-crawl "
                    f"all {len(kb.get('confluence_sources') or [])} registered source(s)."
                )
                rc1, rc2, _ = st.columns([1, 1, 4])
                with rc1:
                    if st.button("Yes, refresh", type="primary", key="confirm_refresh_yes"):
                        st.session_state[f"confirm_refresh_{sel_id}"] = False
                        sources = kb.get("confluence_sources") or []
                        if not sources:
                            st.error("No Confluence sources registered. Add a URL first.")
                        else:
                            from pipeline.config import settings
                            if not settings.confluence_base_url or not settings.confluence_api_token:
                                st.error(
                                    "Confluence connection not configured. "
                                    "Set `CONFLUENCE_BASE_URL` and `CONFLUENCE_API_TOKEN` in your `.env` file, "
                                    "or use the **Confluence** import page."
                                )
                            else:
                                try:
                                    from pipeline.mongo_store import get_staging
                                    cleared = get_staging().clear_by_kb(sel_id)

                                    from pipeline.confluence import ConfluenceCrawler
                                    crawler = ConfluenceCrawler(
                                        base_url=settings.confluence_base_url,
                                        auth_type=settings.confluence_auth_type or "cloud",
                                        email=settings.confluence_email or "",
                                        api_token=settings.confluence_api_token,
                                        strip_wiki_prefix=True,
                                    )

                                    total_staged = 0
                                    import io as _io
                                    import json as _json
                                    import time as _time
                                    from pipeline.ingest import ingest_jsonl

                                    progress = st.progress(0.0, text="Starting refresh…")
                                    for i, src in enumerate(sources):
                                        url  = src.get("url", "")
                                        tags = src.get("tags") or []
                                        progress.progress(i / len(sources), text=f"Crawling {url}…")
                                        pages = crawler.crawl(
                                            page_url=url,
                                            max_depth=kb.get("max_depth", -1),
                                            extra_tags=tags,
                                        )
                                        if pages:
                                            ts       = int(_time.time())
                                            fname    = f"{_slug(kb_name)}_{ts}.jsonl"
                                            lines    = [_json.dumps(crawler.to_record(p), ensure_ascii=False) for p in pages]
                                            buf      = _io.BytesIO(("\n".join(lines) + "\n").encode())
                                            buf.name = fname
                                            result   = ingest_jsonl(source=buf, batch_name=fname, extra_tags=tags, kb_id=sel_id)
                                            total_staged += result.get("total_chunks", 0)

                                    progress.progress(1.0, text="Refresh complete!")
                                    _invalidate()
                                    st.success(
                                        f"Refreshed **{kb_name}**: cleared {cleared} old doc(s), "
                                        f"staged **{total_staged}** new chunk(s) from {len(sources)} source(s)."
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Refresh failed: {exc}")
                with rc2:
                    if st.button("Cancel", key="confirm_refresh_no"):
                        st.session_state[f"confirm_refresh_{sel_id}"] = False
                        st.rerun()

            # ── Delete confirm ────────────────────────────────────────────────
            if st.session_state.get(f"confirm_del_kb_{sel_id}"):
                st.warning(
                    f"Delete **{kb_name}**? This removes the KB record but does not delete "
                    "staging documents — they remain in the review queue."
                )
                dc1, dc2, _ = st.columns([1, 1, 4])
                with dc1:
                    if st.button("Yes, delete", type="primary", key="confirm_del_kb_yes"):
                        from pipeline.mongo_store import get_kb_store
                        get_kb_store().delete(sel_id)
                        _invalidate()
                        st.session_state.kb_selected = None
                        st.session_state[f"confirm_del_kb_{sel_id}"] = False
                        st.rerun()
                with dc2:
                    if st.button("Cancel", key="confirm_del_kb_no"):
                        st.session_state[f"confirm_del_kb_{sel_id}"] = False
                        st.rerun()

            # ── Edit form ─────────────────────────────────────────────────────
            if st.session_state.kb_edit_open:
                with st.form("edit_kb_form", border=True):
                    st.markdown("**Edit Knowledge Base**")
                    e_name = st.text_input("Name *", value=kb.get("name", ""))
                    e_desc = st.text_area("Description", value=kb.get("description", ""), height=60)

                    if source_type == "confluence":
                        e_depth = st.number_input("Max depth (-1 = all)", min_value=-1, value=kb.get("max_depth", -1))
                        e_cron  = st.text_input("Refresh schedule (cron)", value=kb.get("refresh_cron") or "")
                    elif source_type in ("web",):
                        e_urls = st.text_area(
                            "Seed URLs (one per line)",
                            value="\n".join(kb.get("confluence_urls") or []),
                            height=80,
                            help="Stored for reference — run your crawler externally.",
                        )
                    elif source_type == "jsonl":
                        e_file_name = st.text_input("File name", value=kb.get("file_name", ""))

                    with st.expander("Chunking config"):
                        current_strategy = kb.get("chunk_strategy") or "Global default"
                        strategy_options = ["Global default", "heading", "character"]
                        e_strategy = st.radio(
                            "Chunk strategy",
                            strategy_options,
                            index=strategy_options.index(current_strategy) if current_strategy in strategy_options else 0,
                            horizontal=True,
                        )
                        e_max_chars = st.number_input(
                            "Max chars per chunk (0 = global default)",
                            min_value=0,
                            value=kb.get("chunk_max_chars") or 0,
                        )
                        e_overlap = st.number_input(
                            "Overlap chars (0 = global default)",
                            min_value=0,
                            value=kb.get("chunk_overlap_chars") or 0,
                        )

                    saved = st.form_submit_button("Save", type="primary")
                    if saved:
                        if not e_name.strip():
                            st.error("Name is required.")
                        else:
                            try:
                                from pipeline.mongo_store import get_kb_store
                                update_kwargs: dict = {
                                    "name": e_name.strip(),
                                    "description": e_desc.strip(),
                                    "chunk_strategy": None if e_strategy == "Global default" else e_strategy,
                                    "chunk_max_chars": int(e_max_chars) if e_max_chars else None,
                                    "chunk_overlap_chars": int(e_overlap) if e_overlap else None,
                                }
                                if source_type == "confluence":
                                    update_kwargs["max_depth"]    = int(e_depth)
                                    update_kwargs["refresh_cron"] = e_cron.strip() or None
                                elif source_type == "web":
                                    update_kwargs["confluence_urls"] = [u.strip() for u in e_urls.splitlines() if u.strip()]
                                elif source_type == "jsonl":
                                    update_kwargs["file_name"] = e_file_name.strip()
                                get_kb_store().update(sel_id, **update_kwargs)
                                _invalidate()
                                st.session_state.kb_edit_open = False
                                st.success("Knowledge Base updated.")
                                st.rerun()
                            except Exception as exc:
                                if "duplicate" in str(exc).lower() or "E11000" in str(exc):
                                    st.error(f"Name **{e_name.strip()}** already exists.")
                                else:
                                    st.error(str(exc))

            # ── Stats chips ───────────────────────────────────────────────────
            status    = kb.get("status", "empty")
            doc_count = len(kb.get("doc_ids") or [])

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Status", status)
            s2.metric("Source type", source_type)
            s3.metric("Staged docs", doc_count)
            s4.metric("Last updated", _fmt_date(kb.get("last_updated")))

            st.divider()

            # ── Confluence details ─────────────────────────────────────────────
            if source_type == "confluence":
                sources = kb.get("confluence_sources") or []

                # Source table
                col_src_hd, col_src_btn = st.columns([4, 1])
                with col_src_hd:
                    st.markdown("**Confluence sources**")
                with col_src_btn:
                    if st.button("＋ Add URL", key="add_src_btn"):
                        st.session_state["add_src_open"] = not st.session_state.get("add_src_open", False)

                if not sources:
                    st.info("No sources registered. Click **＋ Add URL** to add a Confluence page tree.")
                else:
                    for idx, src in enumerate(sources):
                        with st.container(border=True):
                            r1, r2 = st.columns([5, 1])
                            with r1:
                                st.markdown(f"`{src.get('url', '')}`")
                                if src.get("description"):
                                    st.caption(src["description"])
                                if src.get("tags"):
                                    st.caption("Tags: " + ", ".join(src["tags"]))
                            with r2:
                                if st.button("✕", key=f"rm_src_{sel_id}_{idx}", help="Remove this source"):
                                    updated = [s for i, s in enumerate(sources) if i != idx]
                                    from pipeline.mongo_store import get_kb_store
                                    get_kb_store().update(sel_id, confluence_sources=updated)
                                    _invalidate()
                                    st.rerun()

                # Add source form
                if st.session_state.get("add_src_open"):
                    with st.form("add_src_form", border=True):
                        st.markdown("**Add Confluence source**")
                        new_src_url  = st.text_input("Page URL *", placeholder="https://…/pages/123456/…")
                        new_src_desc = st.text_input("Description", placeholder="e.g. API reference for v2")
                        new_src_tags = st.text_input("Tags (comma-separated)", placeholder="api, internal, v2")
                        if st.form_submit_button("Add", type="primary"):
                            if not new_src_url.strip():
                                st.error("URL is required.")
                            else:
                                tags = [t.strip() for t in new_src_tags.split(",") if t.strip()]
                                updated = sources + [{"url": new_src_url.strip(), "description": new_src_desc.strip(), "tags": tags}]
                                from pipeline.mongo_store import get_kb_store
                                get_kb_store().update(sel_id, confluence_sources=updated)
                                _invalidate()
                                st.session_state["add_src_open"] = False
                                st.rerun()

                col_depth, col_cron = st.columns(2)
                col_depth.metric("Max depth", kb.get("max_depth", -1))
                col_cron.metric("Refresh schedule", kb.get("refresh_cron") or "—")

                if kb.get("last_ingested_at"):
                    st.caption(f"Last ingested: {_fmt_date(kb.get('last_ingested_at'))}")

            # ── Web crawler details ───────────────────────────────────────────
            elif source_type == "web":
                if kb.get("confluence_urls"):
                    st.markdown("**Seed URLs**")
                    for url in kb.get("confluence_urls") or []:
                        st.code(url, language=None)
                st.info(
                    "🌐 Run your external web crawler against the seed URLs above and import "
                    "the resulting JSONL on the **Add Document** page."
                )

            # ── JSONL details ─────────────────────────────────────────────────
            elif source_type == "jsonl":
                if kb.get("file_name"):
                    st.metric("File", kb["file_name"])
                st.markdown("To add more content, go to **Add Document → Bulk JSONL Import** and select this KB.")

            # ── File details ──────────────────────────────────────────────────
            elif source_type == "file":
                st.info("📄 Upload documents on the **Add Document** page and assign them to this KB.")

            st.divider()

            # ── Tabs: Staged / Pushed ─────────────────────────────────────────
            tab_staged, tab_pushed = st.tabs(["Staged documents", "Pushed documents"])

            with tab_staged:
                try:
                    from pipeline.mongo_store import get_staging
                    staged = get_staging().list_all(kb_id=sel_id)
                    if not staged:
                        st.info("No staged documents for this Knowledge Base yet.")
                    else:
                        import pandas as pd
                        rows = [
                            {
                                "Title":  d.get("title", "—"),
                                "Status": d.get("status", ""),
                                "Chunks": d.get("chunk_count", 0),
                                "Score":  f"{float(d.get('quality_score', 0)):.0%}",
                                "Staged": _fmt_date(d.get("staged_at")),
                                "doc_id": d.get("doc_id", ""),
                            }
                            for d in staged
                        ]
                        df = pd.DataFrame(rows)
                        st.dataframe(df.drop(columns=["doc_id"]), hide_index=True)

                        # JSONL download
                        if st.button("⬇️ Download staged chunks as JSONL", key="dl_staged_jsonl"):
                            try:
                                chunks = get_staging().get_chunks_by_kb(sel_id, status=None)
                                if not chunks:
                                    st.warning("No chunks to export.")
                                else:
                                    import json as _json
                                    lines = "\n".join(_json.dumps(c, ensure_ascii=False) for c in chunks)
                                    fname = f"{_slug(kb_name)}_{int(time.time())}.jsonl"
                                    st.download_button(
                                        label=f"⬇️ Download {fname}",
                                        data=lines.encode("utf-8"),
                                        file_name=fname,
                                        mime="application/x-ndjson",
                                    )
                            except Exception as exc:
                                st.error(f"Export failed: {exc}")
                except Exception as exc:
                    st.warning(f"Could not load staged docs: {exc}")

            with tab_pushed:
                try:
                    from pipeline.mongo_store import get_ledger
                    pushed = get_ledger().list_docs(kb_name=kb_name)
                    if not pushed:
                        st.info("No pushed documents for this Knowledge Base yet.")
                    else:
                        import pandas as pd
                        rows_p = [
                            {
                                "Title":       d.get("title", "—"),
                                "Source type": d.get("source_type", "—"),
                                "Chunks":      d.get("chunk_count", 0),
                                "Drift":       d.get("drift_status", "—"),
                                "Pushed":      _fmt_date(d.get("pushed_at")),
                            }
                            for d in pushed
                        ]
                        st.dataframe(pd.DataFrame(rows_p), hide_index=True)

                        # JSONL download from staging (pushed status)
                        if st.button("⬇️ Download pushed chunks as JSONL", key="dl_pushed_jsonl"):
                            try:
                                from pipeline.mongo_store import get_staging
                                chunks = get_staging().get_chunks_by_kb(sel_id, status="pushed")
                                if not chunks:
                                    st.warning("No pushed chunks in staging — they may have been removed after push.")
                                else:
                                    import json as _json
                                    lines = "\n".join(_json.dumps(c, ensure_ascii=False) for c in chunks)
                                    fname = f"{_slug(kb_name)}_{int(time.time())}.jsonl"
                                    st.download_button(
                                        label=f"⬇️ Download {fname}",
                                        data=lines.encode("utf-8"),
                                        file_name=fname,
                                        mime="application/x-ndjson",
                                    )
                            except Exception as exc:
                                st.error(f"Export failed: {exc}")
                except Exception as exc:
                    st.warning(f"Could not load pushed docs: {exc}")
