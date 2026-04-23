"""Use Case Ledger — track ingested content per use case and agent filter."""
from __future__ import annotations

import io
import json

import streamlit as st

st.title("Use Case Ledger")
st.caption(
    "Track what has been accepted and ingested into the vector DB per use case "
    "and agent filter. Register Confluence sources for periodic refresh, and "
    "download chunks as JSONL for external embedding pipelines."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_entries() -> list[dict]:
    from pipeline.mongo_store import get_usecase_ledger
    return get_usecase_ledger().list_entries()


@st.cache_data(ttl=30)
def _load_usecases() -> list[str]:
    from pipeline.mongo_store import get_usecase_ledger
    return get_usecase_ledger().get_distinct_usecases()


@st.cache_data(ttl=30)
def _load_agent_filters(usecase_id: str) -> list[str]:
    from pipeline.mongo_store import get_usecase_ledger
    return get_usecase_ledger().get_agent_filters_for_usecase(usecase_id)


@st.cache_data(ttl=30)
def _load_kb_docs(usecase_id: str, agent_filter: str) -> list[dict]:
    from pipeline.mongo_store import get_ledger
    return get_ledger().list_docs_by_usecase(usecase_id, agent_filter, limit=500)


@st.cache_data(ttl=30)
def _load_confluence_sources() -> list[dict]:
    from pipeline.mongo_store import get_usecase_ledger
    return get_usecase_ledger().list_confluence_sources()


# ── Drift helpers ─────────────────────────────────────────────────────────────


def _compute_drift(
    snapshot: list[dict], current: list[dict]
) -> dict:
    """
    Compare a stored page snapshot against the current page list.
    Returns {added, removed, changed} — each a list of page dicts.
    """
    snap_by_id = {p["page_id"]: p for p in snapshot}
    curr_by_id = {p["page_id"]: p for p in current}

    added   = [p for pid, p in curr_by_id.items() if pid not in snap_by_id]
    removed = [p for pid, p in snap_by_id.items() if pid not in curr_by_id]
    changed = [
        {"old": snap_by_id[pid], "new": curr}
        for pid, curr in curr_by_id.items()
        if pid in snap_by_id and curr["version"] != snap_by_id[pid]["version"]
    ]
    return {"added": added, "removed": removed, "changed": changed}


# ── Main tabs ─────────────────────────────────────────────────────────────────

try:
    all_entries = _load_entries()
except Exception as exc:
    st.error(f"Could not connect to MongoDB: {exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

tab_ledger, tab_sources, tab_bulk, tab_export = st.tabs([
    "Ledger",
    "Confluence Sources",
    "Bulk Import",
    "Export JSONL",
])


# =============================================================================
# Tab 1 — Ledger / Health view
# =============================================================================

with tab_ledger:

    # ── Use case selector ─────────────────────────────────────────────────────
    try:
        usecase_options = _load_usecases()
    except Exception as exc:
        st.error(f"Could not load use cases: {exc}")
        usecase_options = []

    if not usecase_options:
        st.info(
            "No use case ledger entries yet. "
            "Ingest documents with a **Use case ID** and **Agent filter** set, "
            "then push them to the knowledge base."
        )
    else:
        col_uc, col_af = st.columns(2)
        with col_uc:
            sel_usecase = st.selectbox("Use case ID", usecase_options, key="ledger_uc")
        with col_af:
            agent_options = _load_agent_filters(sel_usecase) if sel_usecase else []
            sel_agent = st.selectbox(
                "Agent filter",
                agent_options or ["—"],
                key="ledger_af",
            )

        if sel_usecase and sel_agent and sel_agent != "—":
            import pandas as pd

            # ── Load data ─────────────────────────────────────────────────────
            try:
                kb_docs = _load_kb_docs(sel_usecase, sel_agent)
            except Exception as exc:
                st.error(f"Could not load documents: {exc}")
                kb_docs = []

            # Entry from usecase ledger (has chunk_count, last_pushed_at)
            uc_entry = next(
                (e for e in all_entries
                 if e.get("usecase_id") == sel_usecase and e.get("agent_filter") == sel_agent),
                {},
            )

            # ── Health metrics ────────────────────────────────────────────────
            st.subheader("Knowledge Base Health")

            total_chunks = uc_entry.get("chunk_count", 0)
            total_docs   = len(kb_docs)
            last_pushed  = (uc_entry.get("last_pushed_at") or "")[:10] or "—"

            # Quality and drift aggregates from kb_docs
            n_stale   = sum(1 for d in kb_docs if d.get("drift_status") == "stale")
            n_deleted = sum(1 for d in kb_docs if d.get("drift_status") == "deleted")
            n_current = sum(1 for d in kb_docs if d.get("drift_status") == "current")
            avg_quality = (
                round(sum(float(d.get("quality_score") or 0) for d in kb_docs) / total_docs, 2)
                if total_docs else 0.0
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Searchable chunks",  f"{total_chunks:,}",
                      help="Chunks currently indexed in the vector store for this use case")
            m2.metric("Documents",          total_docs,
                      help="Pushed documents contributing to this use case")
            m3.metric("Last pushed",        last_pushed)
            m4.metric("Avg quality",        f"{avg_quality:.0%}",
                      help="Average fraction of clean chunks across all pushed documents")

            # Drift and staleness callouts
            if n_stale or n_deleted:
                drift_parts = []
                if n_stale:
                    drift_parts.append(f"**{n_stale}** stale (source changed since last push)")
                if n_deleted:
                    drift_parts.append(f"**{n_deleted}** deleted (source file no longer exists)")
                st.warning("Content drift detected — " + "  ·  ".join(drift_parts)
                           + ".  Go to **KB Health** to re-ingest.")
            elif n_current:
                st.success(f"All {n_current} document source(s) are current.")

            # ── Confluence source for this use case ───────────────────────────
            st.divider()
            st.subheader("Confluence Source")

            try:
                all_sources = _load_confluence_sources()
            except Exception:
                all_sources = []

            uc_source = next(
                (s for s in all_sources
                 if s.get("usecase_id") == sel_usecase and s.get("agent_filter") == sel_agent),
                None,
            )

            if uc_source:
                STATUS_ICONS = {"idle": "✅", "running": "⏳", "failed": "❌"}
                status = uc_source.get("refresh_status", "idle")
                last_r = (uc_source.get("last_refresh_at") or "")[:16].replace("T", " ") or "never"
                next_r = (uc_source.get("next_refresh_at") or "")[:16].replace("T", " ") or "—"
                cron   = uc_source.get("refresh_cron") or "manual only"

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Pages (last crawl)", uc_source.get("crawled_page_count", "—"))
                sc2.metric("Last crawled",       last_r)
                sc3.metric("Next crawl",         next_r)
                sc4.metric("Status",             STATUS_ICONS.get(status, "?") + " " + status)

                st.caption(
                    f"Schedule: `{cron}`  ·  "
                    f"{len(uc_source.get('page_urls') or [])} root URL(s)  ·  "
                    f"max depth: {uc_source.get('max_depth', -1)}"
                )

                if uc_source.get("refresh_error"):
                    st.error(f"Last refresh error: {uc_source['refresh_error']}")

                rc1, rc2 = st.columns([1, 4])
                with rc1:
                    if st.button("Refresh now", key="health_refresh_btn"):
                        with st.spinner("Refreshing…"):
                            try:
                                from pipeline.refresh_scheduler import trigger_refresh_now
                                trigger_refresh_now(sel_usecase, sel_agent)
                                st.success("Refresh complete.")
                                st.cache_data.clear()
                            except Exception as exc:
                                st.error(f"Refresh failed: {exc}")
            else:
                st.info(
                    "No Confluence source registered for this use case. "
                    "Go to the **Confluence Sources** tab to register one."
                )

            # ── Pushed documents table ────────────────────────────────────────
            st.divider()
            st.subheader("Pushed Documents")

            if not kb_docs:
                st.info("No documents pushed for this use case / agent filter yet.")
            else:
                st.caption(f"{total_docs:,} document(s)  ·  {total_chunks:,} total chunks")

                DRIFT_ICONS = {
                    "current": "✅",
                    "stale":   "⚠️",
                    "deleted": "🗑️",
                    "unknown": "❓",
                }
                doc_rows = []
                for doc in kb_docs:
                    pushed = (doc.get("pushed_at") or "")[:10]
                    drift  = doc.get("drift_status", "unknown")
                    tags   = doc.get("tags") or []
                    doc_rows.append({
                        "Status":      DRIFT_ICONS.get(drift, "❓") + " " + drift,
                        "Title":       doc.get("title") or doc.get("source_path") or "Untitled",
                        "Type":        doc.get("source_type", ""),
                        "Chunks":      doc.get("chunk_count", 0),
                        "Quality":     round(float(doc.get("quality_score") or 0), 3),
                        "Tags":        "; ".join(tags) if tags else "",
                        "Pushed":      pushed,
                    })

                st.dataframe(
                    pd.DataFrame(doc_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Quality": st.column_config.ProgressColumn(
                            "Quality", min_value=0.0, max_value=1.0, format="%.2f",
                        ),
                        "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
                    },
                )

        else:
            # Summary table when no specific use case is selected
            st.subheader("All Use Cases")
            if all_entries:
                import pandas as pd
                rows = []
                for e in all_entries:
                    last_pushed = (e.get("last_pushed_at") or "")[:10] or "—"
                    rows.append({
                        "Use case ID":  e.get("usecase_id", ""),
                        "Agent filter": e.get("agent_filter", ""),
                        "KB":           e.get("kb_name", "default"),
                        "Chunks":       e.get("chunk_count", 0),
                        "Last pushed":  last_pushed,
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
                    },
                )
            else:
                st.info("No use cases found.")


# =============================================================================
# Tab 2 — Confluence Sources
# =============================================================================

with tab_sources:
    st.subheader("Registered Confluence Sources")
    st.caption(
        "Register Confluence page URLs per use case and agent filter. "
        "The scheduler will re-crawl and re-push them on the configured cron schedule."
    )

    try:
        sources = _load_confluence_sources()
    except Exception as exc:
        st.error(f"Could not load sources: {exc}")
        sources = []

    if sources:
        import pandas as pd

        STATUS_ICONS = {"idle": "✅", "running": "⏳", "failed": "❌"}
        src_rows = []
        for s in sources:
            last_r = (s.get("last_refresh_at") or "")[:19].replace("T", " ") or "—"
            next_r = (s.get("next_refresh_at") or "")[:19].replace("T", " ") or "—"
            status = s.get("refresh_status", "idle")
            src_rows.append({
                "Use case ID":    s.get("usecase_id", ""),
                "Agent filter":   s.get("agent_filter", ""),
                "KB":             s.get("kb_name", "default"),
                "Root URLs":      len(s.get("page_urls") or []),
                "Pages (crawled)": s.get("crawled_page_count", "—"),
                "Cron":           s.get("refresh_cron") or "—",
                "Status":         STATUS_ICONS.get(status, "?") + " " + status,
                "Last refresh":   last_r,
                "Next refresh":   next_r,
            })

        st.dataframe(
            pd.DataFrame(src_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── Manual refresh trigger ────────────────────────────────────────────

        st.caption("Trigger an immediate refresh for a registered source:")
        trig_col1, trig_col2, trig_col3 = st.columns([2, 2, 1])
        trig_options = [f"{s['usecase_id']} / {s['agent_filter']}" for s in sources]
        with trig_col1:
            trig_sel = st.selectbox(
                "Source to refresh", trig_options,
                key="trig_sel", label_visibility="collapsed",
            )
        with trig_col3:
            if st.button("Refresh now", key="trig_btn"):
                idx = trig_options.index(trig_sel)
                chosen = sources[idx]
                with st.spinner(f"Refreshing {trig_sel} …"):
                    try:
                        from pipeline.refresh_scheduler import trigger_refresh_now
                        trigger_refresh_now(chosen["usecase_id"], chosen["agent_filter"])
                        st.success("Refresh complete.")
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(f"Refresh failed: {exc}")

        # ── Drift check ───────────────────────────────────────────────────────

        st.divider()
        st.subheader("Drift check")
        st.caption(
            "Compare the page versions stored during the last crawl against "
            "what Confluence currently has — without re-fetching content."
        )

        drift_options = trig_options
        dc_col1, dc_col2 = st.columns([3, 1])
        with dc_col1:
            drift_sel = st.selectbox(
                "Source to check", drift_options,
                key="drift_sel", label_visibility="collapsed",
            )
        with dc_col2:
            check_drift = st.button("Check drift", key="drift_btn")

        if check_drift:
            idx = drift_options.index(drift_sel)
            chosen = sources[idx]
            source_id = chosen["source_id"]
            page_urls = chosen.get("page_urls") or []
            max_depth = chosen.get("max_depth", -1)

            snapshot = []
            try:
                from pipeline.mongo_store import get_usecase_ledger
                snapshot = get_usecase_ledger().get_crawl_snapshot(source_id)
            except Exception as exc:
                st.error(f"Could not load stored snapshot: {exc}")

            if not snapshot:
                st.warning(
                    "No page snapshot stored for this source yet. "
                    "Run a full refresh first so a baseline is recorded."
                )
            else:
                from pipeline.config import settings
                if not settings.confluence_base_url or not settings.confluence_api_token:
                    st.warning(
                        "Confluence credentials are not configured in settings. "
                        "Set `CONFLUENCE_BASE_URL` and `CONFLUENCE_API_TOKEN` in your `.env` file."
                    )
                else:
                    with st.spinner("Fetching current page metadata from Confluence…"):
                        try:
                            from pipeline.confluence import ConfluenceCrawler
                            crawler = ConfluenceCrawler(
                                base_url=settings.confluence_base_url,
                                auth_type=settings.confluence_auth_type,
                                email=settings.confluence_email,
                                api_token=settings.confluence_api_token,
                                verify_ssl=settings.confluence_verify_ssl,
                            )
                            current_pages: list[dict] = []
                            for url in page_urls:
                                try:
                                    current_pages.extend(
                                        crawler.crawl_metadata(url, max_depth=max_depth)
                                    )
                                except Exception as exc:
                                    st.warning(f"Could not check {url}: {exc}")
                        except Exception as exc:
                            st.error(f"Could not connect to Confluence: {exc}")
                            current_pages = []

                    if current_pages:
                        drift = _compute_drift(snapshot, current_pages)
                        total_now  = len(current_pages)
                        total_snap = len(snapshot)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Pages at last crawl", total_snap)
                        mc2.metric("Pages now",           total_now,
                                   delta=total_now - total_snap,
                                   delta_color="normal")
                        mc3.metric("Changed",  len(drift["changed"]),
                                   delta=len(drift["changed"]) or None, delta_color="inverse")
                        mc4.metric("Removed",  len(drift["removed"]),
                                   delta=-len(drift["removed"]) if drift["removed"] else None,
                                   delta_color="inverse")

                        if not any([drift["added"], drift["removed"], drift["changed"]]):
                            st.success("No drift detected — Confluence content matches the last crawl.")
                        else:
                            if drift["added"]:
                                with st.expander(f"🆕 {len(drift['added'])} new page(s) since last crawl"):
                                    for p in drift["added"]:
                                        st.markdown(f"- **{p['title']}** (v{p['version']})")

                            if drift["removed"]:
                                with st.expander(f"🗑️ {len(drift['removed'])} page(s) removed since last crawl"):
                                    for p in drift["removed"]:
                                        st.markdown(f"- **{p['title']}** (was v{p['version']})")

                            if drift["changed"]:
                                with st.expander(f"✏️ {len(drift['changed'])} page(s) updated since last crawl"):
                                    for entry in drift["changed"]:
                                        old = entry["old"]
                                        new = entry["new"]
                                        st.markdown(
                                            f"- **{new['title']}** — "
                                            f"v{old['version']} → v{new['version']}  "
                                            f"*(last modified: {new['last_modified'][:10] if new['last_modified'] else '?'})*"
                                        )

                            st.info(
                                "Run **Refresh now** above to re-crawl and re-ingest the updated content."
                            )

        if any(s.get("refresh_error") for s in sources):
            with st.expander("Refresh errors"):
                for s in sources:
                    if s.get("refresh_error"):
                        st.code(
                            f"{s['usecase_id']} / {s['agent_filter']}:\n{s['refresh_error']}",
                            language=None,
                        )
    else:
        st.info("No Confluence sources registered yet. Use the **Bulk Import** or **Register** form below.")

    st.divider()

    # ── Register / update form ────────────────────────────────────────────────

    st.subheader("Register or update a Confluence source")

    with st.form("confluence_source_form"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            f_usecase = st.text_input(
                "Use case ID *",
                placeholder="GENAI1597_SSOP",
            )
        with f_col2:
            f_agent = st.text_input(
                "Agent filter *",
                placeholder="ssop_cloud_operations_knowledge_agent",
            )

        f_kb = st.text_input("KB name", value="default")

        f_urls_raw = st.text_area(
            "Confluence page URLs *",
            placeholder=(
                "https://mycompany.atlassian.net/wiki/spaces/SPACE/pages/12345678/Page-Title\n"
                "https://…"
            ),
            height=120,
            help="One URL per line. The crawler will recursively fetch child pages.",
        )

        f_col3, f_col4 = st.columns(2)
        with f_col3:
            f_max_depth = st.number_input(
                "Max crawl depth (-1 = unlimited)",
                min_value=-1,
                value=-1,
                step=1,
            )
        with f_col4:
            f_cron = st.text_input(
                "Refresh schedule (cron expression)",
                placeholder="0 3 * * 1  (Mon 3 AM)",
                help="Standard 5-field cron. Leave empty to disable auto-refresh.",
            )

        f_extra_tags = st.text_input(
            "Extra tags (comma-separated)",
            placeholder="internal, ssop",
        )

        submitted = st.form_submit_button("Save source")

    if submitted:
        errors = []
        if not f_usecase.strip():
            errors.append("Use case ID is required.")
        if not f_agent.strip():
            errors.append("Agent filter is required.")
        page_urls = [u.strip() for u in f_urls_raw.splitlines() if u.strip()]
        if not page_urls:
            errors.append("At least one Confluence page URL is required.")
        if f_cron.strip():
            try:
                from croniter import croniter
                if not croniter.is_valid(f_cron.strip()):
                    errors.append(f"Invalid cron expression: {f_cron.strip()!r}")
            except ImportError:
                pass

        if errors:
            for e in errors:
                st.error(e)
        else:
            extra_tags = [t.strip() for t in f_extra_tags.split(",") if t.strip()]
            try:
                from pipeline.mongo_store import get_usecase_ledger
                get_usecase_ledger().upsert_confluence_source(
                    usecase_id=f_usecase.strip(),
                    agent_filter=f_agent.strip(),
                    kb_name=f_kb.strip() or "default",
                    page_urls=page_urls,
                    max_depth=int(f_max_depth),
                    extra_tags=extra_tags,
                    refresh_cron=f_cron.strip() or None,
                )
                st.success(
                    f"Saved Confluence source for {f_usecase.strip()} / {f_agent.strip()}."
                )
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"Could not save source: {exc}")


# =============================================================================
# Tab 3 — Bulk Import
# =============================================================================

_MANIFEST_EXAMPLE = [
    {
        "name": "SSOP Operations",
        "usecase_id": "GENAI1597_SSOP",
        "agent_filter": "ssop_cloud_operations_agent",
        "kb_name": "default",
        "page_urls": [
            "https://mycompany.atlassian.net/wiki/spaces/OPS/pages/12345678/Operations"
        ],
        "max_depth": -1,
        "tags": ["internal", "ssop", "operations"],
        "refresh_cron": "0 3 * * 1",
    }
]

with tab_bulk:
    st.subheader("Bulk Import from Manifest")
    st.caption(
        "Upload a JSON manifest file to register multiple Confluence sources at once "
        "and optionally crawl them all immediately."
    )

    with st.expander("Manifest format (click to see example)"):
        st.markdown(
            "A JSON array where each entry defines one Confluence source. "
            "All fields except `usecase_id`, `agent_filter`, and `page_urls` are optional."
        )
        st.code(json.dumps(_MANIFEST_EXAMPLE, indent=2), language="json")
        st.download_button(
            "Download example manifest",
            data=json.dumps(_MANIFEST_EXAMPLE, indent=2).encode(),
            file_name="confluence_manifest_example.json",
            mime="application/json",
            key="dl_example",
        )

    manifest_file = st.file_uploader(
        "Upload manifest JSON",
        type=["json"],
        key="manifest_upload",
    )

    if manifest_file:
        try:
            raw_bytes = manifest_file.read()
            entries = json.loads(raw_bytes)
            if not isinstance(entries, list):
                st.error("Manifest must be a JSON array (list of source entries).")
                st.stop()
        except json.JSONDecodeError as exc:
            st.error(f"Could not parse JSON: {exc}")
            st.stop()

        # ── Validate ──────────────────────────────────────────────────────────
        validation_errors: list[str] = []
        for i, e in enumerate(entries, 1):
            label = e.get("name") or e.get("usecase_id") or f"Entry {i}"
            if not e.get("usecase_id"):
                validation_errors.append(f"{label}: missing `usecase_id`")
            if not e.get("agent_filter"):
                validation_errors.append(f"{label}: missing `agent_filter`")
            if not e.get("page_urls"):
                validation_errors.append(f"{label}: missing `page_urls`")
            cron = e.get("refresh_cron")
            if cron:
                try:
                    from croniter import croniter
                    if not croniter.is_valid(cron):
                        validation_errors.append(f"{label}: invalid cron expression {cron!r}")
                except ImportError:
                    pass

        if validation_errors:
            for err in validation_errors:
                st.error(err)
        else:
            # ── Preview table ─────────────────────────────────────────────────
            import pandas as pd

            rows = []
            for e in entries:
                rows.append({
                    "Name":          e.get("name", ""),
                    "Use case ID":   e.get("usecase_id", ""),
                    "Agent filter":  e.get("agent_filter", ""),
                    "KB":            e.get("kb_name", "default"),
                    "Root URLs":     len(e.get("page_urls") or []),
                    "Depth":         e.get("max_depth", -1),
                    "Tags":          ", ".join(e.get("tags") or []),
                    "Cron":          e.get("refresh_cron") or "—",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"{len(entries)} source(s) in manifest.")

            # ── Action buttons ────────────────────────────────────────────────
            bc1, bc2 = st.columns(2)

            with bc1:
                if st.button(
                    "📋 Register all sources",
                    key="bulk_register",
                    use_container_width=True,
                    help="Save all sources to MongoDB without crawling. You can crawl them later.",
                ):
                    from pipeline.mongo_store import get_usecase_ledger
                    uc_ledger = get_usecase_ledger()
                    ok = failed = 0
                    for e in entries:
                        label = e.get("name") or e["usecase_id"]
                        try:
                            uc_ledger.upsert_confluence_source(
                                usecase_id=e["usecase_id"],
                                agent_filter=e["agent_filter"],
                                kb_name=e.get("kb_name", "default"),
                                page_urls=e["page_urls"],
                                max_depth=e.get("max_depth", -1),
                                extra_tags=e.get("tags") or [],
                                refresh_cron=e.get("refresh_cron") or None,
                            )
                            ok += 1
                        except Exception as exc:
                            st.error(f"**{label}**: {exc}")
                            failed += 1
                    if ok:
                        st.success(
                            f"Registered {ok} source(s)."
                            + (f" {failed} failed." if failed else "")
                        )
                    st.cache_data.clear()

            with bc2:
                if st.button(
                    "🚀 Register & Crawl all",
                    key="bulk_crawl",
                    type="primary",
                    use_container_width=True,
                    help=(
                        "Register all sources, then immediately crawl and stage "
                        "each one. May take several minutes for large page trees."
                    ),
                ):
                    from pipeline.mongo_store import get_usecase_ledger
                    from pipeline.refresh_scheduler import trigger_refresh_now
                    uc_ledger = get_usecase_ledger()

                    progress_bar = st.progress(0.0)
                    status_msg   = st.empty()
                    ok = failed  = 0

                    for i, e in enumerate(entries):
                        label = e.get("name") or e["usecase_id"]
                        status_msg.caption(
                            f"Processing **{label}** ({i + 1} / {len(entries)})…"
                        )
                        try:
                            uc_ledger.upsert_confluence_source(
                                usecase_id=e["usecase_id"],
                                agent_filter=e["agent_filter"],
                                kb_name=e.get("kb_name", "default"),
                                page_urls=e["page_urls"],
                                max_depth=e.get("max_depth", -1),
                                extra_tags=e.get("tags") or [],
                                refresh_cron=e.get("refresh_cron") or None,
                            )
                            trigger_refresh_now(e["usecase_id"], e["agent_filter"])
                            ok += 1
                        except Exception as exc:
                            st.error(f"**{label}**: {exc}")
                            failed += 1
                        progress_bar.progress((i + 1) / len(entries))

                    status_msg.empty()
                    if ok:
                        st.success(
                            f"Done — {ok} source(s) crawled and staged."
                            + (f" {failed} failed." if failed else "")
                            + " Go to **Review Queue** to approve and push."
                        )
                    st.cache_data.clear()


# =============================================================================
# Tab 4 — Export JSONL
# =============================================================================

with tab_export:
    st.subheader("Export chunks as JSONL")
    st.caption(
        "Download all chunks for a use case and agent filter as a pipeline-schema "
        "JSONL file. Use this to send content to an external embedding pipeline."
    )

    try:
        export_usecases = _load_usecases()
    except Exception as exc:
        st.error(f"Could not load use cases: {exc}")
        export_usecases = []

    if not export_usecases:
        st.info("No use case ledger entries found. Ingest and push documents first.")
    else:
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            ex_usecase = st.selectbox("Use case ID", export_usecases, key="export_uc")
        with ex_col2:
            ex_agent_opts = _load_agent_filters(ex_usecase) if ex_usecase else []
            ex_agent = st.selectbox(
                "Agent filter",
                ex_agent_opts or ["—"],
                key="export_af",
            )

        ex_status = st.radio(
            "Include documents with status",
            ["pushed only", "all approved"],
            horizontal=True,
            key="export_status",
        )
        status_val = "pushed" if ex_status == "pushed only" else None

        if ex_usecase and ex_agent and ex_agent != "—":
            if st.button("Generate JSONL export", key="export_btn"):
                with st.spinner("Fetching chunks …"):
                    try:
                        from pipeline.mongo_store import get_staging
                        staging = get_staging()
                        chunk_dicts = staging.get_chunks_by_usecase(
                            ex_usecase, ex_agent, status=status_val
                        )

                        if not chunk_dicts:
                            st.warning(
                                f"No chunks found for usecase_id={ex_usecase!r}, "
                                f"agent_filter={ex_agent!r}, status={status_val!r}."
                            )
                        else:
                            lines = [
                                json.dumps(c, ensure_ascii=False, default=str)
                                for c in chunk_dicts
                            ]
                            jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")

                            st.success(f"Ready: {len(chunk_dicts):,} chunks.")
                            st.download_button(
                                label=f"Download {len(chunk_dicts):,} chunks as JSONL",
                                data=jsonl_bytes,
                                file_name=f"export_{ex_usecase}_{ex_agent}.jsonl",
                                mime="application/x-ndjson",
                                key="export_download",
                            )
                    except Exception as exc:
                        st.error(f"Export failed: {exc}")
