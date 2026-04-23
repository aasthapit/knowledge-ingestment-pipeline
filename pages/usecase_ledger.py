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


# ── Main tabs ─────────────────────────────────────────────────────────────────

try:
    all_entries = _load_entries()
except Exception as exc:
    st.error(f"Could not connect to MongoDB: {exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

tab_ledger, tab_sources, tab_export = st.tabs([
    "Ledger",
    "Confluence Sources",
    "Export JSONL",
])


# =============================================================================
# Tab 1 — Ledger
# =============================================================================

with tab_ledger:
    st.subheader("Use Case Inventory")

    if not all_entries:
        st.info(
            "No use case ledger entries yet. "
            "Ingest JSONL files with `usecase_id` and `agent_filter` set, "
            "then push them to the vector DB."
        )
    else:
        import pandas as pd

        rows = []
        for e in all_entries:
            last_pushed = (e.get("last_pushed_at") or "")[:19].replace("T", " ")
            rows.append({
                "Use case ID":   e.get("usecase_id", ""),
                "Agent filter":  e.get("agent_filter", ""),
                "KB":            e.get("kb_name", "default"),
                "Chunks":        e.get("chunk_count", 0),
                "Last pushed":   last_pushed,
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
            },
        )

    st.divider()
    st.subheader("Document Details")
    st.caption("Select a use case and agent filter to see the documents in the knowledge base.")

    try:
        usecase_options = _load_usecases()
    except Exception as exc:
        st.error(f"Could not load use cases: {exc}")
        usecase_options = []

    if not usecase_options:
        st.info("No use cases found.")
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
            try:
                kb_docs = _load_kb_docs(sel_usecase, sel_agent)
            except Exception as exc:
                st.error(f"Could not load documents: {exc}")
                kb_docs = []

            if not kb_docs:
                st.info("No documents pushed for this use case / agent filter yet.")
            else:
                st.caption(f"{len(kb_docs):,} document(s)")

                DRIFT_ICONS = {
                    "current": "✅",
                    "stale":   "⚠️",
                    "deleted": "🗑️",
                    "unknown": "❓",
                }
                doc_rows = []
                for doc in kb_docs:
                    pushed = (doc.get("pushed_at") or "")[:19].replace("T", " ")
                    drift  = doc.get("drift_status", "unknown")
                    tags   = doc.get("tags") or []
                    doc_rows.append({
                        "Status":      DRIFT_ICONS.get(drift, "❓") + " " + drift,
                        "Title":       doc.get("title") or doc.get("source_path") or "Untitled",
                        "Source type": doc.get("source_type", ""),
                        "Chunks":      doc.get("chunk_count", 0),
                        "Quality":     round(float(doc.get("quality_score") or 0), 3),
                        "Tags":        "; ".join(tags) if tags else "",
                        "Pushed at":   pushed,
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


# =============================================================================
# Tab 2 — Confluence Sources
# =============================================================================

with tab_sources:
    st.subheader("Registered Confluence Sources")
    st.caption(
        "Register Confluence page URLs per use case and agent filter. "
        "The scheduler will re-crawl and re-push them on the configured cron schedule."
    )

    # ── Registered sources table ──────────────────────────────────────────────

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
                "Use case ID":   s.get("usecase_id", ""),
                "Agent filter":  s.get("agent_filter", ""),
                "KB":            s.get("kb_name", "default"),
                "Pages":         len(s.get("page_urls") or []),
                "Cron":          s.get("refresh_cron") or "—",
                "Status":        STATUS_ICONS.get(status, "?") + " " + status,
                "Last refresh":  last_r,
                "Next refresh":  next_r,
            })

        st.dataframe(
            pd.DataFrame(src_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Per-row manual trigger
        st.caption("Trigger an immediate refresh for a registered source:")
        trig_col1, trig_col2, trig_col3 = st.columns([2, 2, 1])
        trig_options = [f"{s['usecase_id']} / {s['agent_filter']}" for s in sources]
        with trig_col1:
            trig_sel = st.selectbox("Source to refresh", trig_options, key="trig_sel", label_visibility="collapsed")
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

        if any(s.get("refresh_error") for s in sources):
            with st.expander("Refresh errors"):
                for s in sources:
                    if s.get("refresh_error"):
                        st.code(
                            f"{s['usecase_id']} / {s['agent_filter']}:\n{s['refresh_error']}",
                            language=None,
                        )
    else:
        st.info("No Confluence sources registered yet.")

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
            placeholder="https://mycompany.atlassian.net/wiki/spaces/SPACE/pages/12345678/Page-Title\nhttps://…",
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
                pass  # croniter not installed yet — skip validation

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
# Tab 3 — Export JSONL
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
                            # Build JSONL bytes in memory
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
