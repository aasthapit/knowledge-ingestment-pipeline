"""Knowledge Base Health — drift detection and stale document management."""
from __future__ import annotations

import streamlit as st

st.title("🔍 Knowledge Base Health")
st.caption(
    "Track which documents in your knowledge base are current, stale, or deleted. "
    "Run a drift check to compare source files against what was last pushed."
)

# ── Load ledger ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_stats(kb_name: str | None) -> dict:
    from pipeline.mongo_store import get_ledger
    return get_ledger().get_stats(kb_name=kb_name or None)


@st.cache_data(ttl=30)
def _load_kb_names() -> list[str]:
    from pipeline.mongo_store import get_ledger
    return ["All KBs"] + get_ledger().get_kb_names()


@st.cache_data(ttl=30)
def _load_docs(kb_name: str | None, drift_status: str | None) -> list[dict]:
    from pipeline.mongo_store import get_ledger
    return get_ledger().list_docs(
        kb_name=kb_name or None,
        drift_status=drift_status or None,
        limit=500,
    )


# ── KB selector ───────────────────────────────────────────────────────────────

try:
    kb_options = _load_kb_names()
except Exception as exc:
    st.error(f"Could not connect to MongoDB: {exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URL` is set in your `.env` file.")
    st.stop()

selected_kb_label = st.selectbox("Knowledge base", kb_options, index=0)
selected_kb = None if selected_kb_label == "All KBs" else selected_kb_label

# ── Summary metrics ───────────────────────────────────────────────────────────

stats = _load_stats(selected_kb)
drift = stats.get("drift_counts", {})

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total documents", f"{stats['total_docs']:,}")
m2.metric("Total chunks",    f"{stats['total_chunks']:,}")
m3.metric("✅ Current",      drift.get("current", 0))
m4.metric("⚠️ Stale",        drift.get("stale",   0),   delta_color="inverse")
m5.metric("🗑️ Deleted",      drift.get("deleted", 0),   delta_color="inverse")

if stats["last_push"]:
    st.caption(f"Last push: {stats['last_push'][:19].replace('T', ' ')} UTC")

st.divider()

# ── Drift check button ────────────────────────────────────────────────────────

col_run, col_info = st.columns([1, 3])

with col_run:
    run_check = st.button(
        "🔄  Check for changes",
        type="primary",
        width="stretch",
        help="Compares source file modification times and sizes against the last recorded push.",
    )

with col_info:
    st.info(
        "Drift detection works by comparing the file's modification time and size "
        "to what was recorded when it was last pushed.  "
        "URL-sourced documents show **Unknown** — re-fetch them to detect changes.",
        icon="ℹ️",
    )

if run_check:
    progress = st.progress(0.0, text="Checking…")

    def _cb(done: int, total: int) -> None:
        if total and total > 0:
            progress.progress(done / total, text=f"Checked {done:,} of {total:,}…")

    try:
        from pipeline.mongo_store import get_ledger
        tally = get_ledger().run_drift_check(kb_name=selected_kb, progress_cb=_cb)
        progress.progress(1.0, text="Done!")
        _load_stats.clear()
        _load_docs.clear()
        st.success(
            f"Drift check complete — "
            f"**{tally['current']}** current, "
            f"**{tally['stale']}** stale, "
            f"**{tally['deleted']}** deleted, "
            f"**{tally['unknown']}** unknown."
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Drift check failed: {exc}")

st.divider()

# ── Document table ────────────────────────────────────────────────────────────

DRIFT_ICONS = {
    "current": "✅",
    "stale":   "⚠️",
    "deleted": "🗑️",
    "unknown": "❓",
}

filter_options = ["All", "✅ Current", "⚠️ Stale", "🗑️ Deleted", "❓ Unknown"]
filter_label   = st.segmented_control(
    "Filter by drift status",
    filter_options,
    default="All",
    label_visibility="collapsed",
)

status_map = {
    "✅ Current": "current",
    "⚠️ Stale":   "stale",
    "🗑️ Deleted": "deleted",
    "❓ Unknown": "unknown",
}
drift_filter = status_map.get(filter_label)

try:
    docs = _load_docs(selected_kb, drift_filter)
except Exception as exc:
    st.error(f"Could not load documents: {exc}")
    docs = []

if not docs:
    st.info("No documents found. Push some documents to the knowledge base first.")
else:
    st.caption(f"Showing {len(docs):,} document{'s' if len(docs) != 1 else ''}")

    for doc in docs:
        drift_status = doc.get("drift_status", "unknown")
        icon = DRIFT_ICONS.get(drift_status, "❓")
        title = doc.get("title") or doc.get("source_path") or doc.get("doc_id", "Untitled")
        kb    = doc.get("kb_name", "default")
        pushed_at = (doc.get("pushed_at") or "")[:19].replace("T", " ")
        chunk_count = doc.get("chunk_count", 0)
        source_type = doc.get("source_type", "")
        quality_score = float(doc.get("quality_score", 0))

        with st.container(border=True):
            header_col, action_col = st.columns([5, 1])

            with header_col:
                st.markdown(f"**{icon} {title}**")
                meta_parts = [f"`{kb}`", f"{chunk_count} chunks", source_type or ""]
                if pushed_at:
                    meta_parts.append(f"pushed {pushed_at} UTC")
                st.caption("  ·  ".join(p for p in meta_parts if p))

                # Source path / URL
                source_path = doc.get("source_path", "")
                url = doc.get("url", "")
                if url:
                    st.caption(f"🔗 {url}")
                elif source_path and source_path != doc.get("title", ""):
                    st.caption(f"📁 `{source_path}`")

                # Drift note
                checked_at = (doc.get("drift_checked_at") or "")[:19].replace("T", " ")
                if drift_status == "stale":
                    st.warning(
                        "Source file has changed since last push. "
                        "Re-ingest to update the knowledge base.",
                        icon="⚠️",
                    )
                elif drift_status == "deleted":
                    st.error(
                        "Source file no longer exists. "
                        "You may want to remove this document from the knowledge base.",
                        icon="🗑️",
                    )
                if checked_at:
                    st.caption(f"Last checked: {checked_at} UTC")

            with action_col:
                doc_id = doc.get("doc_id", "")

                if drift_status == "stale" and source_path and source_type not in ("url", "jsonl"):
                    if st.button(
                        "Re-ingest",
                        key=f"reingest_{doc_id}",
                        type="primary",
                        width="stretch",
                        help="Re-process the source file and update the knowledge base.",
                    ):
                        try:
                            from pipeline.ingest import ingest_document
                            from pipeline.review import push_approved
                            tags = doc.get("tags") or []
                            new_result = ingest_document(
                                source=source_path,
                                extra_tags=tags,
                                auto_push=True,
                                kb_name=kb,
                            )
                            st.success(f"Re-ingested: {new_result['chunk_count']} chunks pushed.")
                            _load_stats.clear()
                            _load_docs.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Re-ingest failed: {exc}")

                if drift_status == "deleted":
                    if st.button(
                        "Remove",
                        key=f"remove_{doc_id}",
                        type="secondary",
                        width="stretch",
                        help="Remove this document record from the ledger.",
                    ):
                        try:
                            from pipeline.mongo_store import get_ledger
                            get_ledger().delete_doc(doc_id)
                            st.success("Record removed from ledger.")
                            _load_stats.clear()
                            _load_docs.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Remove failed: {exc}")

                # Single drift check for this document
                if st.button(
                    "Check",
                    key=f"check_{doc_id}",
                    width="stretch",
                    help="Check drift status for this document only.",
                ):
                    try:
                        from pipeline.mongo_store import get_ledger
                        new_status = get_ledger().check_drift_one(doc_id)
                        _load_stats.clear()
                        _load_docs.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Check failed: {exc}")
