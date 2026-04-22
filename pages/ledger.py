"""Knowledge Base Ledger — audit trail of every document pushed to the KB."""
from __future__ import annotations

import csv
import io

import streamlit as st

st.title("📒 Knowledge Base Ledger")
st.caption(
    "Audit trail of every document pushed to the knowledge base — "
    "titles, sources, versions, quality scores, and drift status."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_snapshots(limit: int = 50) -> list[dict]:
    from pipeline.mongo_store import get_ledger
    return get_ledger().list_snapshots(limit=limit)


@st.cache_data(ttl=30)
def _load_snapshot(snapshot_id: str) -> dict | None:
    from pipeline.mongo_store import get_ledger
    return get_ledger().get_snapshot(snapshot_id)


@st.cache_data(ttl=30)
def _load_kb_names() -> list[str]:
    from pipeline.mongo_store import get_ledger
    return ["All KBs"] + get_ledger().get_kb_names()


@st.cache_data(ttl=30)
def _load_stats(kb_name: str | None) -> dict:
    from pipeline.mongo_store import get_ledger
    return get_ledger().get_stats(kb_name=kb_name or None)


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
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

selected_kb_label = st.selectbox("Knowledge base", kb_options, index=0)
selected_kb = None if selected_kb_label == "All KBs" else selected_kb_label

# ── Summary metrics ───────────────────────────────────────────────────────────

stats = _load_stats(selected_kb)

m1, m2, m3 = st.columns(3)
m1.metric("Total documents", f"{stats['total_docs']:,}")
m2.metric("Total chunks",    f"{stats['total_chunks']:,}")
if stats.get("last_push"):
    m3.metric("Last push", stats["last_push"][:10])
else:
    m3.metric("Last push", "—")

st.divider()

# ── Drift status filter ───────────────────────────────────────────────────────

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

# ── Load documents ────────────────────────────────────────────────────────────

try:
    docs = _load_docs(selected_kb, drift_filter)
except Exception as exc:
    st.error(f"Could not load ledger: {exc}")
    docs = []

if not docs:
    st.info("No documents found. Push some documents to the knowledge base first.")
    st.stop()

st.caption(f"Showing {len(docs):,} document{'s' if len(docs) != 1 else ''}")

# ── Download CSV ──────────────────────────────────────────────────────────────

_CSV_COLS = [
    "title", "source_path", "source_type", "kb_name",
    "chunk_count", "quality_score", "tags", "pushed_at", "drift_status",
]

def _build_csv(records: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLS, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        row = {k: rec.get(k, "") for k in _CSV_COLS}
        if isinstance(row["tags"], list):
            row["tags"] = "; ".join(row["tags"])
        pushed_at = row.get("pushed_at", "")
        if isinstance(pushed_at, str) and len(pushed_at) > 19:
            row["pushed_at"] = pushed_at[:19]
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")

from datetime import date
st.download_button(
    "⬇️ Download CSV",
    data=_build_csv(docs),
    file_name=f"ledger_{date.today()}.csv",
    mime="text/csv",
)

st.divider()

# ── Document table ────────────────────────────────────────────────────────────

DRIFT_ICONS = {
    "current": "✅",
    "stale":   "⚠️",
    "deleted": "🗑️",
    "unknown": "❓",
}

# Build a flat list of rows for the DataFrame
import pandas as pd

rows = []
for doc in docs:
    pushed_at = (doc.get("pushed_at") or "")[:19].replace("T", " ")
    ingested_at = (doc.get("ingested_at") or "")[:10]
    tags = doc.get("tags") or []
    drift = doc.get("drift_status", "unknown")
    rows.append({
        "Status":        DRIFT_ICONS.get(drift, "❓") + " " + drift,
        "Title":         doc.get("title") or doc.get("source_path") or "Untitled",
        "Source type":   doc.get("source_type", ""),
        "KB":            doc.get("kb_name", "default"),
        "Chunks":        doc.get("chunk_count", 0),
        "Quality":       round(float(doc.get("quality_score", 0)), 3),
        "Tags":          "; ".join(tags) if tags else "",
        "Pushed at":     pushed_at,
        "Ingested":      ingested_at,
    })

df = pd.DataFrame(rows)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Quality": st.column_config.ProgressColumn(
            "Quality",
            min_value=0.0,
            max_value=1.0,
            format="%.2f",
        ),
        "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
    },
)

st.divider()

# ── Push history / snapshots ──────────────────────────────────────────────────

st.subheader("Push history")
st.caption("Each row is a point-in-time snapshot recorded when approved docs were pushed.")

try:
    snapshots = _load_snapshots(50)
except Exception as exc:
    st.error(f"Could not load snapshots: {exc}")
    snapshots = []

if not snapshots:
    st.info("No push snapshots yet. Snapshots are recorded automatically on each push.")
else:
    snap_rows = []
    for s in snapshots:
        created_at = (s.get("created_at") or "")[:19].replace("T", " ")
        snap_rows.append({
            "Snapshot ID":  s.get("snapshot_id", "")[:8] + "…",
            "Pushed at":    created_at + " UTC",
            "Docs pushed":  s.get("pushed_doc_count", 0),
            "Total docs":   s.get("total_docs", 0),
            "Total chunks": s.get("total_chunks", 0),
            "_id":          s.get("snapshot_id", ""),
        })

    snap_df = pd.DataFrame(snap_rows)
    selected = st.dataframe(
        snap_df.drop(columns=["_id"]),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    sel_rows = selected.selection.get("rows", []) if hasattr(selected, "selection") else []
    if sel_rows:
        chosen_id = snap_rows[sel_rows[0]]["_id"]
        snap_detail = _load_snapshot(chosen_id)
        if snap_detail:
            created_at = (snap_detail.get("created_at") or "")[:19].replace("T", " ")
            st.markdown(f"**Snapshot {chosen_id[:8]}…** — {created_at} UTC")
            st.caption(
                f"{snap_detail.get('pushed_doc_count', 0)} docs pushed · "
                f"{snap_detail.get('total_docs', 0)} total docs · "
                f"{snap_detail.get('total_chunks', 0)} total chunks"
            )

            snap_doc_rows = []
            for d in snap_detail.get("docs", []):
                tags = d.get("tags") or []
                snap_doc_rows.append({
                    "Title":       d.get("title") or d.get("source_path") or "Untitled",
                    "Source type": d.get("source_type", ""),
                    "KB":          d.get("kb_name", "default"),
                    "Chunks":      d.get("chunk_count", 0),
                    "Quality":     round(float(d.get("quality_score", 0) or 0), 3),
                    "Tags":        "; ".join(tags) if tags else "",
                    "Pushed at":   (d.get("pushed_at") or "")[:19].replace("T", " "),
                })

            snap_doc_df = pd.DataFrame(snap_doc_rows)

            # CSV download for this snapshot
            buf = io.StringIO()
            snap_doc_df.to_csv(buf, index=False)
            st.download_button(
                f"⬇️ Download snapshot CSV",
                data=buf.getvalue().encode("utf-8"),
                file_name=f"snapshot_{chosen_id[:8]}.csv",
                mime="text/csv",
            )

            st.dataframe(
                snap_doc_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Quality": st.column_config.ProgressColumn(
                        "Quality", min_value=0.0, max_value=1.0, format="%.2f",
                    ),
                    "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
                },
            )
