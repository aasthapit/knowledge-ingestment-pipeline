"""Knowledge Bases — named document sources (Confluence, JSONL, or web crawler)."""
from __future__ import annotations

import streamlit as st

st.title("📂 Knowledge Bases")
st.caption(
    "A Knowledge Base is a named source of documents — a Confluence page tree, a JSONL file upload, "
    "or the output of an external web crawler. KBs are shared across corpora; one KB can feed many corpora."
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


STATUS_COLOR = {"empty": "gray", "staging": "orange", "ready": "green"}
SOURCE_ICON  = {"confluence": "🔗", "jsonl": "📦", "web": "🌐"}

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
        if st.button("＋ New", use_container_width=True):
            st.session_state.kb_create_open = not st.session_state.kb_create_open

    # ── Create form ───────────────────────────────────────────────────────────
    if st.session_state.kb_create_open:
        with st.form("create_kb_form", border=True):
            st.markdown("**Create Knowledge Base**")
            new_name = st.text_input("Name *", placeholder="e.g. openshift-docs")
            new_desc = st.text_area("Description", height=60)
            new_type = st.radio(
                "Source type",
                ["confluence", "jsonl", "web"],
                horizontal=True,
                key="create_kb_type",
            )

            if new_type == "confluence":
                new_urls = st.text_area(
                    "Confluence page URLs (one per line) *",
                    placeholder="https://mycompany.atlassian.net/wiki/spaces/OPS/pages/123",
                    height=80,
                )
                new_depth = st.number_input("Max depth (-1 = all)", min_value=-1, value=-1)
                new_cron  = st.text_input(
                    "Refresh schedule (cron expression, optional)",
                    placeholder="0 2 * * 1  (every Monday at 2 AM)",
                )
            elif new_type == "web":
                new_urls = st.text_area(
                    "Seed URLs  *(optional, for reference)*",
                    placeholder="https://docs.example.com/",
                    height=60,
                    help="These are stored for reference only — run your external web crawler and import the resulting JSONL via Add Document.",
                )
                new_depth = -1
                new_cron  = ""
                st.info(
                    "🌐 **Web KB** — run your own web crawler against these URLs, "
                    "export the results as JSONL, then import it on the **Add Document** page "
                    "and select this KB. Supported URL field names: `page_url`, `sourceURL`, `source_url`, `url`."
                )
            else:
                new_urls  = ""
                new_depth = -1
                new_cron  = ""

            submitted = st.form_submit_button("Create", type="primary")
            if submitted:
                if not new_name.strip():
                    st.error("Name is required.")
                elif new_type == "confluence" and not new_urls.strip():
                    st.error("At least one Confluence URL is required.")
                else:
                    try:
                        from pipeline.mongo_store import get_kb_store
                        urls = [u.strip() for u in new_urls.splitlines() if u.strip()]
                        kb_id = get_kb_store().create(
                            name=new_name.strip(),
                            source_type=new_type,
                            description=new_desc.strip(),
                            confluence_urls=urls,
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
        ["All", "Confluence", "JSONL", "Web"],
        default="All",
        label_visibility="collapsed",
    )
    filter_map = {"All": None, "Confluence": "confluence", "JSONL": "jsonl", "Web": "web"}
    try:
        filtered_kbs = _load_kbs(source_type=filter_map.get(type_filter or "All"))
    except Exception:
        filtered_kbs = all_kbs

    # ── KB list ───────────────────────────────────────────────────────────────
    if not filtered_kbs:
        st.info("No Knowledge Bases yet. Click **＋ New** to create one.")
    else:
        for kb in filtered_kbs:
            kid  = kb["kb_id"]
            icon = SOURCE_ICON.get(kb.get("source_type", ""), "📂")
            status = kb.get("status", "empty")
            color  = STATUS_COLOR.get(status, "gray")
            label  = f"{icon} **{kb['name']}**  \n:{color}[{status}] · {len(kb.get('doc_ids') or [])} docs"
            btn_type = "primary" if st.session_state.kb_selected == kid else "secondary"
            if st.button(label, key=f"sel_kb_{kid}", use_container_width=True, type=btn_type):
                st.session_state.kb_selected = kid
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
            # ── Header ────────────────────────────────────────────────────────
            hd1, hd2 = st.columns([3, 1])
            with hd1:
                icon = SOURCE_ICON.get(kb.get("source_type", ""), "📂")
                st.subheader(f"{icon} {kb['name']}")
                if kb.get("description"):
                    st.caption(kb["description"])
            with hd2:
                if st.button("Delete", use_container_width=True, type="secondary"):
                    st.session_state[f"confirm_del_kb_{sel_id}"] = True

            if st.session_state.get(f"confirm_del_kb_{sel_id}"):
                st.warning(
                    f"Delete **{kb['name']}**? This removes the KB record but does not delete "
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

            # ── Stats chips ───────────────────────────────────────────────────
            status     = kb.get("status", "empty")
            color      = STATUS_COLOR.get(status, "gray")
            source_type = kb.get("source_type", "")
            doc_count  = len(kb.get("doc_ids") or [])

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Status", status)
            s2.metric("Source type", source_type)
            s3.metric("Staged docs", doc_count)
            s4.metric("Last updated", _fmt_date(kb.get("last_updated")))

            st.divider()

            # ── Confluence details ─────────────────────────────────────────────
            if source_type == "confluence":
                st.markdown("**Confluence URLs**")
                for url in kb.get("confluence_urls") or []:
                    st.code(url, language=None)

                col_depth, col_cron = st.columns(2)
                col_depth.metric("Max depth", kb.get("max_depth", -1))
                col_cron.metric(
                    "Refresh schedule",
                    kb.get("refresh_cron") or "—",
                )

                if kb.get("last_ingested_at"):
                    st.caption(f"Last ingested: {_fmt_date(kb.get('last_ingested_at'))}")

                st.divider()
                st.markdown(
                    "To crawl this KB, go to the **Confluence** page and select it as the target."
                )

            # ── Web crawler details ───────────────────────────────────────────
            elif source_type == "web":
                if kb.get("confluence_urls"):  # stored as seed URLs for reference
                    st.markdown("**Seed URLs**")
                    for url in kb.get("confluence_urls") or []:
                        st.code(url, language=None)
                st.info(
                    "🌐 Run your external web crawler against the seed URLs above and import "
                    "the resulting JSONL on the **Add Document** page. "
                    "Supported URL fields: `page_url`, `sourceURL`, `source_url`, `url`."
                )

            # ── JSONL details ─────────────────────────────────────────────────
            elif source_type == "jsonl":
                if kb.get("file_name"):
                    st.metric("File", kb["file_name"])
                st.markdown(
                    "To add more content, go to **Add Document → Bulk JSONL Import** and select this KB."
                )

            # ── Staging docs tab ──────────────────────────────────────────────
            st.subheader("Staged documents")
            try:
                from pipeline.mongo_store import get_staging
                staged = get_staging().list_all(kb_id=sel_id)
                if not staged:
                    st.info("No staged documents for this Knowledge Base yet.")
                else:
                    import pandas as pd
                    rows = [
                        {
                            "Title":   d.get("title", "—"),
                            "Status":  d.get("status", ""),
                            "Chunks":  d.get("chunk_count", 0),
                            "Score":   f"{float(d.get('quality_score', 0)):.0%}",
                            "Staged":  _fmt_date(d.get("staged_at")),
                            "doc_id":  d.get("doc_id", ""),
                        }
                        for d in staged
                    ]
                    df = pd.DataFrame(rows)
                    st.dataframe(df.drop(columns=["doc_id"]), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Could not load staged docs: {exc}")
