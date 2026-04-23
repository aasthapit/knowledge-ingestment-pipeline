"""Corpus Management — named document collections scoped to a use case."""
from __future__ import annotations

import streamlit as st

st.title("📦 Corpus Management")
st.caption(
    "Named document collections scoped to a use case and agent filter. "
    "Create corpora, inspect which documents belong to each, and remove documents in bulk."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_corpora() -> list[dict]:
    from pipeline.mongo_store import get_corpus_store
    return get_corpus_store().list_all()


@st.cache_data(ttl=30)
def _load_corpus(corpus_id: str) -> dict | None:
    from pipeline.mongo_store import get_corpus_store
    return get_corpus_store().get(corpus_id)


@st.cache_data(ttl=30)
def _load_changelog(corpus_id: str, limit: int = 100) -> list[dict]:
    from pipeline.mongo_store import get_corpus_store
    return get_corpus_store().get_changelog(corpus_id, limit=limit)


def _invalidate(corpus_id: str | None = None) -> None:
    _load_corpora.clear()
    if corpus_id:
        _load_corpus.clear()
        _load_changelog.clear()


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16] if iso else "—"


# ── Connection guard ──────────────────────────────────────────────────────────

try:
    all_corpora = _load_corpora()
except Exception as _exc:
    st.error(f"Could not connect to MongoDB: {_exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

# ── Session state defaults ────────────────────────────────────────────────────

if "corpus_selected" not in st.session_state:
    st.session_state.corpus_selected = None
if "corpus_create_open" not in st.session_state:
    st.session_state.corpus_create_open = False
if "corpus_edit_open" not in st.session_state:
    st.session_state.corpus_edit_open = False

# ── Layout ────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 3], gap="large")

# =============================================================================
# Left — corpus list + create button
# =============================================================================

with left:
    col_hd, col_btn = st.columns([2, 1])
    with col_hd:
        st.subheader("Corpora")
    with col_btn:
        if st.button("＋ New", use_container_width=True):
            st.session_state.corpus_create_open = not st.session_state.corpus_create_open
            st.session_state.corpus_edit_open = False

    # ── Create form ───────────────────────────────────────────────────────────
    if st.session_state.corpus_create_open:
        with st.form("create_corpus_form", border=True):
            st.markdown("**Create Corpus**")
            new_name = st.text_input("Name *", placeholder="e.g. support-kb-v2")
            new_desc = st.text_area("Description", height=68)
            c1, c2 = st.columns(2)
            with c1:
                new_kb = st.text_input("KB Names (comma-separated)", value="default")
            with c2:
                new_uc = st.text_input("Use Case ID", placeholder="support")
            new_af = st.text_input("Agent Filter", placeholder="support-agent-v1")
            submitted = st.form_submit_button("Create", type="primary")
            if submitted:
                if not new_name.strip():
                    st.error("Name is required.")
                else:
                    try:
                        from pipeline.mongo_store import get_corpus_store
                        kb_list = [k.strip() for k in new_kb.split(",") if k.strip()]
                        cid = get_corpus_store().create(
                            name=new_name.strip(),
                            description=new_desc.strip(),
                            kb_names=kb_list or ["default"],
                            usecase_id=new_uc.strip(),
                            agent_filter=new_af.strip(),
                        )
                        _invalidate()
                        st.session_state.corpus_selected = cid
                        st.session_state.corpus_create_open = False
                        st.success(f"Created corpus **{new_name.strip()}**")
                        st.rerun()
                    except Exception as exc:
                        if "duplicate" in str(exc).lower() or "E11000" in str(exc):
                            st.error(f"Corpus name **{new_name.strip()}** already exists.")
                        else:
                            st.error(str(exc))

    # ── Corpus list ───────────────────────────────────────────────────────────
    if not all_corpora:
        st.info("No corpora yet. Click **＋ New** to create one.")
    else:
        for c in all_corpora:
            cid = c["corpus_id"]
            is_selected = st.session_state.corpus_selected == cid
            label = f"**{c['name']}**  \n{c['doc_count']} docs · {c['chunk_count']} chunks"
            if c.get("usecase_id"):
                label += f"  \nuc: `{c['usecase_id']}`"
            btn_type = "primary" if is_selected else "secondary"
            if st.button(label, key=f"sel_{cid}", use_container_width=True, type=btn_type):
                st.session_state.corpus_selected = cid
                st.session_state.corpus_edit_open = False
                st.rerun()

# =============================================================================
# Right — corpus detail
# =============================================================================

with right:
    sel_id = st.session_state.corpus_selected

    if sel_id is None:
        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:240px;border:2px dashed #ccc;border-radius:8px;"
            "color:#888;font-size:0.9rem'>Select a corpus to view details</div>",
            unsafe_allow_html=True,
        )
    else:
        corpus = _load_corpus(sel_id)
        if corpus is None:
            st.warning("Corpus not found — it may have been deleted.")
            st.session_state.corpus_selected = None
        else:
            # ── Header ────────────────────────────────────────────────────────
            hd1, hd2 = st.columns([3, 1])
            with hd1:
                st.subheader(corpus["name"])
                if corpus.get("description"):
                    st.caption(corpus["description"])
            with hd2:
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("Edit", use_container_width=True):
                        st.session_state.corpus_edit_open = not st.session_state.corpus_edit_open
                with btn2:
                    if st.button("Delete", use_container_width=True, type="secondary"):
                        st.session_state[f"confirm_delete_{sel_id}"] = True

            if st.session_state.get(f"confirm_delete_{sel_id}"):
                st.warning(
                    f"Delete corpus **{corpus['name']}**? "
                    "This removes the corpus record but does **not** delete the documents themselves."
                )
                dc1, dc2, _ = st.columns([1, 1, 4])
                with dc1:
                    if st.button("Yes, delete", type="primary", key="confirm_del_yes"):
                        from pipeline.mongo_store import get_corpus_store
                        get_corpus_store().delete(sel_id)
                        _invalidate()
                        st.session_state.corpus_selected = None
                        st.session_state[f"confirm_delete_{sel_id}"] = False
                        st.rerun()
                with dc2:
                    if st.button("Cancel", key="confirm_del_no"):
                        st.session_state[f"confirm_delete_{sel_id}"] = False
                        st.rerun()

            # ── Edit form ─────────────────────────────────────────────────────
            if st.session_state.corpus_edit_open:
                with st.form("edit_corpus_form", border=True):
                    st.markdown("**Edit Corpus**")
                    e_desc = st.text_area("Description", value=corpus.get("description", ""), height=68)
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        e_kb = st.text_input("KB Names (comma-separated)", value=", ".join(corpus.get("kb_names") or []))
                    with ec2:
                        e_uc = st.text_input("Use Case ID", value=corpus.get("usecase_id", ""))
                    e_af = st.text_input("Agent Filter", value=corpus.get("agent_filter", ""))
                    saved = st.form_submit_button("Save", type="primary")
                    if saved:
                        from pipeline.mongo_store import get_corpus_store
                        kb_list = [k.strip() for k in e_kb.split(",") if k.strip()]
                        get_corpus_store().update(
                            corpus_id=sel_id,
                            description=e_desc.strip(),
                            kb_names=kb_list or ["default"],
                            usecase_id=e_uc.strip(),
                            agent_filter=e_af.strip(),
                        )
                        _invalidate(sel_id)
                        st.session_state.corpus_edit_open = False
                        st.success("Corpus updated.")
                        st.rerun()

            # ── Meta chips ────────────────────────────────────────────────────
            chips: list[str] = []
            for kb in corpus.get("kb_names") or []:
                chips.append(f"`{kb}`")
            if corpus.get("usecase_id"):
                chips.append(f"uc: `{corpus['usecase_id']}`")
            if corpus.get("agent_filter"):
                chips.append(f"agent: `{corpus['agent_filter']}`")
            if chips:
                st.markdown("  ·  ".join(chips))

            # ── Stats ─────────────────────────────────────────────────────────
            s1, s2, s3 = st.columns(3)
            s1.metric("Documents", corpus.get("doc_count", 0))
            s2.metric("Chunks", corpus.get("chunk_count", 0))
            s3.metric("Last updated", _fmt_date(corpus.get("last_updated")))

            st.divider()

            # ── Tabs ──────────────────────────────────────────────────────────
            tab_docs, tab_log = st.tabs([
                f"Documents ({corpus.get('doc_count', 0)})",
                "Changelog",
            ])

            # ── Documents tab ─────────────────────────────────────────────────
            with tab_docs:
                doc_ids: list[str] = corpus.get("doc_ids") or []
                if not doc_ids:
                    st.info(
                        "No documents in this corpus yet. Push documents with this corpus's "
                        "use case ID and agent filter to populate it."
                    )
                else:
                    selected_to_remove = st.multiselect(
                        "Select documents to remove",
                        options=doc_ids,
                        format_func=lambda x: x,
                        placeholder="Choose doc IDs…",
                        key=f"docs_to_remove_{sel_id}",
                    )
                    if selected_to_remove:
                        if st.button(
                            f"🗑️ Remove {len(selected_to_remove)} document(s) from corpus",
                            type="primary",
                            key="do_remove_docs",
                        ):
                            from pipeline.mongo_store import get_corpus_store
                            get_corpus_store().remove_docs(
                                corpus_id=sel_id,
                                doc_ids=selected_to_remove,
                                chunk_ids=[],
                            )
                            _invalidate(sel_id)
                            st.success(f"Removed {len(selected_to_remove)} document(s).")
                            st.rerun()

                    import pandas as pd
                    df = pd.DataFrame({"Document ID": doc_ids})
                    st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Changelog tab ─────────────────────────────────────────────────
            with tab_log:
                changelog = _load_changelog(sel_id)
                if not changelog:
                    st.info("No changes recorded yet.")
                else:
                    import pandas as pd
                    rows = [
                        {
                            "Action": entry.get("action", ""),
                            "Title": entry.get("title") or "—",
                            "Doc ID": entry.get("doc_id", ""),
                            "When": _fmt_date(entry.get("timestamp")),
                        }
                        for entry in changelog
                    ]
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
