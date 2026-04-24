"""Corpus Management — collections of Knowledge Bases for data preparation and export."""
from __future__ import annotations

import re
import time

import streamlit as st

st.title("📦 Corpus Management")
st.caption(
    "A corpus is a named collection of Knowledge Bases. "
    "Group your KBs together and download all their staged content as a single JSONL file."
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
def _load_kbs() -> list[dict]:
    from pipeline.mongo_store import get_kb_store
    return get_kb_store().list_all()


@st.cache_data(ttl=60)
def _load_corpus_docs(corpus_id: str, kb_ids_tuple: tuple) -> list[dict]:
    from pipeline.mongo_store import get_ledger
    return get_ledger().list_docs_by_kb_ids(list(kb_ids_tuple))


def _invalidate(corpus_id: str | None = None) -> None:
    _load_corpora.clear()
    _load_corpus_docs.clear()
    if corpus_id:
        _load_corpus.clear()


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


# ── Connection guard ──────────────────────────────────────────────────────────

try:
    all_corpora = _load_corpora()
except Exception as _exc:
    st.error(f"Could not connect to MongoDB: {_exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────

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
        if st.button("＋ New", width="stretch"):
            st.session_state.corpus_create_open = not st.session_state.corpus_create_open
            st.session_state.corpus_edit_open = False

    # ── Create form ───────────────────────────────────────────────────────────
    if st.session_state.corpus_create_open:
        try:
            available_kbs = _load_kbs()
        except Exception:
            available_kbs = []

        kb_options = {kb["name"]: kb["kb_id"] for kb in available_kbs}

        with st.form("create_corpus_form", border=True):
            st.markdown("**Create Corpus**")
            new_name = st.text_input("Name *", placeholder="e.g. support-kb-v2")
            new_desc = st.text_area("Description", height=60)
            new_uc   = st.text_input("Use Case ID", placeholder="GENAI1597_SSOP")
            new_af   = st.text_input("Agent Filter", placeholder="support-agent-v1")

            selected_kb_names = st.multiselect(
                "Knowledge Bases",
                options=list(kb_options.keys()),
                placeholder="Add KBs to this corpus…",
            )

            submitted = st.form_submit_button("Create", type="primary")
            if submitted:
                if not new_name.strip():
                    st.error("Name is required.")
                else:
                    try:
                        from pipeline.mongo_store import get_corpus_store
                        kb_ids = [kb_options[n] for n in selected_kb_names]
                        cid = get_corpus_store().create(
                            name=new_name.strip(),
                            description=new_desc.strip(),
                            usecase_id=new_uc.strip() or None,
                            agent_filter=new_af.strip() or None,
                            kb_ids=kb_ids,
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
            cid      = c["corpus_id"]
            kb_count = len(c.get("kb_ids") or [])
            label    = f"**{c['name']}**  \n{kb_count} KB{'s' if kb_count != 1 else ''}"
            if c.get("usecase_id"):
                label += f"  \nuc: `{c['usecase_id']}`"
            btn_type = "primary" if st.session_state.corpus_selected == cid else "secondary"
            if st.button(label, key=f"sel_{cid}", width="stretch", type=btn_type):
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
            corpus_name = corpus["name"]

            # ── Header ────────────────────────────────────────────────────────
            hd1, hd2 = st.columns([3, 1])
            with hd1:
                st.subheader(corpus_name)
                if corpus.get("description"):
                    st.caption(corpus["description"])
            with hd2:
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("Edit", width="stretch"):
                        st.session_state.corpus_edit_open = not st.session_state.corpus_edit_open
                with btn2:
                    if st.button("Delete", width="stretch", type="secondary"):
                        st.session_state[f"confirm_delete_{sel_id}"] = True

            if st.session_state.get(f"confirm_delete_{sel_id}"):
                st.warning(
                    f"Delete corpus **{corpus_name}**? "
                    "This removes the corpus record; Knowledge Bases and staged documents are not affected."
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
                try:
                    available_kbs = _load_kbs()
                except Exception:
                    available_kbs = []

                kb_opts       = {kb["name"]: kb["kb_id"] for kb in available_kbs}
                kb_id_to_name = {kb["kb_id"]: kb["name"] for kb in available_kbs}

                current_kb_names = [
                    kb_id_to_name[kid]
                    for kid in (corpus.get("kb_ids") or [])
                    if kid in kb_id_to_name
                ]

                with st.form("edit_corpus_form", border=True):
                    st.markdown("**Edit Corpus**")
                    e_desc = st.text_area("Description", value=corpus.get("description", ""), height=60)
                    e_uc   = st.text_input("Use Case ID", value=corpus.get("usecase_id", ""))
                    e_af   = st.text_input("Agent Filter", value=corpus.get("agent_filter", ""))
                    e_kbs  = st.multiselect(
                        "Knowledge Bases",
                        options=list(kb_opts.keys()),
                        default=current_kb_names,
                    )
                    saved = st.form_submit_button("Save", type="primary")
                    if saved:
                        from pipeline.mongo_store import get_corpus_store
                        new_kb_ids = [kb_opts[n] for n in e_kbs]
                        get_corpus_store().update(
                            corpus_id=sel_id,
                            description=e_desc.strip() or None,
                            usecase_id=e_uc.strip() or None,
                            agent_filter=e_af.strip() or None,
                            kb_ids=new_kb_ids,
                        )
                        _invalidate(sel_id)
                        st.session_state.corpus_edit_open = False
                        st.success("Corpus updated.")
                        st.rerun()

            # ── Meta chips ────────────────────────────────────────────────────
            chips: list[str] = []
            if corpus.get("usecase_id"):
                chips.append(f"uc: `{corpus['usecase_id']}`")
            if corpus.get("agent_filter"):
                chips.append(f"agent: `{corpus['agent_filter']}`")
            if chips:
                st.markdown("  ·  ".join(chips))

            # ── Stats ─────────────────────────────────────────────────────────
            try:
                available_kbs = _load_kbs()
            except Exception:
                available_kbs = []

            kb_id_to_obj = {kb["kb_id"]: kb for kb in available_kbs}
            kb_count     = len(corpus.get("kb_ids") or [])

            s1, s2 = st.columns(2)
            s1.metric("Knowledge Bases", kb_count)
            s2.metric("Last updated", _fmt_date(corpus.get("last_updated")))

            st.divider()

            # ── Tabs ──────────────────────────────────────────────────────────
            corpus_kb_ids       = corpus.get("kb_ids") or []
            corpus_kb_ids_tuple = tuple(sorted(corpus_kb_ids))

            tab_kbs, tab_docs, tab_export = st.tabs(["Knowledge Bases", "Documents", "Export"])

            # ── KBs tab ───────────────────────────────────────────────────────
            with tab_kbs:
                if not corpus_kb_ids:
                    st.info("No Knowledge Bases in this corpus yet. Click **Edit** to add KBs.")
                else:
                    import pandas as pd
                    rows = []
                    for kid in corpus_kb_ids:
                        kb = kb_id_to_obj.get(kid)
                        if kb:
                            rows.append({
                                "Name":        kb.get("name", kid),
                                "Type":        kb.get("source_type", "—"),
                                "Status":      kb.get("status", "—"),
                                "Staged docs": len(kb.get("doc_ids") or []),
                                "kb_id":       kid,
                            })
                        else:
                            rows.append({"Name": kid, "Type": "—", "Status": "—", "Staged docs": 0, "kb_id": kid})
                    df = pd.DataFrame(rows)
                    st.dataframe(df.drop(columns=["kb_id"]), hide_index=True)

                    kbs_to_remove = st.multiselect(
                        "Remove KBs from corpus",
                        options=[r["Name"] for r in rows],
                        placeholder="Select KBs to remove…",
                        key=f"rm_kbs_{sel_id}",
                    )
                    if kbs_to_remove:
                        name_to_id = {r["Name"]: r["kb_id"] for r in rows}
                        ids_to_remove = [name_to_id[n] for n in kbs_to_remove]
                        if st.button(f"Remove {len(kbs_to_remove)} KB(s)", type="primary", key="do_rm_kbs"):
                            from pipeline.mongo_store import get_corpus_store
                            get_corpus_store().remove_kbs(sel_id, ids_to_remove)
                            _invalidate(sel_id)
                            st.success(f"Removed {len(kbs_to_remove)} KB(s).")
                            st.rerun()

            # ── Documents tab ─────────────────────────────────────────────────
            with tab_docs:
                try:
                    pushed_docs_list = _load_corpus_docs(sel_id, corpus_kb_ids_tuple)
                except Exception as exc:
                    pushed_docs_list = []
                    st.warning(f"Could not load pushed documents: {exc}")

                if not pushed_docs_list:
                    st.info("No pushed documents in this corpus yet.")
                else:
                    search_q = st.text_input("Filter by title or KB", placeholder="Search…", key=f"doc_search_{sel_id}")
                    import pandas as pd
                    rows_d = [
                        {
                            "Title":       d.get("title", "—"),
                            "KB":          d.get("kb_name", "—"),
                            "Source type": d.get("source_type", "—"),
                            "Chunks":      d.get("chunk_count", 0),
                            "Drift":       d.get("drift_status", "—"),
                            "Pushed":      _fmt_date(d.get("pushed_at")),
                        }
                        for d in pushed_docs_list
                    ]
                    df_d = pd.DataFrame(rows_d)
                    if search_q:
                        mask = (
                            df_d["Title"].str.contains(search_q, case=False, na=False)
                            | df_d["KB"].str.contains(search_q, case=False, na=False)
                        )
                        df_d = df_d[mask]
                    st.caption(f"{len(df_d)} document(s)")
                    st.dataframe(df_d, hide_index=True)

            # ── Export tab ────────────────────────────────────────────────────
            with tab_export:
                st.markdown(
                    "Download all staged content from every Knowledge Base in this corpus as a single JSONL file."
                )

                if not corpus_kb_ids:
                    st.info("Add at least one Knowledge Base before exporting.")
                else:
                    export_status = st.radio(
                        "Include chunks with status",
                        ["all (staged + approved + pushed)", "approved", "pushed"],
                        horizontal=True,
                        key=f"export_status_{sel_id}",
                    )
                    status_map = {
                        "all (staged + approved + pushed)": None,
                        "approved": "approved",
                        "pushed": "pushed",
                    }
                    chosen_status = status_map[export_status]

                    if st.button("⬇️ Prepare corpus JSONL", type="primary", key=f"export_btn_{sel_id}"):
                        with st.spinner("Collecting chunks…"):
                            try:
                                import json as _json
                                from pipeline.mongo_store import get_staging
                                staging = get_staging()
                                all_chunks: list[dict] = []
                                for kid in corpus_kb_ids:
                                    all_chunks.extend(staging.get_chunks_by_kb(kid, status=chosen_status))

                                if not all_chunks:
                                    st.warning(
                                        f"No chunks found with status={chosen_status!r} across "
                                        f"{len(corpus_kb_ids)} KB(s). Stage and approve content first."
                                    )
                                else:
                                    import uuid as _uuid
                                    doc_id   = str(_uuid.uuid4())
                                    uc       = corpus.get("usecase_id") or ""
                                    af       = corpus.get("agent_filter") or ""
                                    lines    = "\n".join(
                                        _json.dumps({**c, "document_id": doc_id, "usecase_id": uc, "agent_filter": af}, ensure_ascii=False)
                                        for c in all_chunks
                                    )
                                    fname = f"{_slug(corpus_name)}_{int(time.time())}.jsonl"
                                    st.download_button(
                                        label=f"⬇️ Download {fname}  ({len(all_chunks):,} chunks)",
                                        data=lines.encode("utf-8"),
                                        file_name=fname,
                                        mime="application/x-ndjson",
                                        key=f"dl_corpus_{sel_id}",
                                    )
                            except Exception as exc:
                                st.error(f"Export failed: {exc}")

                    st.divider()

                    # Vector store push (secondary / advanced)
                    with st.expander("Vector store push (advanced)"):
                        st.caption(
                            "Push this corpus to a vector store for semantic search. "
                            "Requires a vector store to be configured."
                        )
                        try:
                            from pipeline.mongo_store import get_vs_config_store
                            available_vs = get_vs_config_store().list_all()
                        except Exception:
                            available_vs = []

                        if not available_vs:
                            st.info("No vector stores configured. Add one on the **Vector Stores** page.")
                        else:
                            vs_opts  = {vs["name"]: vs["vs_id"] for vs in available_vs}
                            vs_id_to = {vs["vs_id"]: vs["name"] for vs in available_vs}
                            current_vs = vs_id_to.get(corpus.get("vector_store_id", ""), "(none)")

                            push_vs_name = st.selectbox(
                                "Target vector store",
                                options=list(vs_opts.keys()),
                                index=list(vs_opts.keys()).index(current_vs) if current_vs in vs_opts else 0,
                                key=f"push_vs_{sel_id}",
                            )
                            if st.button("🚀 Push to vector store", key=f"push_btn_{sel_id}"):
                                vs_id = vs_opts[push_vs_name]
                                # Save vector store choice
                                from pipeline.mongo_store import get_corpus_store
                                get_corpus_store().update(corpus_id=sel_id, vector_store_id=vs_id)
                                with st.spinner("Embedding and pushing…"):
                                    try:
                                        from pipeline.review import push_approved
                                        result = push_approved(corpus_id=sel_id)
                                        st.success(
                                            f"Pushed **{result.get('pushed_docs', 0)}** document(s) — "
                                            f"**{result.get('pushed_chunks', 0):,}** sections indexed."
                                        )
                                        if result.get("errors"):
                                            for err in result["errors"]:
                                                st.error(err)
                                        _invalidate(sel_id)
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(f"Push failed: {exc}")
