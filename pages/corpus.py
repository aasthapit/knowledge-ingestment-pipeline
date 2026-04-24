"""Corpus Management — collections of Knowledge Bases scoped to a use case."""
from __future__ import annotations

import streamlit as st

st.title("📦 Corpus Management")
st.caption(
    "A corpus is a named collection of Knowledge Bases. It carries the use case ID, "
    "agent filter, and target vector DB. Pushing a corpus embeds and indexes all approved "
    "documents from its Knowledge Bases."
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


@st.cache_data(ttl=30)
def _load_vs_configs() -> list[dict]:
    from pipeline.mongo_store import get_vs_config_store
    return get_vs_config_store().list_all()


def _invalidate(corpus_id: str | None = None) -> None:
    _load_corpora.clear()
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
        if st.button("＋ New", use_container_width=True):
            st.session_state.corpus_create_open = not st.session_state.corpus_create_open
            st.session_state.corpus_edit_open = False

    # ── Create form ───────────────────────────────────────────────────────────
    if st.session_state.corpus_create_open:
        try:
            available_kbs = _load_kbs()
            available_vs  = _load_vs_configs()
        except Exception:
            available_kbs = []
            available_vs  = []

        kb_options = {kb["name"]: kb["kb_id"] for kb in available_kbs}
        vs_options = {vs["name"]: vs["vs_id"] for vs in available_vs}

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
                help="Select one or more Knowledge Bases to include.",
            )
            selected_vs_name = st.selectbox(
                "Vector Store",
                options=list(vs_options.keys()) or ["(none)"],
                help="Which vector DB to push this corpus to.",
            )

            submitted = st.form_submit_button("Create", type="primary")
            if submitted:
                vs_id = vs_options.get(selected_vs_name) if selected_vs_name != "(none)" else None
                if not new_name.strip():
                    st.error("Name is required.")
                elif vs_id is None:
                    st.error("A vector store is required. Add one on the Vector Stores page first.")
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
                            vector_store_id=vs_id,
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
            kb_count = len(c.get("kb_ids") or [])
            label = f"**{c['name']}**  \n{kb_count} KB{'s' if kb_count != 1 else ''}"
            if c.get("usecase_id"):
                label += f"  \nuc: `{c['usecase_id']}`"
            btn_type = "primary" if st.session_state.corpus_selected == cid else "secondary"
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
                    available_vs  = _load_vs_configs()
                except Exception:
                    available_kbs = []
                    available_vs  = []

                kb_opts = {kb["name"]: kb["kb_id"] for kb in available_kbs}
                vs_opts = {vs["name"]: vs["vs_id"] for vs in available_vs}
                kb_id_to_name = {kb["kb_id"]: kb["name"] for kb in available_kbs}

                current_kb_names = [
                    kb_id_to_name[kid]
                    for kid in (corpus.get("kb_ids") or [])
                    if kid in kb_id_to_name
                ]
                current_vs_name  = next(
                    (vs["name"] for vs in available_vs if vs["vs_id"] == corpus.get("vector_store_id")),
                    list(vs_opts.keys())[0] if vs_opts else "(none)",
                )

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
                    e_vs = st.selectbox(
                        "Vector Store",
                        options=list(vs_opts.keys()) or ["(none)"],
                        index=list(vs_opts.keys()).index(current_vs_name)
                        if current_vs_name in vs_opts
                        else 0,
                    )
                    saved = st.form_submit_button("Save", type="primary")
                    if saved:
                        new_vs_id = vs_opts.get(e_vs) if e_vs != "(none)" else None
                        if new_vs_id is None:
                            st.error("A vector store is required.")
                        else:
                            from pipeline.mongo_store import get_corpus_store
                            new_kb_ids = [kb_opts[n] for n in e_kbs]
                            get_corpus_store().update(
                                corpus_id=sel_id,
                                description=e_desc.strip() or None,
                                usecase_id=e_uc.strip() or None,
                                agent_filter=e_af.strip() or None,
                                kb_ids=new_kb_ids,
                                vector_store_id=new_vs_id,
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
                available_vs  = _load_vs_configs()
            except Exception:
                available_kbs = []
                available_vs  = []

            kb_id_to_obj = {kb["kb_id"]: kb for kb in available_kbs}
            vs_id_to_obj = {vs["vs_id"]: vs for vs in available_vs}

            kb_count = len(corpus.get("kb_ids") or [])
            vs_name  = vs_id_to_obj.get(corpus.get("vector_store_id", ""), {}).get("name", "—")

            s1, s2, s3 = st.columns(3)
            s1.metric("Knowledge Bases", kb_count)
            s2.metric("Vector Store",    vs_name)
            s3.metric("Last updated",    _fmt_date(corpus.get("last_updated")))

            st.divider()

            # ── Tabs ──────────────────────────────────────────────────────────
            tab_kbs, tab_push = st.tabs(["Knowledge Bases", "Push"])

            # ── KBs tab ───────────────────────────────────────────────────────
            with tab_kbs:
                corpus_kb_ids = corpus.get("kb_ids") or []
                if not corpus_kb_ids:
                    st.info(
                        "No Knowledge Bases in this corpus yet. "
                        "Click **Edit** to add KBs."
                    )
                else:
                    import pandas as pd
                    rows = []
                    for kid in corpus_kb_ids:
                        kb = kb_id_to_obj.get(kid)
                        if kb:
                            rows.append({
                                "Name":         kb.get("name", kid),
                                "Type":         kb.get("source_type", "—"),
                                "Status":       kb.get("status", "—"),
                                "Staged docs":  len(kb.get("doc_ids") or []),
                                "kb_id":        kid,
                            })
                        else:
                            rows.append({
                                "Name": kid, "Type": "—", "Status": "—",
                                "Staged docs": 0, "kb_id": kid,
                            })
                    df = pd.DataFrame(rows)
                    st.dataframe(df.drop(columns=["kb_id"]), use_container_width=True, hide_index=True)

                    # Quick-remove KBs
                    kbs_to_remove = st.multiselect(
                        "Remove KBs from corpus",
                        options=[r["Name"] for r in rows],
                        placeholder="Select KBs to remove…",
                        key=f"rm_kbs_{sel_id}",
                    )
                    if kbs_to_remove:
                        name_to_id = {r["Name"]: r["kb_id"] for r in rows}
                        ids_to_remove = [name_to_id[n] for n in kbs_to_remove]
                        if st.button(
                            f"Remove {len(kbs_to_remove)} KB(s)",
                            type="primary",
                            key="do_rm_kbs",
                        ):
                            from pipeline.mongo_store import get_corpus_store
                            get_corpus_store().remove_kbs(sel_id, ids_to_remove)
                            _invalidate(sel_id)
                            st.success(f"Removed {len(kbs_to_remove)} KB(s).")
                            st.rerun()

            # ── Push tab ──────────────────────────────────────────────────────
            with tab_push:
                st.markdown(
                    "Push embeds all approved documents from this corpus's Knowledge Bases "
                    f"and writes them to **{vs_name}**."
                )

                if not corpus_kb_ids:
                    st.info("Add at least one Knowledge Base before pushing.")
                elif not corpus.get("vector_store_id"):
                    st.warning("Set a Vector Store target on this corpus before pushing.")
                else:
                    col_push, _ = st.columns([2, 3])
                    with col_push:
                        if st.button("🚀 Push corpus", type="primary", width="stretch"):
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
