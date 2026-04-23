"""Review Queue — inspect, approve, reject, and push staged documents."""
import json
import math
import re

import streamlit as st

from pipeline.chunker import _split_large_chunk

# ── Helpers ───────────────────────────────────────────────────────────────────

STATUS_ICON  = {"approved": "✅", "pending_review": "⏳", "rejected": "❌"}
STATUS_LABEL = {"approved": "Approved", "pending_review": "Needs Review", "rejected": "Rejected"}
STATUS_COLOR = {"approved": "green",    "pending_review": "orange",        "rejected": "red"}

SOURCE_ICON  = {
    "pdf": "📄", "docx": "📝", "pptx": "📊", "html": "🌐",
    "url": "🔗", "markdown": "📋", "text": "📃",
}


def _badge(status: str) -> str:
    icon  = STATUS_ICON.get(status, "❓")
    label = STATUS_LABEL.get(status, status)
    return f"{icon} {label}"


def _age_label(age_days: int | None) -> str:
    if age_days is None:
        return "age unknown"
    if age_days < 30:
        return f"{age_days}d old"
    if age_days < 365:
        return f"{age_days // 30}mo old"
    return f"{age_days // 365}y old"


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clean_content(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    return "\n".join(line.rstrip() for line in text.splitlines())


def _split_by_delimiter(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def _load_docs(kb_id: str | None = None):
    from pipeline.mongo_store import get_staging
    return get_staging().list_all(kb_id=kb_id)


@st.cache_data(ttl=30)
def _load_kbs() -> list[dict]:
    from pipeline.mongo_store import get_kb_store
    return get_kb_store().list_all()


@st.cache_data(ttl=30)
def _load_corpora() -> list[dict]:
    from pipeline.mongo_store import get_corpus_store
    return get_corpus_store().list_all()


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("📋 Review Queue")
st.caption("Inspect documents before they go into the knowledge base. Approve or reject each one.")

# ── KB filter ─────────────────────────────────────────────────────────────────

try:
    all_kbs = _load_kbs()
except Exception:
    all_kbs = []

kb_filter_opts = {"All Knowledge Bases": None}
kb_filter_opts.update({kb["name"]: kb["kb_id"] for kb in all_kbs})

col_kb_filter, _ = st.columns([2, 3])
with col_kb_filter:
    kb_filter_label = st.selectbox(
        "Filter by Knowledge Base",
        options=list(kb_filter_opts.keys()),
        label_visibility="collapsed",
    )
active_kb_id = kb_filter_opts.get(kb_filter_label)

# ── Metrics ───────────────────────────────────────────────────────────────────
try:
    all_docs = _load_docs(kb_id=active_kb_id)
except Exception as exc:
    st.error(f"Could not load staged documents: {exc}")
    st.caption(
        "Check that MongoDB is reachable and that `MONGODB_URI` (or `MONGODB_HOST`) "
        "is set correctly in your `.env` file."
    )
    st.stop()

pending  = [d for d in all_docs if d.get("status") == "pending_review"]
approved = [d for d in all_docs if d.get("status") == "approved"]
rejected = [d for d in all_docs if d.get("status") == "rejected"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Needs Review",     len(pending),  delta=f"{len(pending)} waiting" if pending else None, delta_color="inverse")
m2.metric("Approved",         len(approved), help="Ready to push to knowledge base")
m3.metric("Rejected",         len(rejected))
m4.metric("Total Staged",     len(all_docs))

# Corpus options — needed both in the push panel and in per-doc push buttons
try:
    all_corpora = _load_corpora()
except Exception:
    all_corpora = []
corpus_opts = {c["name"]: c["corpus_id"] for c in all_corpora}
push_corpus_label: str = list(corpus_opts.keys())[0] if corpus_opts else ""

# ── Push all approved ─────────────────────────────────────────────────────────
if approved:
    st.divider()

    push_col, info_col = st.columns([3, 2])
    with push_col:
        push_corpus_label = st.selectbox(
            "Push to corpus",
            options=list(corpus_opts.keys()) or ["(no corpora — create one first)"],
            help="The corpus defines the use case, agent filter, and target vector store.",
            key="push_corpus_select",
        )
        push_corpus_id = corpus_opts.get(push_corpus_label)

        if push_corpus_id and st.button(
            f"🚀  Push {len(approved)} approved document{'s' if len(approved) != 1 else ''} to Knowledge Base",
            type="primary",
            width="stretch",
        ):
            with st.spinner("Embedding and pushing …"):
                from pipeline import review as rev
                result = rev.push_approved(corpus_id=push_corpus_id)
            if result.get("errors"):
                for err in result["errors"]:
                    st.error(err)
            st.success(
                f"✅  Pushed **{result.get('pushed_docs', 0)}** document(s) — "
                f"**{result.get('pushed_chunks', 0):,}** sections added to the knowledge base."
            )
            st.cache_data.clear()
            st.rerun()
    with info_col:
        st.caption(
            "Pushing embeds the document sections and makes them searchable. "
            "The corpus determines the use case context and target vector DB."
        )

# ── Filter tabs ───────────────────────────────────────────────────────────────
st.divider()

filter_options = {
    "All":          all_docs,
    "Needs Review": pending,
    "Approved":     approved,
    "Rejected":     rejected,
}
selected_filter = st.segmented_control(
    "Show",
    list(filter_options.keys()),
    default="All",
    label_visibility="collapsed",
)
visible_docs = filter_options.get(selected_filter or "All", all_docs)

if not visible_docs:
    st.info("Nothing here yet.")
    st.stop()

# ── Document cards ────────────────────────────────────────────────────────────
for doc in visible_docs:
    doc_id      = doc.get("doc_id", "")
    title       = doc.get("title", "Untitled")
    status      = doc.get("status", "")
    score       = _safe_float(doc.get("quality_score", 0))
    chunks      = _safe_int(doc.get("chunk_count", 0))
    src_type    = doc.get("source_type", "")
    flags       = doc.get("quality_flags") or []
    author      = doc.get("author", "")
    pages       = doc.get("page_count", "")
    url         = doc.get("url", "")
    kb_id_doc   = doc.get("kb_id") or ""
    age_days    = doc.get("age_days")
    is_stale    = doc.get("is_stale", False)
    n_short     = _safe_int(doc.get("chunks_too_short", 0))
    n_long      = _safe_int(doc.get("chunks_too_long", 0))
    n_bplate    = _safe_int(doc.get("chunks_boilerplate", 0))

    src_icon = SOURCE_ICON.get(src_type, "📎")
    kb_id_to_name = {kb["kb_id"]: kb["name"] for kb in all_kbs}

    with st.container(border=True):
        # ── Row 1: title + status badge ───────────────────────────────────
        h_col, s_col = st.columns([5, 1])
        with h_col:
            st.markdown(f"**{title}**")
            meta_parts = [f"{src_icon} {src_type.upper()}" if src_type else ""]
            if author:
                meta_parts.append(f"by {author}")
            if pages:
                meta_parts.append(f"{pages} pages")
            meta_parts.append(f"{chunks} sections")
            if kb_id_doc:
                kb_label = kb_id_to_name.get(kb_id_doc, kb_id_doc[:8])
                meta_parts.append(f"📂 {kb_label}")
            st.caption("  ·  ".join(p for p in meta_parts if p))
        with s_col:
            color = STATUS_COLOR.get(status, "gray")
            st.markdown(
                f"<span style='color:{color};font-weight:bold'>{_badge(status)}</span>",
                unsafe_allow_html=True,
            )

        # ── Row 2: quality signals ────────────────────────────────────────
        sig_parts: list[str] = []
        if chunks:
            clean = chunks - (n_short + n_long + n_bplate)
            sig_parts.append(f"✅ {clean} clean")
        if n_short:
            sig_parts.append(f"⬇️ {n_short} too short")
        if n_long:
            sig_parts.append(f"⬆️ {n_long} too long")
        if n_bplate:
            sig_parts.append(f"📋 {n_bplate} boilerplate")

        age_label = _age_label(age_days)
        age_badge = f"🕐 {age_label}"
        if is_stale:
            age_badge = f"⚠️ {age_label} — stale"

        q_col, age_col = st.columns([3, 2])
        with q_col:
            if sig_parts:
                st.caption("  ·  ".join(sig_parts))
        with age_col:
            st.caption(age_badge)

        # ── Row 3: flags (if any) ─────────────────────────────────────────
        if flags:
            with st.expander(f"⚠️  {len(flags)} quality note{'s' if len(flags) != 1 else ''}"):
                for flag in flags:
                    st.write(f"• {flag}")

        # ── Row 4: section editor ─────────────────────────────────────────
        _PAGE_SIZE = 10
        page_key     = f"sp_{doc_id}"
        selected_key = f"sel_{doc_id}"
        if page_key     not in st.session_state: st.session_state[page_key]     = 0
        if selected_key not in st.session_state: st.session_state[selected_key] = set()

        with st.expander(f"✏️  Sections ({chunks})"):
            try:
                from pipeline import review as rev
                detail     = rev.get_doc_detail(doc_id)
                all_chunks = (detail or {}).get("sample_chunks", [])
                if not all_chunks:
                    st.caption("No sections available.")
                else:
                    chunk_filter = st.segmented_control(
                        "Show",
                        ["All", "Issues", "Clean"],
                        default="All",
                        key=f"cf_{doc_id}",
                    )
                    if chunk_filter == "Issues":
                        all_chunks = [
                            c for c in all_chunks
                            if (c.get("metadata") or {}).get("quality_flags", [])
                        ]
                    elif chunk_filter == "Clean":
                        all_chunks = [
                            c for c in all_chunks
                            if not (c.get("metadata") or {}).get("quality_flags", [])
                        ]

                    if not all_chunks:
                        st.caption(
                            f"No {'issue' if chunk_filter == 'Issues' else 'clean'} sections."
                        )
                    else:
                        total_pages = max(1, math.ceil(len(all_chunks) / _PAGE_SIZE))
                        page        = min(st.session_state[page_key], total_pages - 1)
                        st.session_state[page_key] = page
                        page_chunks = all_chunks[page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE]

                        # ── Pagination controls ───────────────────────────────
                        pc1, pc2, pc3 = st.columns([1, 3, 1])
                        with pc1:
                            if st.button("← Prev", key=f"prev_{doc_id}", disabled=page == 0,
                                         width="stretch"):
                                st.session_state[page_key] = page - 1
                                st.rerun()
                        with pc2:
                            start = page * _PAGE_SIZE + 1
                            end   = min(start + _PAGE_SIZE - 1, len(all_chunks))
                            st.caption(
                                f"Sections {start}–{end} of {len(all_chunks)}"
                                + (f"  ·  page {page + 1}/{total_pages}" if total_pages > 1 else "")
                            )
                        with pc3:
                            if st.button("Next →", key=f"next_{doc_id}",
                                         disabled=page >= total_pages - 1,
                                         width="stretch"):
                                st.session_state[page_key] = page + 1
                                st.rerun()

                        # ── Section rows ──────────────────────────────────────
                        for chunk in page_chunks:
                            chunk_id  = chunk.get("chunk_id", "")
                            section   = chunk.get("section", "—")
                            content   = chunk.get("content", "")
                            tags      = chunk.get("tags", [])
                            cit       = (chunk.get("metadata") or {}).get("citation", {})
                            page_no   = cit.get("page_number")
                            chk_flags = (chunk.get("metadata") or {}).get("quality_flags", [])

                            with st.container(border=True):
                                chk_col, body_col = st.columns([1, 12])

                                # Select checkbox for break-out
                                with chk_col:
                                    is_sel = chunk_id in st.session_state[selected_key]
                                    if st.checkbox(
                                        "select", value=is_sel,
                                        key=f"chk_{chunk_id}",
                                        label_visibility="collapsed",
                                    ):
                                        st.session_state[selected_key].add(chunk_id)
                                    else:
                                        st.session_state[selected_key].discard(chunk_id)

                                with body_col:
                                    # Heading
                                    heading = section.split(" > ")[-1] if " > " in section else section
                                    st.markdown(f"**{heading}**")
                                    if " > " in section:
                                        st.caption(section)
                                    if page_no:
                                        st.caption(f"Page {page_no}")
                                    if chk_flags:
                                        st.caption("⚠️ " + "  ·  ".join(chk_flags))

                                    # Content (collapsed) — editable
                                    with st.expander("Content", expanded=False):
                                        edited = st.text_area(
                                            "Edit content",
                                            value=content,
                                            height=200,
                                            key=f"content_{chunk_id}",
                                            label_visibility="collapsed",
                                        )
                                        c1, c2 = st.columns(2)
                                        with c1:
                                            if st.button("💾 Save content", key=f"save_content_{chunk_id}"):
                                                rev.update_chunk(doc_id, chunk_id, {"content": edited.strip()})
                                                st.toast("Content updated", icon="💾")
                                                st.cache_data.clear()
                                                st.rerun()
                                        with c2:
                                            if st.button("🧹 Clean formatting", key=f"clean_{chunk_id}"):
                                                cleaned = _clean_content(edited)
                                                rev.update_chunk(doc_id, chunk_id, {"content": cleaned})
                                                st.toast("Formatting cleaned", icon="🧹")
                                                st.cache_data.clear()
                                                st.rerun()

                                        with st.expander("✂️ Split into subchunks", expanded=False):
                                            split_mode = st.radio(
                                                "Split by",
                                                ["Blank lines (\\n\\n)", "Character limit"],
                                                key=f"smode_{chunk_id}",
                                                horizontal=True,
                                            )
                                            max_chars = 1000
                                            if "Character limit" in split_mode:
                                                max_chars = st.number_input(
                                                    "Max chars", 200, 5000, 1000,
                                                    key=f"maxc_{chunk_id}",
                                                )

                                            if st.button("Preview", key=f"prev_{chunk_id}"):
                                                parts = (
                                                    _split_by_delimiter(edited)
                                                    if "Blank" in split_mode
                                                    else _split_large_chunk(edited, max_chars, 0)
                                                )
                                                st.caption(f"{len(parts)} part(s)")
                                                for i, p in enumerate(parts):
                                                    st.text(f"[{i + 1}] {p[:120]}{'…' if len(p) > 120 else ''}")

                                            if st.button("✅ Confirm split", key=f"conf_{chunk_id}", type="primary"):
                                                parts = (
                                                    _split_by_delimiter(edited)
                                                    if "Blank" in split_mode
                                                    else _split_large_chunk(edited, max_chars, 0)
                                                )
                                                if len(parts) < 2:
                                                    st.warning("Need at least 2 parts to split. Use Save Content instead.")
                                                else:
                                                    new_ids = rev.split_chunk(doc_id, chunk_id, parts)
                                                    if new_ids:
                                                        st.toast(f"Split into {len(new_ids)} subchunks", icon="✂️")
                                                        st.cache_data.clear()
                                                        st.rerun()
                                                    else:
                                                        st.error("Split failed — chunk not found.")

                                    # Tag editor
                                    tag_str = st.text_input(
                                        "Tags",
                                        value=", ".join(tags),
                                        key=f"tags_{chunk_id}",
                                        placeholder="tag1, tag2, …",
                                    )
                                    if st.button("💾 Save tags", key=f"save_tags_{chunk_id}"):
                                        new_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
                                        rev.update_chunk(doc_id, chunk_id, {"tags": new_tags})
                                        st.toast("Tags updated", icon="💾")
                                        st.cache_data.clear()
                                        st.rerun()

                        # ── Break-out control ─────────────────────────────────
                        selected_ids = list(st.session_state.get(selected_key, set()))
                        if selected_ids:
                            st.divider()
                            st.caption(f"{len(selected_ids)} section(s) selected for break-out")
                            new_title = st.text_input(
                                "New document title",
                                key=f"split_title_{doc_id}",
                                placeholder=f"{title} — split",
                            )
                            if st.button(
                                f"✂️  Break out {len(selected_ids)} section(s) into new document",
                                key=f"split_{doc_id}",
                                type="primary",
                            ):
                                if new_title.strip():
                                    new_id = rev.split_doc(doc_id, selected_ids, new_title.strip())
                                    if new_id:
                                        st.toast(f"Created: {new_title.strip()}", icon="✂️")
                                        st.session_state[selected_key] = set()
                                        st.cache_data.clear()
                                        st.rerun()
                                    else:
                                        st.error("Break-out failed — source document not found.")
                                else:
                                    st.warning("Enter a title for the new document.")

            except Exception as exc:
                st.caption(f"Could not load sections: {exc}")

        # ── Row 5: action buttons ─────────────────────────────────────────
        btn_cols = st.columns([1, 1, 1, 3])

        # Approve
        if status != "approved":
            if btn_cols[0].button("✅  Approve", key=f"approve_{doc_id}", width="stretch"):
                from pipeline import review as rev
                rev.approve_doc(doc_id)
                st.toast(f"Approved: {title}", icon="✅")
                st.cache_data.clear()
                st.rerun()
        else:
            btn_cols[0].button("✅  Approved", key=f"noop_approve_{doc_id}", disabled=True, width="stretch")

        # Push single doc — uses the corpus selected in the push panel above
        if status == "approved":
            _single_corpus_id = corpus_opts.get(push_corpus_label) if approved else None
            if btn_cols[1].button("🚀  Push now", key=f"push_{doc_id}", width="stretch",
                                  disabled=not _single_corpus_id):
                with st.spinner(f"Pushing {title} …"):
                    from pipeline import review as rev
                    res = rev.push_approved(corpus_id=_single_corpus_id, doc_id=doc_id)
                if res.get("errors"):
                    st.error(res["errors"][0])
                else:
                    st.toast(f"Pushed {res.get('pushed_chunks', 0)} sections", icon="🚀")
                st.cache_data.clear()
                st.rerun()

        # Reject with reason popover
        if status != "rejected":
            with btn_cols[2].popover("❌  Reject", width="stretch"):
                reason = st.text_input(
                    "Reason *(optional)*",
                    key=f"reason_{doc_id}",
                    placeholder="duplicate, poor quality, wrong topic…",
                )
                if st.button("Confirm rejection", key=f"confirm_reject_{doc_id}", type="primary"):
                    from pipeline import review as rev
                    rev.reject_doc(doc_id, reason=reason)
                    st.toast(f"Rejected: {title}", icon="❌")
                    st.cache_data.clear()
                    st.rerun()
        else:
            reject_reason = doc.get("reject_reason", "")
            btn_cols[2].button(
                "❌  Rejected", key=f"noop_reject_{doc_id}",
                disabled=True, width="stretch",
                help=f"Reason: {reject_reason}" if reject_reason else None,
            )

        # Source link (if URL)
        if url:
            btn_cols[3].markdown(f"[🔗 View source]({url})")
