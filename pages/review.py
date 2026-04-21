"""Review Queue — inspect, approve, reject, and push staged documents."""
import json

import streamlit as st

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


def _score_bar(score: float) -> str:
    filled = round(score * 10)
    return "█" * filled + "░" * (10 - filled)


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


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)          # refresh every 5 s on rerun
def _load_docs():
    from pipeline import review as rev
    return rev.list_all_docs()


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("📋 Review Queue")
st.caption("Inspect documents before they go into the knowledge base. Approve or reject each one.")

# ── Metrics ───────────────────────────────────────────────────────────────────
try:
    all_docs = _load_docs()
except Exception as exc:
    st.error(f"Could not connect to Redis: {exc}")
    st.stop()

pending  = [d for d in all_docs if d.get("status") == "pending_review"]
approved = [d for d in all_docs if d.get("status") == "approved"]
rejected = [d for d in all_docs if d.get("status") == "rejected"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Needs Review",     len(pending),  delta=f"{len(pending)} waiting" if pending else None, delta_color="inverse")
m2.metric("Approved",         len(approved), help="Ready to push to knowledge base")
m3.metric("Rejected",         len(rejected))
m4.metric("Total Staged",     len(all_docs))

# ── Push all approved ─────────────────────────────────────────────────────────
if approved:
    st.divider()
    col_push, col_info = st.columns([2, 3])
    with col_push:
        if st.button(
            f"🚀  Push {len(approved)} approved document{'s' if len(approved) != 1 else ''} to Knowledge Base",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Embedding and pushing …"):
                from pipeline import review as rev
                result = rev.push_approved()
            if result["errors"]:
                for err in result["errors"]:
                    st.error(err)
            st.success(
                f"✅  Pushed **{result['pushed_docs']}** document(s) — "
                f"**{result['pushed_chunks']:,}** sections added to the knowledge base."
            )
            st.cache_data.clear()
            st.rerun()
    with col_info:
        st.caption(
            "Pushing embeds the document sections and makes them searchable. "
            "This may take a minute for large documents."
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
    doc_id   = doc.get("doc_id", "")
    title    = doc.get("title", "Untitled")
    status   = doc.get("status", "")
    score    = _safe_float(doc.get("quality_score", 0))
    chunks   = _safe_int(doc.get("chunk_count", 0))
    src_type = doc.get("source_type", "")
    flags    = doc.get("quality_flags") or []
    author   = doc.get("author", "")
    pages    = doc.get("page_count", "")
    url      = doc.get("url", "")

    src_icon = SOURCE_ICON.get(src_type, "📎")

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
            st.caption("  ·  ".join(p for p in meta_parts if p))
        with s_col:
            color = STATUS_COLOR.get(status, "gray")
            st.markdown(
                f"<span style='color:{color};font-weight:bold'>{_badge(status)}</span>",
                unsafe_allow_html=True,
            )

        # ── Row 2: quality bar ────────────────────────────────────────────
        q_col, _ = st.columns([2, 3])
        with q_col:
            st.progress(score, text=f"Quality  {_score_bar(score)}  {score:.0%}")

        # ── Row 3: flags (if any) ─────────────────────────────────────────
        if flags:
            with st.expander(f"⚠️  {len(flags)} quality note{'s' if len(flags) != 1 else ''}"):
                for flag in flags:
                    st.write(f"• {flag}")

        # ── Row 4: sample sections ────────────────────────────────────────
        with st.expander("🔍  Preview document sections"):
            try:
                from pipeline import review as rev
                detail = rev.get_doc_detail(doc_id)
                samples = (detail or {}).get("sample_chunks", [])
                if samples:
                    for i, chunk in enumerate(samples, 1):
                        section  = chunk.get("section", "—")
                        content  = chunk.get("content", "")
                        tags     = chunk.get("tags", [])
                        cit      = (chunk.get("metadata") or {}).get("citation", {})
                        page_no  = cit.get("page_number")

                        st.markdown(f"**Section {i} of {chunks}** — *{section}*")
                        if page_no:
                            st.caption(f"Page {page_no} of {cit.get('page_count', '?')}")
                        st.text(content[:500] + ("…" if len(content) > 500 else ""))
                        if tags:
                            st.caption("Tags: " + ", ".join(f"`{t}`" for t in tags))
                        if i < len(samples):
                            st.divider()
                else:
                    st.caption("No preview available.")
            except Exception as exc:
                st.caption(f"Could not load preview: {exc}")

        # ── Row 5: action buttons ─────────────────────────────────────────
        btn_cols = st.columns([1, 1, 1, 3])

        # Approve
        if status != "approved":
            if btn_cols[0].button("✅  Approve", key=f"approve_{doc_id}", use_container_width=True):
                from pipeline import review as rev
                rev.approve_doc(doc_id)
                st.toast(f"Approved: {title}", icon="✅")
                st.cache_data.clear()
                st.rerun()
        else:
            btn_cols[0].button("✅  Approved", key=f"noop_approve_{doc_id}", disabled=True, use_container_width=True)

        # Push single doc
        if status == "approved":
            if btn_cols[1].button("🚀  Push now", key=f"push_{doc_id}", use_container_width=True):
                with st.spinner(f"Pushing {title} …"):
                    from pipeline import review as rev
                    res = rev.push_approved(doc_id=doc_id)
                if res["errors"]:
                    st.error(res["errors"][0])
                else:
                    st.toast(f"Pushed {res['pushed_chunks']} sections", icon="🚀")
                st.cache_data.clear()
                st.rerun()

        # Reject with reason popover
        if status != "rejected":
            with btn_cols[2].popover("❌  Reject", use_container_width=True):
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
                disabled=True, use_container_width=True,
                help=f"Reason: {reject_reason}" if reject_reason else None,
            )

        # Source link (if URL)
        if url:
            btn_cols[3].markdown(f"[🔗 View source]({url})")
