"""Search — ask a question, get sourced answers from the knowledge base."""
import streamlit as st

# ── Helpers ───────────────────────────────────────────────────────────────────

def _relevance_label(score: float) -> tuple[str, str]:
    """Return (colour, label) for a normalised 0–1 similarity score."""
    if score >= 0.80:
        return "green", "Highly relevant"
    if score >= 0.60:
        return "orange", "Relevant"
    return "gray", "Possibly relevant"


def _source_display(result: dict) -> str:
    """Friendly one-line source attribution."""
    cit = result.get("citation", {})
    parts: list[str] = []

    title = cit.get("title") or result.get("title", "")
    if title:
        parts.append(f"**{title}**")

    page = cit.get("page_number")
    total = cit.get("page_count")
    if page:
        parts.append(f"page {page}" + (f" of {total}" if total else ""))

    url = cit.get("url")
    if url:
        parts.append(f"[View source]({url})")
    else:
        path = cit.get("source_path") or result.get("source", "")
        if path:
            # Show just the filename, not the full path
            from pathlib import Path as _P
            parts.append(f"`{_P(path).name}`")

    author = cit.get("author")
    if author:
        parts.append(f"by {author}")

    return "  ·  ".join(parts) if parts else "Unknown source"


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("🔎 Search Knowledge Base")
st.caption("Ask a question in plain language — the knowledge base will find the most relevant sections.")

# ── Use case filter ───────────────────────────────────────────────────────────
uc_options = ["All use cases"]
ag_options_map: dict[str, list[str]] = {}
try:
    from pipeline.mongo_store import get_usecase_ledger
    _uc_ledger = get_usecase_ledger()
    uc_options += _uc_ledger.get_distinct_usecases()
    for _uc in uc_options[1:]:
        ag_options_map[_uc] = _uc_ledger.get_agent_filters_for_usecase(_uc)
except Exception:
    pass

uc_cols = st.columns([2, 2, 1])
with uc_cols[0]:
    selected_usecase = st.selectbox(
        "Use case",
        uc_options,
        help="Filter results to chunks belonging to a specific use case.",
    )
with uc_cols[1]:
    ag_options = ["All agents"] + ag_options_map.get(selected_usecase, [])
    selected_agent = st.selectbox(
        "Agent",
        ag_options,
        help="Narrow results to a specific agent persona.",
        disabled=(selected_usecase == "All use cases"),
    )

# ── Search form ───────────────────────────────────────────────────────────────
with st.form("search_form", border=False):
    query = st.text_input(
        "Your question",
        placeholder="How do I configure network policies in OpenShift?",
        label_visibility="collapsed",
    )
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    with f2:
        source_type = st.selectbox(
            "Document type",
            ["All types", "pdf", "docx", "pptx", "html", "url", "markdown"],
            help="Restrict results to a specific document type",
        )
    with f3:
        st.write("")  # spacer
        submitted = st.form_submit_button("Search", type="primary", width="stretch")

# ── Tag filter (outside form so it's more accessible) ─────────────────────────
tags_filter_raw = st.text_input(
    "Filter by tags  *(optional)*",
    placeholder="openshift, networking",
    help="Comma-separated — results must match at least one tag",
    label_visibility="visible",
)
tag_filter = [t.strip() for t in tags_filter_raw.split(",") if t.strip()] or None

# ── Run search ────────────────────────────────────────────────────────────────
if submitted:
    if not query.strip():
        st.warning("Please enter a question to search.")
        st.stop()

    with st.spinner("Searching…"):
        try:
            # Resolve allowed chunk_ids for use case filtering
            uc_chunk_ids: set[str] | None = None
            if selected_usecase != "All use cases":
                try:
                    from pipeline.mongo_store import get_usecase_ledger as _get_ucl
                    _agent = None if selected_agent == "All agents" else selected_agent
                    if _agent:
                        uc_chunk_ids = set(
                            _get_ucl().get_chunk_ids(selected_usecase, _agent)
                        )
                    else:
                        # Union of all agents for this use case
                        _ids: list[str] = []
                        for _ag in ag_options_map.get(selected_usecase, []):
                            _ids.extend(_get_ucl().get_chunk_ids(selected_usecase, _ag))
                        uc_chunk_ids = set(_ids)
                except Exception:
                    uc_chunk_ids = None

            # Over-fetch when filtering so we hit the target top_k after filtering
            fetch_k = top_k * 5 if uc_chunk_ids is not None else top_k

            from pipeline.ingest import query_vectorstore
            results = query_vectorstore(
                question=query.strip(),
                top_k=fetch_k,
                tag_filter=tag_filter,
                source_type=None if source_type == "All types" else source_type,
            )

            # Post-retrieval use case filter
            if uc_chunk_ids is not None:
                results = [r for r in results if r.get("chunk_id") in uc_chunk_ids]
                results = results[:top_k]

        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.stop()

    st.divider()

    if not results:
        st.info("No results found. Try different keywords, or check that documents have been pushed to the knowledge base.")
        st.stop()

    st.caption(f"Found **{len(results)}** relevant section{'s' if len(results) != 1 else ''} for: *{query}*")

    # ── Result cards ──────────────────────────────────────────────────────────
    for i, r in enumerate(results, 1):
        norm_score = float(r.get("normalized_score", 0))
        colour, rel_label = _relevance_label(norm_score)
        section = r.get("section", "")
        content = r.get("content", "")
        tags    = r.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        with st.container(border=True):
            # Header row
            hdr_left, hdr_right = st.columns([4, 1])
            with hdr_left:
                if section:
                    # Bold the last part of the breadcrumb (the section name)
                    parts = [p.strip() for p in section.split(">")]
                    if len(parts) > 1:
                        crumb = " › ".join(parts[:-1])
                        st.markdown(f"{crumb} › **{parts[-1]}**")
                    else:
                        st.markdown(f"**{section}**")
            with hdr_right:
                st.markdown(
                    f"<div style='text-align:right;color:{colour}'>"
                    f"<b>{norm_score:.0%}</b><br>"
                    f"<small>{rel_label}</small></div>",
                    unsafe_allow_html=True,
                )

            # Content preview
            preview = content[:600]
            if len(content) > 600:
                preview += "…"
            st.markdown(preview)

            # Source line
            st.caption(_source_display(r))

            # Tags
            if tags:
                st.caption("Tags: " + "  ".join(f"`{t}`" for t in tags))

            # Full content in expander
            if len(content) > 600:
                with st.expander("Read full section"):
                    st.markdown(content)
