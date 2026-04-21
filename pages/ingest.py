"""Add Document — upload a file or provide a URL."""
import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Helpers ───────────────────────────────────────────────────────────────────

ACCEPTED_TYPES = ["pdf", "docx", "pptx", "html", "htm", "txt", "md"]

TYPE_LABELS = {
    "pdf":  "PDF",
    "docx": "Word",
    "pptx": "PowerPoint",
    "html": "HTML",
    "htm":  "HTML",
    "txt":  "Text",
    "md":   "Markdown",
}


def _score_bar(score: float) -> str:
    """Emoji progress bar for quality score."""
    filled = round(score * 10)
    return "█" * filled + "░" * (10 - filled)


def _relevance_label(score: float) -> tuple[str, str]:
    """(emoji, label) for quality score."""
    if score >= 0.80:
        return "✅", "Excellent"
    if score >= 0.60:
        return "🟡", "Good"
    if score >= 0.40:
        return "🟠", "Fair — review recommended"
    return "🔴", "Poor — review required"


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("➕ Add a Document")
st.caption(
    "Upload a file, paste a web address, or import a JSONL bulk export. "
    "The system will read it, check its quality, and queue it for the knowledge base."
)

# ── Source input ──────────────────────────────────────────────────────────────
tab_file, tab_url, tab_jsonl = st.tabs(["📄 Upload a File", "🔗 From a Web Address", "📦 Bulk JSONL Import"])

uploaded_file = None
url_input = ""
jsonl_file = None

with tab_file:
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=ACCEPTED_TYPES,
        help="Supported formats: PDF, Word (.docx), PowerPoint (.pptx), HTML, plain text, Markdown",
        label_visibility="collapsed",
    )
    if uploaded_file:
        ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
        st.caption(f"📎 **{uploaded_file.name}** · {TYPE_LABELS.get(ext, ext.upper())} · {uploaded_file.size / 1024:.0f} KB")

with tab_url:
    url_input = st.text_input(
        "Web address",
        placeholder="https://docs.example.com/guide",
        label_visibility="collapsed",
    )
    if url_input:
        st.caption(f"🔗 {url_input}")

with tab_jsonl:
    st.markdown(
        "Import a bulk JSONL file — for example, a dataset previously exported from "
        "this pipeline or crawled with `crawl_ocp_docs.py`."
    )
    jsonl_file = st.file_uploader(
        "Choose a .jsonl file",
        type=["jsonl"],
        help="Each line must be a valid JSON object. Supports the crawler schema (text + page_url) and the pipeline export schema (content + source).",
        label_visibility="collapsed",
        key="jsonl_uploader",
    )

    if jsonl_file:
        # ── Preview ──────────────────────────────────────────────────────────
        try:
            from pipeline.jsonl_importer import peek_jsonl
            preview = peek_jsonl(jsonl_file, n=5)

            schema_label = {
                "crawler":  "Crawler format  (page_url · text · section_breadcrumbs)",
                "pipeline": "Pipeline export format  (source · content · section · tags)",
                "unknown":  "Unknown / custom schema",
            }.get(preview["schema"], preview["schema"])

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Schema detected", preview["schema"].title())
            col_b.metric("Unique sources (preview)", preview["unique_sources"])
            col_c.metric("Pre-computed embeddings", "Yes" if preview["has_embeddings"] else "No",
                         help="If Yes, the file already contains embedding vectors — no API calls needed to push.")
            st.caption(schema_label)

            with st.expander("Preview first 5 records"):
                for i, chunk in enumerate(preview["sample_chunks"], 1):
                    st.markdown(f"**{i}.** *{chunk.section or chunk.title}*")
                    st.text(chunk.content[:300] + ("…" if len(chunk.content) > 300 else ""))
                    if i < len(preview["sample_chunks"]):
                        st.divider()

        except Exception as exc:
            st.warning(f"Could not preview file: {exc}")
            preview = None

        st.divider()

        # ── JSONL-specific controls ───────────────────────────────────────────
        jsonl_tags_raw = st.text_input(
            "Extra tags  *(optional)*",
            placeholder="openshift, internal, 4.18",
            help="Comma-separated tags applied to every chunk in this import.",
            key="jsonl_tags",
        )
        jsonl_tags = [t.strip() for t in jsonl_tags_raw.split(",") if t.strip()]

        if not preview or not preview.get("has_embeddings"):
            st.info(
                "⚡ This file has no pre-computed embeddings. "
                "Clicking **Import** will stage the chunks; "
                "you'll embed and push them from the Review Queue. "
                "For a 20 000-chunk file this takes a few minutes and uses your OpenAI API.",
                icon="ℹ️",
            )

        jsonl_kb_name = st.text_input(
            "Knowledge base name",
            value="default",
            placeholder="e.g. openshift-4.18, internal-docs",
            help="Logical name used for ledger grouping and drift tracking.",
            key="jsonl_kb_name",
        )

        if st.button("📦  Import JSONL", type="primary", use_container_width=True, key="jsonl_submit"):
            import io as _io
            jsonl_file.seek(0)
            file_bytes = _io.BytesIO(jsonl_file.read())
            file_bytes.name = jsonl_file.name

            progress_bar = st.progress(0.0, text="Parsing…")

            def _progress(done: int, total: int) -> None:
                if total and total > 0:
                    progress_bar.progress(done / total, text=f"Parsed {done:,} of {total:,} chunks…")
                else:
                    progress_bar.progress(min(done / 25000, 0.99), text=f"Parsed {done:,} chunks…")

            try:
                from pipeline.ingest import ingest_jsonl
                result = ingest_jsonl(
                    source=file_bytes,
                    batch_name=jsonl_file.name,
                    extra_tags=jsonl_tags,
                    progress_cb=_progress,
                    kb_name=jsonl_kb_name.strip() or "default",
                )
                progress_bar.progress(1.0, text="Done!")
                st.session_state["last_jsonl_import"] = result
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    # ── JSONL result card ─────────────────────────────────────────────────────
    if "last_jsonl_import" in st.session_state:
        r = st.session_state["last_jsonl_import"]
        st.success(
            f"**{r['batch_name']}** imported successfully — "
            f"**{r['total_chunks']:,}** sections from **{r['unique_sources']:,}** source{'s' if r['unique_sources'] != 1 else ''}."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Sections imported", f"{r['total_chunks']:,}")
        m2.metric("Unique sources",    f"{r['unique_sources']:,}")
        m3.metric("Schema",            r["schema"].title())

        embed_note = ""
        if r.get("has_embeddings"):
            embed_note = "✅ Pre-computed embeddings reused — ready to push immediately."
        elif r.get("has_partial_embeddings"):
            embed_note = "⚠️ Partial embeddings — missing vectors will be computed on push."
        else:
            embed_note = "ℹ️ No embeddings — sections will be embedded when you push from the Review Queue."
        st.info(embed_note)
        st.info(f"Go to **Review Queue** → **Push to Knowledge Base** to make these sections searchable.\n\nBatch ID: `{r['doc_id']}`")

        if st.button("Clear result", key="clear_jsonl"):
            del st.session_state["last_jsonl_import"]
            st.rerun()

# ── Tags / Advanced / Submit — only for File and URL tabs ────────────────────
# (JSONL tab has its own self-contained flow above)
if jsonl_file:
    st.stop()

# ── Tags / KB name ────────────────────────────────────────────────────────────
st.divider()
col_tags, col_kb = st.columns([3, 1])
with col_tags:
    tags_raw = st.text_input(
        "Tags  *(optional)*",
        placeholder="finance, q1-2024, internal",
        help="Comma-separated keywords that describe this document. These will appear in search results.",
    )
with col_kb:
    kb_name = st.text_input(
        "Knowledge base",
        value="default",
        placeholder="default",
        help="Logical knowledge base name for grouping and drift tracking.",
    )
tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
if tags:
    st.caption("Will be tagged: " + "  ".join(f"`{t}`" for t in tags))

# ── Advanced ──────────────────────────────────────────────────────────────────
with st.expander("⚙️  Advanced options"):
    quality_threshold = st.slider(
        "Quality threshold",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help=(
            "Documents scoring below this are sent for manual review instead of "
            "being auto-approved. Lower = more permissive, higher = stricter."
        ),
    )
    auto_push = st.toggle(
        "Push directly to knowledge base if quality is good enough",
        value=False,
        help=(
            "If on, high-quality documents skip the review step and become "
            "searchable immediately. If off, you approve them in the Review Queue."
        ),
    )

st.divider()

# ── Submit ────────────────────────────────────────────────────────────────────
source_ready = uploaded_file is not None or bool(url_input.strip())

if st.button(
    "Add to Knowledge Base",
    type="primary",
    use_container_width=True,
    disabled=not source_ready,
):
    tmp_path = None
    try:
        with st.status("Processing document…", expanded=True) as proc_status:

            if uploaded_file:
                suffix = Path(uploaded_file.name).suffix or ".bin"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                st.write(f"Converting **{uploaded_file.name}** …")
                source_arg = tmp_path
            else:
                st.write(f"Fetching **{url_input}** …")
                source_arg = url_input.strip()

            from pipeline.ingest import ingest_document
            result = ingest_document(
                source=source_arg,
                extra_tags=tags,
                quality_threshold=quality_threshold,
                auto_push=auto_push,
                kb_name=kb_name.strip() or "default",
            )

            if result["quality_passed"]:
                label = (
                    "✅ Added to knowledge base"
                    if auto_push else
                    "✅ Approved — ready to push to knowledge base"
                )
                proc_status.update(label=label, state="complete")
            else:
                proc_status.update(
                    label="📋 Queued for review",
                    state="complete",
                    expanded=True,
                )

            st.session_state["last_ingest"] = result

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

elif not source_ready:
    st.caption("Upload a file or enter a web address to get started.")

# ── Result card ───────────────────────────────────────────────────────────────
if "last_ingest" in st.session_state:
    result = st.session_state["last_ingest"]
    score  = float(result.get("quality_score", 0))
    passed = result.get("quality_passed", False)
    emoji, label = _relevance_label(score)

    st.divider()

    with st.container(border=True):
        # Header
        if passed:
            st.success(
                f"**{result['title']}** was added successfully."
                + (" It's now searchable." if auto_push else " Approve it in the Review Queue to make it searchable.")
            )
        else:
            st.warning(
                f"**{result['title']}** has been queued for review. "
                "Visit the **Review Queue** to inspect it and decide whether to approve it."
            )

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Quality Score", f"{score:.0%}", help="How well-structured this document is")
        m2.metric("Sections found", result.get("chunk_count", 0), help="Number of logical sections the document was split into")
        m3.metric("Tags applied", len(result.get("tags", [])))

        # Quality bar
        st.markdown(
            f"**Quality** {emoji} {label}  \n"
            f"`{_score_bar(score)}`  {score:.0%}"
        )

        # Tags
        if result.get("tags"):
            st.write("**Tags:** " + "  ".join(f"`{t}`" for t in result["tags"]))

        # Quality flags
        if result.get("flags"):
            with st.expander("📋 Quality notes — why this document needs review"):
                for flag in result["flags"]:
                    st.write(f"• {flag}")

        # Next-step guidance
        if not passed:
            st.info(
                f"📋 Go to [Review Queue](review) to inspect, approve, or reject this document.\n\n"
                f"Document reference: `{result['doc_id']}`"
            )
        elif not auto_push:
            st.info(
                "✅ This document is approved. Go to [Review Queue](review) and click "
                "**Push to Knowledge Base** to make it searchable."
            )

    # Clear button
    if st.button("Clear result", type="secondary"):
        del st.session_state["last_ingest"]
        st.rerun()
