"""Add Document — upload a file or provide a URL, scoped to a Knowledge Base."""
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


@st.cache_data(ttl=30)
def _load_kbs() -> list[dict]:
    from pipeline.mongo_store import get_kb_store
    return get_kb_store().list_all()


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("➕ Add a Document")
st.caption(
    "Upload a file, paste a web address, or import a JSONL bulk export into a Knowledge Base. "
    "The system will read it, check its quality, and queue it for review."
)

# ── Knowledge Base selector ───────────────────────────────────────────────────

try:
    all_kbs = _load_kbs()
except Exception as _exc:
    all_kbs = []
    st.warning(f"Could not load Knowledge Bases: {_exc}")

kb_options = {kb["name"]: kb["kb_id"] for kb in all_kbs}

if not kb_options:
    st.warning(
        "No Knowledge Bases found. "
        "Create one on the **Knowledge Bases** page before importing documents."
    )
    kb_options = {"(create a KB first)": None}

selected_kb_name = st.selectbox(
    "Knowledge Base *",
    options=list(kb_options.keys()),
    help="All documents imported here will be associated with this Knowledge Base.",
)
selected_kb_id = kb_options.get(selected_kb_name)

if not selected_kb_id:
    st.stop()

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
        "this pipeline or crawled with an external tool."
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

        # ── Field mapper ──────────────────────────────────────────────────────
        _schema_unknown = not preview or preview.get("schema") == "unknown"
        _avail_keys = preview.get("available_keys", []) if preview else []
        field_map_inputs: dict[str, str] = {}

        _PIPELINE_FIELDS = [
            ("content",  "Content *",  "The main text body of each chunk — required."),
            ("source",   "Source",     "URL or file path that identifies the document."),
            ("title",    "Title",      "Document or page title."),
            ("section",  "Section",    "Heading path / breadcrumb (can be a list)."),
            ("chunk_id", "Chunk ID",   "Unique identifier per chunk; auto-generated if omitted."),
            ("tags",     "Tags",       "List or comma-separated string of tags."),
            ("embedding","Embedding",  "Pre-computed float vector; skips re-embedding if present."),
        ]

        with st.expander(
            "Field mapper — convert key names to pipeline fields",
            expanded=_schema_unknown,
        ):
            if not _avail_keys:
                st.caption("Upload a JSONL file to see available keys.")
            else:
                st.caption(
                    "Map your JSONL keys to the fields this pipeline expects. "
                    "Leave a field as **(none)** to skip it."
                )
                _none_opt = "(none)"
                _key_opts  = [_none_opt] + _avail_keys

                field_map_inputs = {}
                cols = st.columns(2)
                for i, (fld, label, tip) in enumerate(_PIPELINE_FIELDS):
                    with cols[i % 2]:
                        _default = fld if fld in _avail_keys else _none_opt
                        chosen = st.selectbox(
                            label,
                            _key_opts,
                            index=_key_opts.index(_default),
                            help=tip,
                            key=f"fm_{fld}",
                        )
                        if chosen != _none_opt:
                            field_map_inputs[fld] = chosen

                if field_map_inputs and preview and preview.get("sample_records"):
                    st.divider()
                    st.caption("Preview — first record through your mapping:")
                    try:
                        from pipeline.jsonl_importer import peek_jsonl as _peek
                        _mapped_preview = _peek(jsonl_file, n=1, field_map=field_map_inputs)
                        if _mapped_preview["sample_chunks"]:
                            c = _mapped_preview["sample_chunks"][0]
                            st.markdown(f"**Title:** {c.title or '—'}")
                            st.markdown(f"**Source:** {c.source or '—'}")
                            st.markdown(f"**Section:** {c.section or '—'}")
                            st.text(c.content[:400] + ("…" if len(c.content) > 400 else ""))
                    except Exception as _prev_exc:
                        st.warning(f"Preview error: {_prev_exc}")

                st.divider()
                st.caption("Save this mapping as a reusable named schema:")
                _sc1, _sc2 = st.columns([3, 1])
                with _sc1:
                    _schema_name = st.text_input(
                        "Schema name",
                        placeholder="my_export_format",
                        label_visibility="collapsed",
                        key="fm_schema_name",
                    )
                with _sc2:
                    if st.button("Save schema", key="fm_save_schema"):
                        if not _schema_name.strip():
                            st.error("Enter a schema name.")
                        elif not field_map_inputs:
                            st.error("Map at least one field first.")
                        elif "content" not in field_map_inputs:
                            st.error("The content field is required.")
                        else:
                            try:
                                from pipeline.jsonl_importer import save_custom_schema
                                save_custom_schema(
                                    name=_schema_name.strip(),
                                    field_map=field_map_inputs,
                                )
                                st.success(f"Schema '{_schema_name.strip()}' saved.")
                            except Exception as _se:
                                st.error(f"Could not save schema: {_se}")

        _active_field_map: dict[str, str] | None = (
            field_map_inputs if field_map_inputs and "content" in field_map_inputs else None
        )

        st.divider()

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
                "you'll embed and push them from the Review Queue.",
                icon="ℹ️",
            )

        if st.button("📦  Import JSONL", type="primary", width="stretch", key="jsonl_submit"):
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
                    kb_id=selected_kb_id,
                    field_map=_active_field_map,
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
        st.info(f"Go to **Review Queue** → push to make these sections searchable.\n\nBatch ID: `{r['doc_id']}`")

        if st.button("Clear result", key="clear_jsonl"):
            del st.session_state["last_jsonl_import"]
            st.rerun()

# ── Tags / Advanced / Submit — only for File and URL tabs ────────────────────
if jsonl_file:
    st.stop()

st.divider()

tags_raw = st.text_input(
    "Tags  *(optional)*",
    placeholder="finance, q1-2024, internal",
    help="Comma-separated keywords that describe this document.",
)
tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
if tags:
    st.caption("Will be tagged: " + "  ".join(f"`{t}`" for t in tags))

with st.expander("⚙️  Advanced options"):
    auto_push = st.toggle(
        "Push directly to knowledge base if all quality checks pass",
        value=False,
        help=(
            "If on, documents with no quality flags skip the review step. "
            "Requires a corpus with this KB to be configured for the target vector store."
        ),
    )

st.divider()

source_ready = uploaded_file is not None or bool(url_input.strip())

if st.button(
    "Add to Knowledge Base",
    type="primary",
    width="stretch",
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
                auto_push=auto_push,
                kb_id=selected_kb_id,
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

    st.divider()

    with st.container(border=True):
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

        chunk_count = result.get("chunk_count", 0)
        age_days    = result.get("age_days")
        is_stale    = result.get("is_stale", False)

        if age_days is not None:
            age_label = (
                f"{age_days}d" if age_days < 30
                else f"{age_days // 30}mo" if age_days < 365
                else f"{age_days // 365}y"
            )
            age_display = f"⚠️ {age_label} — stale" if is_stale else f"✅ {age_label} old"
        else:
            age_display = "age unknown"

        m1, m2, m3 = st.columns(3)
        m1.metric("Sections found", chunk_count)
        m2.metric("Clean sections", f"{score:.0%}")
        m3.metric("Content age", age_display)

        if result.get("tags"):
            st.write("**Tags:** " + "  ".join(f"`{t}`" for t in result["tags"]))

        if result.get("flags"):
            with st.expander("📋 Quality notes — why this document needs review"):
                for flag in result["flags"]:
                    st.write(f"• {flag}")

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

    if st.button("Clear result", type="secondary"):
        del st.session_state["last_ingest"]
        st.rerun()
