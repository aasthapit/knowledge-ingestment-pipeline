"""
ingest.py
High-level orchestration for the knowledge ingestion pipeline.

Two entry points are available:

``ingest_document(source, …)``
    New path — uses Docling to convert any supported format (PDF, DOCX,
    PPTX, HTML, URLs, Markdown).  Runs quality assessment, auto-tags
    well-structured documents, and stages everything in Redis for review
    before final push to the vector store.

``ingest_file / ingest_directory``
    Legacy path — Markdown-only, direct to Redis (no staging/review).
    Kept for backward compatibility.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from pipeline import chunker, embedder, exporter, tagger, mongo_store
# redis_store is imported lazily inside functions that need it so that the
# Redis Search module is not required when only using the MongoDB / JSONL path.
from pipeline.config import settings

logger = logging.getLogger(__name__)


def ingest_file(
    path: str | Path,
    tags: list[str] | None = None,
    export_jsonl: bool = True,
    skip_redis: bool = False,
    jsonl_path: str | Path | None = None,
) -> list[chunker.Chunk]:
    """
    Full pipeline for a single markdown file.

    Steps
    -----
    1. Chunk the document by headings (respecting front-matter tags).
    2. Merge any *extra* CLI tags.
    3. Embed all chunks.
    4. Optionally export to JSONL.
    5. Upsert into Redis (unless skip_redis=True).

    Returns the list of :class:`~pipeline.chunker.Chunk` objects.
    """
    settings.validate()
    path = Path(path)
    extra_tags = tags or []

    logger.info("Chunking: %s", path)
    chunks = chunker.chunk_markdown_file(
        path,
        extra_tags=extra_tags,
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap_chars,
    )
    logger.info("  → %d chunks", len(chunks))

    if not chunks:
        logger.warning("No chunks produced from %s", path)
        return chunks

    logger.info("Embedding %d chunks …", len(chunks))
    vectors = embedder.embed_chunks(chunks)

    if export_jsonl:
        out = exporter.export_jsonl(chunks, embeddings=vectors, output_path=jsonl_path)
        logger.info("JSONL saved: %s", out)

    if not skip_redis:
        from pipeline import redis_store
        redis_store.create_index()
        redis_store.upsert_chunks(chunks, vectors)

    return chunks


def ingest_directory(
    directory: str | Path,
    glob: str = "**/*.md",
    tags: list[str] | None = None,
    export_jsonl: bool = True,
    skip_redis: bool = False,
) -> list[chunker.Chunk]:
    """
    Ingest all markdown files matching *glob* under *directory*.
    All chunks are batched into a single JSONL file per run.
    """
    settings.validate()
    directory = Path(directory)
    md_files = sorted(directory.glob(glob))

    if not md_files:
        logger.warning("No files matched '%s' under %s", glob, directory)
        return []

    all_chunks: list[chunker.Chunk] = []
    all_vectors: list[list[float]] = []

    for md_file in md_files:
        logger.info("Processing: %s", md_file)
        file_chunks = chunker.chunk_markdown_file(
            md_file,
            extra_tags=tags or [],
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap_chars,
        )
        if not file_chunks:
            continue
        vectors = embedder.embed_chunks(file_chunks)
        all_chunks.extend(file_chunks)
        all_vectors.extend(vectors)

    logger.info("Total chunks across all files: %d", len(all_chunks))

    if all_chunks:
        if export_jsonl:
            exporter.export_jsonl(all_chunks, embeddings=all_vectors)
        if not skip_redis:
            from pipeline import redis_store
            redis_store.create_index()
            redis_store.upsert_chunks(all_chunks, all_vectors)

    return all_chunks


# ---------------------------------------------------------------------------
# New path: Docling-powered multi-format ingestion with quality gate
# ---------------------------------------------------------------------------

def ingest_document(
    source: str | Path,
    extra_tags: list[str] | None = None,
    auto_push: bool = False,
    kb_id: str | None = None,
    corpus_id: str | None = None,
    manifest_id: str | None = None,
    chunk_strategy: str | None = None,
    chunk_max_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
) -> dict:
    """
    Ingest any supported document format through the full pipeline.

    Steps
    -----
    1. Convert with Docling (PDF, DOCX, PPTX, HTML, URL, Markdown).
    2. Auto-tag from headings.
    3. Chunk with HybridChunker (preserving citation metadata per chunk).
    4. Assess quality per chunk: size, boilerplate, and document recency.
       - All chunks clean + not stale → status ``approved``
       - Any flag → status ``pending_review`` (requires human decision)
    5. Stage all chunks in MongoDB.
    6. If ``auto_push=True`` and quality passes, immediately embed and push.

    Parameters
    ----------
    source:
        File path or HTTP/HTTPS URL.
    extra_tags:
        Additional tags merged with the auto-generated ones.
    auto_push:
        If True, auto-approved documents are embedded and pushed immediately.
        Requires ``corpus_id`` to resolve the vector store target.
    kb_id:
        Knowledge Base this document belongs to.
    corpus_id:
        Required when ``auto_push=True`` — provides usecase/agent context
        and vector store target.

    Returns
    -------
    dict
        ``{doc_id, title, quality_score, quality_passed, status,
           chunk_count, tags, flags, age_days, is_stale}``
    """
    from pipeline.converter import convert_document
    from pipeline.quality import assess_document, extract_tags
    from pipeline.review import push_approved

    # 1 — Convert
    converted = convert_document(source)
    citation = converted.citation

    # 2 — Auto-tag from headings (always, not gated on quality)
    tags = list(extra_tags or [])
    for t in extract_tags(converted.markdown, title=citation.title):
        if t not in tags:
            tags.append(t)

    # 3 — Chunk
    effective_strategy = chunk_strategy or "heading"
    effective_max_chars = chunk_max_chars or settings.chunk_max_chars
    effective_overlap = chunk_overlap_chars or settings.chunk_overlap_chars

    if effective_strategy == "character":
        chunks = chunker.chunk_character(
            converted.markdown,
            source=str(source),
            extra_tags=tags,
            max_chars=effective_max_chars,
            overlap=effective_overlap,
        )
        for c in chunks:
            c.metadata["citation"] = citation.to_dict()
    else:
        chunks = chunker.chunk_docling(
            converted,
            tags=tags,
            max_tokens=settings.docling_max_tokens,
        )

    if not chunks:
        logger.warning("No chunks produced from '%s'.", source)

    # 4 — Quality assessment (chunk-aware + recency)
    result = assess_document(chunks, citation)

    # Annotate each chunk dict with its per-chunk quality flags
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        d = chunk.to_dict()
        per_chunk = result.chunk_flags.get(i)
        if per_chunk:
            d.setdefault("metadata", {})["quality_flags"] = per_chunk
        chunk_dicts.append(d)

    # 5 — Stage in MongoDB
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(source)))
    status = "approved" if result.passed else "pending_review"

    meta = {
        "doc_id":             doc_id,
        "title":              citation.title,
        "source_path":        citation.source_path,
        "source_type":        citation.source_type,
        "author":             citation.author or "",
        "created_date":       citation.created_date or "",
        "url":                citation.url or "",
        "page_count":         citation.page_count or 0,
        "quality_score":      round(result.score, 4),
        "quality_passed":     int(result.passed),
        "quality_flags":      json.dumps(result.flags),
        "suggested_tags":     json.dumps(tags),
        "chunk_count":        len(chunks),
        "status":             status,
        "kb_id":              kb_id or None,
        "age_days":           result.age_days,
        "is_stale":           result.is_stale,
        "chunks_too_short":   result.chunks_too_short,
        "chunks_too_long":    result.chunks_too_long,
        "chunks_boilerplate": result.chunks_boilerplate,
    }

    staging = mongo_store.get_staging()
    staging.enqueue(doc_id, meta, chunk_dicts)

    if result.passed:
        staging.approve(doc_id)
        logger.info(
            "Auto-approved '%s' (score=%.2f) — %d chunks staged.",
            citation.title, result.score, len(chunks),
        )
    else:
        logger.warning(
            "Flagged for review: '%s'. Issues: %s",
            citation.title, " | ".join(result.flags),
        )

    if manifest_id:
        try:
            import hashlib
            from pipeline.manifests import get_manifest_manager
            version_id = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            get_manifest_manager().add_entry(
                manifest_id=manifest_id,
                doc_id=doc_id,
                object_id=doc_id,
                file_id=doc_id,
                version_id=version_id,
                source_type=citation.source_type,
                source_ref=str(source),
                title=citation.title,
                kb_id=kb_id,
                status=status,
            )
        except Exception as _manifest_exc:
            logger.warning("Could not associate doc with manifest: %s", _manifest_exc)

    # 6 — Optional immediate push (requires corpus_id for context)
    if auto_push and result.passed and corpus_id:
        push_result = push_approved(corpus_id=corpus_id, doc_id=doc_id)
        logger.info("Auto-pushed: %s", push_result)

    return {
        "doc_id":           doc_id,
        "title":            citation.title,
        "quality_score":    round(result.score, 4),
        "quality_passed":   result.passed,
        "status":           status,
        "chunk_count":      len(chunks),
        "tags":             tags,
        "flags":            result.flags,
        "age_days":         result.age_days,
        "is_stale":         result.is_stale,
    }


# ---------------------------------------------------------------------------
# JSONL import path
# ---------------------------------------------------------------------------

def ingest_jsonl(
    source,
    batch_name: str | None = None,
    extra_tags: list[str] | None = None,
    progress_cb=None,
    kb_id: str | None = None,
    field_map: dict[str, str] | None = None,
    tags_static: list[str] | None = None,
    section_join: str = " > ",
    manifest_id: str | None = None,
) -> dict:
    """
    Import a JSONL chunk file into the staging area under a specific Knowledge Base.

    Supports both the crawler schema (``text`` + ``page_url``) and the
    pipeline exporter schema (``content`` + ``source``).  Auto-detects
    which schema is in use from the first record.

    Pre-computed embeddings in pipeline-schema files are reused automatically.

    Parameters
    ----------
    source:
        File path, URL string, or file-like object (e.g. Streamlit BytesIO).
    batch_name:
        Human-readable label for this import batch.
    extra_tags:
        Additional tags applied to every chunk.
    progress_cb:
        Optional ``progress_cb(done: int, total: int)`` for progress updates.
    kb_id:
        Knowledge Base this import belongs to.

    Returns
    -------
    dict
        ``{doc_id, batch_name, schema, total_chunks, unique_sources,
           has_embeddings, has_partial_embeddings}``
    """
    from pipeline.jsonl_importer import import_jsonl as _import

    return _import(
        source=source,
        batch_name=batch_name,
        extra_tags=extra_tags,
        progress_cb=progress_cb,
        kb_id=kb_id,
        field_map=field_map,
        tags_static=tags_static,
        section_join=section_join,
        manifest_id=manifest_id,
    )


def export_kb_jsonl(
    kb_id: str,
    output_path: str | Path | None = None,
    status: str | None = "pushed",
) -> dict:
    """
    Export all chunks for a Knowledge Base to a JSONL file.

    Parameters
    ----------
    kb_id:        Knowledge Base ID.
    output_path:  Write path. Defaults to a timestamped file in JSONL_OUTPUT_DIR.
    status:       Filter docs by status (default: ``"pushed"``).

    Returns
    -------
    dict
        ``{"chunk_count": int, "output_path": str}``
    """
    from pipeline.exporter import export_chunks_as_jsonl

    staging = mongo_store.get_staging()
    chunk_dicts = staging.get_chunks_by_kb(kb_id, status=status)

    if not chunk_dicts:
        raise ValueError(f"No chunks found for kb_id={kb_id!r}, status={status!r}.")

    out = export_chunks_as_jsonl(chunk_dicts, output_path=output_path)
    logger.info("Exported %d chunks for kb_id=%s → %s", len(chunk_dicts), kb_id, out)
    return {"chunk_count": len(chunk_dicts), "output_path": str(out)}


# ---------------------------------------------------------------------------
# Legacy path — Markdown only, direct to Redis, no staging/review
# ---------------------------------------------------------------------------

def query_vectorstore(
    question: str,
    top_k: int = 5,
    tag_filter: list[str] | None = None,
    source_type: str | None = None,
    vs_id: str | None = None,
    agent_filter: str | None = None,
) -> list[dict]:
    """
    Embed *question* and search whichever vector backend is configured.

    Normalises scores to a 0–1 similarity value (higher = more relevant)
    regardless of backend, and adds a ``normalized_score`` key to every
    result dict for consistent UI display.

    Parameters
    ----------
    tag_filter:   List of tag strings — at least one must match.
    source_type:  Restrict to a specific source type (e.g. ``"pdf"``).
    vs_id:        Target vector store ID. Defaults to the built-in Redis.
    agent_filter: Optional agent label for downstream filtering.
    """
    settings.validate()

    all_tags = list(tag_filter or [])
    if source_type:
        all_tags.append(source_type)
    redis_tag_filter = "@tags:{" + "|".join(all_tags) + "}" if all_tags else None

    if vs_id:
        from pipeline.mongo_store import get_vs_config_store
        from pipeline.vector_store import get_vector_store_client
        vs_config = get_vs_config_store().get(vs_id) or {}
        client = get_vector_store_client(vs_config)
        if client.handles_own_embedding:
            # Text-based backend (e.g. Tachyon) — skip local embedding, pass raw text.
            results = client.search(
                query_vector=[],
                top_k=top_k,
                tag_filter=redis_tag_filter,
                agent_filter=agent_filter,
                query_text=question,
            )
        else:
            vectors = embedder.embed_texts([question])
            results = client.search(
                vectors[0], top_k=top_k, tag_filter=redis_tag_filter, agent_filter=agent_filter
            )
    else:
        # Default: built-in Redis
        from pipeline import redis_store
        vectors = embedder.embed_texts([question])
        results = redis_store.search(vectors[0], top_k=top_k, tag_filter=redis_tag_filter)

    for r in results:
        raw = float(r.get("score", 1.0))
        r["normalized_score"] = round(max(0.0, min(1.0, 1.0 - raw)), 4)
        tags_raw = r.get("tags", "")
        if isinstance(tags_raw, str):
            r["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()]
        if "citation" not in r:
            r["citation"] = {
                "source_path": r.get("source", ""),
                "source_type": "",
                "title": r.get("title", ""),
            }

    return results


def query(
    question: str,
    top_k: int = 5,
    tag_filter: str | None = None,
) -> list[dict]:
    """
    Embed *question* and return the top-k most similar chunks from Redis.

    Parameters
    ----------
    question:   Natural language question / search string.
    top_k:      Number of results.
    tag_filter: Optional RediSearch tag filter, e.g. ``"@tags:{python}"``.
    """
    settings.validate()
    from pipeline import redis_store
    vectors = embedder.embed_texts([question])
    results = redis_store.search(vectors[0], top_k=top_k, tag_filter=tag_filter)
    return results
