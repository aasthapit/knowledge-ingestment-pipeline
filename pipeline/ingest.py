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
    quality_threshold: float | None = None,
    auto_push: bool = False,
    kb_name: str = "default",
) -> dict:
    """
    Ingest any supported document format through the full pipeline.

    Steps
    -----
    1. Convert with Docling (PDF, DOCX, PPTX, HTML, URL, Markdown).
    2. Assess structural quality.
    3. Auto-tag from headings if quality passes.
    4. Chunk with HybridChunker (preserving citation metadata per chunk).
    5. Stage all chunks in Redis (StagingStore).
       - Quality PASS → status ``approved``
       - Quality FAIL → status ``pending_review`` (requires human decision)
    6. If ``auto_push=True`` and quality passes, immediately embed and push
       to the configured vector backend.

    Parameters
    ----------
    source:
        File path or HTTP/HTTPS URL.
    extra_tags:
        Additional tags merged with the auto-generated ones.
    quality_threshold:
        Override ``settings.quality_threshold`` for this call.
    auto_push:
        If True, auto-approved documents are embedded and pushed immediately
        without waiting for an explicit ``review push`` command.
    kb_name:
        Logical knowledge base name for ledger grouping and drift tracking.
        Defaults to ``"default"``.

    Returns
    -------
    dict
        ``{doc_id, title, quality_score, quality_passed, status,
           chunk_count, tags, flags}``
    """
    from pipeline.converter import convert_document
    from pipeline.quality import assess_quality, extract_tags
    from pipeline.review import push_approved

    threshold = quality_threshold if quality_threshold is not None else settings.quality_threshold

    # 1 — Convert
    converted = convert_document(source)
    citation = converted.citation

    # 2 — Quality assessment
    result = assess_quality(converted)

    # 3 — Determine tags
    tags = list(extra_tags or [])
    if result.passed:
        # Merge auto-suggested tags (quality assessor already extracted them)
        for t in result.suggested_tags:
            if t not in tags:
                tags.append(t)

    # 4 — Chunk
    chunks = chunker.chunk_docling(
        converted,
        tags=tags,
        max_tokens=settings.docling_max_tokens,
    )

    if not chunks:
        logger.warning("No chunks produced from '%s'.", source)

    # 5 — Stage in Redis
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(source)))
    status = "approved" if result.passed else "pending_review"

    meta = {
        "doc_id":           doc_id,
        "title":            citation.title,
        "source_path":      citation.source_path,
        "source_type":      citation.source_type,
        "author":           citation.author or "",
        "created_date":     citation.created_date or "",
        "url":              citation.url or "",
        "page_count":       citation.page_count or 0,
        "quality_score":    round(result.score, 4),
        "quality_passed":   int(result.passed),
        "quality_flags":    json.dumps(result.flags),
        "suggested_tags":   json.dumps(tags),
        "chunk_count":      len(chunks),
        "status":           status,
        "kb_name":          kb_name,
    }

    staging = mongo_store.get_staging()
    staging.enqueue(doc_id, meta, [c.to_dict() for c in chunks])

    if result.passed:
        staging.approve(doc_id)
        logger.info(
            "Auto-approved '%s' (score=%.2f) — %d chunks staged.",
            citation.title, result.score, len(chunks),
        )
    else:
        logger.warning(
            "Flagged for review: '%s' (score=%.2f). "
            "Run 'review list' to inspect.",
            citation.title, result.score,
        )

    # 6 — Optional immediate push
    if auto_push and result.passed:
        push_result = push_approved(doc_id=doc_id)
        logger.info("Auto-pushed: %s", push_result)


    return {
        "doc_id":          doc_id,
        "title":           citation.title,
        "quality_score":   round(result.score, 4),
        "quality_passed":  result.passed,
        "status":          status,
        "chunk_count":     len(chunks),
        "tags":            tags,
        "flags":           result.flags,
    }


# ---------------------------------------------------------------------------
# JSONL import path
# ---------------------------------------------------------------------------

def ingest_jsonl(
    source,
    batch_name: str | None = None,
    extra_tags: list[str] | None = None,
    progress_cb=None,
    kb_name: str = "default",
    usecase_id: str | None = None,
    agent_filter: str | None = None,
    require_usecase: bool = False,
    field_map: dict[str, str] | None = None,
    tags_static: list[str] | None = None,
    section_join: str = " > ",
) -> dict:
    """
    Import a JSONL chunk file into the staging area.

    Supports both the crawler schema (``text`` + ``page_url``) and the
    pipeline exporter schema (``content`` + ``source``).  Auto-detects
    which schema is in use from the first record.

    Pre-computed embeddings in pipeline-schema files are reused automatically
    so you don't pay for re-embedding.

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
    kb_name:
        Logical knowledge base name for ledger grouping and drift tracking.
    usecase_id:
        Business use-case identifier. Required for crawler-schema files or
        when ``require_usecase=True``.
    agent_filter:
        Target agent/persona identifier.
    require_usecase:
        When True, raise ``ValueError`` if usecase_id or agent_filter are missing.

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
        kb_name=kb_name,
        usecase_id=usecase_id,
        agent_filter=agent_filter,
        require_usecase=require_usecase,
        field_map=field_map,
        tags_static=tags_static,
        section_join=section_join,
    )


def export_usecase_jsonl(
    usecase_id: str,
    agent_filter: str,
    output_path: str | Path | None = None,
    status: str | None = "pushed",
) -> dict:
    """
    Export all chunks for a (usecase_id, agent_filter) pair to a JSONL file.

    Parameters
    ----------
    usecase_id:    Business use-case identifier.
    agent_filter:  Target agent/persona identifier.
    output_path:   Write path. Defaults to a timestamped file in JSONL_OUTPUT_DIR.
    status:        Filter docs by status (default: ``"pushed"``). Pass ``None``
                   to include all statuses (approved + pending + pushed).

    Returns
    -------
    dict
        ``{"chunk_count": int, "output_path": str}``
    """
    from pipeline.exporter import export_chunks_as_jsonl

    staging = mongo_store.get_staging()
    chunk_dicts = staging.get_chunks_by_usecase(usecase_id, agent_filter, status=status)

    if not chunk_dicts:
        raise ValueError(
            f"No chunks found for usecase_id={usecase_id!r}, "
            f"agent_filter={agent_filter!r}, status={status!r}."
        )

    out = export_chunks_as_jsonl(chunk_dicts, output_path=output_path)
    logger.info(
        "Exported %d chunks for usecase=%s agent=%s → %s",
        len(chunk_dicts), usecase_id, agent_filter, out,
    )
    return {"chunk_count": len(chunk_dicts), "output_path": str(out)}


# ---------------------------------------------------------------------------
# Legacy path — Markdown only, direct to Redis, no staging/review
# ---------------------------------------------------------------------------

def query_vectorstore(
    question: str,
    top_k: int = 5,
    tag_filter: list[str] | None = None,
    source_type: str | None = None,
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
    """
    settings.validate()
    vectors = embedder.embed_texts([question])
    vec = vectors[0]
    backend = settings.vector_backend

    if backend == "qdrant":
        from pipeline import qdrant_store
        results = qdrant_store.search(
            vec, top_k=top_k,
            tag_filter=tag_filter,
            source_type_filter=source_type,
        )
        # Qdrant cosine similarity: 1 = identical, already in ~[0,1] for text
        for r in results:
            r["normalized_score"] = round(max(0.0, float(r.get("score", 0))), 4)
    else:
        # Redis cosine distance: 0 = identical, ~[0,2] range
        from pipeline import redis_store
        redis_tag_filter = None
        if tag_filter:
            redis_tag_filter = "@tags:{" + "|".join(tag_filter) + "}"
        results = redis_store.search(vec, top_k=top_k, tag_filter=redis_tag_filter)
        for r in results:
            raw = float(r.get("score", 1.0))
            r["normalized_score"] = round(max(0.0, min(1.0, 1.0 - raw)), 4)
            # Normalise tags from Redis string → list
            tags_raw = r.get("tags", "")
            if isinstance(tags_raw, str):
                r["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()]
            # Qdrant-style citation stub so UI code is uniform
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
