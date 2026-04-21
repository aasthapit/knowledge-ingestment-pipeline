"""
jsonl_importer.py
Imports pre-existing JSONL chunk files into the pipeline staging area.

Two schemas are auto-detected:

  "crawler"
      Produced by crawl_ocp_docs.py.
      Key fields: text, page_url, page_name, section_breadcrumbs (list),
                  section_heading, chunk_id, agent_filter, usecase_id.
      No embeddings — must be embedded on push.

  "pipeline"
      Produced by pipeline/exporter.py.
      Key fields: content, source, title, section, tags, metadata,
                  chunk_id, embedding (optional — reused if present).

The entire JSONL file is staged as a single import batch in Redis
(one StagingStore entry) so the review queue stays clean.
"""
from __future__ import annotations

import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Generator

from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

def detect_schema(record: dict) -> str:
    """Return "crawler", "pipeline", or "unknown" for a single record."""
    if "text" in record and "page_url" in record:
        return "crawler"
    if "content" in record and "source" in record:
        return "pipeline"
    return "unknown"


# ---------------------------------------------------------------------------
# Record → Chunk mapping
# ---------------------------------------------------------------------------

def _section_from_crawler(rec: dict) -> str:
    """Build a breadcrumb section string from crawler fields."""
    crumbs = rec.get("section_breadcrumbs") or []
    # breadcrumbs can be a list or a JSON-encoded string
    if isinstance(crumbs, str):
        try:
            crumbs = json.loads(crumbs)
        except Exception:
            crumbs = [crumbs] if crumbs else []
    heading = rec.get("section_heading", "")
    parts = [c for c in crumbs if c] + ([heading] if heading and heading not in crumbs else [])
    title = rec.get("page_name", "")
    return " > ".join([title] + parts) if parts else title


def map_record(
    rec: dict,
    schema: str,
    extra_tags: list[str] | None = None,
) -> tuple[Chunk, list[float] | None]:
    """
    Map a single JSONL record to a (Chunk, embedding_or_None) tuple.

    The returned embedding is non-None only for pipeline-schema records
    that already contain a pre-computed ``embedding`` vector.
    """
    extra_tags = extra_tags or []
    embedding: list[float] | None = None

    if schema == "crawler":
        tags = [
            t for t in [
                rec.get("agent_filter", ""),
                rec.get("usecase_id", ""),
                rec.get("data_classification", ""),
            ] + extra_tags
            if t and t.strip()
        ]
        chunk = Chunk(
            chunk_id=rec.get("chunk_id") or rec.get("id") or str(uuid.uuid4()),
            source=rec.get("page_url") or rec.get("source_file", ""),
            title=rec.get("page_name", ""),
            section=_section_from_crawler(rec),
            content=rec.get("text", ""),
            tags=list(dict.fromkeys(tags)),   # deduplicate, preserve order
            metadata={
                "citation": {
                    "source_path": rec.get("page_url", ""),
                    "source_type": "url",
                    "title": rec.get("page_name", ""),
                    "url": rec.get("page_url"),
                    "char_count": rec.get("char_count"),
                    "word_count": rec.get("word_count"),
                }
            },
        )

    elif schema == "pipeline":
        raw_embedding = rec.get("embedding")
        if isinstance(raw_embedding, list) and raw_embedding:
            embedding = raw_embedding
        all_tags = list(rec.get("tags") or []) + extra_tags
        chunk = Chunk(
            chunk_id=rec.get("chunk_id") or str(uuid.uuid4()),
            source=rec.get("source", ""),
            title=rec.get("title", ""),
            section=rec.get("section", ""),
            content=rec.get("content", ""),
            tags=list(dict.fromkeys(all_tags)),
            metadata=rec.get("metadata") or {},
        )

    else:
        # Best-effort for unknown schema — grab any text-like field
        text = (
            rec.get("text")
            or rec.get("content")
            or rec.get("body")
            or rec.get("chunk")
            or ""
        )
        chunk = Chunk(
            chunk_id=rec.get("chunk_id") or rec.get("id") or str(uuid.uuid4()),
            source=rec.get("url") or rec.get("source") or rec.get("source_path", ""),
            title=rec.get("title") or rec.get("page_name", ""),
            section=rec.get("section") or rec.get("section_heading", ""),
            content=text,
            tags=list(rec.get("tags") or []) + extra_tags,
            metadata={},
        )

    return chunk, embedding


# ---------------------------------------------------------------------------
# Peek / preview
# ---------------------------------------------------------------------------

def peek_jsonl(
    source: str | Path | io.IOBase,
    n: int = 5,
) -> dict[str, Any]:
    """
    Read the first *n* records of a JSONL file and return a preview dict:

    .. code-block:: python

        {
            "schema":          "crawler" | "pipeline" | "unknown",
            "has_embeddings":  bool,
            "sample_records":  list[dict],   # first n raw records
            "sample_chunks":   list[Chunk],  # mapped Chunk objects
            "unique_sources":  int,          # estimated from first n records
        }
    """
    samples_raw: list[dict] = []
    sources_seen: set[str] = set()

    def _lines():
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as fh:
                for line in fh:
                    yield line
        else:
            # BytesIO / file-like object — seek to start first
            try:
                source.seek(0)
            except Exception:
                pass
            for raw_line in source:
                if isinstance(raw_line, bytes):
                    yield raw_line.decode("utf-8")
                else:
                    yield raw_line

    for i, line in enumerate(_lines()):
        if i >= n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            samples_raw.append(rec)
            src = rec.get("page_url") or rec.get("source") or rec.get("url", "")
            if src:
                sources_seen.add(src)
        except json.JSONDecodeError:
            continue

    if not samples_raw:
        return {"schema": "unknown", "has_embeddings": False, "sample_records": [], "sample_chunks": [], "unique_sources": 0}

    schema = detect_schema(samples_raw[0])
    has_embeddings = schema == "pipeline" and bool(samples_raw[0].get("embedding"))

    sample_chunks = []
    for rec in samples_raw:
        chunk, _ = map_record(rec, schema)
        sample_chunks.append(chunk)

    return {
        "schema":         schema,
        "has_embeddings": has_embeddings,
        "sample_records": samples_raw,
        "sample_chunks":  sample_chunks,
        "unique_sources": len(sources_seen),
    }


# ---------------------------------------------------------------------------
# Full import → Redis staging
# ---------------------------------------------------------------------------

def import_jsonl(
    source: str | Path | io.IOBase,
    batch_name: str | None = None,
    extra_tags: list[str] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """
    Parse an entire JSONL file and stage all chunks as a single import batch.

    The batch is stored as one entry in :class:`~pipeline.redis_store.StagingStore`
    with ``status="approved"`` (JSONL imports are treated as pre-vetted data).

    Parameters
    ----------
    source:
        File path (str / Path) or a file-like object (e.g. ``BytesIO`` from
        Streamlit's file uploader).
    batch_name:
        Human-readable label for this import. Defaults to the filename.
    extra_tags:
        Additional tags applied to every chunk.
    progress_cb:
        Optional ``progress_cb(done: int, total: int)`` called after each
        1 000-chunk batch to allow UI progress updates.
        ``total`` is ``-1`` when the file size is unknown.

    Returns
    -------
    dict
        ``{doc_id, batch_name, schema, total_chunks, unique_sources,
           has_embeddings, has_partial_embeddings}``
    """
    from pipeline import redis_store

    extra_tags = extra_tags or []
    chunks_dicts: list[dict] = []
    embeddings_map: dict[str, list[float]] = {}   # chunk_id → vector (pre-computed)

    unique_sources: set[str] = set()
    schema_detected = "unknown"
    has_any_embedding = False
    has_all_embeddings = True
    total = 0

    # Resolve batch name
    if not batch_name:
        if isinstance(source, (str, Path)):
            batch_name = Path(source).name
        else:
            batch_name = getattr(source, "name", "jsonl_import")

    # Stream + map
    def _line_iter():
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as fh:
                yield from fh
        else:
            try:
                source.seek(0)
            except Exception:
                pass
            for raw in source:
                yield raw.decode("utf-8") if isinstance(raw, bytes) else raw

    for line in _line_iter():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Detect schema from first valid record
        if schema_detected == "unknown":
            schema_detected = detect_schema(rec)

        chunk, embedding = map_record(rec, schema_detected, extra_tags=extra_tags)
        if not chunk.content.strip():
            continue

        src = rec.get("page_url") or rec.get("source") or rec.get("url", "")
        if src:
            unique_sources.add(src)

        if embedding:
            has_any_embedding = True
            embeddings_map[chunk.chunk_id] = embedding
        else:
            has_all_embeddings = False

        chunks_dicts.append(chunk.to_dict())
        total += 1

        if progress_cb and total % 1000 == 0:
            progress_cb(total, -1)

    if not chunks_dicts:
        raise ValueError("No valid records found in JSONL file.")

    # Stage as a single batch
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, batch_name + str(total)))

    meta: dict[str, Any] = {
        "title":                  f"JSONL import — {batch_name}",
        "source_path":            batch_name,
        "source_type":            "jsonl",
        "schema_type":            schema_detected,
        "chunk_count":            total,
        "unique_sources":         len(unique_sources),
        "has_embeddings":         int(has_all_embeddings),
        "has_partial_embeddings": int(has_any_embedding and not has_all_embeddings),
        "quality_score":          1.0,
        "quality_passed":         1,
        "quality_flags":          "[]",
        "status":                 "approved",
    }

    # If all embeddings are pre-computed, attach them into the chunk dicts
    if has_all_embeddings and embeddings_map:
        for cd in chunks_dicts:
            emb = embeddings_map.get(cd.get("chunk_id", ""))
            if emb:
                cd["_embedding"] = emb

    staging = redis_store.get_staging()
    staging.enqueue(doc_id, meta, chunks_dicts)
    staging.approve(doc_id)

    logger.info(
        "Staged JSONL batch '%s': %d chunks, %d unique sources, schema=%s, embeddings=%s",
        batch_name, total, len(unique_sources), schema_detected,
        "all" if has_all_embeddings else ("partial" if has_any_embedding else "none"),
    )

    return {
        "doc_id":                  doc_id,
        "batch_name":              batch_name,
        "schema":                  schema_detected,
        "total_chunks":            total,
        "unique_sources":          len(unique_sources),
        "has_embeddings":          has_all_embeddings,
        "has_partial_embeddings":  has_any_embedding,
    }
