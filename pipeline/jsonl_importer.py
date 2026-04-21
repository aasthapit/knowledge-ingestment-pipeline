"""
jsonl_importer.py
Imports pre-existing JSONL chunk files into the pipeline staging area.

Two built-in schemas are auto-detected:

  "crawler"
      Produced by crawl_ocp_docs.py.
      Key fields: text, page_url, page_name, section_breadcrumbs (list),
                  section_heading, chunk_id, agent_filter, usecase_id.

  "pipeline"
      Produced by pipeline/exporter.py.
      Key fields: content, source, title, section, tags, metadata,
                  chunk_id, embedding (optional — reused if present).

Custom schemas can be defined in schemas.yaml at the project root.
They are checked before the built-in schemas and use the same field-mapping
format documented in that file.
"""
from __future__ import annotations

import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable

import yaml

from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom schema loader
# ---------------------------------------------------------------------------

_SCHEMAS_FILE = Path(__file__).resolve().parent.parent / "schemas.yaml"
_custom_schemas: list[dict] | None = None


def _load_custom_schemas() -> list[dict]:
    """Load and cache schemas.yaml. Returns empty list if file is absent or empty."""
    global _custom_schemas
    if _custom_schemas is not None:
        return _custom_schemas
    if not _SCHEMAS_FILE.exists():
        _custom_schemas = []
        return _custom_schemas
    try:
        data = yaml.safe_load(_SCHEMAS_FILE.read_text(encoding="utf-8")) or {}
        _custom_schemas = data.get("schemas") or []
    except Exception as exc:
        logger.warning("Could not load schemas.yaml: %s", exc)
        _custom_schemas = []
    return _custom_schemas


def reload_schemas() -> None:
    """Force a re-read of schemas.yaml (useful in long-running processes)."""
    global _custom_schemas
    _custom_schemas = None


def _resolve_field(rec: dict, field_path: str) -> Any:
    """
    Get a value from a record using a dot-notated field path.

    Examples:
        "title"           → rec["title"]
        "_links.webui"    → rec["_links"]["webui"]
    """
    parts = field_path.split(".")
    val: Any = rec
    for part in parts:
        if not isinstance(val, dict):
            return None
        val = val.get(part)
    return val


def _detect_custom(record: dict) -> str | None:
    """
    Return the name of the first matching custom schema, or None.
    """
    for schema_def in _load_custom_schemas():
        detect = schema_def.get("detect", {})
        required = detect.get("required") or []
        exclude  = detect.get("exclude") or []

        if not required:
            continue
        if all(r in record for r in required) and not any(e in record for e in exclude):
            return schema_def["name"]
    return None


def _map_custom(rec: dict, schema_name: str, extra_tags: list[str]) -> tuple[Chunk, list[float] | None]:
    """Map a record using a named custom schema definition."""
    schema_def = next(
        (s for s in _load_custom_schemas() if s["name"] == schema_name), None
    )
    if schema_def is None:
        raise ValueError(f"Custom schema '{schema_name}' not found in schemas.yaml")

    fields       = schema_def.get("fields", {})
    tags_static  = schema_def.get("tags_static") or []
    section_join = schema_def.get("section_join", " > ")

    def _get(field: str) -> Any:
        path = fields.get(field)
        return _resolve_field(rec, path) if path else None

    # Content
    content = _get("content") or ""

    # Source
    source = _get("source") or ""

    # Section — join if it's a list
    section_raw = _get("section")
    if isinstance(section_raw, list):
        section = section_join.join(str(s) for s in section_raw if s)
    else:
        section = str(section_raw) if section_raw else ""

    # Tags
    rec_tags = _get("tags") or []
    if isinstance(rec_tags, str):
        rec_tags = [t.strip() for t in rec_tags.split(",") if t.strip()]
    all_tags = list(dict.fromkeys(list(rec_tags) + list(tags_static) + list(extra_tags)))

    # Embedding
    raw_emb = _get("embedding")
    embedding: list[float] | None = raw_emb if isinstance(raw_emb, list) and raw_emb else None

    chunk = Chunk(
        chunk_id=_get("chunk_id") or rec.get("id") or str(uuid.uuid4()),
        source=str(source),
        title=str(_get("title") or ""),
        section=section,
        content=str(content),
        tags=all_tags,
        metadata={},
    )
    return chunk, embedding


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

def detect_schema(record: dict) -> str:
    """
    Return the schema name for a single record.

    Checks custom schemas (schemas.yaml) first, then the built-in
    "crawler" and "pipeline" schemas, then falls back to "unknown".
    """
    custom = _detect_custom(record)
    if custom:
        return custom
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

    # Custom schema defined in schemas.yaml
    if schema not in ("crawler", "pipeline", "unknown"):
        return _map_custom(rec, schema, extra_tags)

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
    kb_name: str = "default",
) -> dict[str, Any]:
    """
    Parse an entire JSONL file and stage all chunks as a single import batch.

    The batch is stored as one entry in :class:`~pipeline.mongo_store.MongoStagingStore`
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
    kb_name:
        Logical knowledge base name for ledger grouping and drift tracking.

    Returns
    -------
    dict
        ``{doc_id, batch_name, schema, total_chunks, unique_sources,
           has_embeddings, has_partial_embeddings}``
    """
    from pipeline import mongo_store

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
        "kb_name":                kb_name,
    }

    # If all embeddings are pre-computed, attach them into the chunk dicts
    if has_all_embeddings and embeddings_map:
        for cd in chunks_dicts:
            emb = embeddings_map.get(cd.get("chunk_id", ""))
            if emb:
                cd["_embedding"] = emb

    staging = mongo_store.get_staging()
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
