"""
jsonl_importer.py
Imports pre-existing JSONL chunk files into the pipeline staging area.

Two built-in schemas are auto-detected:

  "crawler"
      Generic web-crawler output.
      Detection trigger: record contains "text" AND one of
        "page_url", "sourceURL", or "source_url".
      Key fields: text, page_url (or sourceURL / source_url), page_name,
                  section_breadcrumbs (list), section_heading, chunk_id.

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


def _map_with_fieldmap(
    rec: dict,
    fields: dict[str, str],
    extra_tags: list[str],
    tags_static: list[str],
    section_join: str,
) -> tuple[Chunk, list[float] | None]:
    """
    Core field-mapping logic used by both named custom schemas (schemas.yaml)
    and inline field_map dicts supplied at import time.

    ``fields`` maps pipeline field names to dot-notated paths in the source
    record, e.g. ``{"content": "body.text", "source": "_links.webui"}``.
    """
    def _get(field: str) -> Any:
        path = fields.get(field)
        return _resolve_field(rec, path) if path else None

    content = _get("content") or ""
    source  = _get("source") or ""

    section_raw = _get("section")
    if isinstance(section_raw, list):
        section = section_join.join(str(s) for s in section_raw if s)
    else:
        section = str(section_raw) if section_raw else ""

    rec_tags = _get("tags") or []
    if isinstance(rec_tags, str):
        rec_tags = [t.strip() for t in rec_tags.split(",") if t.strip()]
    all_tags = list(dict.fromkeys(list(rec_tags) + list(tags_static) + list(extra_tags)))

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


def _map_custom(rec: dict, schema_name: str, extra_tags: list[str]) -> tuple[Chunk, list[float] | None]:
    """Map a record using a named custom schema definition from schemas.yaml."""
    schema_def = next(
        (s for s in _load_custom_schemas() if s["name"] == schema_name), None
    )
    if schema_def is None:
        raise ValueError(f"Custom schema '{schema_name}' not found in schemas.yaml")

    return _map_with_fieldmap(
        rec,
        fields=schema_def.get("fields", {}),
        extra_tags=extra_tags,
        tags_static=schema_def.get("tags_static") or [],
        section_join=schema_def.get("section_join", " > "),
    )


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
    _url_fields = ("page_url", "sourceURL", "source_url")
    if "text" in record and any(k in record for k in _url_fields):
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
        url = (
            rec.get("page_url")
            or rec.get("sourceURL")
            or rec.get("source_url")
            or rec.get("source_file", "")
        )
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
            source=url,
            title=rec.get("page_name", ""),
            section=_section_from_crawler(rec),
            content=rec.get("text", ""),
            tags=list(dict.fromkeys(tags)),
            metadata={
                "citation": {
                    "source_path": url,
                    "source_type": "url",
                    "title": rec.get("page_name", ""),
                    "url": url,
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
            source=(
                rec.get("url")
                or rec.get("sourceURL")
                or rec.get("source_url")
                or rec.get("source")
                or rec.get("source_path", "")
            ),
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
    field_map: dict[str, str] | None = None,
    tags_static: list[str] | None = None,
    section_join: str = " > ",
) -> dict[str, Any]:
    """
    Read the first *n* records of a JSONL file and return a preview dict.

    When ``field_map`` is supplied the sample chunks are generated using
    that mapping instead of the auto-detected schema, so the UI can show
    a live preview of a custom mapping before committing to an import.

    Returns
    -------
    dict with keys:
        schema          – auto-detected schema name
        has_embeddings  – whether the first record contains an embedding vector
        sample_records  – first n raw dicts
        sample_chunks   – mapped Chunk objects (using field_map when supplied)
        unique_sources  – distinct source values seen in the sample
        available_keys  – sorted list of all top-level keys across sample records
    """
    samples_raw: list[dict] = []
    sources_seen: set[str] = set()
    all_keys: set[str] = set()

    def _lines():
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as fh:
                for line in fh:
                    yield line
        else:
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
            all_keys.update(rec.keys())
            src = (
                rec.get("page_url")
                or rec.get("sourceURL")
                or rec.get("source_url")
                or rec.get("source")
                or rec.get("url", "")
            )
            if src:
                sources_seen.add(src)
        except json.JSONDecodeError:
            continue

    if not samples_raw:
        return {
            "schema": "unknown", "has_embeddings": False,
            "sample_records": [], "sample_chunks": [], "unique_sources": 0,
            "available_keys": [],
        }

    schema = detect_schema(samples_raw[0])
    has_embeddings = schema == "pipeline" and bool(samples_raw[0].get("embedding"))

    sample_chunks = []
    for rec in samples_raw:
        if field_map:
            chunk, _ = _map_with_fieldmap(
                rec, field_map,
                extra_tags=[],
                tags_static=tags_static or [],
                section_join=section_join,
            )
        else:
            chunk, _ = map_record(rec, schema)
        sample_chunks.append(chunk)

    return {
        "schema":         schema,
        "has_embeddings": has_embeddings,
        "sample_records": samples_raw,
        "sample_chunks":  sample_chunks,
        "unique_sources": len(sources_seen),
        "available_keys": sorted(all_keys),
    }


# ---------------------------------------------------------------------------
# Save a field mapping as a reusable named schema in schemas.yaml
# ---------------------------------------------------------------------------

def save_custom_schema(
    name: str,
    field_map: dict[str, str],
    required_keys: list[str] | None = None,
    exclude_keys: list[str] | None = None,
    section_join: str = " > ",
    tags_static: list[str] | None = None,
) -> None:
    """
    Write (or overwrite) a named custom schema entry in ``schemas.yaml``.

    The schema can then be auto-detected on future imports without the user
    having to set up the field mapping again.

    Parameters
    ----------
    name:           Unique schema name (e.g. ``"my_export_v2"``).
    field_map:      Dict mapping pipeline fields to source keys.
    required_keys:  Keys that must be present in a record to trigger
                    auto-detection.  Defaults to the mapped source keys.
    exclude_keys:   Keys whose presence disqualifies the schema.
    section_join:   Separator for list-valued section fields.
    tags_static:    Tags applied to every chunk under this schema.
    """
    data: dict = {}
    if _SCHEMAS_FILE.exists():
        try:
            data = yaml.safe_load(_SCHEMAS_FILE.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    schemas: list[dict] = data.get("schemas") or []

    # Build the detect block — use mapped source keys as required if not given
    if required_keys is None:
        required_keys = [v for v in field_map.values() if v and "." not in v]

    new_entry: dict[str, Any] = {
        "name": name,
        "detect": {
            "required": required_keys or [],
        },
        "fields": {k: v for k, v in field_map.items() if v},
        "section_join": section_join,
    }
    if exclude_keys:
        new_entry["detect"]["exclude"] = exclude_keys
    if tags_static:
        new_entry["tags_static"] = tags_static

    # Replace existing entry with same name, or append
    idx = next((i for i, s in enumerate(schemas) if s.get("name") == name), None)
    if idx is not None:
        schemas[idx] = new_entry
    else:
        schemas.append(new_entry)

    data["schemas"] = schemas
    _SCHEMAS_FILE.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    # Clear the in-memory cache so the new schema is picked up immediately
    reload_schemas()
    logger.info("Saved custom schema '%s' to %s", name, _SCHEMAS_FILE)


# ---------------------------------------------------------------------------
# Full import → staging
# ---------------------------------------------------------------------------

def import_jsonl(
    source: str | Path | io.IOBase,
    batch_name: str | None = None,
    extra_tags: list[str] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    kb_id: str | None = None,
    field_map: dict[str, str] | None = None,
    tags_static: list[str] | None = None,
    section_join: str = " > ",
    manifest_id: str | None = None,
) -> dict[str, Any]:
    """
    Parse an entire JSONL file and stage all chunks as a single import batch
    associated with a Knowledge Base (``kb_id``).

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

    # Peek at first record to detect schema and fall back usecase fields
    def _peek_first() -> dict:
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            pass
        else:
            try:
                source.seek(0)
            except Exception:
                pass
            for raw in source:
                line = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip()
                if line:
                    try:
                        rec = json.loads(line)
                        source.seek(0)
                        return rec
                    except json.JSONDecodeError:
                        pass
            try:
                source.seek(0)
            except Exception:
                pass
        return {}

    first_rec = _peek_first()

    # Quality signal accumulators (filled during the stream loop below)
    from pipeline.quality import (
        _compute_age_days, _is_boilerplate,
        MIN_CHUNK_CHARS, MAX_CHUNK_CHARS, STALE_THRESHOLD_DAYS,
    )
    n_too_short = n_too_long = n_boilerplate = 0
    record_dates: list[str] = []

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

        # Map the record — use inline field_map if supplied, else auto-detect schema
        if field_map:
            schema_detected = "field_map"
            chunk, embedding = _map_with_fieldmap(
                rec, field_map,
                extra_tags=extra_tags,
                tags_static=tags_static or [],
                section_join=section_join,
            )
        else:
            if schema_detected == "unknown":
                schema_detected = detect_schema(rec)
            chunk, embedding = map_record(rec, schema_detected, extra_tags=extra_tags)

        content = chunk.content.strip()
        if not content:
            continue

        src = (
            rec.get("page_url")
            or rec.get("sourceURL")
            or rec.get("source_url")
            or rec.get("source")
            or rec.get("url", "")
            or chunk.source
        )
        if src:
            unique_sources.add(src)

        if embedding:
            has_any_embedding = True
            embeddings_map[chunk.chunk_id] = embedding
        else:
            has_all_embeddings = False

        # Auto-split oversized records into character-overlap sub-chunks.
        # Skip splitting when a pre-computed embedding exists — splitting would
        # invalidate it.
        if len(content) > MAX_CHUNK_CHARS and not embedding:
            from pipeline.chunker import _split_large_chunk
            from pipeline.config import settings as _settings
            overlap = _settings.chunk_overlap_chars
            parts = [p.strip() for p in _split_large_chunk(content, MAX_CHUNK_CHARS, overlap) if p.strip()]
            n_parts = len(parts)
            for idx, part in enumerate(parts):
                section = chunk.section + (f" [{idx + 1}/{n_parts}]" if n_parts > 1 else "")
                sub = Chunk(
                    source=chunk.source,
                    title=chunk.title,
                    section=section,
                    content=part,
                    tags=list(chunk.tags),
                    metadata=dict(chunk.metadata),
                )
                d = sub.to_dict()
                q_issues: list[str] = []
                if len(part) < MIN_CHUNK_CHARS:
                    q_issues.append("too_short")
                    n_too_short += 1
                if _is_boilerplate(part):
                    q_issues.append("boilerplate")
                    n_boilerplate += 1
                if q_issues:
                    d.setdefault("metadata", {})["quality_flags"] = q_issues
                chunks_dicts.append(d)
                total += 1
        else:
            d = chunk.to_dict()
            char_count = len(content)
            q_issues = []
            if char_count < MIN_CHUNK_CHARS:
                q_issues.append("too_short")
                n_too_short += 1
            elif char_count > MAX_CHUNK_CHARS:
                # Has a pre-computed embedding — flag rather than split
                q_issues.append("too_long")
                n_too_long += 1
            if _is_boilerplate(content):
                q_issues.append("boilerplate")
                n_boilerplate += 1
            if q_issues:
                d.setdefault("metadata", {})["quality_flags"] = q_issues
            chunks_dicts.append(d)
            total += 1

        # Collect date for recency check (try multiple field paths)
        date_val = (
            _resolve_field(rec, "metadata.citation.created_date")
            or _resolve_field(rec, "metadata.confluence.last_modified")
            or rec.get("created_date")
            or rec.get("last_modified")
        )
        if date_val and isinstance(date_val, str):
            record_dates.append(date_val)

        if progress_cb and total % 1000 == 0:
            progress_cb(total, -1)

    if not chunks_dicts:
        raise ValueError("No valid records found in JSONL file.")

    # Stage as a single batch
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, batch_name + str(total)))

    # Compute recency from collected record dates (use average age)
    age_days: int | None = None
    is_stale = False
    if record_dates:
        ages = [_compute_age_days(d) for d in record_dates]
        valid_ages = [a for a in ages if a is not None]
        if valid_ages:
            age_days = int(sum(valid_ages) / len(valid_ages))
            is_stale = age_days > STALE_THRESHOLD_DAYS

    # Build quality flags list and score
    import json as _json
    quality_flags: list[str] = []
    if n_too_short:
        quality_flags.append(
            f"{n_too_short} chunk(s) are too short (< {MIN_CHUNK_CHARS} chars) "
            "— may be section stubs or nav elements."
        )
    if n_too_long:
        quality_flags.append(
            f"{n_too_long} chunk(s) exceed {MAX_CHUNK_CHARS} chars "
            "— risk truncation by the embedding model."
        )
    if n_boilerplate:
        quality_flags.append(
            f"{n_boilerplate} chunk(s) appear to be boilerplate "
            "— navigation menus, table of contents, or template text."
        )
    if is_stale:
        quality_flags.append(
            f"Content is ~{age_days} days old — consider refreshing from the source."
        )

    n_flagged_chunks = sum(1 for cd in chunks_dicts if cd.get("metadata", {}).get("quality_flags"))
    quality_score = round(1.0 - (n_flagged_chunks / max(total, 1)), 4)
    quality_passed = len(quality_flags) == 0
    status = "approved" if quality_passed else "pending_review"

    meta: dict[str, Any] = {
        "title":                  f"JSONL import — {batch_name}",
        "source_path":            batch_name,
        "source_type":            "jsonl",
        "schema_type":            schema_detected,
        "chunk_count":            total,
        "unique_sources":         len(unique_sources),
        "has_embeddings":         int(has_all_embeddings),
        "has_partial_embeddings": int(has_any_embedding and not has_all_embeddings),
        "quality_score":          quality_score,
        "quality_passed":         int(quality_passed),
        "quality_flags":          _json.dumps(quality_flags),
        "status":                 status,
        "kb_id":                  kb_id or None,
        "age_days":               age_days,
        "is_stale":               is_stale,
        "chunks_too_short":       n_too_short,
        "chunks_too_long":        n_too_long,
        "chunks_boilerplate":     n_boilerplate,
    }

    # If all embeddings are pre-computed, attach them into the chunk dicts
    if has_all_embeddings and embeddings_map:
        for cd in chunks_dicts:
            emb = embeddings_map.get(cd.get("chunk_id", ""))
            if emb:
                cd["_embedding"] = emb

    staging = mongo_store.get_staging()
    staging.enqueue(doc_id, meta, chunks_dicts)
    if quality_passed:
        staging.approve(doc_id)

    if manifest_id:
        try:
            from pipeline.manifests import get_manifest_manager
            get_manifest_manager().add_entry(
                manifest_id=manifest_id,
                doc_id=doc_id,
                object_id=doc_id,
                file_id=doc_id,
                version_id=str(total),
                source_type="jsonl",
                source_ref=batch_name or "",
                title=f"JSONL import — {batch_name}",
                kb_id=kb_id,
                status=status,
            )
        except Exception as _manifest_exc:
            logger.warning("Could not associate JSONL batch with manifest: %s", _manifest_exc)

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
