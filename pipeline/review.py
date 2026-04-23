"""
review.py
Orchestrates the human-review workflow that sits between document ingestion
and final push to the production vector store.

Flow:
  1. ingest_document() stages all docs in Redis (StagingStore).
     - Quality PASS → status "approved" (auto-approved, ready to push)
     - Quality FAIL → status "pending_review" (awaits human decision)
  2. User runs ``cli.py review list`` to see what needs attention.
  3. User approves or rejects individual docs.
  4. ``cli.py review push`` embeds all approved docs and upserts them into
     the configured vector backend (Redis RediSearch or Qdrant).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from pipeline import embedder, mongo_store
# redis_store imported lazily so Redis Search module is not required at import time
from pipeline.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _staging() -> mongo_store.MongoStagingStore:
    return mongo_store.get_staging()


# ---------------------------------------------------------------------------
# Listing / inspection
# ---------------------------------------------------------------------------

def list_all_docs() -> list[dict[str, Any]]:
    """
    Return a summary of every staged document.

    Each dict has at minimum:
      doc_id, title, source_path, source_type, status,
      quality_score, quality_flags (list), chunk_count
    """
    return _staging().list_all()


def list_pending_docs() -> list[dict[str, Any]]:
    """Return only documents that are pending human review."""
    all_docs = list_all_docs()
    return [d for d in all_docs if d.get("status") == "pending_review"]


def get_doc_detail(doc_id: str) -> dict[str, Any] | None:
    """
    Return full metadata + first 3 sample chunks for a staged document.
    Returns None if the doc_id does not exist in staging.
    """
    staging = _staging()
    meta = staging.get_doc_meta(doc_id)
    if not meta:
        return None

    # Deserialise quality_flags
    try:
        meta["quality_flags"] = json.loads(meta.get("quality_flags", "[]"))
    except Exception:
        meta["quality_flags"] = []

    chunks = staging.get_chunks(doc_id)
    meta["sample_chunks"] = chunks          # all chunks (UI paginates)
    meta["chunk_count"] = len(chunks)
    meta["doc_id"] = doc_id
    return meta


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

def approve_doc(doc_id: str) -> bool:
    """
    Mark a document as approved.

    Returns True if the document existed in staging, False otherwise.
    """
    staging = _staging()
    if not staging.get_doc_meta(doc_id):
        return False
    staging.approve(doc_id)
    return True


def update_chunk(doc_id: str, chunk_id: str, updates: dict[str, Any]) -> bool:
    """
    Update fields on a single staged chunk (e.g. tags, content, section).

    Returns True if the chunk was found, False otherwise.
    """
    return _staging().update_chunk(chunk_id, updates)


def split_chunk(doc_id: str, chunk_id: str, content_parts: list[str]) -> list[str]:
    """
    Split a single staged chunk into multiple subchunks.

    Each string in *content_parts* becomes a new chunk, inheriting all
    metadata (section, tags, citation, page_number) from the original.
    The original chunk is deleted.

    Returns the list of new chunk_ids, or [] if the operation was a no-op
    (chunk not found, or all content_parts were blank).
    """
    return _staging().split_chunk(doc_id, chunk_id, content_parts)


def split_doc(source_doc_id: str, chunk_ids: list[str], new_title: str) -> str | None:
    """
    Break a subset of chunks out of *source_doc_id* into a new staged document.

    The new document inherits source metadata (source_path, kb_name, …) but
    gets its own doc_id and status ``pending_review``.

    Returns the new doc_id, or None if the source document was not found.
    """
    staging = _staging()
    meta = staging.get_doc_meta(source_doc_id)
    if not meta:
        return None

    new_doc_id = str(uuid.uuid4())
    new_meta = {
        "title":          new_title,
        "source_path":    meta.get("source_path", ""),
        "source_type":    meta.get("source_type", ""),
        "author":         meta.get("author", ""),
        "created_date":   meta.get("created_date", ""),
        "url":            meta.get("url", ""),
        "page_count":     0,
        "quality_score":  meta.get("quality_score", 1.0),
        "quality_passed": meta.get("quality_passed", True),
        "quality_flags":  meta.get("quality_flags", []),
        "suggested_tags": meta.get("suggested_tags", []),
        "schema_type":    meta.get("schema_type", ""),
        "kb_name":        meta.get("kb_name", "default"),
    }
    moved = staging.split_doc(source_doc_id, new_doc_id, chunk_ids, new_meta)
    logger.info(
        "split_doc: moved %d chunks from %s → %s (%s)",
        moved, source_doc_id, new_doc_id, new_title,
    )
    return new_doc_id


def reject_doc(doc_id: str, reason: str = "") -> bool:
    """
    Mark a document as rejected (will not be pushed to vector store).

    Returns True if the document existed in staging, False otherwise.
    """
    staging = _staging()
    if not staging.get_doc_meta(doc_id):
        return False
    staging.reject(doc_id, reason=reason)
    return True


# ---------------------------------------------------------------------------
# Push to vector store
# ---------------------------------------------------------------------------

def push_approved(
    doc_id: str | None = None,
    remove_after_push: bool = False,
) -> dict[str, Any]:
    """
    Embed all approved chunks and upsert them into the configured vector backend.

    Parameters
    ----------
    doc_id:
        If given, push only this specific document (must be approved).
        If None, push all approved documents.
    remove_after_push:
        Whether to delete staging docs/chunks after a successful push.
        Defaults to False so staging data is retained for audit and JSONL export.

    Returns
    -------
    dict
        Summary: ``{"pushed_docs": int, "pushed_chunks": int, "errors": list}``
    """
    from pipeline.chunker import Chunk

    staging = _staging()

    # Decide which docs to push
    if doc_id:
        meta = staging.get_doc_meta(doc_id)
        if not meta:
            return {"pushed_docs": 0, "pushed_chunks": 0, "errors": [f"Doc {doc_id} not found."]}
        if meta.get("status") != "approved":
            return {"pushed_docs": 0, "pushed_chunks": 0, "errors": [f"Doc {doc_id} is not approved (status={meta.get('status')})."]}
        doc_ids = [doc_id]
    else:
        doc_ids = staging.get_approved()

    if not doc_ids:
        logger.info("No approved documents to push.")
        return {"pushed_docs": 0, "pushed_chunks": 0, "errors": []}

    pushed_docs = 0
    pushed_chunks = 0
    pushed_doc_ids: list[str] = []
    errors: list[str] = []

    for did in doc_ids:
        try:
            chunk_dicts = staging.get_chunks(did)
            if not chunk_dicts:
                logger.warning("Doc %s has no staged chunks — skipping.", did)
                continue

            # Reconstruct Chunk objects
            # JSONL imports may have pre-computed embeddings stored in _embedding
            chunks: list[Chunk] = []
            precomputed: list[list[float] | None] = []
            for cd in chunk_dicts:
                c = Chunk(
                    chunk_id=cd.get("chunk_id", str(uuid.uuid4())),
                    source=cd.get("source", ""),
                    title=cd.get("title", ""),
                    section=cd.get("section", ""),
                    content=cd.get("content", ""),
                    tags=cd.get("tags", []),
                    metadata=cd.get("metadata", {}),
                )
                chunks.append(c)
                precomputed.append(cd.get("_embedding"))  # None if not present

            # Embed only chunks that don't already have a vector
            needs_embed = [i for i, v in enumerate(precomputed) if v is None]
            if needs_embed:
                logger.info("Embedding %d/%d chunks for doc %s …", len(needs_embed), len(chunks), did)
                sub_chunks = [chunks[i] for i in needs_embed]
                sub_vectors = embedder.embed_chunks(sub_chunks)
                for idx, vec in zip(needs_embed, sub_vectors):
                    precomputed[idx] = vec
            else:
                logger.info("Reusing %d pre-computed embeddings for doc %s.", len(chunks), did)

            vectors = [v for v in precomputed]   # type: ignore[assignment]

            # Quality score for this doc (stored in staging meta)
            doc_meta = staging.get_doc_meta(did) or {}
            try:
                qs_value = float(doc_meta.get("quality_score", 1.0))
            except (ValueError, TypeError):
                qs_value = 1.0

            # Push to Redis
            from pipeline import redis_store
            redis_store.create_index()
            redis_store.upsert_chunks(chunks, vectors)

            pushed_docs += 1
            pushed_chunks += len(chunks)
            pushed_doc_ids.append(did)

            # Record push in the KB ledger for drift tracking
            uc_id  = doc_meta.get("usecase_id") or None
            ag_flt = doc_meta.get("agent_filter") or None
            try:
                ledger = mongo_store.get_ledger()
                ledger.record_push(
                    doc_id=did,
                    title=doc_meta.get("title", ""),
                    source_path=doc_meta.get("source_path", ""),
                    source_type=doc_meta.get("source_type", ""),
                    url=doc_meta.get("url") or None,
                    chunk_ids=[c.chunk_id for c in chunks],
                    tags=doc_meta.get("suggested_tags") or [],
                    quality_score=qs_value,
                    kb_name=doc_meta.get("kb_name", "default"),
                    usecase_id=uc_id,
                    agent_filter=ag_flt,
                )
            except Exception as ledger_exc:
                logger.warning("Could not record push to ledger: %s", ledger_exc)

            # Update usecase-level ledger when usecase_id + agent_filter are set
            if uc_id and ag_flt:
                try:
                    uc_ledger = mongo_store.get_usecase_ledger()
                    uc_ledger.record_push(
                        usecase_id=uc_id,
                        agent_filter=ag_flt,
                        kb_name=doc_meta.get("kb_name", "default"),
                        doc_ids=[did],
                        chunk_ids=[c.chunk_id for c in chunks],
                    )
                except Exception as uc_exc:
                    logger.warning("Could not update usecase ledger: %s", uc_exc)

            staging.mark_pushed(did)

            if remove_after_push:
                staging.remove_doc(did)
                logger.info("Staging data removed for doc %s.", did)

        except Exception as exc:
            logger.error("Failed to push doc %s: %s", did, exc, exc_info=True)
            errors.append(f"{did}: {exc}")

    if pushed_docs > 0:
        # Store a point-in-time snapshot of the full KB state in MongoDB
        try:
            snap_id = mongo_store.get_ledger().record_snapshot(pushed_doc_ids)
            logger.info("Ledger snapshot %s recorded (%d docs pushed)", snap_id, len(pushed_doc_ids))
        except Exception as snap_exc:
            logger.warning("Could not record ledger snapshot: %s", snap_exc)

        # Optionally also write a CSV file if LEDGER_OUTPUT_DIR is configured
        if settings.ledger_output_dir:
            try:
                from pipeline.exporter import export_ledger_csv
                all_records = mongo_store.get_ledger().list_docs(limit=2000)
                csv_path = export_ledger_csv(all_records)
                logger.info("Ledger CSV written to %s", csv_path)
            except Exception as ledger_exc:
                logger.warning("Could not export ledger CSV: %s", ledger_exc)

    return {
        "pushed_docs":   pushed_docs,
        "pushed_chunks": pushed_chunks,
        "errors":        errors,
    }
