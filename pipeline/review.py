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

from pipeline import embedder, redis_store
from pipeline.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _staging() -> redis_store.StagingStore:
    return redis_store.get_staging()


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
    meta["sample_chunks"] = chunks[:3]
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
    remove_after_push: bool = True,
) -> dict[str, Any]:
    """
    Embed all approved chunks and upsert them into the configured vector backend.

    Parameters
    ----------
    doc_id:
        If given, push only this specific document (must be approved).
        If None, push all approved documents.
    remove_after_push:
        Whether to remove the staging data after a successful push (default True).

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
    errors: list[str] = []

    for did in doc_ids:
        try:
            chunk_dicts = staging.get_chunks(did)
            if not chunk_dicts:
                logger.warning("Doc %s has no staged chunks — skipping.", did)
                continue

            # Reconstruct Chunk objects
            chunks: list[Chunk] = []
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

            # Embed
            logger.info("Embedding %d chunks for doc %s …", len(chunks), did)
            vectors = embedder.embed_chunks(chunks)

            # Quality score for this doc (stored in staging meta)
            doc_meta = staging.get_doc_meta(did) or {}
            try:
                qs_value = float(doc_meta.get("quality_score", 1.0))
            except (ValueError, TypeError):
                qs_value = 1.0
            quality_scores = {c.chunk_id: qs_value for c in chunks}

            # Push to configured backend
            backend = settings.vector_backend
            if backend == "qdrant":
                from pipeline import qdrant_store
                qdrant_store.upsert_chunks(chunks, vectors, quality_scores=quality_scores)
            else:  # redis (default)
                redis_store.create_index()
                redis_store.upsert_chunks(chunks, vectors)

            pushed_docs += 1
            pushed_chunks += len(chunks)

            if remove_after_push:
                staging.remove_doc(did)
                logger.info("Staging data removed for doc %s.", did)

        except Exception as exc:
            logger.error("Failed to push doc %s: %s", did, exc, exc_info=True)
            errors.append(f"{did}: {exc}")

    return {
        "pushed_docs":   pushed_docs,
        "pushed_chunks": pushed_chunks,
        "errors":        errors,
    }
