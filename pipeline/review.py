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
        "kb_id":          meta.get("kb_id"),
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
    corpus_id: str,
    doc_id: str | None = None,
    remove_after_push: bool = False,
) -> dict[str, Any]:
    """
    Embed all approved chunks from a corpus's KBs and push to the corpus's
    configured vector store.

    Parameters
    ----------
    corpus_id:
        The corpus that provides the usecase/agent context and vector store target.
        All approved staging docs whose kb_id belongs to this corpus are pushed.
    doc_id:
        If given, push only this specific document (must be approved).
    remove_after_push:
        Whether to delete staging docs/chunks after a successful push.

    Returns
    -------
    dict
        ``{"pushed_docs": int, "pushed_chunks": int, "errors": list}``
    """
    from pipeline.chunker import Chunk
    from pipeline.vector_store import get_vector_store_client

    staging  = _staging()
    cs       = mongo_store.get_corpus_store()
    vs_store = mongo_store.get_vs_config_store()

    corpus = cs.get(corpus_id)
    if not corpus:
        return {"pushed_docs": 0, "pushed_chunks": 0, "errors": [f"Corpus {corpus_id} not found."]}

    usecase_id   = corpus.get("usecase_id") or None
    agent_filter = corpus.get("agent_filter") or None
    kb_ids       = corpus.get("kb_ids") or []
    vs_id        = corpus.get("vector_store_id") or None

    vs_config = vs_store.get(vs_id) if vs_id else None
    if not vs_config:
        return {
            "pushed_docs": 0, "pushed_chunks": 0,
            "errors": ["No vector store configured for this corpus. Assign one before pushing."],
        }
    vector_client = get_vector_store_client(vs_config)
    skip_embedding = getattr(vector_client, "handles_own_embedding", False)

    # Decide which docs to push
    if doc_id:
        meta = staging.get_doc_meta(doc_id)
        if not meta:
            return {"pushed_docs": 0, "pushed_chunks": 0, "errors": [f"Doc {doc_id} not found."]}
        if meta.get("status") != "approved":
            return {
                "pushed_docs": 0, "pushed_chunks": 0,
                "errors": [f"Doc {doc_id} is not approved (status={meta.get('status')})."],
            }
        doc_ids = [doc_id]
    else:
        # Collect approved docs across all KBs in this corpus
        all_approved = staging.get_approved()
        if kb_ids:
            approved_meta = {d["doc_id"]: d for d in staging.list_all() if d["doc_id"] in all_approved}
            doc_ids = [did for did in all_approved if approved_meta.get(did, {}).get("kb_id") in kb_ids]
        else:
            doc_ids = all_approved

    if not doc_ids:
        logger.info("No approved documents to push for corpus %s.", corpus_id)
        return {"pushed_docs": 0, "pushed_chunks": 0, "errors": []}

    pushed_docs = 0
    pushed_chunks = 0
    pushed_doc_ids: list[str] = []
    errors: list[str] = []

    vector_client.ensure_index()

    for did in doc_ids:
        try:
            chunk_dicts = staging.get_chunks(did)
            if not chunk_dicts:
                logger.warning("Doc %s has no staged chunks — skipping.", did)
                continue

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
                precomputed.append(cd.get("_embedding"))

            if skip_embedding:
                logger.info("Skipping embedding for doc %s — backend handles its own vectorization.", did)
                vectors = [[] for _ in chunks]
            else:
                needs_embed = [i for i, v in enumerate(precomputed) if v is None]
                if needs_embed:
                    logger.info("Embedding %d/%d chunks for doc %s …", len(needs_embed), len(chunks), did)
                    sub_vectors = embedder.embed_chunks([chunks[i] for i in needs_embed])
                    for idx, vec in zip(needs_embed, sub_vectors):
                        precomputed[idx] = vec
                else:
                    logger.info("Reusing %d pre-computed embeddings for doc %s.", len(chunks), did)
                vectors = [v for v in precomputed]  # type: ignore[assignment]

            doc_meta = staging.get_doc_meta(did) or {}
            try:
                qs_value = float(doc_meta.get("quality_score", 1.0))
            except (ValueError, TypeError):
                qs_value = 1.0

            vector_client.upsert_chunks(chunks, vectors)

            pushed_docs += 1
            pushed_chunks += len(chunks)
            pushed_doc_ids.append(did)

            kb_id = doc_meta.get("kb_id") or None
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
                    kb_id=kb_id,
                    usecase_id=usecase_id,
                    agent_filter=agent_filter,
                )
            except Exception as ledger_exc:
                logger.warning("Could not record push to ledger: %s", ledger_exc)

            if usecase_id and agent_filter:
                try:
                    uc_ledger = mongo_store.get_usecase_ledger()
                    uc_ledger.record_push(
                        usecase_id=usecase_id,
                        agent_filter=agent_filter,
                        kb_name=kb_id or "default",
                        doc_ids=[did],
                        chunk_ids=[c.chunk_id for c in chunks],
                    )
                except Exception as uc_exc:
                    logger.warning("Could not update usecase ledger: %s", uc_exc)

            staging.mark_pushed(did)

            try:
                from datetime import datetime as _dt, timezone as _tz
                from pipeline.manifests import get_manifest_manager
                _mm = get_manifest_manager()
                _push_now = _dt.now(_tz.utc)
                for _mf in _mm.find_manifests_by_doc_id(did):
                    _mm.update_entry_status(
                        manifest_id=_mf["manifest_id"],
                        doc_id=did,
                        status="pushed",
                        pushed_at=_push_now,
                    )
            except Exception as _manifest_exc:
                logger.warning("Could not update manifest entry after push: %s", _manifest_exc)

            if remove_after_push:
                staging.remove_doc(did)
                logger.info("Staging data removed for doc %s.", did)

        except Exception as exc:
            logger.error("Failed to push doc %s: %s", did, exc, exc_info=True)
            errors.append(f"{did}: {exc}")

    if pushed_docs > 0:
        try:
            snap_id = mongo_store.get_ledger().record_snapshot(pushed_doc_ids)
            logger.info("Ledger snapshot %s recorded (%d docs pushed)", snap_id, len(pushed_doc_ids))
        except Exception as snap_exc:
            logger.warning("Could not record ledger snapshot: %s", snap_exc)

        if settings.ledger_output_dir:
            try:
                from pipeline.exporter import export_ledger_csv
                csv_path = export_ledger_csv(mongo_store.get_ledger().list_docs(limit=2000))
                logger.info("Ledger CSV written to %s", csv_path)
            except Exception as ledger_exc:
                logger.warning("Could not export ledger CSV: %s", ledger_exc)

    return {
        "pushed_docs":   pushed_docs,
        "pushed_chunks": pushed_chunks,
        "errors":        errors,
    }
