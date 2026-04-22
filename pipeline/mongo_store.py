"""
mongo_store.py
MongoDB-backed staging store and knowledge-base ledger.

Two classes are provided:

``MongoStagingStore``
    Replaces the Redis ``StagingStore`` for the document review workflow.
    Stores document metadata and chunks between ingestion and final push to
    the vector database.  Exposes the same interface as
    :class:`~pipeline.redis_store.StagingStore` so ``review.py`` and the
    Streamlit pages need no structural changes.

``KBLedger``
    Permanent record of every document that has been pushed to the vector
    store.  Enables drift detection: tracks source modification time / size
    (for files) or URL (for web sources) so you can identify documents that
    have changed since they were last indexed.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import quote_plus

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from pipeline.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client / DB helpers (lazily initialised, module-level singleton)
# ---------------------------------------------------------------------------

_client: MongoClient | None = None


def _build_uri() -> str:
    """
    Build a MongoDB connection URI from individual settings fields.

    Uses ``mongodb+srv://`` when ``MONGODB_SRV=true`` (default), which is
    required for Atlas and most cloud-hosted clusters.  Set ``MONGODB_SRV=false``
    for a plain ``mongodb://`` connection with an explicit host:port.
    """
    host = settings.mongodb_host

    creds = ""
    if settings.mongodb_username:
        user = quote_plus(settings.mongodb_username)
        pwd  = quote_plus(settings.mongodb_password) if settings.mongodb_password else ""
        creds = f"{user}:{pwd}@" if pwd else f"{user}@"

    if settings.mongodb_srv:
        # SRV — no port in the URI (resolved from DNS)
        uri = f"mongodb+srv://{creds}{host}/{settings.mongodb_db_name}"
    else:
        uri = f"mongodb://{creds}{host}:{settings.mongodb_port}/{settings.mongodb_db_name}"

    params: list[str] = []
    if settings.mongodb_auth_source:
        params.append(f"authSource={quote_plus(settings.mongodb_auth_source)}")
    if not settings.mongodb_tls and not settings.mongodb_srv:
        # SRV connections always use TLS; only disable for plain mongodb:// URIs
        params.append("tls=false")

    if params:
        uri += "?" + "&".join(params)

    return uri


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        if settings.mongodb_uri:
            # Full URI supplied — use it directly, ignore individual fields
            uri = settings.mongodb_uri
        else:
            uri = _build_uri()
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client


def _get_db() -> Database:
    return _get_client()[settings.mongodb_db_name]


def _coll_name(base: str) -> str:
    """Apply the configured collection prefix to a base name."""
    prefix = settings.mongodb_collection_prefix
    return f"{prefix}{base}" if prefix else base


# ---------------------------------------------------------------------------
# MongoStagingStore
# ---------------------------------------------------------------------------

class MongoStagingStore:
    """
    Staging store backed by MongoDB.

    Collections used:
    - ``staging_docs``   — one document per ingested item (metadata + status)
    - ``staging_chunks`` — normalised chunk records, linked by ``doc_id``

    Status values: ``pending_review``, ``approved``, ``rejected``, ``pushed``
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or _get_db()
        self._docs: Collection   = self._db[_coll_name("staging_docs")]
        self._chunks: Collection = self._db[_coll_name("staging_chunks")]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes the first time the store is used."""
        self._docs.create_index("status")
        self._docs.create_index("ingested_at")
        self._chunks.create_index("doc_id")
        self._chunks.create_index("chunk_id", unique=True, sparse=True)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def enqueue(
        self,
        doc_id: str,
        meta: dict[str, Any],
        chunks: list[dict],
    ) -> None:
        """
        Stage a document for review.

        ``meta`` should contain at least: title, source_path, source_type,
        quality_score, quality_passed, quality_flags, chunk_count, status.
        ``chunks`` is the list of serialised Chunk dicts.
        """
        now = datetime.now(timezone.utc)

        # Coerce types stored as strings by the Redis path
        def _bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, int):
                return bool(val)
            if isinstance(val, str):
                return val.lower() in ("1", "true", "yes")
            return False

        def _float(val: Any, default: float = 0.0) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        def _int(val: Any, default: int = 0) -> int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        def _list(val: Any) -> list:
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return []
            return []

        doc = {
            "_id":                   doc_id,
            "title":                 meta.get("title", ""),
            "source_path":           meta.get("source_path", ""),
            "source_type":           meta.get("source_type", ""),
            "author":                meta.get("author", ""),
            "created_date":          meta.get("created_date", ""),
            "url":                   meta.get("url", ""),
            "page_count":            _int(meta.get("page_count", 0)),
            "quality_score":         _float(meta.get("quality_score", 0.0)),
            "quality_passed":        _bool(meta.get("quality_passed", False)),
            "quality_flags":         _list(meta.get("quality_flags", "[]")),
            "suggested_tags":        _list(meta.get("suggested_tags", "[]")),
            "chunk_count":           _int(meta.get("chunk_count", len(chunks))),
            "status":                meta.get("status", "pending_review"),
            "schema_type":           meta.get("schema_type", ""),
            "unique_sources":        _int(meta.get("unique_sources", 0)),
            "has_embeddings":        _bool(meta.get("has_embeddings", False)),
            "has_partial_embeddings": _bool(meta.get("has_partial_embeddings", False)),
            "kb_name":               meta.get("kb_name", "default"),
            "ingested_at":           now,
            "approved_at":           None,
            "pushed_at":             None,
            "reject_reason":         None,
        }

        # Upsert so re-ingest of the same doc_id refreshes the record
        self._docs.replace_one({"_id": doc_id}, doc, upsert=True)

        # Replace chunk records for this doc
        if chunks:
            self._chunks.delete_many({"doc_id": doc_id})
            chunk_docs = []
            for c in chunks:
                cd = dict(c)
                cd["doc_id"] = doc_id
                # Use chunk_id as _id if available
                if "chunk_id" in cd:
                    cd["_id"] = cd.pop("chunk_id")
                chunk_docs.append(cd)
            self._chunks.insert_many(chunk_docs, ordered=False)

        logger.debug("Staged doc %s (%d chunks)", doc_id, len(chunks))

    def approve(self, doc_id: str) -> None:
        """Mark a document as approved and ready to push."""
        self._docs.update_one(
            {"_id": doc_id},
            {"$set": {"status": "approved", "approved_at": datetime.now(timezone.utc)}},
        )

    def reject(self, doc_id: str, reason: str = "") -> None:
        """Mark a document as rejected."""
        self._docs.update_one(
            {"_id": doc_id},
            {"$set": {
                "status": "rejected",
                "reject_reason": reason,
            }},
        )

    def mark_pushed(self, doc_id: str) -> None:
        """Mark a document as pushed to the vector store."""
        self._docs.update_one(
            {"_id": doc_id},
            {"$set": {"status": "pushed", "pushed_at": datetime.now(timezone.utc)}},
        )

    def remove_doc(self, doc_id: str) -> None:
        """Remove a document and its chunks from staging (called after push)."""
        self._docs.delete_one({"_id": doc_id})
        self._chunks.delete_many({"doc_id": doc_id})

    # ------------------------------------------------------------------
    # Read operations (mirrors Redis StagingStore interface)
    # ------------------------------------------------------------------

    def get_doc_meta(self, doc_id: str) -> dict[str, Any] | None:
        """Return document metadata dict, or None if not found."""
        doc = self._docs.find_one({"_id": doc_id})
        if doc is None:
            return None
        # Normalise _id back to doc_id for compatibility
        doc["doc_id"] = doc.pop("_id")
        # Serialise datetime fields to ISO strings for UI compatibility
        for key in ("ingested_at", "approved_at", "pushed_at"):
            val = doc.get(key)
            if isinstance(val, datetime):
                doc[key] = val.isoformat()
        return doc

    def get_chunks(self, doc_id: str) -> list[dict]:
        """Return all chunk dicts for a document, restoring chunk_id."""
        chunks = []
        for cd in self._chunks.find({"doc_id": doc_id}):
            cd["chunk_id"] = str(cd.pop("_id"))
            cd.pop("doc_id", None)
            chunks.append(cd)
        return chunks

    def get_pending(self) -> list[str]:
        """Return doc_ids with status pending_review."""
        return [d["_id"] for d in self._docs.find({"status": "pending_review"}, {"_id": 1})]

    def get_approved(self) -> list[str]:
        """Return doc_ids with status approved."""
        return [d["_id"] for d in self._docs.find({"status": "approved"}, {"_id": 1})]

    def list_all(self) -> list[dict[str, Any]]:
        """
        Return a summary list of every staged document (newest first).

        Each dict has: doc_id, title, source_path, source_type, status,
        quality_score, quality_flags, chunk_count.
        """
        results = []
        for doc in self._docs.find(
            {},
            sort=[("ingested_at", DESCENDING)],
        ):
            doc["doc_id"] = doc.pop("_id")
            for key in ("ingested_at", "approved_at", "pushed_at"):
                val = doc.get(key)
                if isinstance(val, datetime):
                    doc[key] = val.isoformat()
            results.append(doc)
        return results


# ---------------------------------------------------------------------------
# KBLedger — permanent record of pushed documents for drift detection
# ---------------------------------------------------------------------------

class KBLedger:
    """
    Permanent ledger of every document that has been pushed to the vector store.

    Collection: ``kb_documents``

    Each document record carries enough information to detect drift:
    - For file sources: ``source_mtime`` (last-modified timestamp) and
      ``source_size`` (bytes).
    - For URL sources: the ``url`` field itself (content-change detection
      requires a separate fetch).

    Drift statuses:
    - ``current``  — source unchanged since last push
    - ``stale``    — source has changed (mtime/size differs)
    - ``deleted``  — source file no longer exists
    - ``unknown``  — cannot determine (URL sources, missing fields)
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or _get_db()
        self._coll: Collection = self._db[_coll_name("kb_documents")]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._coll.create_index("kb_name")
        self._coll.create_index("source_path")
        self._coll.create_index("drift_status")
        self._coll.create_index("pushed_at")

    def record_push(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        source_type: str,
        url: str | None,
        chunk_ids: list[str],
        tags: list[str],
        quality_score: float,
        kb_name: str = "default",
    ) -> None:
        """
        Record (or update) a document in the ledger after a successful push.

        Captures source file mtime/size if the source is a local file path
        that currently exists on disk.
        """
        now = datetime.now(timezone.utc)
        source_mtime: float | None = None
        source_size: int | None = None

        if source_path and source_type not in ("url", "jsonl"):
            try:
                stat = os.stat(source_path)
                source_mtime = stat.st_mtime
                source_size  = stat.st_size
            except OSError:
                pass   # file may have been deleted / moved already

        record = {
            "_id":             doc_id,
            "title":           title,
            "source_path":     source_path,
            "source_type":     source_type,
            "url":             url or "",
            "chunk_ids":       chunk_ids,
            "chunk_count":     len(chunk_ids),
            "tags":            tags,
            "quality_score":   quality_score,
            "kb_name":         kb_name,
            "pushed_at":       now,
            "drift_status":    "current",
            "drift_checked_at": None,
            "source_mtime":    source_mtime,
            "source_size":     source_size,
        }

        self._coll.replace_one({"_id": doc_id}, record, upsert=True)
        logger.debug("Ledger: recorded push for doc %s (kb=%s)", doc_id, kb_name)

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift_one(self, doc_id: str) -> str:
        """
        Check drift status for a single document and update the record.

        Returns the new drift status string.
        """
        now = datetime.now(timezone.utc)
        rec = self._coll.find_one({"_id": doc_id})
        if rec is None:
            return "unknown"

        source_path = rec.get("source_path", "")
        source_type = rec.get("source_type", "")

        if source_type in ("url", "jsonl") or not source_path:
            # Cannot determine drift without fetching the URL
            status = "unknown"
        else:
            try:
                stat = os.stat(source_path)
                stored_mtime = rec.get("source_mtime")
                stored_size  = rec.get("source_size")

                if stored_mtime is None and stored_size is None:
                    status = "unknown"
                elif (
                    abs(stat.st_mtime - (stored_mtime or 0)) > 1.0
                    or stat.st_size != stored_size
                ):
                    status = "stale"
                else:
                    status = "current"
            except FileNotFoundError:
                status = "deleted"
            except OSError:
                status = "unknown"

        self._coll.update_one(
            {"_id": doc_id},
            {"$set": {"drift_status": status, "drift_checked_at": now}},
        )
        return status

    def run_drift_check(
        self,
        kb_name: str | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """
        Run drift check for all (or a named KB's) documents.

        Returns a tally: ``{"current": N, "stale": N, "deleted": N, "unknown": N}``
        """
        query: dict[str, Any] = {}
        if kb_name:
            query["kb_name"] = kb_name

        doc_ids = [d["_id"] for d in self._coll.find(query, {"_id": 1})]
        total = len(doc_ids)
        tally: dict[str, int] = {"current": 0, "stale": 0, "deleted": 0, "unknown": 0}

        for i, did in enumerate(doc_ids, 1):
            status = self.check_drift_one(did)
            tally[status] = tally.get(status, 0) + 1
            if progress_cb:
                progress_cb(i, total)

        logger.info(
            "Drift check complete (kb=%s): %s",
            kb_name or "all", tally,
        )
        return tally

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_kb_names(self) -> list[str]:
        """Return a sorted list of all distinct kb_name values."""
        return sorted(self._coll.distinct("kb_name"))

    def get_stats(self, kb_name: str | None = None) -> dict[str, Any]:
        """
        Return aggregate stats for one KB or all KBs.

        Returns:
          total_docs, total_chunks, drift_counts (dict), last_push (ISO str | None)
        """
        match: dict[str, Any] = {}
        if kb_name:
            match["kb_name"] = kb_name

        pipeline = [
            *(([{"$match": match}]) if match else []),
            {
                "$group": {
                    "_id": None,
                    "total_docs":   {"$sum": 1},
                    "total_chunks": {"$sum": "$chunk_count"},
                    "last_push":    {"$max": "$pushed_at"},
                }
            },
        ]

        agg = list(self._coll.aggregate(pipeline))
        if not agg:
            return {
                "total_docs":   0,
                "total_chunks": 0,
                "drift_counts": {"current": 0, "stale": 0, "deleted": 0, "unknown": 0},
                "last_push":    None,
            }

        row = agg[0]
        last_push = row.get("last_push")
        if isinstance(last_push, datetime):
            last_push = last_push.isoformat()

        # Drift counts
        drift_pipeline = [
            *(([{"$match": match}]) if match else []),
            {"$group": {"_id": "$drift_status", "n": {"$sum": 1}}},
        ]
        drift_counts: dict[str, int] = {
            "current": 0, "stale": 0, "deleted": 0, "unknown": 0,
        }
        for d in self._coll.aggregate(drift_pipeline):
            drift_counts[d["_id"] or "unknown"] = d["n"]

        return {
            "total_docs":   row["total_docs"],
            "total_chunks": row["total_chunks"],
            "drift_counts": drift_counts,
            "last_push":    last_push,
        }

    def list_docs(
        self,
        kb_name: str | None = None,
        drift_status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Return a list of document ledger records (newest push first).

        Optionally filter by ``kb_name`` and/or ``drift_status``.
        """
        query: dict[str, Any] = {}
        if kb_name:
            query["kb_name"] = kb_name
        if drift_status:
            query["drift_status"] = drift_status

        results = []
        for doc in self._coll.find(
            query,
            sort=[("pushed_at", DESCENDING)],
            limit=limit,
        ):
            doc["doc_id"] = doc.pop("_id")
            for key in ("pushed_at", "drift_checked_at"):
                val = doc.get(key)
                if isinstance(val, datetime):
                    doc[key] = val.isoformat()
            results.append(doc)
        return results

    def delete_doc(self, doc_id: str) -> bool:
        """Remove a document from the ledger (e.g. after it's been deleted from the KB)."""
        result = self._coll.delete_one({"_id": doc_id})
        return result.deleted_count > 0


# ---------------------------------------------------------------------------
# Module-level singletons (lazy)
# ---------------------------------------------------------------------------

_staging: MongoStagingStore | None = None
_ledger: KBLedger | None = None


def get_staging() -> MongoStagingStore:
    """Return (or create) the shared MongoStagingStore instance."""
    global _staging
    if _staging is None:
        _staging = MongoStagingStore()
    return _staging


def get_ledger() -> KBLedger:
    """Return (or create) the shared KBLedger instance."""
    global _ledger
    if _ledger is None:
        _ledger = KBLedger()
    return _ledger
