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
import uuid
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
    if settings.mongodb_tls_insecure:
        params.append("tlsAllowInvalidCertificates=true")
    if settings.mongodb_auth_mechanism:
        params.append(f"authMechanism={quote_plus(settings.mongodb_auth_mechanism)}")

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
        self._docs: Collection   = self._db[_coll_name(settings.mongodb_coll_staging_docs)]
        self._chunks: Collection = self._db[_coll_name(settings.mongodb_coll_staging_chunks)]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes the first time the store is used."""
        self._docs.create_index("status")
        self._docs.create_index("ingested_at")
        self._docs.create_index("usecase_id")
        self._docs.create_index("agent_filter")
        self._docs.create_index([("usecase_id", ASCENDING), ("agent_filter", ASCENDING)])
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
            "usecase_id":            meta.get("usecase_id") or None,
            "agent_filter":          meta.get("agent_filter") or None,
            "ingested_at":           now,
            "approved_at":           None,
            "pushed_at":             None,
            "reject_reason":         None,
        }

        # Upsert so re-ingest of the same doc_id refreshes the record
        self._docs.replace_one({"_id": doc_id}, doc, upsert=True)

        # Replace chunk records for this doc.
        # Use bulk upsert (ReplaceOne) rather than delete+insert so that
        # re-ingesting a source whose chunk count changed (and therefore
        # produces a new doc_id) never hits E11000 on pre-existing chunk_ids.
        if chunks:
            from pymongo import ReplaceOne
            ops = []
            for c in chunks:
                cd = dict(c)
                cd["doc_id"] = doc_id
                if "chunk_id" in cd:
                    cd["_id"] = cd.pop("chunk_id")
                ops.append(ReplaceOne({"_id": cd["_id"]}, cd, upsert=True))
            self._chunks.bulk_write(ops, ordered=False)

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

    def update_chunk(self, chunk_id: str, updates: dict[str, Any]) -> bool:
        """Update fields on a single staged chunk. Returns True if the chunk was found."""
        result = self._chunks.update_one({"_id": chunk_id}, {"$set": updates})
        return result.matched_count > 0

    def split_doc(
        self,
        source_doc_id: str,
        new_doc_id: str,
        chunk_ids: list[str],
        new_meta: dict[str, Any],
    ) -> int:
        """
        Move *chunk_ids* from *source_doc_id* into a freshly created doc
        (*new_doc_id*).  Updates the source doc's chunk_count.

        Returns the number of chunks actually moved.
        """
        now = datetime.now(timezone.utc)
        doc: dict[str, Any] = {
            "_id":                    new_doc_id,
            "title":                  new_meta.get("title", ""),
            "source_path":            new_meta.get("source_path", ""),
            "source_type":            new_meta.get("source_type", ""),
            "author":                 new_meta.get("author", ""),
            "created_date":           new_meta.get("created_date", ""),
            "url":                    new_meta.get("url", ""),
            "page_count":             new_meta.get("page_count", 0),
            "quality_score":          new_meta.get("quality_score", 1.0),
            "quality_passed":         new_meta.get("quality_passed", True),
            "quality_flags":          new_meta.get("quality_flags", []),
            "suggested_tags":         new_meta.get("suggested_tags", []),
            "chunk_count":            len(chunk_ids),
            "status":                 "pending_review",
            "schema_type":            new_meta.get("schema_type", ""),
            "unique_sources":         0,
            "has_embeddings":         False,
            "has_partial_embeddings": False,
            "kb_name":                new_meta.get("kb_name", "default"),
            "usecase_id":             new_meta.get("usecase_id") or None,
            "agent_filter":           new_meta.get("agent_filter") or None,
            "ingested_at":            now,
            "approved_at":            None,
            "pushed_at":              None,
            "reject_reason":          None,
        }
        self._docs.insert_one(doc)

        result = self._chunks.update_many(
            {"_id": {"$in": chunk_ids}},
            {"$set": {"doc_id": new_doc_id}},
        )
        moved = result.modified_count

        remaining = self._chunks.count_documents({"doc_id": source_doc_id})
        self._docs.update_one(
            {"_id": source_doc_id},
            {"$set": {"chunk_count": remaining}},
        )
        return moved

    def split_chunk(
        self,
        doc_id: str,
        source_chunk_id: str,
        content_parts: list[str],
    ) -> list[str]:
        """
        Replace *source_chunk_id* with N new chunks, one per entry in
        *content_parts*.  All metadata (section, tags, citation, page_number)
        is inherited from the original chunk unchanged.

        Returns the list of new chunk_ids, or [] if the operation was a
        no-op (chunk not found, or all content_parts were blank).
        """
        import copy

        original = self._chunks.find_one({"_id": source_chunk_id})
        if not original:
            return []

        clean_parts = [p.strip() for p in content_parts if p.strip()]
        if not clean_parts:
            return []

        new_ids: list[str] = []
        for part in clean_parts:
            new_chunk = copy.deepcopy(original)
            new_chunk["_id"] = str(uuid.uuid4())
            new_chunk["content"] = part
            self._chunks.insert_one(new_chunk)
            new_ids.append(new_chunk["_id"])

        self._chunks.delete_one({"_id": source_chunk_id})

        new_count = self._chunks.count_documents({"doc_id": doc_id})
        self._docs.update_one({"_id": doc_id}, {"$set": {"chunk_count": new_count}})

        return new_ids

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


    def get_chunks_by_usecase(
        self,
        usecase_id: str,
        agent_filter: str,
        status: str | None = "pushed",
    ) -> list[dict]:
        """
        Return all chunk dicts for docs matching (usecase_id, agent_filter).

        Optionally filter by doc status (default: pushed docs only).
        Used for JSONL export to external embedding pipelines.
        """
        query: dict[str, Any] = {
            "usecase_id":   usecase_id,
            "agent_filter": agent_filter,
        }
        if status:
            query["status"] = status
        doc_ids = [d["_id"] for d in self._docs.find(query, {"_id": 1})]
        if not doc_ids:
            return []

        chunks = []
        for cd in self._chunks.find({"doc_id": {"$in": doc_ids}}):
            cd["chunk_id"] = str(cd.pop("_id"))
            cd.pop("doc_id", None)
            chunks.append(cd)
        return chunks


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
        self._coll: Collection = self._db[_coll_name(settings.mongodb_coll_kb_documents)]
        self._snaps: Collection = self._db[_coll_name(settings.mongodb_coll_kb_snapshots)]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._coll.create_index("kb_name")
        self._coll.create_index("source_path")
        self._coll.create_index("drift_status")
        self._coll.create_index("pushed_at")
        self._coll.create_index("usecase_id")
        self._coll.create_index("agent_filter")
        self._coll.create_index([("usecase_id", ASCENDING), ("agent_filter", ASCENDING)])
        self._snaps.create_index("created_at")

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
        usecase_id: str | None = None,
        agent_filter: str | None = None,
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
            "usecase_id":      usecase_id or None,
            "agent_filter":    agent_filter or None,
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

    def list_docs_by_usecase(
        self,
        usecase_id: str,
        agent_filter: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return kb_documents records for a specific usecase+agent pair (newest first)."""
        query: dict[str, Any] = {"usecase_id": usecase_id, "agent_filter": agent_filter}
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

    def record_snapshot(self, pushed_doc_ids: list[str]) -> str:
        """
        Store a point-in-time snapshot of the full KB ledger state.

        Called automatically by push_approved() after a successful push.
        Captures lightweight doc summaries (no chunk content) so the full
        KB composition at each push event is queryable later.

        Returns the snapshot_id.
        """
        now = datetime.now(timezone.utc)
        snapshot_id = str(uuid.uuid4())

        # Collect current state of every doc in the ledger
        doc_summaries = []
        for doc in self._coll.find(
            {},
            {
                "title": 1, "source_path": 1, "source_type": 1,
                "kb_name": 1, "chunk_count": 1, "quality_score": 1,
                "tags": 1, "pushed_at": 1, "drift_status": 1,
            },
        ):
            doc_id = doc.pop("_id")
            pushed_at = doc.get("pushed_at")
            if isinstance(pushed_at, datetime):
                doc["pushed_at"] = pushed_at.isoformat()
            doc["doc_id"] = doc_id
            doc_summaries.append(doc)

        total_chunks = sum(d.get("chunk_count", 0) for d in doc_summaries)

        record: dict[str, Any] = {
            "_id":              snapshot_id,
            "created_at":       now,
            "pushed_doc_ids":   pushed_doc_ids,
            "pushed_doc_count": len(pushed_doc_ids),
            "total_docs":       len(doc_summaries),
            "total_chunks":     total_chunks,
            "docs":             doc_summaries,
        }
        self._snaps.insert_one(record)
        logger.debug("Ledger snapshot %s recorded (%d total docs)", snapshot_id, len(doc_summaries))
        return snapshot_id

    def list_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent snapshots (newest first), without the full docs list."""
        results = []
        for snap in self._snaps.find(
            {},
            {"docs": 0},  # exclude the docs array for the summary list
            sort=[("created_at", DESCENDING)],
            limit=limit,
        ):
            snap["snapshot_id"] = snap.pop("_id")
            created_at = snap.get("created_at")
            if isinstance(created_at, datetime):
                snap["created_at"] = created_at.isoformat()
            results.append(snap)
        return results

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        """Return a full snapshot record including the docs list, or None."""
        snap = self._snaps.find_one({"_id": snapshot_id})
        if snap is None:
            return None
        snap["snapshot_id"] = snap.pop("_id")
        created_at = snap.get("created_at")
        if isinstance(created_at, datetime):
            snap["created_at"] = created_at.isoformat()
        return snap

    def delete_doc(self, doc_id: str) -> bool:
        """Remove a document from the ledger (e.g. after it's been deleted from the KB)."""
        result = self._coll.delete_one({"_id": doc_id})
        return result.deleted_count > 0


# ---------------------------------------------------------------------------
# UsecaseLedger — tracks pushed content per (usecase_id, agent_filter) pair
# ---------------------------------------------------------------------------

class UsecaseLedger:
    """
    Tracks what has been pushed to the vector DB per (usecase_id, agent_filter) pair,
    and manages Confluence page source registrations with refresh schedules.

    Collections used:
    - ``usecase_ledger``            — pushed chunk inventory per usecase+agent
    - ``usecase_confluence_sources`` — registered page URLs and refresh schedules
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or _get_db()
        self._coll: Collection    = self._db[_coll_name(settings.mongodb_coll_usecase_ledger)]
        self._sources: Collection = self._db[_coll_name(settings.mongodb_coll_usecase_confluence)]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._coll.create_index("usecase_id")
        self._coll.create_index("agent_filter")
        self._coll.create_index(
            [("usecase_id", ASCENDING), ("agent_filter", ASCENDING)], unique=True
        )
        self._sources.create_index("usecase_id")
        self._sources.create_index("agent_filter")
        self._sources.create_index(
            [("usecase_id", ASCENDING), ("agent_filter", ASCENDING)], unique=True
        )
        self._sources.create_index("next_refresh_at")

    @staticmethod
    def _make_id(usecase_id: str, agent_filter: str) -> str:
        return f"{usecase_id}|{agent_filter}"

    # ------------------------------------------------------------------
    # Usecase ledger write operations
    # ------------------------------------------------------------------

    def record_push(
        self,
        usecase_id: str,
        agent_filter: str,
        kb_name: str,
        doc_ids: list[str],
        chunk_ids: list[str],
    ) -> None:
        """Upsert the usecase ledger entry after a successful vector push."""
        now = datetime.now(timezone.utc)
        entry_id = self._make_id(usecase_id, agent_filter)
        self._coll.update_one(
            {"_id": entry_id},
            {
                "$set": {
                    "usecase_id":      usecase_id,
                    "agent_filter":    agent_filter,
                    "kb_name":         kb_name,
                    "last_pushed_at":  now,
                    "updated_at":      now,
                    "chunk_count":     0,  # recalculated below
                },
                "$setOnInsert": {"created_at": now},
                "$addToSet": {
                    "doc_ids":   {"$each": doc_ids},
                    "chunk_ids": {"$each": chunk_ids},
                },
            },
            upsert=True,
        )
        # Recalculate chunk_count from the actual stored array length
        entry = self._coll.find_one({"_id": entry_id}, {"chunk_ids": 1})
        if entry:
            count = len(entry.get("chunk_ids") or [])
            self._coll.update_one({"_id": entry_id}, {"$set": {"chunk_count": count}})

    def remove_chunks(
        self,
        usecase_id: str,
        agent_filter: str,
        chunk_ids: list[str],
    ) -> None:
        """Remove specific chunk_ids from a usecase ledger entry."""
        entry_id = self._make_id(usecase_id, agent_filter)
        self._coll.update_one(
            {"_id": entry_id},
            {
                "$pull": {"chunk_ids": {"$in": chunk_ids}},
                "$set":  {"updated_at": datetime.now(timezone.utc)},
            },
        )
        entry = self._coll.find_one({"_id": entry_id}, {"chunk_ids": 1})
        if entry:
            count = len(entry.get("chunk_ids") or [])
            self._coll.update_one({"_id": entry_id}, {"$set": {"chunk_count": count}})

    # ------------------------------------------------------------------
    # Usecase ledger read operations
    # ------------------------------------------------------------------

    def get_entry(self, usecase_id: str, agent_filter: str) -> dict[str, Any] | None:
        """Return the ledger entry for a usecase+agent pair, or None."""
        entry = self._coll.find_one({"_id": self._make_id(usecase_id, agent_filter)})
        if entry is None:
            return None
        entry.pop("_id", None)
        for key in ("last_pushed_at", "last_ingested_at", "created_at", "updated_at"):
            val = entry.get(key)
            if isinstance(val, datetime):
                entry[key] = val.isoformat()
        return entry

    def list_entries(self) -> list[dict[str, Any]]:
        """Return all usecase ledger entries (summary — excludes full chunk_ids list)."""
        results = []
        for entry in self._coll.find({}, {"chunk_ids": 0}, sort=[("updated_at", DESCENDING)]):
            entry.pop("_id", None)
            for key in ("last_pushed_at", "last_ingested_at", "created_at", "updated_at"):
                val = entry.get(key)
                if isinstance(val, datetime):
                    entry[key] = val.isoformat()
            results.append(entry)
        return results

    def get_distinct_usecases(self) -> list[str]:
        """Return sorted list of distinct usecase_id values."""
        return sorted(v for v in self._coll.distinct("usecase_id") if v)

    def get_agent_filters_for_usecase(self, usecase_id: str) -> list[str]:
        """Return sorted agent_filter values for a specific usecase_id."""
        return sorted(
            v for v in self._coll.distinct("agent_filter", {"usecase_id": usecase_id}) if v
        )

    def get_chunk_ids(self, usecase_id: str, agent_filter: str) -> list[str]:
        """Return the live chunk_ids for a usecase+agent pair."""
        entry = self._coll.find_one(
            {"_id": self._make_id(usecase_id, agent_filter)}, {"chunk_ids": 1}
        )
        return list(entry.get("chunk_ids") or []) if entry else []

    # ------------------------------------------------------------------
    # Confluence source management
    # ------------------------------------------------------------------

    def upsert_confluence_source(
        self,
        usecase_id: str,
        agent_filter: str,
        kb_name: str,
        page_urls: list[str],
        max_depth: int = -1,
        extra_tags: list[str] | None = None,
        refresh_cron: str | None = None,
    ) -> None:
        """Register or update Confluence page URLs for a usecase+agent pair."""
        now = datetime.now(timezone.utc)
        next_refresh: datetime | None = None
        if refresh_cron:
            try:
                from croniter import croniter
                next_refresh = croniter(refresh_cron, now).get_next(datetime)
            except Exception:
                pass

        self._sources.update_one(
            {"usecase_id": usecase_id, "agent_filter": agent_filter},
            {
                "$set": {
                    "usecase_id":      usecase_id,
                    "agent_filter":    agent_filter,
                    "kb_name":         kb_name,
                    "page_urls":       page_urls,
                    "max_depth":       max_depth,
                    "extra_tags":      extra_tags or [],
                    "refresh_cron":    refresh_cron or None,
                    "next_refresh_at": next_refresh,
                    "updated_at":      now,
                },
                "$setOnInsert": {
                    "created_at":      now,
                    "last_refresh_at": None,
                    "refresh_status":  "idle",
                    "refresh_error":   None,
                },
            },
            upsert=True,
        )

    def add_url_to_confluence_source(
        self, usecase_id: str, agent_filter: str, url: str
    ) -> None:
        """Add a single URL to the page_urls list for a source (no-op if already present)."""
        self._sources.update_one(
            {"usecase_id": usecase_id, "agent_filter": agent_filter},
            {"$addToSet": {"page_urls": url}},
        )

    def remove_url_from_confluence_source(
        self, usecase_id: str, agent_filter: str, url: str
    ) -> None:
        """Remove a single URL from the page_urls list for a source."""
        self._sources.update_one(
            {"usecase_id": usecase_id, "agent_filter": agent_filter},
            {"$pull": {"page_urls": url}},
        )

    def get_confluence_source(
        self, usecase_id: str, agent_filter: str
    ) -> dict[str, Any] | None:
        """Return the Confluence source config for a usecase+agent pair, or None."""
        doc = self._sources.find_one({"usecase_id": usecase_id, "agent_filter": agent_filter})
        if doc is None:
            return None
        doc["source_id"] = str(doc.pop("_id"))
        for key in ("last_refresh_at", "next_refresh_at", "created_at", "updated_at"):
            val = doc.get(key)
            if isinstance(val, datetime):
                doc[key] = val.isoformat()
        return doc

    def list_confluence_sources(self) -> list[dict[str, Any]]:
        """Return all registered Confluence source configs (excludes page snapshots)."""
        results = []
        for doc in self._sources.find(
            {}, {"crawled_pages": 0}, sort=[("usecase_id", ASCENDING)]
        ):
            doc["source_id"] = str(doc.pop("_id"))
            for key in ("last_refresh_at", "next_refresh_at", "created_at", "updated_at"):
                val = doc.get(key)
                if isinstance(val, datetime):
                    doc[key] = val.isoformat()
            results.append(doc)
        return results

    def get_sources_due_for_refresh(self) -> list[dict[str, Any]]:
        """Return source configs where next_refresh_at <= now and not currently running."""
        now = datetime.now(timezone.utc)
        query = {
            "next_refresh_at": {"$lte": now},
            "refresh_status":  {"$ne": "running"},
        }
        results = []
        for doc in self._sources.find(query):
            doc["source_id"] = str(doc.pop("_id"))
            results.append(doc)
        return results

    def mark_refresh_running(self, source_id: str) -> None:
        from bson import ObjectId
        self._sources.update_one(
            {"_id": ObjectId(source_id)},
            {"$set": {"refresh_status": "running", "refresh_error": None}},
        )

    def mark_refresh_done(self, source_id: str) -> None:
        from bson import ObjectId
        now = datetime.now(timezone.utc)
        self._sources.update_one(
            {"_id": ObjectId(source_id)},
            {"$set": {
                "refresh_status":  "idle",
                "last_refresh_at": now,
                "refresh_error":   None,
            }},
        )

    def mark_refresh_failed(self, source_id: str, error: str) -> None:
        from bson import ObjectId
        self._sources.update_one(
            {"_id": ObjectId(source_id)},
            {"$set": {
                "refresh_status": "failed",
                "refresh_error":  error,
                "next_refresh_at": None,
            }},
        )

    def update_next_refresh(self, source_id: str, cron_expr: str) -> None:
        """Compute and store the next fire time from a cron expression."""
        from bson import ObjectId
        from croniter import croniter
        now = datetime.now(timezone.utc)
        next_dt = croniter(cron_expr, now).get_next(datetime)
        self._sources.update_one(
            {"_id": ObjectId(source_id)},
            {"$set": {"next_refresh_at": next_dt}},
        )

    def record_crawl_snapshot(self, source_id: str, pages: list[dict]) -> None:
        """
        Store page metadata snapshot after a crawl for drift tracking.
        Each entry: ``{page_id, title, version, last_modified}``.
        """
        from bson import ObjectId
        self._sources.update_one(
            {"_id": ObjectId(source_id)},
            {"$set": {
                "crawled_pages":      pages,
                "crawled_page_count": len(pages),
                "updated_at":         datetime.now(timezone.utc),
            }},
        )

    def get_crawl_snapshot(self, source_id: str) -> list[dict]:
        """Return the stored page metadata snapshot for a source, or []."""
        from bson import ObjectId
        doc = self._sources.find_one(
            {"_id": ObjectId(source_id)}, {"crawled_pages": 1}
        )
        return (doc or {}).get("crawled_pages") or []


# ---------------------------------------------------------------------------
# Module-level singletons (lazy)
# ---------------------------------------------------------------------------

_staging: MongoStagingStore | None = None
_ledger: KBLedger | None = None
_usecase_ledger: UsecaseLedger | None = None


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


def get_usecase_ledger() -> UsecaseLedger:
    """Return (or create) the shared UsecaseLedger instance."""
    global _usecase_ledger
    if _usecase_ledger is None:
        _usecase_ledger = UsecaseLedger()
    return _usecase_ledger
