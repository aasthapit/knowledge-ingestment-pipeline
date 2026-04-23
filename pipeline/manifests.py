"""
manifests.py
Document Manifest system — named, versioned snapshots of corpus state.

A manifest groups source documents (Confluence pages, JSONL imports, file
uploads) into an operable record with full provenance per entry: doc_id,
version_id, file_id, object_id, and lifecycle timestamps.

Key operations
--------------
snapshot_corpus_to_manifest   Save current kb_documents state as a frozen manifest.
create_manifest_from_sources  Pre-populate a manifest with pending source refs.
ingest_from_manifest          Re-crawl / re-import Confluence sources.
remove_manifest_docs          Delete chunks from Redis + KB ledger.
diff_manifests                Compute added / removed / changed entries between two manifests.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from pipeline.config import settings
from pipeline.mongo_store import _get_db, _coll_name

logger = logging.getLogger(__name__)


class ManifestManager:
    """
    Manages document manifests.

    Collection: ``doc_manifests``

    Manifest statuses
    -----------------
    open      Still accepting new entries.
    frozen    Immutable corpus snapshot.
    archived  Soft-deleted; excluded from default listing.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or _get_db()
        self._coll: Collection = self._db[_coll_name(settings.mongodb_coll_doc_manifests)]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._coll.create_index("corpus_id")
        self._coll.create_index("status")
        self._coll.create_index("created_at")
        self._coll.create_index("entries.doc_id")
        self._coll.create_index("entries.source_ref")

    # ------------------------------------------------------------------
    # Create / Mutate
    # ------------------------------------------------------------------

    def create_manifest(
        self,
        name: str,
        corpus_id: str | None = None,
        description: str = "",
        created_by: str = "system",
        tags: list[str] | None = None,
        status: str = "open",
    ) -> str:
        """Create a new manifest and return its manifest_id."""
        now = datetime.now(timezone.utc)
        manifest_id = str(uuid.uuid4())
        self._coll.insert_one({
            "_id":          manifest_id,
            "name":         name,
            "description":  description,
            "corpus_id":    corpus_id or None,
            "status":       status,
            "created_at":   now,
            "updated_at":   now,
            "frozen_at":    None,
            "created_by":   created_by,
            "entries":      [],
            "entry_count":  0,
            "pushed_count": 0,
            "tags":         tags or [],
        })
        logger.debug("Created manifest %s (%s)", manifest_id, name)
        return manifest_id

    def freeze_manifest(self, manifest_id: str) -> bool:
        """Set status='frozen'. Returns True if manifest was found."""
        now = datetime.now(timezone.utc)
        result = self._coll.update_one(
            {"_id": manifest_id},
            {"$set": {"status": "frozen", "frozen_at": now, "updated_at": now}},
        )
        return result.matched_count > 0

    def archive_manifest(self, manifest_id: str) -> bool:
        """Set status='archived'. Returns True if manifest was found."""
        result = self._coll.update_one(
            {"_id": manifest_id},
            {"$set": {"status": "archived", "updated_at": datetime.now(timezone.utc)}},
        )
        return result.matched_count > 0

    def rename_manifest(self, manifest_id: str, name: str) -> bool:
        """Update manifest name. Returns True if manifest was found."""
        result = self._coll.update_one(
            {"_id": manifest_id},
            {"$set": {"name": name, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.matched_count > 0

    # ------------------------------------------------------------------
    # Entry Management
    # ------------------------------------------------------------------

    def add_entry(
        self,
        manifest_id: str,
        doc_id: str,
        object_id: str,
        file_id: str,
        version_id: str,
        source_type: str,
        source_ref: str,
        title: str,
        kb_id: str | None = None,
        status: str = "pending",
        staged_at: datetime | None = None,
        pushed_at: datetime | None = None,
    ) -> bool:
        """
        Upsert a manifest entry by doc_id.

        Raises ValueError when the manifest is frozen or archived.
        Returns True if manifest was found.
        """
        manifest = self._coll.find_one({"_id": manifest_id}, {"status": 1})
        if not manifest:
            return False
        if manifest.get("status") in ("frozen", "archived"):
            raise ValueError(
                f"Manifest {manifest_id!r} is {manifest['status']} and cannot be modified."
            )

        now = datetime.now(timezone.utc)
        entry: dict[str, Any] = {
            "doc_id":      doc_id,
            "object_id":   object_id,
            "file_id":     file_id,
            "version_id":  version_id,
            "status":      status,
            "source_type": source_type,
            "source_ref":  source_ref,
            "title":       title,
            "kb_id":       kb_id or None,
            "staged_at":   staged_at or now,
            "pushed_at":   pushed_at,
            "removed_at":  None,
        }

        # Try updating existing entry
        result = self._coll.update_one(
            {"_id": manifest_id, "entries.doc_id": doc_id},
            {
                "$set": {
                    "entries.$[e].object_id":   object_id,
                    "entries.$[e].file_id":     file_id,
                    "entries.$[e].version_id":  version_id,
                    "entries.$[e].status":      status,
                    "entries.$[e].source_type": source_type,
                    "entries.$[e].source_ref":  source_ref,
                    "entries.$[e].title":       title,
                    "entries.$[e].staged_at":   staged_at or now,
                    "entries.$[e].pushed_at":   pushed_at,
                    "updated_at":               now,
                }
            },
            array_filters=[{"e.doc_id": doc_id}],
        )
        if result.matched_count == 0:
            # New entry
            self._coll.update_one(
                {"_id": manifest_id},
                {"$push": {"entries": entry}, "$set": {"updated_at": now}},
            )
        self._recalc_counts(manifest_id)
        return True

    def update_entry_status(
        self,
        manifest_id: str,
        doc_id: str,
        status: str,
        pushed_at: datetime | None = None,
        removed_at: datetime | None = None,
    ) -> bool:
        """Update entry status and optional timestamps. Returns True if entry was found."""
        updates: dict[str, Any] = {
            "entries.$[e].status": status,
            "updated_at": datetime.now(timezone.utc),
        }
        if pushed_at is not None:
            updates["entries.$[e].pushed_at"] = pushed_at
        if removed_at is not None:
            updates["entries.$[e].removed_at"] = removed_at

        result = self._coll.update_one(
            {"_id": manifest_id, "entries.doc_id": doc_id},
            {"$set": updates},
            array_filters=[{"e.doc_id": doc_id}],
        )
        if result.matched_count > 0:
            self._recalc_counts(manifest_id)
        return result.matched_count > 0

    def remove_entry(self, manifest_id: str, doc_id: str) -> bool:
        """Hard-remove an entry from the entries array. Returns True if removed."""
        result = self._coll.update_one(
            {"_id": manifest_id},
            {
                "$pull": {"entries": {"doc_id": doc_id}},
                "$set":  {"updated_at": datetime.now(timezone.utc)},
            },
        )
        if result.modified_count > 0:
            self._recalc_counts(manifest_id)
        return result.modified_count > 0

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    def snapshot_corpus_to_manifest(
        self,
        corpus_id: str,
        manifest_name: str,
        created_by: str = "system",
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Save the current pushed state of all KBs in a corpus as a frozen manifest.

        Entries represent all JSONL docs in the corpus at the time of snapshot.
        Returns the new manifest_id.
        """
        from pipeline.mongo_store import get_ledger, get_corpus_store

        corpus = get_corpus_store().get(corpus_id)
        if not corpus:
            raise ValueError(f"Corpus {corpus_id!r} not found.")

        kb_ids = corpus.get("kb_ids") or []
        manifest_id = self.create_manifest(
            name=manifest_name,
            corpus_id=corpus_id,
            description=description or f"Snapshot of corpus '{corpus.get('name', corpus_id)}'",
            created_by=created_by,
            tags=tags or [],
            status="open",
        )

        ledger = get_ledger()
        now = datetime.now(timezone.utc)
        entries: list[dict[str, Any]] = []

        # Collect all pushed docs that belong to any KB in this corpus
        query: dict[str, Any] = {"kb_id": {"$in": kb_ids}} if kb_ids else {}
        for doc in ledger._coll.find(query):
            doc_id       = str(doc.get("_id", ""))
            pushed_at_raw = doc.get("pushed_at")
            pushed_dt: datetime | None = None
            if isinstance(pushed_at_raw, datetime):
                pushed_dt = pushed_at_raw
            elif isinstance(pushed_at_raw, str):
                try:
                    pushed_dt = datetime.fromisoformat(pushed_at_raw)
                except Exception:
                    pass

            source_ref = doc.get("source_path", "") or doc.get("url", "")
            version_id = hashlib.sha256(
                f"{doc_id}:{source_ref}:{pushed_at_raw}".encode()
            ).hexdigest()[:16]

            entries.append({
                "doc_id":      doc_id,
                "object_id":   doc_id,
                "file_id":     doc_id,
                "version_id":  version_id,
                "status":      "pushed",
                "source_type": doc.get("source_type", ""),
                "source_ref":  source_ref,
                "title":       doc.get("title", ""),
                "kb_id":       doc.get("kb_id") or None,
                "staged_at":   pushed_dt or now,
                "pushed_at":   pushed_dt or now,
                "removed_at":  None,
            })

        pushed_count = sum(1 for e in entries if e["status"] == "pushed")
        self._coll.update_one(
            {"_id": manifest_id},
            {
                "$set": {
                    "entries":      entries,
                    "entry_count":  len(entries),
                    "pushed_count": pushed_count,
                    "updated_at":   now,
                }
            },
        )
        self.freeze_manifest(manifest_id)
        logger.info(
            "Snapshot manifest %s: %d docs from corpus %s", manifest_id, len(entries), corpus_id
        )
        return manifest_id

    def create_manifest_from_sources(
        self,
        name: str,
        source_refs: list[str],
        source_type: str,
        corpus_id: str | None = None,
        kb_id: str | None = None,
        created_by: str = "system",
        tags: list[str] | None = None,
        description: str = "",
    ) -> str:
        """Create an open manifest pre-populated with pending source refs."""
        manifest_id = self.create_manifest(
            name=name,
            corpus_id=corpus_id,
            description=description,
            created_by=created_by,
            tags=tags or [],
        )

        entries = [
            {
                "doc_id":      "",
                "object_id":   "",
                "file_id":     "",
                "version_id":  "",
                "status":      "pending",
                "source_type": source_type,
                "source_ref":  ref,
                "title":       ref,
                "kb_id":       kb_id or None,
                "staged_at":   None,
                "pushed_at":   None,
                "removed_at":  None,
            }
            for ref in source_refs
        ]

        if entries:
            now = datetime.now(timezone.utc)
            self._coll.update_one(
                {"_id": manifest_id},
                {
                    "$set": {
                        "entries":      entries,
                        "entry_count":  len(entries),
                        "pushed_count": 0,
                        "updated_at":   now,
                    }
                },
            )
        return manifest_id

    def ingest_from_manifest(
        self,
        manifest_id: str,
        kb_id: str | None = None,
        extra_tags: list[str] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Re-import all Confluence sources in a manifest into a Knowledge Base.

        File-upload and JSONL entries without a crawlable source_ref are counted
        as skipped — they require manual re-upload.

        Returns
        -------
        dict
            ``{"ingested": int, "skipped": int, "errors": list[str]}``
        """
        import io as _io
        import json as _json
        from pipeline.ingest import ingest_jsonl

        manifest = self.get_manifest(manifest_id)
        if not manifest:
            return {"ingested": 0, "skipped": 0, "errors": [f"Manifest {manifest_id!r} not found."]}

        # Use explicit kb_id; fall back to the first entry's kb_id
        resolved_kb_id = kb_id or next(
            (e.get("kb_id") for e in (manifest.get("entries") or []) if e.get("kb_id")),
            None,
        )

        entries = manifest.get("entries") or []
        confluence_entries = [e for e in entries if e.get("source_type") == "confluence"]
        skipped = len(entries) - len(confluence_entries)
        ingested = 0
        errors: list[str] = []

        if not confluence_entries:
            return {"ingested": 0, "skipped": skipped, "errors": errors}

        if not settings.confluence_base_url or not settings.confluence_api_token:
            return {
                "ingested": 0,
                "skipped": len(entries),
                "errors": ["Confluence credentials not configured (CONFLUENCE_BASE_URL / CONFLUENCE_API_TOKEN)."],
            }

        from pipeline.confluence import ConfluenceCrawler
        crawler = ConfluenceCrawler(
            base_url=settings.confluence_base_url,
            auth_type=settings.confluence_auth_type,
            email=settings.confluence_email,
            api_token=settings.confluence_api_token,
            verify_ssl=settings.confluence_verify_ssl,
        )

        total = len(confluence_entries)
        for i, entry in enumerate(confluence_entries):
            source_ref = entry.get("source_ref", "")
            if not source_ref:
                skipped += 1
                continue
            try:
                pages = crawler.crawl(source_ref)
                if not pages:
                    skipped += 1
                    continue

                lines = [
                    _json.dumps({
                        "text":                p.content_text,
                        "page_url":            p.url,
                        "page_name":           p.title,
                        "section_breadcrumbs": p.ancestors,
                        "section_heading":     "",
                        "chunk_id":            "",
                    })
                    for p in pages
                ]
                buf = _io.BytesIO("\n".join(lines).encode("utf-8"))
                ingest_jsonl(
                    source=buf,
                    batch_name=source_ref[:80],
                    extra_tags=extra_tags,
                    kb_id=resolved_kb_id,
                    manifest_id=manifest_id,
                )
                ingested += 1
            except Exception as exc:
                errors.append(f"{source_ref}: {exc}")
                logger.warning("ingest_from_manifest: error for %s: %s", source_ref, exc)

            if progress_cb:
                progress_cb(i + 1, total)

        return {"ingested": ingested, "skipped": skipped, "errors": errors}

    def remove_manifest_docs(
        self,
        manifest_id: str,
        doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Remove corpus documents for this manifest.

        Deletes chunk vectors from Redis, removes records from kb_documents,
        updates the usecase ledger, and marks all referencing manifest entries
        as 'removed'.

        Parameters
        ----------
        doc_ids:
            Specific doc_ids to remove. Defaults to all pushed entries.
        """
        from pipeline.mongo_store import get_ledger, get_usecase_ledger
        from pipeline import redis_store

        manifest = self.get_manifest(manifest_id)
        if not manifest:
            return {"removed_docs": 0, "removed_chunks": 0, "errors": [f"Manifest {manifest_id!r} not found."]}

        entries = manifest.get("entries") or []
        if doc_ids is not None:
            target = {e["doc_id"] for e in entries if e.get("doc_id") in doc_ids}
        else:
            target = {e["doc_id"] for e in entries if e.get("status") == "pushed" and e.get("doc_id")}

        ledger    = get_ledger()
        uc_ledger = get_usecase_ledger()
        now       = datetime.now(timezone.utc)

        removed_docs   = 0
        removed_chunks = 0
        errors: list[str] = []

        for did in target:
            try:
                kb_rec = ledger._coll.find_one(
                    {"_id": did}, {"chunk_ids": 1, "usecase_id": 1, "agent_filter": 1}
                )
                if kb_rec:
                    chunk_ids = kb_rec.get("chunk_ids") or []
                    if chunk_ids:
                        try:
                            cnt = redis_store.delete_chunks(chunk_ids)
                            removed_chunks += cnt
                        except Exception as exc:
                            errors.append(f"Redis delete {did}: {exc}")

                        uc  = kb_rec.get("usecase_id")
                        ag  = kb_rec.get("agent_filter")
                        if uc and ag:
                            try:
                                uc_ledger.remove_chunks(uc, ag, chunk_ids)
                            except Exception as exc:
                                errors.append(f"Usecase ledger {did}: {exc}")

                    ledger.delete_doc(did)
                    removed_docs += 1

                # Mark entry as removed in every manifest that references this doc_id
                for mf in self.find_manifests_by_doc_id(did):
                    try:
                        self.update_entry_status(
                            mf["manifest_id"], did, "removed", removed_at=now
                        )
                    except Exception as exc:
                        errors.append(f"Manifest entry update {mf['manifest_id']}/{did}: {exc}")

            except Exception as exc:
                errors.append(f"Remove {did}: {exc}")
                logger.warning("remove_manifest_docs: error removing %s: %s", did, exc)

        return {"removed_docs": removed_docs, "removed_chunks": removed_chunks, "errors": errors}

    def diff_manifests(
        self,
        manifest_id_a: str,
        manifest_id_b: str,
    ) -> dict[str, Any]:
        """
        Diff two manifests by doc_id and version_id.

        Returns
        -------
        dict with keys:
            added     — in B but not in A
            removed   — in A but not in B
            changed   — same doc_id, different version_id
            unchanged — same doc_id, same version_id
        """
        def _ser(entry: dict) -> dict:
            out = dict(entry)
            for k in ("staged_at", "pushed_at", "removed_at"):
                v = out.get(k)
                if isinstance(v, datetime):
                    out[k] = v.isoformat()
            return out

        mf_a = self.get_manifest(manifest_id_a)
        mf_b = self.get_manifest(manifest_id_b)

        a_map = {e["doc_id"]: e for e in (mf_a or {}).get("entries", []) if e.get("doc_id")}
        b_map = {e["doc_id"]: e for e in (mf_b or {}).get("entries", []) if e.get("doc_id")}

        added: list[dict] = []
        changed: list[dict] = []
        unchanged: list[dict] = []

        for did, entry in b_map.items():
            if did not in a_map:
                added.append(_ser(entry))
            elif entry.get("version_id") != a_map[did].get("version_id"):
                changed.append({"before": _ser(a_map[did]), "after": _ser(entry)})
            else:
                unchanged.append(_ser(entry))

        removed = [_ser(e) for did, e in a_map.items() if did not in b_map]

        return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}

    # ------------------------------------------------------------------
    # Read Operations
    # ------------------------------------------------------------------

    def get_manifest(self, manifest_id: str) -> dict[str, Any] | None:
        """Return full manifest including entries, or None."""
        doc = self._coll.find_one({"_id": manifest_id})
        if doc is None:
            return None
        return self._serialize(doc)

    def list_manifests(
        self,
        corpus_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return manifest summaries (entries array excluded), newest first."""
        query: dict[str, Any] = {}
        if corpus_id:
            query["corpus_id"] = corpus_id
        if status:
            query["status"] = status
        results = []
        for doc in self._coll.find(
            query, {"entries": 0}, sort=[("created_at", DESCENDING)], limit=limit
        ):
            results.append(self._serialize(doc))
        return results

    def find_manifests_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        """Return summaries of all manifests containing a given doc_id."""
        results = []
        for doc in self._coll.find({"entries.doc_id": doc_id}, {"entries": 0}):
            results.append(self._serialize(doc))
        return results

    def get_manifest_entries(
        self,
        manifest_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return only the entries array for a manifest."""
        doc = self._coll.find_one({"_id": manifest_id}, {"entries": 1})
        if not doc:
            return []
        entries = doc.get("entries") or []
        if status:
            entries = [e for e in entries if e.get("status") == status]
        for entry in entries:
            for k in ("staged_at", "pushed_at", "removed_at"):
                v = entry.get(k)
                if isinstance(v, datetime):
                    entry[k] = v.isoformat()
        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recalc_counts(self, manifest_id: str) -> None:
        """Recompute entry_count and pushed_count from the live entries array."""
        self._coll.update_one(
            {"_id": manifest_id},
            [
                {
                    "$set": {
                        "entry_count": {"$size": "$entries"},
                        "pushed_count": {
                            "$size": {
                                "$filter": {
                                    "input": "$entries",
                                    "cond": {"$eq": ["$$this.status", "pushed"]},
                                }
                            }
                        },
                    }
                }
            ],
        )

    @staticmethod
    def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
        """Normalize _id → manifest_id and serialize datetimes."""
        doc["manifest_id"] = doc.pop("_id")
        for k in ("created_at", "updated_at", "frozen_at"):
            v = doc.get(k)
            if isinstance(v, datetime):
                doc[k] = v.isoformat()
        for entry in doc.get("entries") or []:
            for k in ("staged_at", "pushed_at", "removed_at"):
                v = entry.get(k)
                if isinstance(v, datetime):
                    entry[k] = v.isoformat()
        return doc


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_manifest_manager: ManifestManager | None = None


def get_manifest_manager() -> ManifestManager:
    """Return (or create) the shared ManifestManager instance."""
    global _manifest_manager
    if _manifest_manager is None:
        _manifest_manager = ManifestManager()
    return _manifest_manager
