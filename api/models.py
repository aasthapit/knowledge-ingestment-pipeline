from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any


# ── Shared ────────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str


# ── Stats / Status ─────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    indexed_chunks: int
    pending_review: int
    approved: int
    pushed_today: int
    recent_pushes: list[dict[str, Any]] = []


class ServiceStatus(BaseModel):
    ok: bool
    url: str | None = None
    detail: str | None = None


class StatusResponse(BaseModel):
    redis: ServiceStatus
    mongodb: ServiceStatus
    embeddings: ServiceStatus
    config: dict[str, Any]
    kb_stats: dict[str, Any]


# ── Knowledge Base ─────────────────────────────────────────────────────────────

class CreateKBRequest(BaseModel):
    name: str
    source_type: str                      # "confluence" | "jsonl" | "web"
    description: str = ""
    confluence_urls: list[str] = []
    max_depth: int = -1
    refresh_cron: str | None = None
    file_name: str | None = None
    file_ref: str | None = None


class UpdateKBRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    confluence_urls: list[str] | None = None
    max_depth: int | None = None
    refresh_cron: str | None = None
    file_name: str | None = None
    file_ref: str | None = None


# ── Vector Store Config ────────────────────────────────────────────────────────

class CreateVectorStoreRequest(BaseModel):
    name: str
    vs_type: str = "custom"              # "redis" or "custom"
    endpoint: str = ""
    api_key: str = ""
    collection: str = ""
    extra: dict[str, Any] = {}


class UpdateVectorStoreRequest(BaseModel):
    name: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    collection: str | None = None
    extra: dict[str, Any] | None = None


# ── Corpus ─────────────────────────────────────────────────────────────────────

class CreateCorpusRequest(BaseModel):
    name: str
    description: str = ""
    usecase_id: str = ""
    agent_filter: str = ""
    kb_ids: list[str] = []
    vector_store_id: str = "default"


class UpdateCorpusRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    usecase_id: str | None = None
    agent_filter: str | None = None
    vector_store_id: str | None = None


class CorpusKBRequest(BaseModel):
    kb_ids: list[str]


# ── Ingest ─────────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str
    tags: list[str] = []
    kb_id: str | None = None
    corpus_id: str | None = None        # required for auto_push
    auto_push: bool = False


class IngestResult(BaseModel):
    doc_id: str
    quality_score: float
    quality_passed: bool
    quality_flags: list[str] = []
    chunk_count: int
    tags: list[str] = []
    detected_schema: str | None = None


class SaveSchemaRequest(BaseModel):
    name: str
    field_map: dict[str, str]
    required_keys: list[str] = []
    tags_static: list[str] = []
    section_join: str = " > "


# ── Review ─────────────────────────────────────────────────────────────────────

class RejectRequest(BaseModel):
    reason: str = ""


class UpdateChunkRequest(BaseModel):
    content: str | None = None
    tags: list[str] | None = None
    section: str | None = None


class SplitDocRequest(BaseModel):
    chunk_ids: list[str]
    new_title: str


class SplitChunkRequest(BaseModel):
    content_parts: list[str]


class PushRequest(BaseModel):
    corpus_id: str
    doc_id: str | None = None
    remove_after_push: bool = False


class PushResult(BaseModel):
    pushed_docs: int
    pushed_chunks: int
    failed_docs: int


# ── Search ─────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    tag_filter: list[str] = []
    usecase_id: str | None = None
    agent_filter: str | None = None
    source_type: str | None = None


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    source: str
    title: str
    section: str
    tags: list[str]
    score: float
    page_number: int | None = None


# ── Confluence ─────────────────────────────────────────────────────────────────

class ConfluenceCrawlRequest(BaseModel):
    base_url: str
    auth_type: str = "cloud"
    email: str | None = None
    api_token: str | None = None
    ssl_verify: bool = False
    page_url: str
    max_depth: int = -1
    kb_id: str | None = None            # target Knowledge Base
    tags: list[str] = []


# ── Ledger ─────────────────────────────────────────────────────────────────────

class DriftCheckResult(BaseModel):
    current: int
    stale: int
    deleted: int
    unknown: int


# ── Manifests ─────────────────────────────────────────────────────────────────

class CreateManifestRequest(BaseModel):
    name: str
    corpus_id: str | None = None
    description: str = ""
    tags: list[str] = []


class SnapshotManifestRequest(BaseModel):
    corpus_id: str
    manifest_name: str


class CreateFromSourcesRequest(BaseModel):
    name: str
    source_refs: list[str]
    source_type: str
    corpus_id: str | None = None
    kb_id: str | None = None
    description: str = ""
    tags: list[str] = []


class DiffManifestsRequest(BaseModel):
    manifest_id_a: str
    manifest_id_b: str


class RemoveManifestDocsRequest(BaseModel):
    doc_ids: list[str] | None = None


# ── Use Case (legacy — kept for existing usecase ledger / search compat) ───────

class UpsertConfluenceSourceRequest(BaseModel):
    page_urls: list[str]
    kb_name: str = "default"
    max_depth: int = -1
    refresh_cron: str | None = None
    extra_tags: list[str] = []
