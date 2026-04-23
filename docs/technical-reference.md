# Technical Reference — Knowledge Ingestion Pipeline

## Architecture

Three layers. Each layer can be replaced or extended independently.

```
┌─────────────────────────────────────────────────────────────┐
│  Interface Layer                                             │
│  app.py (Streamlit)        cli.py (Click)                   │
│  pages/{home, ingest, confluence, review, search,           │
│          drift, ledger, manifests, status, usecase_ledger}  │
└────────────────────────────┬────────────────────────────────┘
                             │  function calls
┌────────────────────────────▼────────────────────────────────┐
│  Pipeline Core                                               │
│  ingest.py ─── converter.py ─── chunker.py                  │
│      │              │                │                      │
│  quality.py    (Docling)      HybridChunker                 │
│      │                                                       │
│  review.py ─── embedder.py ─── redis_store.py               │
│      │                                                       │
│  mongo_store.py ─── manifests.py                            │
│      │                                                       │
│  confluence.py ─── jsonl_importer.py                        │
│  refresh_scheduler.py                                        │
└──────────────┬──────────────────────────┬───────────────────┘
               │ pymongo                  │ redis-py
┌──────────────▼──────────┐  ┌────────────▼────────────────┐
│  MongoDB                │  │  Redis Stack                │
│  staging_docs           │  │  FLAT index, COSINE         │
│  staging_chunks         │  │  FLOAT32, 1536 dims         │
│  kb_documents           │  │  JSON documents             │
│  usecase_ledger         │  └─────────────────────────────┘
│  usecase_confluence_... │
│  doc_manifests          │
│  push_snapshots         │
└─────────────────────────┘
```

---

## Module Reference

### `pipeline/config.py`

Singleton settings class. All values loaded from `.env` on import via `python-dotenv`. Accessed throughout the codebase as `from pipeline.config import settings`.

- Validates only the fields required for the selected embedding provider (not all at startup)
- MongoDB can be configured via a single URI (`MONGODB_URI`) or component fields — URI takes precedence
- Custom HTTP embedding endpoint accepts headers as a JSON string (`EMBEDDING_CUSTOM_HEADERS`)

---

### `pipeline/converter.py`

Converts a source (file path or URL) into a `ConvertedDocument`.

```python
@dataclass
class Citation:
    source_path: str        # original file path or URL
    source_type: str        # pdf | docx | pptx | html | url | markdown | text
    title: str
    page_count: int | None
    page_number: int | None
    author: str | None
    created_date: str | None
    url: str | None

@dataclass
class ConvertedDocument:
    citation: Citation
    docling_doc: Any        # DoclingDocument or None (Markdown bypass)
    markdown: str           # full Markdown export from Docling

convert_document(source: str | Path) -> ConvertedDocument
```

**Source type detection:** extension-based (`pdf`, `docx`, `pptx`, `html`/`htm`/`xhtml`) or URL scheme. Markdown files bypass Docling entirely (direct read).

**Metadata extraction:** title from Docling doc metadata → doc name → filename. Page count, author, creation date extracted where available from the document's native metadata.

---

### `pipeline/chunker.py`

Splits a `ConvertedDocument` into `Chunk` objects, one per retrievable section.

```python
@dataclass
class Chunk:
    chunk_id: str           # UUID4
    source: str             # file path or URL
    title: str              # document title (H1 or filename)
    section: str            # heading breadcrumb, e.g. "Intro > Setup > Step 1"
    content: str            # plain text of this chunk
    tags: list[str]
    metadata: dict          # includes citation as dict, plus any front-matter fields

chunk_markdown(text, source, extra_tags, max_chars=2000, overlap=200) -> list[Chunk]
chunk_docling(converted_doc, tags, max_tokens=512) -> list[Chunk]
```

**Markdown path:** Parses front-matter YAML → splits by heading levels → builds breadcrumb stack → enforces `max_chars` with `overlap` character overlap across splits.

**Docling path:** Uses `HybridChunker` (merges logically related small chunks) → extracts heading breadcrumbs from metadata → retrieves page numbers from `DoclingDocument` → stores full `Citation` in `metadata["citation"]`.

**Breadcrumb format:** `"H1 title > H2 section > H3 subsection"` — the last segment is the immediate heading; earlier segments are context.

---

### `pipeline/quality.py`

Assesses a list of chunks against a `Citation` and returns a `QualityResult`.

```python
@dataclass
class QualityResult:
    score: float                        # 0.0–1.0, fraction of clean chunks
    passed: bool                        # score == 1.0 and not is_stale
    flags: list[str]                    # document-level flag names
    chunk_flags: dict[int, list[str]]   # {chunk_index: [flag_names]}
    age_days: int | None
    is_stale: bool
    chunks_too_short: int
    chunks_too_long: int
    chunks_boilerplate: int

assess_document(chunks: list[Chunk], citation: Citation) -> QualityResult
extract_tags(markdown: str, title: str, extra: list[str]) -> list[str]
```

**Per-chunk checks:**
- `too_short`: `len(content) < 100`
- `too_long`: `len(content) > 2000`
- `boilerplate`: regex patterns (nav menus, TOC, login prompts, copyright) + ≥8 short lines averaging < 30 chars. Code blocks are exempt.

**Document-level check:**
- `stale`: age > 180 days. Age computed from `citation.created_date` (tries 4 date formats) or file `mtime` as fallback.

**Tag extraction:** H1–H3 headings from Markdown, lowercased, filtered by `[a-z0-9]+` pattern, deduplicated against a stop-word list, capped at 10.

**Auto-approval rule:** `passed = (score == 1.0) AND (not is_stale)`. Note: the `QUALITY_THRESHOLD` env var is loaded by config but not currently used in `assess_document()`.

---

### `pipeline/embedder.py`

Wraps four embedding providers behind a single interface.

```python
embed_texts(texts: list[str]) -> list[list[float]]
embed_chunks(chunks: list[Chunk]) -> list[list[float]]
```

Providers selected by `EMBEDDING_PROVIDER`:
- `openai` — OpenAI Embeddings API (`text-embedding-3-small` default)
- `azure` — Azure OpenAI Embeddings API
- `sentence-transformers` — local model, loads on first call (no caching between calls)
- `custom` — any OpenAI-compatible HTTP endpoint; tries `data[0].embedding` then `embeddings[0]` response shapes

Inputs are batched at `EMBED_BATCH_SIZE` (default 32). No retry logic; no embedding cache.

---

### `pipeline/ingest.py`

Orchestrator. Routes all ingest paths and provides the primary entry points.

```python
ingest_document(source, extra_tags, auto_push, kb_name, usecase_id, agent_filter,
                manifest_id) -> dict
# Returns: doc_id, quality_score, quality_passed, quality_flags, chunk_count, tags

ingest_jsonl(source, batch_name, extra_tags, kb_name, usecase_id, agent_filter,
             require_usecase, field_map) -> dict
# Returns: doc_id, schema, total, too_short, too_long, boilerplate, stale, 
#          has_embeddings, usecase_id, agent_filter

query_vectorstore(question, top_k, tag_filter) -> list[dict]
# Returns: chunk_id, content, source, title, section, tags, score (0–1)
```

**`ingest_document` flow:**
1. `converter.convert_document(source)` → `ConvertedDocument`
2. `quality.extract_tags(markdown)` → auto-tags
3. `chunker.chunk_docling(converted_doc)` → `list[Chunk]`
4. `quality.assess_document(chunks, citation)` → `QualityResult`
5. `MongoStagingStore.enqueue(doc_id, meta, chunks)` → staged
6. If `auto_push=True` and `quality_passed`: immediately call `review.push_approved(doc_id)`

**`query_vectorstore` flow:**
1. Embed the question
2. `redis_store.search(vector, top_k, tag_filter)` → raw results
3. Normalize scores: `score = 1 - (raw_distance / 2)` (cosine distance 0–2 → similarity 0–1)

---

### `pipeline/mongo_store.py`

Three store classes, each a thin wrapper around a MongoDB collection.

#### `MongoStagingStore`

Collection pair: `staging_docs` + `staging_chunks`. Manages the pending → approved → pushed state machine.

```python
enqueue(doc_id, meta, chunks) -> None
approve(doc_id) -> None
reject(doc_id, reason) -> None
split_doc(source_doc_id, new_doc_id, chunk_ids, new_meta) -> int
split_chunk(doc_id, source_chunk_id, content_parts) -> list[str]
get_doc_meta(doc_id) -> dict | None
get_chunks(doc_id) -> list[dict]
list_all() -> list[dict]
```

Type coercion on enqueue normalises JSON-string booleans, floats, and lists for MongoDB compatibility.

#### `KBLedger`

Collection: `kb_documents`. Permanent record of every pushed document.

```python
record_push(doc_id, title, source_path, chunk_ids, source_type, kb_name,
            quality_score, tags, usecase_id, agent_filter) -> None
check_drift_one(doc_id) -> str   # "current" | "stale" | "deleted" | "unknown"
run_drift_check(kb_name) -> dict[str, int]
get_stats(kb_name) -> dict
record_snapshot(pushed_doc_ids) -> str
list_docs(kb_name, drift_status) -> list[dict]
list_snapshots() -> list[dict]
```

**Drift detection logic:**
- `file` / `markdown` sources: compare current `os.stat().st_mtime` and `st_size` against stored values → `current` / `stale` / `deleted`
- `confluence` sources: defer to `ConfluenceCrawler.crawl_metadata()` version comparison
- `url` / anything else: always `unknown`

#### `UsecaseLedger`

Collection pair: `usecase_ledger` + `usecase_confluence_sources`. Maintains live chunk-ID inventory per (usecase_id, agent_filter) pair and manages Confluence source scheduling.

```python
record_push(usecase_id, agent_filter, kb_name, doc_ids, chunk_ids) -> None
get_chunk_ids(usecase_id, agent_filter) -> list[str]
upsert_confluence_source(usecase_id, agent_filter, page_urls, refresh_cron) -> None
get_sources_due_for_refresh() -> list[dict]
update_crawl_snapshot(source_id, pages_meta) -> None
get_crawl_snapshot(source_id) -> list[dict]
```

---

### `pipeline/redis_store.py`

Vector index and staging queue over Redis Stack (RediSearch).

#### Index schema

```
VectorField: embedding  ALGORITHM FLAT  TYPE FLOAT32  DIM {EMBEDDING_DIMENSIONS}  DISTANCE_METRIC COSINE
TextField:   source, title, section, content
TagField:    tags
```

Chunk stored as JSON document at key `doc:{chunk_id}`. Embedding stored as packed binary (`struct.pack(f'{n}f', *vector)`) for efficiency.

#### Core functions

```python
create_index(client) -> None
upsert_chunks(chunks, embeddings, client, pipeline_size) -> None   # batched pipeline
search(query_vector, top_k, tag_filter) -> list[dict]              # KNN + optional tag pre-filter
delete_chunks(chunk_ids, client) -> int
update_tags(chunk_id, tags, client) -> None
get_chunk(chunk_id, client) -> dict | None
```

**Search:** KNN query with `RETURN` fields `source title section content tags __vector_score`. Tag filter is RediSearch syntax (`@tags:{tag1|tag2}`). Score normalised: `1 - (raw_score / 2)`.

#### `StagingStore` (Redis-backed, legacy)

Redis data model:
- `review:queue` — FIFO list of doc_ids
- `review:doc:{id}` — hash of doc metadata
- `review:chunks:{id}` — list of JSON-serialised chunk dicts
- Sets `review:pending`, `review:approved`, `review:rejected`, `review:pushed` for status tracking

The MongoDB-backed `MongoStagingStore` is the primary staging store. This Redis-backed version is retained for compatibility.

---

### `pipeline/confluence.py`

```python
@dataclass
class ConfluencePage:
    page_id: str
    title: str
    space_key: str
    url: str
    content_text: str       # HTML → plain text via BeautifulSoup
    ancestors: list[str]    # parent titles, root → immediate parent
    labels: list[str]
    version: int
    author: str
    last_modified: str

class ConfluenceCrawler:
    crawl(page_url, max_depth=-1) -> list[ConfluencePage]
    crawl_metadata(page_url, max_depth=-1) -> list[dict]   # lightweight, no body
    fetch_pages_by_ids(page_ids) -> list[ConfluencePage]
    to_record(page) -> dict                                # pipeline JSONL schema
    export_jsonl(pages, output_path) -> Path
```

**Authentication:** Cloud (`email` + `API token`, HTTP Basic) or Server/DC (`Personal Access Token`, Bearer header).

**Page ID extraction:** Cloud URLs contain `/pages/{id}/` as a path segment; Server URLs use `?pageId={id}` query parameter.

**Crawl:** DFS walk — fetches child page IDs recursively via `GET /rest/api/content/{id}/child/page`, then batch-fetches bodies with `expand=body.storage,ancestors,metadata.labels,version,space`.

**HTML → text:** BeautifulSoup with lxml parser. Block elements (`p`, `h1`–`h6`, `li`, `tr`) become newlines. `<code>` and `ac:plain-text-body` blocks become triple-backtick fenced code.

**Canonical URL:** `{base_url}/wiki/spaces/{space_key}/pages/{page_id}/{slug}`

---

### `pipeline/jsonl_importer.py`

```python
detect_schema(record: dict) -> str
# Returns: "crawler" | "pipeline" | "{custom_name}" | "unknown"

map_record(rec, schema, extra_tags) -> tuple[Chunk, list[float] | None]
# Returns the mapped Chunk and pre-computed embedding (or None)

import_jsonl(source, batch_name, extra_tags, kb_name, usecase_id, agent_filter,
             require_usecase, field_map) -> dict

peek_jsonl(source, n=5, field_map) -> dict
# Returns: schema, fields_found, samples (list of dicts), has_embeddings

save_custom_schema(name, field_map, required_keys, tags_static, section_join) -> None
```

**Schema detection order:**
1. Custom schemas from `schemas.yaml` — checked in order; first match wins
2. `crawler` — has both `text` and `page_url`
3. `pipeline` — has both `content` and `source`
4. `unknown` — best-effort using any available fields

**Custom schema format** (`schemas.yaml`):

```yaml
schemas:
  - name: my_schema
    detect:
      required: [field1, field2]   # ALL must be present
      exclude: [field3]            # ANY disqualifies
    fields:
      content:  body               # target: source_field
      source:   url
      title:    page_title
      section:  category
      tags:     labels
      embedding: vector
    tags_static: [tag1, tag2]
    section_join: " > "
```

Field mappings support dot-notation for nested fields: `"metadata.citation.url"`.

**Batch staging:** All records in a JSONL file become a single staging document with a UUID5 batch ID (`namespace_dns(batch_name + str(count))`). Quality flags are accumulated across records and compared against thresholds. If any flag is set, the batch enters review; otherwise auto-approved.

---

### `pipeline/review.py`

Orchestrates the staging → push transition.

```python
list_all_docs() -> list[dict]
list_pending_docs() -> list[dict]
get_doc_detail(doc_id) -> dict | None    # includes full chunks list

approve_doc(doc_id) -> bool
reject_doc(doc_id, reason) -> bool
split_doc(source_doc_id, chunk_ids, new_title) -> str | None
split_chunk(doc_id, chunk_id, content_parts) -> list[str]
update_chunk(doc_id, chunk_id, updates) -> bool

push_approved(doc_id=None, remove_after_push=False) -> dict
# Returns: pushed_docs, pushed_chunks, failed_docs
```

**`push_approved` flow:**
1. Fetch all approved doc_ids from `MongoStagingStore` (or single `doc_id`)
2. Reconstruct `Chunk` objects from staged dicts
3. Identify chunks with pre-computed `_embedding` field → reuse without API call
4. Embed remaining chunks via `embedder.embed_chunks()`
5. `redis_store.upsert_chunks(chunks, embeddings)` — batched pipeline writes
6. `KBLedger.record_push(doc_id, ...)` — permanent record
7. `UsecaseLedger.record_push(usecase_id, agent_filter, ...)` — chunk-ID inventory update
8. `ManifestManager` — update entry status to `pushed` in any referencing manifests
9. `KBLedger.record_snapshot(pushed_doc_ids)` — point-in-time snapshot
10. Optional CSV ledger export to `LEDGER_OUTPUT_DIR`
11. Mark doc as `pushed` in staging (or remove if `remove_after_push=True`)

---

### `pipeline/manifests.py`

```python
class ManifestManager:
    create_manifest(name, usecase_id, agent_filter, kb_name, description, tags,
                    created_by) -> str

    add_entry(manifest_id, doc_id, object_id, file_id, version_id, source_type,
              source_ref, title, status) -> bool
    update_entry_status(manifest_id, doc_id, status, pushed_at, removed_at) -> bool
    remove_entry(manifest_id, doc_id) -> bool

    freeze_manifest(manifest_id) -> bool
    archive_manifest(manifest_id) -> bool

    snapshot_corpus_to_manifest(usecase_id, agent_filter, manifest_name) -> str
    create_manifest_from_sources(name, source_refs, source_type, usecase_id,
                                 agent_filter, ...) -> str
    ingest_from_manifest(manifest_id, extra_tags, auto_push) -> dict
    remove_manifest_docs(manifest_id, doc_ids) -> dict
    diff_manifests(manifest_id_a, manifest_id_b) -> dict

    get_manifest(manifest_id) -> dict | None
    list_manifests(usecase_id, agent_filter, status) -> list[dict]
    find_manifests_by_doc_id(doc_id) -> list[dict]
```

**Manifest state machine:** `open → frozen → archived`

**Entry status progression:** `pending → staged → approved → pushed → removed`

**`snapshot_corpus_to_manifest`:** Queries `KBLedger.list_docs()` for all pushed docs matching (usecase_id, agent_filter). Computes `version_id = sha256(doc_id + source_ref + pushed_at)[:16]`. Creates manifest, adds entries, freezes immediately.

**`diff_manifests`:** Builds `{doc_id → entry}` maps for both manifests. Categorises as:
- `added` — in B, not in A
- `removed` — in A, not in B
- `changed` — in both, different `version_id`
- `unchanged` — in both, same `version_id`

**`remove_manifest_docs`:** For each doc_id: `redis_store.delete_chunks(chunk_ids)` → `KBLedger` delete → `UsecaseLedger` chunk removal → all open manifests entry status → `removed`.

---

### `pipeline/refresh_scheduler.py`

Background APScheduler running in a daemon thread. Polls MongoDB every 5 minutes for Confluence sources due for refresh.

```python
start_scheduler() -> None    # idempotent, guarded by threading.Lock
stop_scheduler() -> None
trigger_refresh_now(usecase_id, agent_filter, on_step) -> None
```

**Incremental refresh logic:**
1. Load prior crawl snapshot from `UsecaseLedger.get_crawl_snapshot(source_id)`
2. `ConfluenceCrawler.crawl_metadata(page_url)` → current page versions (no body)
3. If snapshot exists: diff versions → identify changed or new page IDs
4. `ConfluenceCrawler.fetch_pages_by_ids(changed_ids)` → bodies of changed pages only
5. `ingest.ingest_jsonl(pages_as_jsonl, auto_push=True)` → stage + push without review
6. `UsecaseLedger.update_crawl_snapshot(source_id, current_metadata)`
7. Compute `next_refresh_at` from cron expression via `croniter`

**Concurrency guard:** `refresh_status` field in MongoDB set to `running` before starting; set to `done` or `failed` after. Prevents duplicate runs if scheduler fires while a previous refresh is still active.

---

### `pipeline/tagger.py`

Thin utilities for tag management.

```python
apply_tags(chunks, tags) -> list[Chunk]       # adds tags, deduplicates (order-preserving)
remove_tags(chunks, tags) -> list[Chunk]
filter_chunks_by_tag(chunks, required_tags, match_all=False) -> list[Chunk]
retag_in_redis(chunk_ids, add_tags, remove_tags_list) -> None  # fetch-modify-write per chunk
```

---

### `pipeline/exporter.py`

Serialises chunks and ledger records to JSONL and CSV.

```python
export_jsonl(chunks, embeddings, output_path) -> Path
export_ledger_csv(records, output_path) -> Path
export_chunks_as_jsonl(chunks, output_path) -> Path   # for usecase JSONL export
load_jsonl(path) -> list[dict]
```

Auto-generates timestamped filenames (`chunks_YYYYMMDD_HHMMSS.jsonl`) if `output_path` is not provided.

---

## Data Models

### Redis document (`doc:{chunk_id}`)

```json
{
  "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
  "source":   "https://docs.example.com/guide",
  "title":    "Guide Title",
  "section":  "Installation > Docker",
  "content":  "Run the following command to start the container...",
  "tags":     "docker,install,GENAI1597_SSOP,ssop_agent",
  "embedding": "<binary FLOAT32 blob>"
}
```

Tags stored as comma-separated string for RediSearch `TAG` field compatibility.

### MongoDB `staging_docs`

```json
{
  "_id":              "uuid-string",
  "title":            "Document Title",
  "source_path":      "/path/to/file.pdf",
  "source_type":      "pdf",
  "usecase_id":       "GENAI1597_SSOP",
  "agent_filter":     "ssop_agent",
  "kb_name":          "default",
  "quality_score":    0.85,
  "quality_passed":   false,
  "quality_flags":    ["too_short", "boilerplate"],
  "status":           "pending_review",
  "age_days":         45,
  "is_stale":         false,
  "chunks_too_short": 1,
  "chunks_too_long":  0,
  "chunks_boilerplate": 1,
  "ingested_at":      "2025-04-01T10:00:00",
  "approved_at":      null,
  "pushed_at":        null
}
```

### MongoDB `staging_chunks`

```json
{
  "_id":       "uuid-string",
  "doc_id":    "parent-staging-doc-id",
  "source":    "https://docs.example.com/guide",
  "title":     "Guide Title",
  "section":   "Installation > Docker",
  "content":   "Run the following command...",
  "tags":      ["docker", "install"],
  "metadata":  {
    "citation": {
      "source_path": "...", "source_type": "...", "title": "...",
      "page_count": 12, "page_number": 3, "author": null,
      "created_date": null, "url": null
    }
  },
  "_embedding": [0.012, -0.034, ...]
}
```

### MongoDB `kb_documents`

```json
{
  "_id":          "staging-doc-id",
  "title":        "Document Title",
  "source_path":  "/path/to/file.pdf",
  "source_type":  "pdf",
  "usecase_id":   "GENAI1597_SSOP",
  "agent_filter": "ssop_agent",
  "chunk_ids":    ["uuid1", "uuid2", "uuid3"],
  "quality_score": 1.0,
  "kb_name":      "default",
  "pushed_at":    "2025-04-01T11:00:00",
  "drift_status": "current",
  "source_mtime": 1711966800.0,
  "source_size":  245760,
  "tags":         ["docker", "install"]
}
```

### MongoDB `doc_manifests`

```json
{
  "_id":          "manifest-uuid",
  "name":         "SSOP v2 Corpus",
  "usecase_id":   "GENAI1597_SSOP",
  "agent_filter": "ssop_agent",
  "kb_name":      "default",
  "status":       "frozen",
  "description":  "Corpus snapshot before model upgrade",
  "tags":         ["v2"],
  "entry_count":  42,
  "pushed_count": 40,
  "created_at":   "2025-04-01T09:00:00",
  "frozen_at":    "2025-04-01T12:00:00",
  "entries": [
    {
      "doc_id":      "uuid",
      "object_id":   "mongo-object-id",
      "file_id":     "confluence-page-id",
      "version_id":  "abc123de",
      "status":      "pushed",
      "source_type": "confluence",
      "source_ref":  "https://wiki.example.com/pages/123",
      "title":       "Page Title",
      "staged_at":   "2025-04-01T09:30:00",
      "pushed_at":   "2025-04-01T11:00:00",
      "removed_at":  null
    }
  ]
}
```

---

## Ingestion Paths

Four distinct ingestion paths share the same staging and push infrastructure:

| Path | Entry point | Conversion | Quality check | Auto-approve rule |
|---|---|---|---|---|
| **Document** (file / URL) | `ingest_document()` | Docling | Per-chunk + recency | score == 1.0 and not stale |
| **JSONL bulk** | `ingest_jsonl()` | None (pre-chunked) | Per-chunk + recency | No quality flags in batch |
| **Confluence** | `ConfluenceCrawler` → `ingest_jsonl()` | HTML → text (BS4) | Per-chunk + recency | No quality flags in batch |
| **Legacy Markdown** | `ingest_file()` | None | None | Always auto-approved |

---

## Tag Conventions

Tags serve three distinct roles. All stored together in a single `tags` list per chunk.

| Tag type | Example | Set by |
|---|---|---|
| Content tags | `docker`, `installation` | Auto-extracted from H1–H3 headings by `quality.extract_tags()` |
| Source tags | `confluence`, `openshift-docs` | User-specified at ingest time |
| Scope tags | `GENAI1597_SSOP`, `ssop_agent` | Automatically added from `usecase_id` and `agent_filter` |

At search time, the `tag_filter` parameter maps to a RediSearch TAG query: `@tags:{tag1|tag2}`.

---

## Embedding Providers

| Provider | Config | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | Default. `text-embedding-3-small` → 1536 dims |
| `azure` | `AZURE_OPENAI_*` | Requires deployment name |
| `sentence-transformers` | `EMBEDDING_MODEL` = HuggingFace model name | Runs locally; model loads on every call |
| `custom` | `EMBEDDING_CUSTOM_URL`, `EMBEDDING_CUSTOM_API_KEY`, `EMBEDDING_CUSTOM_HEADERS` | Any OpenAI-compatible endpoint (Ollama, vLLM, etc.) |

---

## Configuration Reference

All settings loaded from `.env` (or environment variables). Full list in `.env.example`.

### Embedding

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `azure` \| `sentence-transformers` \| `custom` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `EMBEDDING_DIMENSIONS` | `1536` | Must match model output and Redis index |
| `OPENAI_API_KEY` | — | Required for `openai` |
| `AZURE_OPENAI_API_KEY` | — | Required for `azure` |
| `AZURE_OPENAI_ENDPOINT` | — | Required for `azure` |
| `AZURE_OPENAI_DEPLOYMENT` | — | Required for `azure` |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embedding API call |
| `EMBEDDING_CUSTOM_URL` | — | Endpoint for `custom` provider |
| `EMBEDDING_CUSTOM_API_KEY` | — | API key for custom endpoint |
| `EMBEDDING_CUSTOM_HEADERS` | — | JSON string of extra HTTP headers |

### Redis

| Variable | Default | Notes |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | |
| `REDIS_INDEX_NAME` | `knowledge_index` | RediSearch index name |
| `REDIS_KEY_PREFIX` | `doc:` | Key prefix for stored chunks |

### MongoDB

| Variable | Default | Notes |
|---|---|---|
| `MONGODB_URI` | — | Overrides all component fields when set |
| `MONGODB_HOST` | `localhost` | |
| `MONGODB_PORT` | `27017` | |
| `MONGODB_USERNAME` | — | Leave empty for no auth |
| `MONGODB_PASSWORD` | — | |
| `MONGODB_TLS` | `true` | Set `false` for plain local instances |
| `MONGODB_TLS_INSECURE` | `false` | Skip cert verification (self-signed certs) |
| `MONGODB_SRV` | `true` | Use DNS SRV (Atlas); set `false` for direct |
| `MONGODB_DB_NAME` | `knowledge_pipeline` | |
| `MONGODB_COLLECTION_PREFIX` | — | e.g. `prod_` to separate environments |

### Pipeline

| Variable | Default | Notes |
|---|---|---|
| `DOCLING_MAX_TOKENS` | `512` | HybridChunker max tokens per chunk |
| `CHUNK_MAX_CHARS` | `2000` | Max chars for Markdown chunker; also `too_long` threshold |
| `CHUNK_OVERLAP_CHARS` | `200` | Overlap between consecutive Markdown chunks |
| `QUALITY_THRESHOLD` | `0.6` | Loaded by config but currently unused in `assess_document()` |
| `JSONL_OUTPUT_DIR` | `./output` | Default JSONL export directory |
| `LEDGER_OUTPUT_DIR` | — | If set, exports CSV after each push |

### Confluence

| Variable | Notes |
|---|---|
| `CONFLUENCE_BASE_URL` | e.g. `https://mycompany.atlassian.net` |
| `CONFLUENCE_AUTH_TYPE` | `cloud` or `server` |
| `CONFLUENCE_EMAIL` | Required for Cloud |
| `CONFLUENCE_API_TOKEN` | API token (Cloud) or PAT (Server) |
| `CONFLUENCE_SSL_VERIFY` | `false` to disable TLS verification for self-signed certs |

---

## Known Limitations

| Area | Limitation |
|---|---|
| **Quality threshold** | `QUALITY_THRESHOLD` env var is loaded but not used in `assess_document()`. The current rule is hard-coded: score == 1.0 required for auto-approval. |
| **Use case scoping at search time** | The search page over-fetches (`top_k * 5`) and post-filters by chunk IDs from MongoDB. Does not use Redis tag-filter KNN, which would be more accurate and efficient. |
| **URL drift detection** | `KBLedger.check_drift_one()` returns `"unknown"` for any URL-sourced document. No content-hash drift detection for generic web pages. |
| **Embedding cache** | No caching. Unchanged Confluence pages are re-embedded on every incremental refresh. |
| **Scheduler isolation** | APScheduler runs in-process. Multi-instance deployments (multiple Streamlit workers) will run duplicate refresh jobs. |
| **Authentication** | No login or role separation on the Streamlit UI. Suitable for single-user or trusted-network deployments only. |
| **Manifest re-ingest** | `ingest_from_manifest()` skips non-Confluence entries silently. File-upload sources cannot be re-ingested from a manifest. |
| **Qdrant backend** | `qdrant_store.py` is archived. The `QDRANT_URL` env var in `.env.example` is no longer functional. |
| **Manifest rollback** | Manifests are audit history and diff tools. There is no `restore_to_manifest()` operation that re-ingests a previous corpus state in a single step. |
