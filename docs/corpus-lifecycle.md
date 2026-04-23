# Corpus Lifecycle — How Content Flows Through the Pipeline

This document covers how a document goes from a URL or file into the knowledge base, how the system keeps track of what's there, and how content is updated or removed over time.

---

## The core flow

```mermaid
flowchart TD
    A([URL / File / Confluence page / JSONL]) --> B["Convert\nDocling extracts text and metadata\nfrom PDF · DOCX · PPTX · HTML · URL"]
    B --> C["Chunk\nSplit into small retrievable sections\nwith heading breadcrumbs and citation"]
    C --> D["Quality check\nAssess every chunk individually"]
    D --> E{"Any flags?"}
    E -->|No flags| F["status = approved\nauto-approved, ready to push"]
    E -->|Flags found| G["status = pending_review\nawaits human decision"]
    F --> H["MongoDB staging\nstaging_docs + staging_chunks"]
    G --> H
    H --> I["Review Queue\napprove · reject · edit chunks · split doc"]
    I --> J["push_approved()\nEmbed chunks · upsert to Redis"]
    J --> K["KB ledger — kb_documents\nRecords chunk IDs, source info, push timestamp"]
    J --> L["Use Case ledger — usecase_ledger\nRecords chunk IDs per (usecase_id, agent_filter)"]
    K --> M([Searchable in Redis vector index])
    L --> M
```

---

## Quality checks

Every chunk is assessed independently after the document is split. The quality score is the fraction of chunks with no flags (0.0 – 1.0).

| Flag | Condition | Why it matters |
|---|---|---|
| **too_short** | Chunk < 100 characters | Navigation stubs and empty headings — retrieval noise |
| **too_long** | Chunk > 2 000 characters | Risk of truncation by the embedding model |
| **boilerplate** | Nav menus, TOC, login prompts, copyright footers | Adds noise, not useful for search |
| **stale** | Source > 6 months old (from creation date or file mtime) | Content may no longer be accurate |

A document is auto-approved only when it has zero flagged chunks. Any flag routes it to human review.

---

## Use case and agent tracking

Every staging document carries two fields that flow through the entire lifecycle:

- **usecase_id** — the business use case this content supports
- **agent_filter** — the specific AI agent or persona it is meant for

These are stored in `staging_docs`, written through to `kb_documents` and `usecase_ledger`, and visible in the Review Queue and Use Case Ledger UI.

The `usecase_ledger` collection maintains a live index of chunk IDs per (usecase_id, agent_filter) pair. This makes it possible to scope search results to a specific agent without re-querying the whole vector index.

---

## JSONL import

When data arrives as JSONL (from a crawler, export, or external system), the importer auto-detects the schema from the first record and maps fields to the internal `Chunk` structure.

```mermaid
flowchart TD
    A([JSONL file]) --> B["Read first record"]
    B --> C{"Match custom schemas\n(schemas.yaml) in order"}
    C -->|match found| D["Apply custom field mapping"]
    C -->|no match| E{Built-in schemas}
    E -->|has 'text' + 'page_url'| F["Crawler schema"]
    E -->|has 'content' + 'source'| G["Pipeline export schema"]
    E -->|nothing matched| H["Best-effort detection"]
    D --> I["Chunk{chunk_id, source, title, section, content, tags}"]
    F --> I
    G --> I
    H --> I
    I --> J["Quality check per chunk\n(size · boilerplate · recency)"]
    J --> K["Enqueue as staging_doc\nstatus = approved or pending_review"]
```

If the JSONL file includes pre-computed `embedding` vectors, they are stored and reused at push time — no embedding API call needed.

---

## MongoDB collections

```mermaid
erDiagram
    staging_docs {
        string _id PK
        string title
        string source_path
        string source_type
        string usecase_id
        string agent_filter
        float quality_score
        boolean quality_passed
        string quality_flags
        string status
        string kb_name
        int age_days
        boolean is_stale
        int chunks_too_short
        int chunks_too_long
        int chunks_boilerplate
        datetime ingested_at
        datetime approved_at
        datetime pushed_at
    }
    staging_chunks {
        string _id PK
        string doc_id FK
        string source
        string title
        string section
        string content
        list tags
        dict metadata
        list _embedding
    }
    kb_documents {
        string _id PK
        string title
        string source_path
        string source_type
        string usecase_id
        string agent_filter
        list chunk_ids
        float quality_score
        string kb_name
        datetime pushed_at
        string drift_status
        float source_mtime
        int source_size
    }
    usecase_ledger {
        string _id PK
        string usecase_id
        string agent_filter
        string kb_name
        list doc_ids
        list chunk_ids
        int chunk_count
        datetime last_pushed_at
    }
    usecase_confluence_sources {
        string _id PK
        string usecase_id
        string agent_filter
        list page_urls
        int max_depth
        list crawled_pages
        string refresh_cron
        datetime last_refresh_at
        datetime next_refresh_at
        string refresh_status
    }
    staging_docs ||--o{ staging_chunks : "has chunks"
    staging_docs ||..o| kb_documents : "becomes ledger entry on push"
    kb_documents }o--|| usecase_ledger : "chunk IDs tracked per use case"
```

---

## Document lifecycle states

```mermaid
stateDiagram-v2
    [*] --> approved : No quality flags — auto-approved
    [*] --> pending_review : One or more flags found
    pending_review --> approved : Engineer approves
    pending_review --> rejected : Engineer rejects
    approved --> pushed : push_approved() — embedded and indexed
    rejected --> [*] : Stays in staging, never pushed
```

---

## Confluence sources and scheduled refresh

Confluence page trees can be registered per use case with a crawl schedule. A background scheduler (APScheduler, 5-minute poll interval) checks for sources due for refresh and re-crawls them automatically.

```mermaid
sequenceDiagram
    participant Scheduler as refresh_scheduler.py
    participant Ledger as UsecaseLedger
    participant Confluence as Confluence REST API
    participant Ingest as ingest.py

    loop Every 5 minutes
        Scheduler->>Ledger: get_sources_due_for_refresh()
        Ledger-->>Scheduler: [sources where next_refresh_at ≤ now]
    end

    Scheduler->>Ledger: mark_refresh_running(source_id)
    Scheduler->>Confluence: crawl page tree (body + metadata)
    Confluence-->>Scheduler: list of ConfluencePage objects
    Scheduler->>Ledger: record_crawl_snapshot(source_id, pages)
    Scheduler->>Ingest: ingest_jsonl(pages as JSONL, usecase_id, agent_filter)
    Ingest-->>Scheduler: staging doc created
    Scheduler->>Ledger: mark_refresh_done(source_id)
    Scheduler->>Ledger: update_next_refresh(source_id, cron_expr)
```

**Drift check (lightweight)** — separate from the full refresh, the UI can compare the stored page snapshot against the current Confluence state by fetching only version metadata (no body content). This shows which pages have been added, removed, or updated without re-crawling.

---

## Keeping the corpus healthy over time

### Adding content

Any new ingest creates new staging records. After review and push, the vector index grows incrementally — no full rebuild needed. The KB ledger records every push with a timestamp so you know exactly when each document entered the index.

### Removing content

Documents can be deleted from the KB Health page. The ledger stores chunk IDs, so removal is exact — only the selected document's vectors are deleted from the search index.

### Detecting staleness

For file-based sources: the ledger records the file's modification time and size at push time. A drift check compares the current file against the stored fingerprint.

For Confluence sources: the stored page-version snapshot is compared against current metadata from Confluence.

### Rolling back

Staging records are kept after push by default. To roll back:

1. Query `kb_documents` for documents pushed after your target date
2. Collect their `chunk_ids`
3. Delete those chunk IDs from the Redis index
4. Re-push the prior versions from staging if they're still there, or re-ingest from source

---

## Configuration

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Max chunk tokens | `DOCLING_MAX_TOKENS` | `512` | HybridChunker max (≈ 400 words) |
| Max chunk chars | `CHUNK_MAX_CHARS` | `2000` | Also the "too_long" flag threshold |
| MongoDB database | `MONGODB_DB_NAME` | `knowledge_pipeline` | All collections live here |
| Collection prefix | `MONGODB_COLLECTION_PREFIX` | `""` | Separate environments, e.g. `prod_` |
| JSONL output dir | `JSONL_OUTPUT_DIR` | `./output` | Where exports are written |
