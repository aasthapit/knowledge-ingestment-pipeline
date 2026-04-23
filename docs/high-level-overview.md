# High-Level Overview — Knowledge Ingestion Pipeline

## What Is This?

The Knowledge Ingestion Pipeline is a tool for building and maintaining knowledge bases that AI agents use to answer questions. It sits between your source documents (Confluence wikis, PDFs, web pages, spreadsheet exports) and the AI that reads them — ensuring that what the AI retrieves is accurate, current, and relevant.

Think of it as a **content supply chain for AI**: raw documents go in one end, curated and searchable knowledge comes out the other.

---

## The Problem It Solves

The naive approach to building an AI knowledge base is to embed all your documents and search them. This works for a prototype but degrades quickly in production:

- **Stale content** accumulates. Pages updated in Confluence six months ago are still in the index with old information.
- **Low-quality chunks** pollute search results. Navigation menus, copyright footers, and empty section headings all end up in the index as if they were real content.
- **No audit trail.** You can't tell what's in the index, when it was added, or whether it's still accurate.
- **No scope control.** Different AI agents serving different use cases all query the same undifferentiated index.

This pipeline adds a structured process around those steps: convert → assess quality → stage for review → approve → embed → index → track.

---

## Who Uses It

**Knowledge Engineers** manage the corpus: they ingest documents, review quality, approve content for publication, monitor health, and manage Confluence refresh schedules.

**AI Agents / Applications** query the knowledge base to retrieve relevant sections when answering user questions. They interact with Redis directly via vector search — the pipeline is invisible to them at query time.

---

## Core Concepts

### The Ingest-Review-Push Lifecycle

Nothing reaches the AI without passing through three stages:

1. **Ingest** — a document is converted, chunked, and assessed for quality. If it passes cleanly, it's auto-approved. If any section is flagged (too short, too long, boilerplate, or stale), it enters the human review queue.
2. **Review** — a knowledge engineer inspects flagged documents, edits chunks if needed, and approves or rejects them.
3. **Push** — approved documents are embedded (converted to vector representations) and written to the Redis search index. Only pushed documents are searchable.

The staging area (MongoDB) acts as a gate: the vector index only ever contains content that a human or automated quality check has approved.

### Chunks, Not Documents

The pipeline doesn't index whole documents — it indexes **chunks**: small, self-contained sections of a document (typically a heading's worth of content). Each chunk carries:
- The section heading breadcrumb (e.g. `Installation > Docker > Prerequisites`)
- The source document title and URL/path
- The page number (for PDFs)
- Tags identifying the use case and content category
- The full citation metadata

This structure makes search results precise and source-attributable. An AI can cite the exact section and page it pulled from.

### Use Case Scoping

Every document can be tagged with a **use case ID** and **agent filter** — identifying which business function it supports and which AI agent it's intended for. These tags flow through the entire lifecycle. At search time, an agent can filter results to only its own content, even when multiple teams share the same infrastructure.

### Corpus Provenance

The pipeline maintains two permanent records:

- **KB Ledger** (`kb_documents`) — records every pushed document with its source path, push timestamp, chunk IDs, and quality score. Used for drift detection and audit.
- **Use Case Ledger** (`usecase_ledger`) — records the live chunk-ID inventory per (use case, agent) pair. Used to scope search results and track coverage per use case.

---

## Key Features

### Multi-Format Ingestion

| Source type | How it's handled |
|---|---|
| PDF, DOCX, PPTX | Converted via Docling — structure preserved |
| HTML, web pages | Fetched and converted via Docling |
| Confluence | Crawled recursively via REST API; ancestor breadcrumbs preserved |
| JSONL bulk files | Auto-detected schema; supports pre-computed embeddings |
| Markdown | Lightweight heading-split; no Docling needed |

### Quality Assessment

Before staging, every chunk in a document is evaluated individually:

- **Too short** (< 100 chars) — likely a navigation stub or empty heading
- **Too long** (> 2 000 chars) — risk of embedding truncation
- **Boilerplate** — navigation menus, TOC entries, copyright footers
- **Stale** — source older than 6 months (from file mtime or document creation date)

The quality score is the fraction of clean chunks. A document with 9 clean sections out of 10 scores 0.9. A score of 1.0 with no staleness flag auto-approves; anything less goes to the review queue.

### Human Review Queue

Staged documents can be inspected before pushing. In the Review Queue UI, a knowledge engineer can:
- See every chunk with its quality flags
- Edit chunk content or section labels
- Add or remove tags
- Split a chunk into smaller pieces
- Break selected chunks out into a separate document
- Approve or reject the document with an optional reason

None of these edits require re-ingestion — all changes apply directly to the staged version.

### Confluence Integration

The pipeline has a first-class Confluence connector:
- Crawls page trees recursively, preserving ancestor breadcrumbs
- Stores a metadata snapshot (page versions) after each crawl
- Compares the snapshot against current Confluence state to detect which pages have been added, removed, or updated — without re-fetching body content
- Supports scheduled auto-refresh via cron expressions (checked every 5 minutes by a background scheduler)
- Incremental refresh: only changed or new pages are re-ingested

### Corpus Versioning with Manifests

A **manifest** is a named, versioned snapshot of a corpus:
- **Snapshot** — freeze the current KB state for a given use case as a named manifest
- **Create from sources** — define the intended corpus upfront as a list of source references
- **Diff** — compare two manifests to see exactly what was added, removed, or changed
- **Re-ingest** — re-crawl all Confluence sources listed in a manifest
- **Remove** — bulk-delete a set of documents from the KB and all referencing manifests

Manifests move through states: `open → frozen → archived`.

### JSONL Schema Flexibility

The importer handles data from any external system via schema detection and custom field mapping:
- Auto-detects built-in schemas (crawler schema, pipeline export schema)
- User-defined schemas saved in `schemas.yaml` match on field presence
- Field mappings support dot-notation for nested values
- Pre-computed embeddings in JSONL are reused at push time (no API call needed)

---

## The User Journey End-to-End

```
1.  Knowledge Engineer opens the Streamlit app at localhost:8501

2.  ADD DOCUMENT
    - Upload file / paste URL / upload JSONL
    - Select use case ID + agent filter
    - System converts, chunks, and quality-checks
    - Result: auto-approved (score 1.0) or sent to review queue

3.  REVIEW QUEUE  (if flagged)
    - Inspect flagged chunks with quality signals
    - Edit content / tags, split chunks, reorganise if needed
    - Approve or reject

4.  PUSH TO KB
    - Click "Push N approved documents"
    - System embeds chunks via OpenAI (or local model)
    - Vectors upserted into Redis search index
    - KB ledger + use case ledger updated
    - Documents are now searchable

5.  AI AGENT QUERIES
    - Agent sends question as text
    - System embeds the question, runs KNN search in Redis
    - Returns top-k chunks with relevance scores and source citations
    - Agent can optionally filter to its own use case

6.  HEALTH MONITORING
    - KB Health page: check all pushed docs for drift
    - Use Case Ledger: view live coverage per (use case, agent)
    - Confluence sources: check last crawl, trigger manual refresh
    - Manifests: snapshot corpus before major changes

7.  ONGOING MAINTENANCE
    - Confluence sources refresh automatically on schedule
    - Stale file sources flagged in drift check
    - Knowledge engineer re-ingests changed sources as needed
```

---

## What Lives Where

| Data | Store | Purpose |
|---|---|---|
| In-flight document chunks during ingest/review | MongoDB `staging_docs` + `staging_chunks` | Human review gate; editable before push |
| Pushed document records + drift fingerprints | MongoDB `kb_documents` | Audit trail; drift detection |
| Chunk-ID inventory per use case | MongoDB `usecase_ledger` | Scoped search; coverage tracking |
| Confluence source metadata + schedules | MongoDB `usecase_confluence_sources` | Refresh scheduling; lightweight drift check |
| Corpus version snapshots | MongoDB `doc_manifests` | Versioning; diff; re-ingest |
| Searchable vector embeddings | Redis Stack (RediSearch) | KNN semantic search |

---

## What This Tool Is Not

- **Not a search API.** It manages the corpus. AI agents query Redis directly.
- **Not a document store.** Original files are not kept — only processed chunks.
- **Not multi-tenant.** The UI has no authentication; it assumes a trusted single-user environment.
- **Not a distributed system.** The Confluence refresh scheduler runs in-process; multi-instance deployments would need coordination.
