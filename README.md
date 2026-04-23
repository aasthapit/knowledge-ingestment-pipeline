# Knowledge Ingestion Pipeline

A tool for building and maintaining AI knowledge bases. It takes documents from various sources — PDFs, Word files, web pages, Confluence wikis — converts them into a searchable format, and gives you a review step before anything reaches your AI agents.

```
Documents (PDF, Word, PowerPoint, HTML, URLs, Confluence, JSONL)
      │
      ▼
  Convert & chunk  ─── breaks content into small, retrievable sections
      │
      ▼
  Quality check    ─── flags sections that are too short, too long, boilerplate, or stale
      │
      ├── all clear → auto-approved
      └── flagged   → sent to human review queue
            │
            ▼
      Review Queue  ─── inspect, approve, reject, or edit before anything goes live
            │
            ▼
      Embed & index ─── sections become searchable vectors in Redis
            │
            ▼
      Knowledge base ─── AI agents query it; ledger tracks what's there and when
```

---

## What problem does this solve?

Most approaches to AI knowledge bases work like this: dump your documents in, embed them all, run a search. That works for demos but becomes unmanageable quickly — stale content accumulates, low-quality chunks degrade search results, and you have no record of what's actually in the index or when it was last updated.

This pipeline adds structure around that process:

- **Review before publish** — nothing reaches the AI until a human (or the quality checker) has signed off
- **Track everything** — every document has a record: what was ingested, when, by whom, for which use case
- **Detect drift** — for Confluence sources, the system checks whether pages have changed since the last crawl and flags what needs refreshing
- **Scope by use case** — content is tagged with which AI agent it belongs to, so search results can be filtered to the right context

---

## Quick Start

### Prerequisites

| Service | Purpose | Local Docker |
|---|---|---|
| MongoDB 7+ | Staging queue + knowledge base ledger | `docker run -p 27017:27017 mongo:7` |
| Redis Stack | Vector search index | `docker run -p 6379:6379 redis/redis-stack-server:latest` |

### Install

```bash
git clone <repo-url>
cd knowledge-ingestment-pipeline

# Recommended — uses uv
uv sync --python python3.13

# Or plain pip
pip install -e .
```

### Configure

```bash
cp .env.example .env   # macOS / Linux
copy .env.example .env # Windows
```

Edit `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-...
MONGODB_URI=mongodb://localhost:27017
MONGODB_TLS=false        # for a plain local MongoDB
REDIS_URL=redis://localhost:6379
```

See [Configuration reference](#configuration-reference) for all options.

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How it works — the UI

### Dashboard

The first thing you see. Shows how many documents are waiting for review, approved and ready to push, and already live in the knowledge base. Quick links to the most common tasks.

---

### Add Document

Three ways to get content in:

**Upload a file** — PDF, Word (.docx), PowerPoint (.pptx), HTML, plain text, or Markdown. Drag and drop, select the use case it belongs to, and the system handles conversion and chunking.

**From a web address** — Paste a URL. The system fetches and converts the page automatically.

**Bulk JSONL import** — For large batches, upload a `.jsonl` file (one JSON object per line). The importer auto-detects the data format and shows a preview before importing. Custom field mappings let you import from any source without changing your data.

Every document can be tagged with a **use case ID** and **agent filter** — this is how you tell the system which AI agent this content is meant for.

---

### Confluence

Connects directly to Confluence (Cloud or Server/Data Center) and crawls a page tree. You provide the URL of a parent page and it fetches all child pages automatically.

Crawled pages go through the same quality check and review workflow as any other document.

**Drift detection** — after a crawl, the system stores a snapshot of what pages existed and at what version. Clicking "Check drift" later compares that snapshot against the current Confluence state without re-fetching content — showing you exactly which pages were added, removed, or updated since the last crawl.

---

### Review Queue

All staged documents appear here, whether auto-approved or flagged for review.

For each document you can see:
- Which use case and agent it's tagged for
- Quality signals: how many sections are too short, too long, or boilerplate
- Content age: a warning if the source is older than 6 months
- The actual chunks that will go into the knowledge base

Actions per document:
- **Approve** — marks it ready to push
- **Reject** — removes it from the queue with an optional reason
- **Edit chunks** — fix content, tags, or section labels before pushing
- **Split** — break a document into separate pieces if needed

**Push to Knowledge Base** — embeds all approved documents and makes them searchable. For JSONL imports that already include embeddings, this step is instant (no API calls needed).

---

### Search

Semantic search over the knowledge base. Type a question in plain language and get back the most relevant sections, with source citations and page numbers.

Filter by use case or agent to scope results to specific content. Filter by tags or document type.

---

### Use Case Ledger

The health dashboard for a specific use case + agent pair. Select a use case to see:

- **Chunks in KB** — how many sections are currently searchable for this agent
- **Documents** — which source documents are contributing
- **Content drift** — whether any sources have changed since they were last pushed
- **Confluence source** — if registered: last crawl date, next scheduled crawl, page count, and a "Refresh now" button
- **Pushed documents table** — every document with its quality score, chunk count, and drift status

The **Confluence Sources** sub-tab lets you register Confluence page trees for a use case, set a crawl schedule (standard cron expression), and trigger manual refreshes.

The **Bulk Import** sub-tab accepts a JSON manifest file to register multiple Confluence sources at once.

The **Export JSONL** sub-tab lets you download all chunks for a use case as a JSONL file — useful for sending content to an external embedding pipeline.

---

### KB Health

Tracks drift for all pushed documents across all knowledge bases. Shows which sources have changed since they were last indexed, which have been deleted, and which are current.

---

### Status

Connection health for Redis, MongoDB, and the embedding provider. Shows the current chunk count and pipeline configuration.

---

## Quality checks

When a document is ingested, the system evaluates every chunk individually and flags:

| Signal | What it means |
|---|---|
| **Too short** (< 100 chars) | Section stubs, navigation fragments, or empty headings — likely useless for search |
| **Too long** (> 2 000 chars) | May get truncated by the embedding model, degrading retrieval quality |
| **Boilerplate** | Navigation menus, table of contents, login prompts, copyright footers — noise |
| **Stale** (> 6 months old) | Content from the source's creation or modification date — worth checking if still accurate |

A document passes automatically when it has no flagged chunks. Any flag sends it to the review queue for a human to decide.

The quality score is the fraction of clean chunks — a document with 8 clean sections out of 10 scores 0.8.

---

## Use cases and agent filters

Every document can be tagged with two fields:

- **Use case ID** — the business use case this content supports (e.g. `GENAI1597_SSOP`)
- **Agent filter** — the specific AI agent or persona this content is meant for (e.g. `ssop_cloud_operations_agent`)

These fields are stored through the entire lifecycle — staging, review, push, and ledger. The Use Case Ledger page shows you the full picture per use case, and the search page lets you scope results to a specific use case so an agent only retrieves content that's relevant to its context.

---

## JSONL import formats

The importer auto-detects which format a file uses from the first record.

**Pipeline schema** — the format this system exports:

```json
{
  "chunk_id":  "abc-123",
  "source":    "https://docs.example.com/guide",
  "title":     "Guide Title",
  "section":   "Installation > Docker",
  "content":   "Run the following command...",
  "tags":      ["docker", "install"],
  "embedding": [0.012, -0.034, 0.019]
}
```

If `embedding` is present, it's reused — no API call needed.

**Crawler schema** — produced by web crawlers (detected by `text` + `page_url`):

```json
{
  "text":                "Content body...",
  "page_url":            "https://docs.example.com/page",
  "page_name":           "Page title",
  "section_breadcrumbs": ["Section", "Subsection"],
  "usecase_id":          "GENAI1597_SSOP",
  "agent_filter":        "ssop_agent"
}
```

**Custom schemas** — define your own field mappings in `schemas.yaml` without writing code:

```yaml
schemas:
  - name: my_docs
    detect:
      required: [body, url]
    fields:
      content:  body
      source:   url
      title:    page_title
      tags:     labels
    tags_static: [internal]
```

---

## Configuration reference

All settings are loaded from `.env` (or environment variables). Copy `.env.example` to `.env` to get started.

### Embedding

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `azure` \| `sentence-transformers` \| `custom` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name |
| `EMBEDDING_DIMENSIONS` | `1536` | Must match the model output |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `AZURE_OPENAI_API_KEY` | — | Required when provider is `azure` |
| `AZURE_OPENAI_ENDPOINT` | — | Required when provider is `azure` |
| `AZURE_OPENAI_DEPLOYMENT` | — | Required when provider is `azure` |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embedding API call |

### Custom embedding endpoint

To use any OpenAI-compatible embedding API (e.g. a local Ollama instance):

| Variable | Description |
|---|---|
| `EMBEDDING_CUSTOM_URL` | Your endpoint, e.g. `http://localhost:11434/v1/embeddings` |
| `EMBEDDING_CUSTOM_API_KEY` | API key if required |
| `EMBEDDING_CUSTOM_HEADERS` | Extra HTTP headers as a JSON object |

### Vector store (Redis)

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Redis Stack connection string |
| `REDIS_INDEX_NAME` | `knowledge_index` | RediSearch index name |
| `REDIS_KEY_PREFIX` | `doc:` | Key prefix for stored chunks |

### MongoDB

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | — | Full connection string — overrides all other MongoDB settings when set |
| `MONGODB_HOST` | `localhost` | Hostname (ignored when MONGODB_URI is set) |
| `MONGODB_PORT` | `27017` | Port (ignored when MONGODB_URI is set) |
| `MONGODB_USERNAME` | — | Leave empty for no auth |
| `MONGODB_PASSWORD` | — | Leave empty for no auth |
| `MONGODB_TLS` | `true` | Set `false` for plain local instances |
| `MONGODB_TLS_INSECURE` | `false` | Skip certificate verification (self-signed certs) |
| `MONGODB_SRV` | `true` | Use DNS SRV discovery — required for Atlas; set `false` for direct connections |
| `MONGODB_DB_NAME` | `knowledge_pipeline` | Database name |
| `MONGODB_COLLECTION_PREFIX` | — | Optional prefix, e.g. `prod_` to separate environments |

### Pipeline

| Variable | Default | Description |
|---|---|---|
| `DOCLING_MAX_TOKENS` | `512` | Max tokens per chunk (≈ 400 words) |
| `CHUNK_MAX_CHARS` | `2000` | Max characters per chunk |
| `CHUNK_OVERLAP_CHARS` | `200` | Overlap between consecutive chunks |
| `JSONL_OUTPUT_DIR` | `./output` | Default JSONL export directory |

### Confluence

| Variable | Description |
|---|---|
| `CONFLUENCE_BASE_URL` | Your Confluence base URL, e.g. `https://mycompany.atlassian.net` |
| `CONFLUENCE_AUTH_TYPE` | `cloud` (email + API token) or `server` (Personal Access Token) |
| `CONFLUENCE_EMAIL` | Required for Cloud auth |
| `CONFLUENCE_API_TOKEN` | API token (Cloud) or Personal Access Token (Server) |

---

## Project layout

```
knowledge-ingestment-pipeline/
├── app.py                    ← Streamlit entry point
├── cli.py                    ← Command-line interface
├── schemas.yaml              ← Custom JSONL field mapping definitions
├── .env.example              ← Copy to .env and fill in
│
├── pages/                    ← UI pages (one file per page)
│   ├── home.py               ← Dashboard
│   ├── ingest.py             ← Add Document
│   ├── confluence.py         ← Confluence crawler
│   ├── review.py             ← Review Queue
│   ├── search.py             ← Semantic search
│   ├── usecase_ledger.py     ← Use Case Ledger (health + Confluence sources + export)
│   ├── ledger.py             ← KB Health / drift detection
│   └── status.py             ← Connection status + configuration
│
└── pipeline/                 ← Core library
    ├── config.py             ← Settings from .env
    ├── converter.py          ← Document conversion (Docling)
    ├── quality.py            ← Quality assessment (chunk size, boilerplate, recency)
    ├── chunker.py            ← Document chunking
    ├── embedder.py           ← Embedding (OpenAI / Azure / local)
    ├── mongo_store.py        ← Staging store, KB ledger, Use Case ledger
    ├── redis_store.py        ← Vector search index (Redis RediSearch)
    ├── ingest.py             ← Ingestion orchestration
    ├── review.py             ← Approve / reject / push workflow
    ├── jsonl_importer.py     ← JSONL import with custom schema support
    ├── confluence.py         ← Confluence REST API crawler
    ├── refresh_scheduler.py  ← Background Confluence refresh scheduler
    └── exporter.py           ← JSONL export
```

---

## CLI reference

```bash
# Ingest a document (file or URL)
python cli.py ingest doc path/to/file.pdf --tags finance

# Import a JSONL bulk file
python cli.py ingest jsonl export.jsonl --tags openshift

# List staged documents
python cli.py review list

# Approve and push a document
python cli.py review approve <doc-id>
python cli.py review push

# Search
python cli.py query "How do I configure persistent storage?"
```

---

## License

MIT
