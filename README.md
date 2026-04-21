# Knowledge Ingestion Pipeline

A multi-format document ingestion pipeline with a quality gate, human review workflow, and Streamlit UI. Converts documents into vector embeddings stored in Redis or Qdrant. Uses MongoDB to track document staging and detect content drift over time.

```
Sources (PDF, DOCX, PPTX, HTML, URL, Markdown, JSONL, Confluence)
      │
      ▼
  Docling converter  ─── extracts text + citation metadata
      │
      ▼
  Quality assessor   ─── scores structure 0–1; auto-tags headings
      │
      ├── score ≥ threshold → auto-approved
      └── score < threshold → pending human review
            │
            ▼
      MongoDB staging  ← Review Queue (approve / reject / inspect)
            │
            ▼
      Embedder (OpenAI / Azure / sentence-transformers)
            │
            ▼
      Vector store (Redis RediSearch  or  Qdrant)
            │
            ▼
      KB Ledger (MongoDB) ← drift detection
```

---

## Quick Start

### 1. Prerequisites

| Service | Purpose | Local Docker command |
|---|---|---|
| MongoDB 7+ | Staging store + KB ledger | `docker run -p 27017:27017 mongo:7` |
| Redis Stack | Vector database (default) | `docker run -p 6379:6379 redis/redis-stack-server:latest` |
| Qdrant *(optional)* | Production vector DB | `docker run -p 6333:6333 qdrant/qdrant` |

### 2. Install

```bash
git clone <repo-url>
cd knowledge-ingestment-pipeline

# Recommended — uses uv
UV_LINK_MODE=copy uv sync --python python3.13

# Or plain pip
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
```

### 3. Configure

```bash
copy .env.example .env   # Windows
cp  .env.example .env    # macOS / Linux
```

Edit `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-...
MONGODB_HOST=localhost
MONGODB_TLS=false        # for a plain local MongoDB
REDIS_URL=redis://localhost:6379
```

See [Configuration reference](#configuration-reference) for all options.

### 4. Run the UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## UI Pages

### Dashboard (`/`)

Live summary of the staging queue and vector store. Shows counts of pending, approved, and pushed documents, plus quick-action buttons to jump to common tasks.

---

### Add Document (`/ingest`)

Three tabs for different input types.

**Upload a File**

Drag and drop or browse for: PDF, Word (.docx), PowerPoint (.pptx), HTML, plain text (.txt), or Markdown (.md).

Docling converts the file automatically — no manual formatting required.

**From a Web Address**

Paste any HTTP/HTTPS URL. The pipeline fetches and converts the page.

**Bulk JSONL Import**

Upload a `.jsonl` file. The importer auto-detects the schema (see [JSONL Schemas](#jsonl-schemas)) and shows a preview of the first 5 records before importing.

**Options (all tabs)**

| Option | Description |
|---|---|
| Tags | Comma-separated keywords applied to every chunk |
| Knowledge base | Logical KB name for grouping and drift tracking (default: `default`) |
| Quality threshold | Override the default 0.6 threshold for this import |
| Push directly | Skip the review step if quality passes |

---

### Confluence (`/confluence`)

Crawl a Confluence page tree directly into the knowledge base.

1. Enter your Confluence base URL (e.g. `https://mycompany.atlassian.net`)
2. Choose **Cloud** (email + API token) or **Server / DC** (Personal Access Token)
3. Paste the URL of the parent page — all sub-pages are fetched automatically
4. Set max depth (`-1` = entire tree), KB name, and any extra tags
5. Choose whether to **stage**, **download as JSONL**, or both
6. Click **Start crawl**

Pages are output in pipeline schema (see [JSONL Schemas](#jsonl-schemas)) and staged directly in the Review Queue.

**Getting a token**

- **Cloud:** [id.atlassian.com → Security → API tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
- **Server/DC:** Your profile → Personal Access Tokens

---

### Review Queue (`/review`)

Lists all staged documents. Filter by status: All, Pending review, Approved, Pushed.

For each document you can:

- **Approve** — marks the document ready to push
- **Reject** — removes it from the queue with an optional reason
- **Inspect** — see quality flags, sample chunks, metadata
- **Push to Knowledge Base** — embeds approved chunks and upserts them into the vector store

Documents that pass the quality threshold are auto-approved on ingest. Only low-scoring documents require manual review here.

---

### Search (`/search`)

Semantic search over the vector store. Results include relevance score, section breadcrumb, source citation, and page number (for PDF/DOCX). Filter by tag.

---

### KB Health (`/drift`)

Tracks which pushed documents are still current.

| Status | Meaning |
|---|---|
| ✅ Current | Source file unchanged since last push |
| ⚠️ Stale | Source file has been modified — re-ingest recommended |
| 🗑️ Deleted | Source file no longer exists |
| ❓ Unknown | URL source (cannot detect changes without fetching) |

Click **Check for changes** to run a full drift scan. Stale file-based documents show a **Re-ingest** button that re-processes and re-pushes in one click.

---

### Status (`/status`)

Connection health for Redis, MongoDB, and OpenAI. Shows index stats, chunk counts, and current configuration.

---

## JSONL Schemas

JSONL files must have one JSON object per line (UTF-8). The importer auto-detects which schema is in use from the first record.

### Built-in: Pipeline schema

Produced by the pipeline's own exporter and the Confluence crawler. Detected when a record contains both `content` and `source`.

```json
{
  "chunk_id":  "abc-123",
  "source":    "https://docs.example.com/guide",
  "title":     "Guide Title",
  "section":   "Guide Title > Installation > Docker",
  "content":   "Run the following command to install...",
  "tags":      ["docker", "install"],
  "metadata":  {},
  "embedding": [0.012, -0.034, 0.019]
}
```

| Field | Required | Notes |
|---|---|---|
| `content` | **yes** | Chunk body text |
| `source` | **yes** | URL or file path |
| `title` | no | Document title |
| `section` | no | Breadcrumb string |
| `chunk_id` | no | Auto-generated UUID if absent |
| `tags` | no | List of strings |
| `metadata` | no | Arbitrary dict, passed through |
| `embedding` | no | Pre-computed float array — **skips API call if present** |

### Built-in: Crawler schema

Produced by `crawl_ocp_docs.py`. Detected when a record contains both `text` and `page_url`.

```json
{
  "chunk_id":            "abc-123",
  "text":                "OpenShift Container Platform installation...",
  "page_url":            "https://docs.openshift.com/container-platform/4.18/...",
  "page_name":           "Installation overview",
  "section_heading":     "Prerequisites",
  "section_breadcrumbs": ["Installation", "Prerequisites"],
  "agent_filter":        "openshift",
  "usecase_id":          "install",
  "data_classification": "public"
}
```

`agent_filter`, `usecase_id`, and `data_classification` become tags automatically.

### Custom schemas

Define your own field mappings in [`schemas.yaml`](schemas.yaml) at the project root. Custom schemas are checked **before** the built-ins.

```yaml
schemas:
  - name: my_docs
    detect:
      required: [body, url]      # ALL must be present to match
      exclude:  [page_url]       # ANY present disqualifies the match
    fields:
      content:  body             # chunk body
      source:   url              # URL or file path
      title:    page_title       # document title
      section:  category         # section string (list joined with section_join)
      chunk_id: id
      tags:     labels           # list of strings or comma-separated string
      embedding: vector          # float array — reused if present
    tags_static: [internal]      # always added to every chunk
    section_join: " > "          # how to join list-type section fields
```

Field paths support dot notation for nested fields: `source: _links.webui` resolves `record["_links"]["webui"]`.

After editing `schemas.yaml` the running Streamlit app picks up the changes on the next import (no restart needed).

---

## Configuration Reference

All settings are loaded from `.env` (or environment variables). Copy `.env.example` to `.env` to get started.

### Embedding

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `azure` \| `sentence-transformers` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name |
| `EMBEDDING_DIMENSIONS` | `1536` | Must match the model output |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `AZURE_OPENAI_API_KEY` | — | Required when provider is `azure` |
| `AZURE_OPENAI_ENDPOINT` | — | Required when provider is `azure` |
| `AZURE_OPENAI_DEPLOYMENT` | — | Required when provider is `azure` |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embedding API call |

### Vector store

| Variable | Default | Description |
|---|---|---|
| `VECTOR_BACKEND` | `redis` | `redis` \| `qdrant` |
| `REDIS_URL` | `redis://localhost:6379` | Redis Stack connection string |
| `REDIS_INDEX_NAME` | `knowledge_index` | RediSearch index name |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `knowledge_base` | Qdrant collection name |

### MongoDB

| Variable | Default | Description |
|---|---|---|
| `MONGODB_HOST` | `localhost` | MongoDB hostname |
| `MONGODB_PORT` | `27017` | MongoDB port |
| `MONGODB_USERNAME` | — | Leave empty for no auth |
| `MONGODB_PASSWORD` | — | Leave empty for no auth |
| `MONGODB_AUTH_SOURCE` | — | Auth database; use `$external` for LDAP/X.509 |
| `MONGODB_TLS` | `true` | Set `false` for plain local instances |
| `MONGODB_DB_NAME` | `knowledge_pipeline` | Database name |
| `MONGODB_COLLECTION_PREFIX` | — | Optional prefix, e.g. `prod_` |

### Pipeline

| Variable | Default | Description |
|---|---|---|
| `QUALITY_THRESHOLD` | `0.6` | Auto-approve above this score (0–1) |
| `DOCLING_MAX_TOKENS` | `512` | Max tokens per chunk (≈ 400 words) |
| `CHUNK_MAX_CHARS` | `2000` | Max chars per chunk (legacy Markdown path) |
| `CHUNK_OVERLAP_CHARS` | `200` | Overlap between consecutive chunks |
| `JSONL_OUTPUT_DIR` | `./output` | Default JSONL export directory |

---

## Project Structure

```
knowledge-ingestment-pipeline/
├── app.py                    ← Streamlit entry point  (streamlit run app.py)
├── cli.py                    ← CLI entry point        (python cli.py --help)
├── schemas.yaml              ← Custom JSONL schema definitions
├── .env.example              ← Copy to .env and fill in
│
├── pages/                    ← Streamlit UI pages
│   ├── home.py               ← Dashboard
│   ├── ingest.py             ← Add Document (file / URL / JSONL)
│   ├── confluence.py         ← Confluence page-tree importer
│   ├── review.py             ← Review Queue
│   ├── search.py             ← Semantic search
│   ├── drift.py              ← KB Health / drift detection
│   └── status.py             ← Connection status + config
│
└── pipeline/                 ← Core library
    ├── config.py             ← Settings loaded from .env
    ├── converter.py          ← Docling document conversion + Citation dataclass
    ├── quality.py            ← Quality scoring (0–1) + auto-tagging
    ├── chunker.py            ← Docling HybridChunker + legacy Markdown chunker
    ├── embedder.py           ← OpenAI / Azure / sentence-transformers
    ├── mongo_store.py        ← MongoStagingStore + KBLedger (drift tracking)
    ├── redis_store.py        ← RediSearch vector index
    ├── qdrant_store.py       ← Qdrant vector index
    ├── ingest.py             ← High-level orchestration
    ├── review.py             ← Approve / reject / push workflow
    ├── jsonl_importer.py     ← JSONL import with custom schema support
    ├── confluence.py         ← Confluence REST API crawler
    ├── exporter.py           ← JSONL export
    └── tagger.py             ← Tag management
```

---

## CLI Reference

```
python cli.py --help
```

### Ingest commands

```bash
# Single file (PDF, DOCX, PPTX, HTML, Markdown, or URL)
python cli.py ingest doc path/to/file.pdf --tags finance --kb-name my-kb

# Single Markdown file (legacy — direct to Redis, no review step)
python cli.py ingest file docs/guide.md --tags internal

# Directory of Markdown files (legacy)
python cli.py ingest dir ./docs --tags team-a

# JSONL bulk import
python cli.py ingest jsonl export.jsonl --tags openshift --kb-name ocp-docs
```

### Review commands

```bash
# List all staged documents
python cli.py review list

# Show detail for one document
python cli.py review show <doc-id>

# Approve a document
python cli.py review approve <doc-id>

# Reject a document
python cli.py review reject <doc-id> --reason "Duplicate content"

# Push all approved documents to the vector store
python cli.py review push

# Push a specific document
python cli.py review push --doc-id <doc-id>
```

### Search

```bash
python cli.py query "How do I configure persistent storage in OpenShift?"
python cli.py query "RBAC roles" --top-k 10 --tag-filter openshift
```

---

## Switching Embedding Models

Update `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` in `.env`. If dimensions change you must drop and recreate the vector index:

**Redis:**
```bash
python cli.py index drop --delete-docs
python cli.py index create
```

**Qdrant:**  Delete the collection in the Qdrant dashboard or via API, then push again.

---

## License

MIT
