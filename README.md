# Knowledge Ingestment Pipeline

A Python pipeline that converts **Markdown documents** into vector embeddings
and stores them in **Redis Enterprise** (RediSearch) for semantic retrieval by
AI agents.

```
Markdown files
      │
      ▼
  Chunker          ← splits by heading, respects YAML front-matter
      │
      ▼
  Embedder         ← OpenAI / Azure OpenAI / sentence-transformers (via .env)
      │
      ├──► JSONL export   (./output/chunks_<ts>.jsonl)
      │
      └──► Redis upsert   (RediSearch JSON + FLAT vector index)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd knowledge-ingestment-pipeline
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env
# Edit .env — set OPENAI_API_KEY and REDIS_URL at minimum
```

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `azure` \| `sentence-transformers` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name |
| `EMBEDDING_DIMENSIONS` | `1536` | Must match the model output |
| `OPENAI_API_KEY` | *(required)* | Your OpenAI secret key |
| `REDIS_URL` | `redis://localhost:6379` | Redis Enterprise connection string |
| `REDIS_INDEX_NAME` | `knowledge_index` | RediSearch index name |
| `CHUNK_MAX_CHARS` | `2000` | Max characters per chunk |

### 3. Ingest a file

```bash
python cli.py ingest file docs/redis-setup.md --tags infrastructure --tags redis
```

### 4. Ingest a directory

```bash
python cli.py ingest dir ./docs --tags internal
```

### 5. Query

```bash
python cli.py query "How do I connect to Redis Enterprise?"
```

---

## Document Tagging

Tags can be set in three ways (all are merged):

### a) YAML front-matter (per document)

```yaml
---
title: My Doc
tags:
  - python
  - howto
---
```

### b) CLI flag (at ingest time)

```bash
python cli.py ingest file my-doc.md --tags team-a --tags draft
```

### c) Post-hoc via the `retag` command

```bash
python cli.py retag <chunk-id-1> <chunk-id-2> --add published --remove draft
```

---

## JSONL Format

Each line in the exported JSONL file contains:

```json
{
  "chunk_id": "uuid",
  "source": "docs/redis-setup.md",
  "title": "Redis Setup Guide",
  "section": "Redis Setup Guide > Installation > Using Docker (local)",
  "content": "...",
  "tags": ["redis", "infrastructure"],
  "metadata": { "ingested_at": 1741000000 },
  "embedding": [0.012, -0.034, ...]
}
```

---

## CLI Reference

```
python cli.py --help

Commands:
  ingest file  Ingest a single markdown file
  ingest dir   Ingest all markdown files in a directory
  query        Semantic search over stored chunks
  retag        Add/remove tags on existing Redis chunks
  index create Create (or confirm) the RediSearch vector index
  index drop   Drop the RediSearch vector index
```

### Options for `ingest file`

| Flag | Description |
|---|---|
| `--tags / -t` | Tag(s) to attach (repeatable) |
| `--no-jsonl` | Skip JSONL export |
| `--no-redis` | Skip Redis upsert (JSONL only) |
| `--output / -o` | Custom JSONL output path |

### Options for `query`

| Flag | Description |
|---|---|
| `--top-k / -k` | Number of results (default 5) |
| `--tag-filter` | RediSearch tag filter, e.g. `"@tags:{python\|redis}"` |
| `--json-out` | Emit raw JSON instead of formatted output |

---

## Switching Embedding Models

Change `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` in `.env`.  
If you switch dimensions you **must** drop and recreate the index:

```bash
python cli.py index drop --delete-docs
python cli.py index create
```

### Supported providers

| Provider | `EMBEDDING_PROVIDER` | Notes |
|---|---|---|
| OpenAI | `openai` | Needs `OPENAI_API_KEY` |
| Azure OpenAI | `azure` | Needs `AZURE_OPENAI_*` vars |
| Local model | `sentence-transformers` | `pip install sentence-transformers` |

---

## Project Structure

```
knowledge-ingestment-pipeline/
├── .env.example          ← copy to .env and fill in
├── cli.py                ← command-line entry point
├── requirements.txt
├── docs/                 ← sample markdown documents
└── pipeline/
    ├── config.py         ← settings loaded from .env
    ├── chunker.py        ← markdown → Chunk objects
    ├── embedder.py       ← Chunk → embedding vectors
    ├── redis_store.py    ← RediSearch index + upsert + search
    ├── exporter.py       ← export/load JSONL files
    ├── tagger.py         ← tag management (in-memory + Redis)
    └── ingest.py         ← orchestration (file / directory / query)
```

---

## License

MIT
