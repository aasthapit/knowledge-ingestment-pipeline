# System Context — Knowledge Ingestion Pipeline

High-level view of who and what interacts with the pipeline.

```mermaid
graph TD
    engineer["👤 Knowledge Engineer\nIngests documents, reviews quality,\nmanages KB health per use case"]
    consumer["👤 AI Agent / App\nQueries the knowledge base\nfor semantic retrieval"]

    pipeline["⚙️ Knowledge Ingestion Pipeline\nConverts · quality-checks · chunks\nembeds · stores · tracks documents"]

    openai["OpenAI / Azure OpenAI\nGenerates text embedding vectors"]
    confluence["Confluence\nSource of structured wiki page trees"]
    mongodb[("MongoDB\nStaging queue · KB ledger\nUse Case ledger · drift tracking")]
    redis[("Redis Stack\nVector search index\n(RediSearch)")]

    engineer -->|"Browser / CLI — uploads files,\nreviews docs, manages use cases"| pipeline
    pipeline -->|"HTTPS REST — embed text chunks"| openai
    pipeline -->|"Confluence REST API — crawl page trees\ncheck drift via version metadata"| confluence
    pipeline -->|"pymongo — stage docs, track pushes,\nstore use case + agent assignments"| mongodb
    pipeline -->|"redis-py — upsert & search vectors"| redis
    consumer -->|"RediSearch — semantic vector search\n(optionally scoped to use case chunk IDs)"| redis
```

## Key concepts

**Use case ID + Agent filter** — every document is tagged with which business use case it supports and which AI agent it's meant for. This flows from ingestion through the review step and into the knowledge base ledger, so search results can be scoped and the health of each use case can be tracked independently.

**Staging vs. live** — nothing reaches the vector index without going through MongoDB staging first. Documents sit in staging (pending, approved, pushed) so they can be reviewed, edited, or rejected before they affect search results.

**Confluence drift** — the pipeline stores a page-version snapshot after each crawl. A lightweight drift check (metadata only, no body fetch) can then detect added, removed, or updated pages without re-crawling everything.

**KB ledger** — MongoDB records every push event with chunk IDs, source fingerprints, and timestamps. This makes it possible to detect staleness, audit what's in the index, and roll back if needed.
