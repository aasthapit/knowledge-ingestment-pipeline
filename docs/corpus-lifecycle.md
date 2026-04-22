# Corpus Lifecycle Management — RAG Data Pipeline

Managing the document corpus is the most underestimated part of a production RAG system. This document covers how data flows from a URL through schema mapping, labeling, and review, ending as versioned JSONL records in MongoDB — and how that same store lets you add, prune, and roll back the corpus over time.

---

## Naive RAG vs. Managed Corpus RAG

Most RAG tutorials stop at "embed your docs and run a vector search." That approach works for demos but breaks in production.

| Dimension | Naive RAG | Managed Corpus RAG |
|---|---|---|
| **Ingestion** | Bulk embed, no review | Quality gate + human review before push |
| **Provenance** | Lost after indexing | Full citation tracked per chunk (source, page, author, date) |
| **Updates** | Re-embed everything | Drift detection — only stale/changed docs re-ingested |
| **Rollback** | Not possible | MongoDB ledger preserves prior push state; staging never deleted by default |
| **Labeling** | Manual or none | Auto-tags from headings + static tags from schema + explicit CLI tags |
| **Schema flexibility** | One format | Pluggable YAML schemas map any JSONL structure to a canonical chunk |
| **Corpus health** | Unknown | Live drift status: `current · stale · deleted · unknown` |
| **Multi-KB** | Single index | `kb_name` scopes staging, ledger, and vector index per knowledge base |

The pipeline in this repo is the managed path. MongoDB serves as the **source of truth** for what has been pushed, when, and in what state — making every operation auditable and reversible.

---

## End-to-End Flow

```mermaid
flowchart TD
    A([URL / File / Confluence page]) --> B["converter.py\nDocling fetches & converts\nPDF · DOCX · PPTX · HTML · URL → Markdown + Citation"]
    B --> C["quality.py\nScores 0–1\nHeadings · richness · metadata\nAuto-extracts tags from H1–H3"]
    C --> D{Quality score\n≥ threshold?}
    D -->|Yes — auto-approved| E["status = approved"]
    D -->|No — flagged| F["status = pending_review"]
    E --> G["mongo_store.py\nstaging_docs + staging_chunks\nStored in MongoDB"]
    F --> G
    G --> H["Review Queue\nStreamlit UI or CLI\napprove · reject · split · edit chunks"]
    H --> I{Decision}
    I -->|Approved| J["review.py — push_approved()\nEmbed chunks (or reuse pre-computed)\nUpsert to Redis / Qdrant"]
    I -->|Rejected| K["status = rejected\nRemains in staging\nNever pushed"]
    J --> L["KBLedger — kb_documents\nRecords chunk_ids · source_mtime · source_size\ndrift_status = current"]
    L --> M([Vector Store — searchable])
```

---

## Step 1 — Fetch URL Data

URLs (and files) enter the pipeline through `converter.py`, which wraps the Docling library to handle any format uniformly.

```mermaid
sequenceDiagram
    actor User
    participant Ingest as ingest.py
    participant Converter as converter.py / Docling
    participant Quality as quality.py
    participant Chunker as chunker.py

    User->>Ingest: ingest_document(url, tags=["prod"], kb_name="ops")
    Ingest->>Converter: convert_document(url)
    Note over Converter: Docling fetches URL,<br/>extracts structure,<br/>renders Markdown
    Converter-->>Ingest: ConvertedDocument<br/>(markdown, Citation{url, title, author, page_count})
    Ingest->>Quality: assess_quality(markdown)
    Quality-->>Ingest: QualityResult{score=0.78, passed=True,<br/>suggested_tags=["api","auth","oauth"]}
    Ingest->>Chunker: chunk_docling(doc, max_tokens=512)
    Chunker-->>Ingest: List[Chunk] — each with section breadcrumb,<br/>content, page_number in metadata
    Ingest->>Ingest: merge tags:<br/>suggested_tags + static_tags + extra_tags
```

Every chunk carries a full `Citation` in its `metadata` field — source URL, title, author, page number — so retrieval results are always attributable.

---

## Step 2 — Apply the YAML Schema

When data arrives as JSONL (from a crawler, external system, or export), `schemas.yaml` maps the foreign field names to the canonical `Chunk` structure without writing code.

### How a schema is defined

```yaml
# schemas.yaml
schemas:
  - name: my_docs
    detect:
      required: [body, url]       # all must be present to match this schema
      exclude: [section_breadcrumbs]  # any present disqualifies the match
    fields:
      content: body               # maps "body" field → chunk content
      source: url                 # maps "url" field  → chunk source
      title: page_title
      section: category
      chunk_id: id
      tags: labels
      embedding: vector           # pre-computed vector — reused, no API call
    tags_static: [internal, v2]   # always added to every chunk
    section_join: " > "           # join list-type section fields with this
```

### Detection and mapping flow

```mermaid
flowchart TD
    A([JSONL file]) --> B["jsonl_importer.py\nRead first record"]
    B --> C{Match custom schemas\nin order}
    C -->|"required fields present\nexclude fields absent"| D["Apply matched schema\nmap fields → Chunk"]
    C -->|No match| E{Built-in schemas}
    E -->|"has 'text' + 'page_url'"| F["Crawler schema"]
    E -->|"has 'content' + 'source'"| G["Pipeline schema\n(this project's export format)"]
    E -->|Nothing matched| H["Unknown — best-effort\nfield detection"]
    D --> I["Chunk{chunk_id, source, title,\nsection, content, tags, metadata}"]
    F --> I
    G --> I
    H --> I
    I --> J["tags = rec_tags + tags_static + extra_tags\n(order-preserving dedup)"]
    J --> K["Enqueue as single staging_doc\nstatus = approved\nchunks → staging_chunks"]
```

The entire JSONL batch is stored as one `staging_doc` record, with each line becoming a row in `staging_chunks`. If the JSONL includes pre-computed `embedding` vectors, they are preserved in `_embedding` and reused at push time — no OpenAI call needed.

---

## Step 3 — Label and Review

Labeling happens automatically during ingestion but can be refined in the review queue before anything reaches the vector store.

### Tag sources (merged in order)

```mermaid
flowchart LR
    A["quality.py\nAuto-tags from H1–H3 headings\n(up to 10 lowercase keywords)"]
    B["schemas.yaml\ntags_static: [internal, v2]\n(always applied per schema)"]
    C["CLI / API\nextra_tags passed at ingest time"]
    D["Merge — order-preserving dedup\nrec_tags → tags_static → extra_tags"]
    A --> D
    B --> D
    C --> D
    D --> E["Chunk.tags stored\nin staging_chunks + vector store"]
```

### Review workflow

```mermaid
sequenceDiagram
    actor Engineer
    participant UI as Streamlit / CLI
    participant Review as review.py
    participant Staging as MongoStagingStore
    participant Tagger as tagger.py

    Engineer->>UI: Open Review Queue
    UI->>Review: list_all_docs()
    Review->>Staging: query staging_docs by status
    Staging-->>Review: [{doc_id, title, quality_score, status, chunk_count, tags}]
    Review-->>UI: doc list
    Engineer->>UI: Inspect doc — view chunks, quality flags
    UI->>Review: get_doc_detail(doc_id)
    Review-->>UI: metadata + chunk samples

    alt Edit chunk before push
        Engineer->>UI: Edit chunk content / tags / section
        UI->>Review: update_chunk(doc_id, chunk_id, {content, tags})
        Review->>Staging: update staging_chunks record
    end

    alt Add / remove tags across chunks
        Engineer->>UI: Retag (add "deprecated", remove "v1")
        UI->>Tagger: apply_tags(chunks, add=["deprecated"], remove=["v1"])
    end

    alt Approve
        Engineer->>UI: Approve
        UI->>Review: approve_doc(doc_id)
        Review->>Staging: status → "approved", approved_at = now()
    else Reject
        Engineer->>UI: Reject with reason
        UI->>Review: reject_doc(doc_id, reason)
        Review->>Staging: status → "rejected"
    else Split
        Engineer->>UI: Move chunks to new doc
        UI->>Review: split_doc(source_id, chunk_ids, new_title)
        Review->>Staging: Create new staging_doc, reassign chunks
    end
```

---

## Step 4 — Store JSONL in MongoDB

MongoDB plays two distinct roles: a mutable **staging store** and a permanent **ledger**.

### Collections

```mermaid
erDiagram
    staging_docs {
        string _id PK
        string title
        string source_path
        string source_type
        float quality_score
        bool quality_passed
        list quality_flags
        list suggested_tags
        string status
        string schema_type
        string kb_name
        datetime ingested_at
        datetime approved_at
        datetime pushed_at
        string reject_reason
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
        string source_path
        string source_type
        list chunk_ids
        list tags
        float quality_score
        string kb_name
        datetime pushed_at
        string drift_status
        float source_mtime
        int source_size
    }
    staging_docs ||--o{ staging_chunks : "has chunks"
    staging_docs ..o| kb_documents : "becomes ledger entry on push"
```

### Document lifecycle

```mermaid
stateDiagram-v2
    [*] --> pending_review : Quality score < threshold
    [*] --> approved : Quality score ≥ threshold
    pending_review --> approved : Engineer approves
    pending_review --> rejected : Engineer rejects
    approved --> pushed : push_approved() — embedded & upserted to vector store
    pushed --> [*] : staging_doc removed (default)\nledger entry persists in kb_documents
    rejected --> [*] : Remains in staging\nnever pushed
```

---

## Corpus Lifecycle Management

The ledger in `kb_documents` is what makes the corpus manageable over time.

### Adding documents

Any new ingest (URL, file, JSONL batch) creates new staging records. After approval and push, the ledger records the chunk IDs and source fingerprint (`mtime`, `size`). The vector store index grows incrementally — no full rebuild needed.

### Pruning documents

```mermaid
flowchart LR
    A["Engineer selects doc\nin KB Health page"] --> B["review.py — remove from ledger\nledger.remove_doc(doc_id)"]
    B --> C["redis_store.delete_chunks(chunk_ids)\nor qdrant_store.delete_chunks(chunk_ids)"]
    C --> D["Ledger entry deleted\nVector store chunks deleted"]
```

Chunk IDs stored in `kb_documents.chunk_ids` make targeted deletion exact — only the pruned document's vectors are removed.

### Drift detection and re-ingestion

```mermaid
sequenceDiagram
    participant Drift as Drift Check (pages/drift.py)
    participant Ledger as KBLedger
    participant FS as File System / URL
    participant Ingest as ingest.py

    Drift->>Ledger: run_drift_check(kb_name)
    loop each kb_documents entry
        Ledger->>FS: stat(source_path) — mtime, size
        alt mtime differs > 1s OR size differs
            Ledger-->>Drift: drift_status = "stale"
        else file missing
            Ledger-->>Drift: drift_status = "deleted"
        else URL source
            Ledger-->>Drift: drift_status = "unknown"
        else unchanged
            Ledger-->>Drift: drift_status = "current"
        end
    end
    Drift-->>Engineer: Summary {current: N, stale: N, deleted: N, unknown: N}
    Engineer->>Drift: Re-ingest stale docs
    Drift->>Ingest: ingest_document(source_path) for each stale doc
    Note over Ingest: New staging_doc created\nOld ledger entry replaced on push
```

### Reverting to a prior corpus state

The staging store is non-destructive by default (`remove_after_push=False` preserves staging records). Combined with the ledger's push timestamps, you can reconstruct what was in the corpus at any point:

1. **Query the ledger** — `kb_documents` records `pushed_at` for every document version. Filter by timestamp to see the corpus state at a given date.
2. **Identify chunks to remove** — Documents pushed after your target timestamp have `pushed_at > target`. Their `chunk_ids` can be deleted from the vector store.
3. **Re-push prior versions** — If staging records are retained, prior chunk content is still in `staging_chunks`. Re-approve and push to restore.

```mermaid
flowchart TD
    A["Target rollback date: T"] --> B["Query kb_documents\nwhere pushed_at > T"]
    B --> C["Collect chunk_ids from\npost-T documents"]
    C --> D["Delete those chunk_ids\nfrom vector store"]
    D --> E{Prior version\nin staging?}
    E -->|Yes| F["Re-approve staging_doc\npush_approved(doc_id)"]
    E -->|No| G["Re-ingest source\n(URL or file)"]
    F --> H([Corpus restored to state at T])
    G --> H
```

---

## Configuration Reference

| Setting | Env var | Default | Purpose |
|---|---|---|---|
| Quality threshold | `QUALITY_THRESHOLD` | `0.6` | Auto-approve cutoff |
| Max chunk tokens | `DOCLING_MAX_TOKENS` | `512` | HybridChunker max |
| Chunk overlap | `CHUNK_OVERLAP_CHARS` | `200` | Sliding window overlap |
| MongoDB database | `MONGODB_DB_NAME` | `knowledge_pipeline` | All collections live here |
| Collection prefix | `MONGODB_COLLECTION_PREFIX` | `""` | Scope per environment (e.g. `prod_`) |
| KB name | `--kb-name` CLI flag | `default` | Logical corpus partition |
| Vector backend | `VECTOR_BACKEND` | `redis` | `redis` or `qdrant` |
| Keep staging after push | `remove_after_push=False` | `True` | Set to False for rollback support |
