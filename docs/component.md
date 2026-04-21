C4Component
    title Component — Pipeline Core

    Container_Ext(ui,  "Streamlit Web App")
    Container_Ext(cli, "CLI Tool")

    System_Boundary(core, "Pipeline Core") {
        Component(ingest,     "ingest.py",          "Orchestrator",          "Entry point for all ingest paths: document, JSONL, legacy Markdown. Routes to converter, quality, chunker, staging.")
        Component(converter,  "converter.py",       "Docling Adapter",       "Converts PDF / DOCX / PPTX / HTML / URL to Markdown + Citation using Docling DocumentConverter")
        Component(quality,    "quality.py",         "Quality Assessor",      "Scores 0–1 from heading depth, section richness, metadata. Auto-tags from H1–H3. Flags docs below threshold.")
        Component(chunker,    "chunker.py",         "Chunker",               "Docling HybridChunker (max 512 tokens) for rich formats; heading-based splitter for Markdown")
        Component(embedder,   "embedder.py",        "Embedder",              "Pluggable: OpenAI · Azure OpenAI · SentenceTransformers. Batched. Skips pre-computed vectors.")
        Component(staging,    "MongoStagingStore",  "Staging Store",         "MongoDB-backed. Status lifecycle: pending_review → approved → pushed. Stores metadata + serialised chunks.")
        Component(ledger,     "KBLedger",           "KB Ledger",             "Permanent push record. Drift detection via file mtime/size. Statuses: current · stale · deleted · unknown.")
        Component(review,     "review.py",          "Review Orchestrator",   "Lists/approves/rejects staged docs. On push: embeds remaining chunks, upserts to vector backend, records in ledger.")
        Component(jsonl_imp,  "jsonl_importer.py",  "JSONL Importer",        "Auto-detects schema (crawler / pipeline / custom). Loads schemas.yaml. Maps records to Chunks. Reuses pre-computed embeddings.")
        Component(confluence, "confluence.py",      "Confluence Crawler",    "Recursive REST API crawl. Converts Confluence HTML storage format to plain text. Outputs pipeline-schema JSONL.")
        Component(redis_s,    "redis_store.py",     "Redis Store",           "RediSearch FLAT index. Upsert JSON + vector. Cosine distance search with tag filter.")
        Component(qdrant_s,   "qdrant_store.py",    "Qdrant Store",          "HNSW collection with cosine similarity. Upsert with quality scores. Filtered search by tag and source type.")
    }

    ContainerDb_Ext(mongo,    "MongoDB")
    ContainerDb_Ext(redis_db, "Redis Stack")
    ContainerDb_Ext(qdrant_db,"Qdrant")
    System_Ext(openai_ext,    "OpenAI / Azure OpenAI")
    System_Ext(confluence_ext,"Confluence")

    Rel(ui,        ingest,    "ingest_document / ingest_jsonl / ingest_file")
    Rel(ui,        review,    "list / approve / reject / push")
    Rel(cli,       ingest,    "ingest doc / jsonl / file / dir")
    Rel(cli,       review,    "review list / approve / reject / push")

    Rel(ingest,    converter, "Convert source file or URL")
    Rel(ingest,    quality,   "Score + auto-tag converted doc")
    Rel(ingest,    chunker,   "Chunk converted doc")
    Rel(ingest,    staging,   "Enqueue metadata + chunks")
    Rel(ingest,    jsonl_imp, "Delegate JSONL import path")

    Rel(confluence,ingest,    "Produces JSONL → ingest_jsonl()")
    Rel(confluence,confluence_ext, "Fetch pages recursively", "REST API v1")

    Rel(review,    staging,   "Read approved doc + chunks")
    Rel(review,    embedder,  "Embed chunks missing vectors")
    Rel(review,    redis_s,   "Upsert to Redis (if backend=redis)")
    Rel(review,    qdrant_s,  "Upsert to Qdrant (if backend=qdrant)")
    Rel(review,    ledger,    "record_push() after successful push")

    Rel(jsonl_imp, staging,   "Enqueue batch as single doc")

    Rel(staging,   mongo,     "staging_docs + staging_chunks collections")
    Rel(ledger,    mongo,     "kb_documents collection")
    Rel(embedder,  openai_ext,"Batch embed API call")
    Rel(redis_s,   redis_db,  "RESP3")
    Rel(qdrant_s,  qdrant_db, "gRPC / HTTP")
