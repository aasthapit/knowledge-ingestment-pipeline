C4Container
    title Container — Knowledge Ingestion Pipeline

    Person(engineer, "Knowledge Engineer")
    Person_Ext(consumer, "AI Agent / App")

    System_Boundary(sys, "Knowledge Ingestion Pipeline") {
        Container(ui,   "Streamlit Web App", "Python / Streamlit 1.40+", "All user-facing workflows: ingest, Confluence import, review queue, search, KB health, status")
        Container(cli,  "CLI Tool",          "Python / Click",           "Batch ingest, review, query — scriptable and CI-friendly")
        Container(core, "Pipeline Core",     "Python library",           "Conversion, quality scoring, chunking, embedding, staging, push orchestration, drift detection")
    }

    ContainerDb(mongo,    "MongoDB",      "MongoDB 7+",           "staging_docs · staging_chunks · kb_documents")
    ContainerDb(redis_db, "Redis Stack",  "Redis + RediSearch",   "FLAT vector index on doc:* keys — cosine distance, 1536 dims")
    ContainerDb(qdrant_db,"Qdrant",       "Qdrant HNSW",          "Cosine similarity collection — optional production backend")

    System_Ext(openai,     "OpenAI / Azure OpenAI")
    System_Ext(confluence, "Confluence")

    Rel(engineer,  ui,        "Uses",                          "HTTPS")
    Rel(engineer,  cli,       "Uses",                          "Shell")
    Rel(ui,        core,      "Calls pipeline functions")
    Rel(cli,       core,      "Calls pipeline functions")
    Rel(core,      mongo,     "Stage docs + chunks; read/write ledger", "pymongo TCP")
    Rel(core,      redis_db,  "Upsert vectors; cosine search",          "RESP3")
    Rel(core,      qdrant_db, "Upsert vectors; filtered search",        "gRPC / HTTP")
    Rel(core,      openai,    "Embed chunks (batched)",                 "HTTPS REST")
    Rel(core,      confluence,"Crawl page trees",                       "HTTPS REST")
    Rel(consumer,  redis_db,  "Semantic search",                        "RediSearch API")
    Rel(consumer,  qdrant_db, "Semantic search",                        "Qdrant API")
