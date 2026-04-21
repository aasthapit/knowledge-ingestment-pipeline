C4Context
    title System Context — Knowledge Ingestion Pipeline

    Person(engineer, "Knowledge Engineer", "Ingests documents, reviews quality, manages KB health")
    Person_Ext(consumer, "AI Agent / App", "Queries the knowledge base for semantic retrieval")

    System(pipeline, "Knowledge Ingestion Pipeline", "Converts, quality-gates, chunks, embeds, and stores documents in a searchable vector knowledge base")

    System_Ext(openai,     "OpenAI / Azure OpenAI",  "Generates text embedding vectors")
    System_Ext(confluence, "Confluence",              "Source of structured wiki page trees")
    System_Ext(mongodb,    "MongoDB",                 "Document staging queue and permanent KB ledger")
    System_Ext(redis,      "Redis Stack",             "Default vector database (RediSearch)")
    System_Ext(qdrant,     "Qdrant",                  "Alternative production vector database")

    Rel(engineer,  pipeline,   "Uploads files, reviews docs, searches KB", "Browser / CLI")
    Rel(consumer,  redis,      "Semantic search queries",                  "RediSearch API")
    Rel(consumer,  qdrant,     "Semantic search queries",                  "Qdrant HTTP/gRPC")
    Rel(pipeline,  openai,     "Embed text chunks",                        "HTTPS REST")
    Rel(pipeline,  confluence, "Crawl page trees",                         "Confluence REST API v1")
    Rel(pipeline,  mongodb,    "Stage docs, track drift",                  "pymongo")
    Rel(pipeline,  redis,      "Upsert and search vectors",                "redis-py")
    Rel(pipeline,  qdrant,     "Upsert and search vectors",                "qdrant-client")
