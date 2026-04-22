---
title: Redis Setup Guide
tags:
  - redis
  - infrastructure
  - getting-started
---

# Redis Setup Guide

This document walks through setting up Redis for use with the knowledge ingestion pipeline.

## Setup Flow

```mermaid
flowchart TD
    A([Start]) --> B{Deployment target?}
    B -->|Local dev| C["Run Redis Stack via Docker"]
    B -->|Production| D["Create Redis Enterprise Cloud account"]
    C --> E["Expose ports 6379 & 8001"]
    D --> F["Create Fixed / Flexible subscription"]
    F --> G["Enable Search & Query module"]
    G --> H["Copy endpoint & password"]
    E --> I["Set REDIS_URL in .env"]
    H --> I
    I --> J["Run: python cli.py index create"]
    J --> K{Output?}
    K -->|"Index ready."| L([Done ✓])
    K -->|WRONGTYPE error| M["Drop index: cli.py index drop --delete-docs\nRecreate: cli.py index create"]
    K -->|Connection refused| N["Check REDIS_URL, firewall, VPN, TLS"]
    M --> J
    N --> I
```

## Prerequisites

- Docker Desktop (for local testing) or a Redis Enterprise account
- Python 3.11+
- A valid OpenAI API key

## Installation

### Using Docker (local)

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

### Redis Enterprise Cloud

1. Create a free account at [Redis Cloud](https://redis.com/try-free/).
2. Create a **Fixed** or **Flexible** subscription.
3. Enable the **Search & Query** module on your database.
4. Copy the public endpoint and password into your `.env` file.

## Configuration

Edit `.env` in the project root:

```env
REDIS_URL=redis://:<password>@<host>:<port>
REDIS_INDEX_NAME=knowledge_index
```

## Verifying the Connection

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant CLI as cli.py
    participant Core as Pipeline Core
    participant Redis as Redis Stack

    Dev->>CLI: python cli.py index create
    CLI->>Core: create_index()
    Core->>Redis: FT.CREATE knowledge_index (FLAT, cosine, 1536 dims)
    alt Index created successfully
        Redis-->>Core: OK
        Core-->>CLI: Index ready
        CLI-->>Dev: "Index ready."
    else Key type conflict
        Redis-->>Core: WRONGTYPE error
        Core-->>CLI: raise RedisError
        CLI-->>Dev: Error message
        Dev->>CLI: python cli.py index drop --delete-docs
        CLI->>Redis: FT.DROPINDEX + DEL doc:*
        Redis-->>CLI: OK
        Dev->>CLI: python cli.py index create
        CLI->>Redis: FT.CREATE ...
        Redis-->>CLI: OK
        CLI-->>Dev: "Index ready."
    end
```

Run the following to confirm the pipeline can reach Redis:

```bash
python cli.py index create
```

You should see:

```
Index ready.
```

## Troubleshooting

### `WRONGTYPE Operation against a key holding the wrong kind of value`

The index key type conflicts with an existing key. Drop the index and recreate:

```bash
python cli.py index drop --delete-docs
python cli.py index create
```

### Connection refused

Check that the `REDIS_URL` in `.env` is correct and that the Redis database
is reachable from your machine (firewall / VPN / TLS settings).
