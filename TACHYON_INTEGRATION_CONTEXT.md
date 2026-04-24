# Tachyon Integration Context

This document is a handoff for a Copilot agent implementing the Tachyon-side function files in the **Tachyon pipeline repo**. Read it top-to-bottom before writing any code.

---

## What This Pipeline Does

`knowledge-ingestment-pipeline` is a Streamlit app that ingests documents into vector stores, manages knowledge bases (KBs) and corpora, and exposes a semantic search UI. Documents flow through:

```
KB (source) → staging (MongoDB) → review/approve → push to vector store
```

A **corpus** ties one or more KBs to a single vector store and a usecase context (`usecase_id`, `agent_filter`). The **vector store** is configured in `vector_stores.yaml` and is resolved at push/search time via a factory.

---

## What Has Already Been Done in This Repo

| File | Change |
|------|--------|
| `vector_stores.yaml` | Full Tachyon config template (see below) |
| `pipeline/tachyon/__init__.py` | Package marker — **you copy your function files here** |
| `pipeline/tachyon_client.py` | Adapter class `TachyonClient` — **you adjust call sites** |
| `pipeline/vector_store.py` | `TachyonVectorStore` fully wired to `TachyonClient` |
| `pipeline/ingest.py` | `query_vectorstore()` skips local embedding for Tachyon |
| `pipeline/mongo_store.py` | `KBLedger.record_push()` accepts `s3_file_id` + `vector_file_id` |

---

## Tachyon Vector Store Config Shape

When a user adds a Tachyon vector store they fill in `vector_stores.yaml`:

```yaml
- id: tachyon-prod
  name: Tachyon Production
  type: tachyon
  collection: my-corpus-name        # corpus/collection identifier in Tachyon
  extra:
    consumer_key:    ${TACHYON_CONSUMER_KEY}
    consumer_secret: ${TACHYON_CONSUMER_SECRET}
    api_key:         ${TACHYON_API_KEY}
    usecase_id:      ${TACHYON_USECASE_ID}
    apigee_url:      ${TACHYON_APIGEE_URL}
    search_url:      ${TACHYON_SEARCH_URL}
    completion_url:  ${TACHYON_COMPLETION_URL}
    cert_path:       ${TACHYON_CERT_PATH}     # path to client .crt
    key_path:        ${TACHYON_KEY_PATH}      # path to client .key
    ca_bundle:       ${TACHYON_CA_BUNDLE}     # path to CA bundle / root cert
```

`${VAR}` references are expanded from `.env` at runtime. Multiple Tachyon entries with different `id` values are supported (one per endpoint/usecase).

The config dict your functions receive at runtime:

```python
{
    "consumer_key":    str,
    "consumer_secret": str,
    "api_key":         str,
    "usecase_id":      str,
    "apigee_url":      str,      # Apigee token endpoint
    "search_url":      str,      # Tachyon search endpoint
    "completion_url":  str,      # Tachyon completion endpoint (future use)
    "cert":            (str, str) | None,  # (cert_path, key_path) tuple for mTLS
    "ca_bundle":       str | None,         # path to CA bundle
}
```

---

## What You Need to Implement

Copy your working function files into `pipeline/tachyon/` in this repo. Then adjust the call sites in `pipeline/tachyon_client.py` to match your actual function signatures.

### `pipeline/tachyon/auth.py`

Must export:

```python
def get_access_token(
    consumer_key: str,
    consumer_secret: str,
    apigee_url: str,
    cert: tuple[str, str] | None,   # (cert_path, key_path) for mTLS
    ca_bundle: str | None,
) -> str:
    """Exchange consumer credentials for a bearer token via Apigee OAuth."""
    ...
```

**Returns:** a bearer token string used in subsequent API calls.

---

### `pipeline/tachyon/search.py`

Must export:

```python
def search_documents(
    query: str,
    top_k: int,
    usecase_id: str,
    collection: str,
    token: str,
    search_url: str,
    api_key: str,
    cert: tuple[str, str] | None,
    ca_bundle: str | None,
) -> list[dict]:
    """Submit a text query to Tachyon and return ranked results."""
    ...
```

**Returns:** a list of result dicts. Each dict **must** contain at least:

```python
{
    "chunk_id": str,      # unique identifier for this result chunk
    "content":  str,      # the text of the result
    "score":    float,    # raw similarity score (lower = more similar, like L2 distance)
                          # OR higher = more similar — document which convention you use
}
```

Optionally include any of these and the UI will render them automatically:

```python
{
    "source":  str,        # source file path or URL
    "title":   str,        # document title
    "section": str,        # breadcrumb path within the document
    "tags":    list[str],  # topic tags
    "citation": {
        "title":       str,
        "url":         str | None,
        "source_path": str,
        "page_number": int | None,
        "page_count":  int | None,
        "author":      str | None,
    },
}
```

**Score convention note:** the pipeline normalises scores with `normalized_score = 1.0 - raw_score` (clamped to [0, 1]). If Tachyon returns a similarity score where higher = better, either invert it before returning or the pipeline will need to be adjusted. Document which convention your function uses.

---

### `pipeline/tachyon/delete.py`

Must export:

```python
def delete_file(
    s3_file_id: str,
    vector_file_id: str,
    token: str,
    api_key: str,
    cert: tuple[str, str] | None,
    ca_bundle: str | None,
) -> None:
    """Remove an S3 uploaded file and its Tachyon vector doc by file ID."""
    ...
```

These IDs come from MongoDB (`kb_documents.s3_file_id` and `kb_documents.vector_file_id`) and are populated by the ingestion flow below.

---

## Ingestion Flow (Follow-Up Plan — Implement Later)

The ingestion plan covers: **JSONL → S3 upload → Tachyon vectorize → track file IDs in MongoDB**.

When you implement it, add two more modules:

### `pipeline/tachyon/upload.py`

```python
def upload_to_s3(
    jsonl_path: str,         # local path to the .jsonl file
    token: str,
    api_key: str,
    cert: tuple[str, str] | None,
    ca_bundle: str | None,
    # ... any additional params your function requires
) -> str:
    """Upload a JSONL file to the S3 bucket and return the file ID."""
    ...
    return s3_file_id   # str
```

### `pipeline/tachyon/vectorize.py`

```python
def vectorize_file(
    s3_file_id: str,
    token: str,
    api_key: str,
    cert: tuple[str, str] | None,
    ca_bundle: str | None,
    # ... any additional params your function requires
) -> str:
    """Trigger Tachyon vectorization for an uploaded S3 file and return the vector file ID."""
    ...
    return vector_file_id   # str
```

After a successful upload + vectorize, call:

```python
from pipeline.mongo_store import get_kb_ledger

get_kb_ledger().record_push(
    doc_id=...,
    title=...,
    source_path=...,
    source_type="jsonl",
    url=None,
    chunk_ids=[],          # Tachyon tracks chunks internally
    tags=[],
    quality_score=1.0,
    kb_id=...,
    kb_name=...,
    usecase_id=...,
    agent_filter=...,
    s3_file_id=s3_file_id,         # ← from upload_to_s3()
    vector_file_id=vector_file_id,  # ← from vectorize_file()
)
```

This stores both IDs in `kb_documents` so `delete_file()` can look them up later.

---

## Adapter Call Sites to Adjust

`pipeline/tachyon_client.py` contains `# COPILOT: ...` comments at every call site. After copying your function files in, update the kwargs in `TachyonClient._token()`, `TachyonClient.search()`, and `TachyonClient.delete()` to match your actual function signatures.

Do **not** change the public method signatures of `TachyonClient` itself — the pipeline calls them directly.

---

## How the Pipeline Calls Your Code

### Search path (triggered from UI or API):

```
pages/search.py
  └─ pipeline/ingest.py: query_vectorstore(question, vs_id="tachyon-prod", ...)
       ├─ TachyonVectorStore detected (handles_own_embedding=True)
       ├─ skips local embedder
       └─ TachyonVectorStore.search(query_vector=[], query_text=question, ...)
            └─ TachyonClient.search(query, top_k, usecase_id, collection)
                 ├─ TachyonClient._token() → auth.get_access_token(...)
                 └─ search.search_documents(query, ..., token=token)
```

### Delete path (triggered from ingestion plan):

```
[ingestion plan code]
  └─ TachyonVectorStore.delete_chunks(chunk_ids)
       └─ looks up s3_file_id + vector_file_id from kb_documents by doc_id
            └─ TachyonClient.delete(s3_file_id, vector_file_id)
                 ├─ TachyonClient._token() → auth.get_access_token(...)
                 └─ delete.delete_file(s3_file_id, vector_file_id, ...)
```

---

## Verification Checklist

- [ ] `from pipeline.tachyon.auth import get_access_token` — imports without error
- [ ] `from pipeline.tachyon.search import search_documents` — imports without error
- [ ] `from pipeline.tachyon.delete import delete_file` — imports without error
- [ ] `from pipeline.tachyon_client import TachyonClient` — imports without error
- [ ] `from pipeline.vector_store import TachyonVectorStore` — imports without error
- [ ] A Tachyon entry added to `vector_stores.yaml` appears in the Vector Stores page
- [ ] A corpus assigned to that Tachyon vector store can be saved
- [ ] A search query via that corpus reaches `search_documents()` (check logs)
- [ ] No calls to the local embedder when searching a Tachyon corpus (check logs for "skips local embedding")
