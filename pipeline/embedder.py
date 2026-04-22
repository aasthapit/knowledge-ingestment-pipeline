"""
embedder.py
Wraps multiple embedding providers behind a single interface.
Provider is selected by EMBEDDING_PROVIDER in .env.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pipeline.config import settings

if TYPE_CHECKING:
    from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    return [item.embedding for item in response.data]


def _embed_azure(texts: list[str]) -> list[list[float]]:
    from openai import AzureOpenAI  # type: ignore

    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version="2024-02-01",
    )
    response = client.embeddings.create(
        model=settings.azure_openai_deployment,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


def _embed_custom(texts: list[str]) -> list[list[float]]:
    """
    POST to any OpenAI-compatible /embeddings endpoint.

    Configure via .env:
        EMBEDDING_PROVIDER=custom
        EMBEDDING_CUSTOM_URL=http://localhost:11434/v1/embeddings
        EMBEDDING_CUSTOM_API_KEY=          # optional Bearer token
        EMBEDDING_CUSTOM_HEADERS={}        # optional extra headers (JSON object)
        EMBEDDING_MODEL=nomic-embed-text   # model name sent in the request body

    Request body:  {"model": EMBEDDING_MODEL, "input": ["text", ...]}
    Accepted response shapes:
        {"data": [{"embedding": [...]}, ...]}   — OpenAI / vLLM / LiteLLM
        {"embeddings": [[...], ...]}             — Ollama native

    To add a Python-native provider instead of an HTTP endpoint:
        1. Define a function: (list[str]) -> list[list[float]]
        2. Add it to _PROVIDERS under a new key string
        3. Set EMBEDDING_PROVIDER=<that key> in .env
    """
    import json
    import urllib.request

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.embedding_custom_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_custom_api_key}"
    headers.update(json.loads(settings.embedding_custom_headers))

    body = json.dumps({"model": settings.embedding_model, "input": texts}).encode()
    req = urllib.request.Request(settings.embedding_custom_url, data=body, headers=headers)
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    if "data" in result:
        return [item["embedding"] for item in result["data"]]
    if "embeddings" in result:
        return result["embeddings"]
    raise ValueError(
        f"Unrecognised embedding response shape from custom endpoint: {list(result.keys())}. "
        "Expected 'data' (OpenAI-style) or 'embeddings' (Ollama-style)."
    )


_PROVIDERS = {
    "openai": _embed_openai,
    "azure": _embed_azure,
    "sentence-transformers": _embed_sentence_transformers,
    "custom": _embed_custom,
}


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text."""
    provider_fn = _PROVIDERS.get(settings.embedding_provider)
    if provider_fn is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            f"Choose from: {list(_PROVIDERS)}"
        )

    batch_size = settings.embed_batch_size
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug("Embedding batch %d-%d / %d", i, i + len(batch), len(texts))
        all_embeddings.extend(provider_fn(batch))

    return all_embeddings


def embed_chunks(chunks: list["Chunk"]) -> list[list[float]]:
    """Embed the ``content`` field of each chunk and return the vectors."""
    texts = [c.content for c in chunks]
    return embed_texts(texts)
