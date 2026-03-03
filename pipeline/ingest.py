"""
ingest.py
High-level orchestration: chunk → embed → (export JSONL) → store in Redis.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pipeline import chunker, embedder, exporter, redis_store, tagger
from pipeline.config import settings

logger = logging.getLogger(__name__)


def ingest_file(
    path: str | Path,
    tags: list[str] | None = None,
    export_jsonl: bool = True,
    skip_redis: bool = False,
    jsonl_path: str | Path | None = None,
) -> list[chunker.Chunk]:
    """
    Full pipeline for a single markdown file.

    Steps
    -----
    1. Chunk the document by headings (respecting front-matter tags).
    2. Merge any *extra* CLI tags.
    3. Embed all chunks.
    4. Optionally export to JSONL.
    5. Upsert into Redis (unless skip_redis=True).

    Returns the list of :class:`~pipeline.chunker.Chunk` objects.
    """
    settings.validate()
    path = Path(path)
    extra_tags = tags or []

    logger.info("Chunking: %s", path)
    chunks = chunker.chunk_markdown_file(
        path,
        extra_tags=extra_tags,
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap_chars,
    )
    logger.info("  → %d chunks", len(chunks))

    if not chunks:
        logger.warning("No chunks produced from %s", path)
        return chunks

    logger.info("Embedding %d chunks …", len(chunks))
    vectors = embedder.embed_chunks(chunks)

    if export_jsonl:
        out = exporter.export_jsonl(chunks, embeddings=vectors, output_path=jsonl_path)
        logger.info("JSONL saved: %s", out)

    if not skip_redis:
        redis_store.create_index()
        redis_store.upsert_chunks(chunks, vectors)

    return chunks


def ingest_directory(
    directory: str | Path,
    glob: str = "**/*.md",
    tags: list[str] | None = None,
    export_jsonl: bool = True,
    skip_redis: bool = False,
) -> list[chunker.Chunk]:
    """
    Ingest all markdown files matching *glob* under *directory*.
    All chunks are batched into a single JSONL file per run.
    """
    settings.validate()
    directory = Path(directory)
    md_files = sorted(directory.glob(glob))

    if not md_files:
        logger.warning("No files matched '%s' under %s", glob, directory)
        return []

    all_chunks: list[chunker.Chunk] = []
    all_vectors: list[list[float]] = []

    for md_file in md_files:
        logger.info("Processing: %s", md_file)
        file_chunks = chunker.chunk_markdown_file(
            md_file,
            extra_tags=tags or [],
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap_chars,
        )
        if not file_chunks:
            continue
        vectors = embedder.embed_chunks(file_chunks)
        all_chunks.extend(file_chunks)
        all_vectors.extend(vectors)

    logger.info("Total chunks across all files: %d", len(all_chunks))

    if all_chunks:
        if export_jsonl:
            exporter.export_jsonl(all_chunks, embeddings=all_vectors)
        if not skip_redis:
            redis_store.create_index()
            redis_store.upsert_chunks(all_chunks, all_vectors)

    return all_chunks


def query(
    question: str,
    top_k: int = 5,
    tag_filter: str | None = None,
) -> list[dict]:
    """
    Embed *question* and return the top-k most similar chunks from Redis.

    Parameters
    ----------
    question:   Natural language question / search string.
    top_k:      Number of results.
    tag_filter: Optional RediSearch tag filter, e.g. ``"@tags:{python}"``.
    """
    settings.validate()
    vectors = embedder.embed_texts([question])
    results = redis_store.search(vectors[0], top_k=top_k, tag_filter=tag_filter)
    return results
