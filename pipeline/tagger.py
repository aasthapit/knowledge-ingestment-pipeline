"""
tagger.py
Utilities for managing tags on chunks — both in-memory (before indexing) and
post-hoc (on documents already stored in Redis).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory tagging (before indexing)
# ---------------------------------------------------------------------------

def apply_tags(chunks: list["Chunk"], tags: list[str]) -> list["Chunk"]:
    """
    Merge *tags* into every chunk's tag list (deduplicating, order-preserving).
    Mutates each chunk in-place and returns the same list.
    """
    for chunk in chunks:
        merged = list(dict.fromkeys(chunk.tags + tags))
        chunk.tags = merged
    return chunks


def remove_tags(chunks: list["Chunk"], tags: list[str]) -> list["Chunk"]:
    """Remove each tag in *tags* from every chunk. Mutates in-place."""
    tag_set = set(tags)
    for chunk in chunks:
        chunk.tags = [t for t in chunk.tags if t not in tag_set]
    return chunks


def filter_chunks_by_tag(
    chunks: list["Chunk"], required_tags: list[str], match_all: bool = False
) -> list["Chunk"]:
    """
    Return chunks that carry *any* (or *all* when match_all=True) of
    the required tags.
    """
    required = set(required_tags)
    result = []
    for chunk in chunks:
        present = set(chunk.tags) & required
        if match_all and present == required:
            result.append(chunk)
        elif not match_all and present:
            result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# Post-hoc tagging (documents already in Redis)
# ---------------------------------------------------------------------------

def retag_in_redis(
    chunk_ids: list[str],
    add_tags: list[str] | None = None,
    remove_tags_list: list[str] | None = None,
) -> None:
    """
    For each chunk_id, fetch the current tags from Redis, apply additions /
    removals, and write back.
    """
    from pipeline import redis_store

    client = redis_store.get_client()
    for cid in chunk_ids:
        doc = redis_store.get_chunk(cid, client=client)
        if doc is None:
            logger.warning("Chunk '%s' not found in Redis; skipping.", cid)
            continue
        current: list[str] = doc.get("tags", [])
        updated = list(dict.fromkeys(current + (add_tags or [])))
        if remove_tags_list:
            rm = set(remove_tags_list)
            updated = [t for t in updated if t not in rm]
        redis_store.update_tags(cid, updated, client=client)
        logger.debug("chunk %s tags → %s", cid, updated)

    logger.info("Re-tagged %d chunks in Redis.", len(chunk_ids))
