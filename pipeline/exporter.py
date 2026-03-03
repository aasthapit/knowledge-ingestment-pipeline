"""
exporter.py
Serialises chunks (with or without embeddings) to JSONL files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.config import settings

if TYPE_CHECKING:
    from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


def export_jsonl(
    chunks: list["Chunk"],
    embeddings: list[list[float]] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Write chunks to a JSONL file.

    Parameters
    ----------
    chunks:      List of :class:`~pipeline.chunker.Chunk` objects.
    embeddings:  Optional list of embedding vectors (same order as chunks).
                 When provided they are included in each record under the
                 key ``"embedding"``.
    output_path: Explicit file path.  If *None*, a file is auto-generated
                 inside ``settings.jsonl_output_dir``.

    Returns
    -------
    Path of the written file.
    """
    if output_path is None:
        settings.jsonl_output_dir.mkdir(parents=True, exist_ok=True)
        import time
        ts = int(time.time())
        output_path = settings.jsonl_output_dir / f"chunks_{ts}.jsonl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for i, chunk in enumerate(chunks):
            record: dict[str, Any] = chunk.to_dict()
            if embeddings is not None:
                record["embedding"] = embeddings[i]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Exported %d chunks → %s", len(chunks), output_path)
    return output_path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts."""
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
