"""
exporter.py
Serialises chunks (with or without embeddings) to JSONL files, and exports
the KB ledger to CSV.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
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


_LEDGER_COLUMNS = [
    "title", "source_path", "source_type", "kb_name",
    "chunk_count", "quality_score", "tags", "pushed_at", "drift_status",
]


def export_ledger_csv(
    records: list[dict],
    output_path: str | Path | None = None,
) -> Path:
    """
    Write KB ledger records to a CSV snapshot file.

    Parameters
    ----------
    records:      List of ledger record dicts (from KBLedger.list_docs).
    output_path:  Explicit file path.  If *None*, uses
                  ``settings.ledger_output_dir``; raises ValueError if
                  that is also unset.

    Returns
    -------
    Path of the written file.
    """
    if output_path is None:
        if settings.ledger_output_dir is None:
            raise ValueError(
                "LEDGER_OUTPUT_DIR is not configured. "
                "Set it in .env or pass output_path explicitly."
            )
        settings.ledger_output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        output_path = settings.ledger_output_dir / f"ledger_{ts}.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LEDGER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in _LEDGER_COLUMNS}
            tags = row.get("tags", "")
            if isinstance(tags, list):
                row["tags"] = "; ".join(tags)
            pushed_at = row.get("pushed_at", "")
            if isinstance(pushed_at, str) and len(pushed_at) > 19:
                row["pushed_at"] = pushed_at[:19]
            writer.writerow(row)

    logger.info("Exported ledger (%d docs) → %s", len(records), output_path)
    return output_path


def export_chunks_as_jsonl(
    chunks: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    """
    Write a list of already-serialised chunk dicts to a pipeline-schema JSONL file.

    Used for usecase exports where chunks are fetched from MongoDB directly
    rather than from Chunk objects.  Each dict is written as-is.

    Parameters
    ----------
    chunks:      List of chunk dicts (as returned by MongoStagingStore.get_chunks_by_usecase).
    output_path: Explicit file path.  If *None*, auto-generated inside JSONL_OUTPUT_DIR.

    Returns
    -------
    Path of the written file.
    """
    if output_path is None:
        settings.jsonl_output_dir.mkdir(parents=True, exist_ok=True)
        import time
        ts = int(time.time())
        output_path = settings.jsonl_output_dir / f"usecase_export_{ts}.jsonl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False, default=str) + "\n")

    logger.info("Exported %d chunk dicts → %s", len(chunks), output_path)
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
