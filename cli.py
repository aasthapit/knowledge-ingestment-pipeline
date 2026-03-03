"""
cli.py
Command-line interface for the knowledge ingestion pipeline.

Usage examples:
    python cli.py ingest file docs/setup.md --tags python --tags redis
    python cli.py ingest dir ./docs --tags internal
    python cli.py query "How do I configure Redis?"
    python cli.py retag CHUNK_ID1 CHUNK_ID2 --add devops --remove draft
    python cli.py index create
    python cli.py index drop
"""
from __future__ import annotations

import json
import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Knowledge Ingestment Pipeline — vectorise markdown → Redis."""


# ---------------------------------------------------------------------------
# ingest sub-group
# ---------------------------------------------------------------------------

@cli.group("ingest")
def ingest_group() -> None:
    """Ingest markdown documents into the pipeline."""


@ingest_group.command("file")
@click.argument("path", type=click.Path(exists=True))
@click.option("--tags", "-t", multiple=True, help="Tags to attach to all chunks.")
@click.option("--no-jsonl", is_flag=True, default=False, help="Skip JSONL export.")
@click.option("--no-redis", is_flag=True, default=False, help="Skip Redis upsert.")
@click.option(
    "--output", "-o", default=None, type=click.Path(), help="Custom JSONL output path."
)
def ingest_file_cmd(
    path: str,
    tags: tuple[str, ...],
    no_jsonl: bool,
    no_redis: bool,
    output: str | None,
) -> None:
    """Ingest a single markdown FILE."""
    from pipeline.ingest import ingest_file

    chunks = ingest_file(
        path=path,
        tags=list(tags),
        export_jsonl=not no_jsonl,
        skip_redis=no_redis,
        jsonl_path=output,
    )
    click.echo(f"Done. {len(chunks)} chunk(s) ingested from {path}.")


@ingest_group.command("dir")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--glob", default="**/*.md", show_default=True, help="File glob pattern.")
@click.option("--tags", "-t", multiple=True, help="Tags to attach to all chunks.")
@click.option("--no-jsonl", is_flag=True, default=False, help="Skip JSONL export.")
@click.option("--no-redis", is_flag=True, default=False, help="Skip Redis upsert.")
def ingest_dir_cmd(
    directory: str,
    glob: str,
    tags: tuple[str, ...],
    no_jsonl: bool,
    no_redis: bool,
) -> None:
    """Ingest all markdown files in DIRECTORY."""
    from pipeline.ingest import ingest_directory

    chunks = ingest_directory(
        directory=directory,
        glob=glob,
        tags=list(tags),
        export_jsonl=not no_jsonl,
        skip_redis=no_redis,
    )
    click.echo(f"Done. {len(chunks)} total chunk(s) ingested from {directory}.")


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------

@cli.command("query")
@click.argument("question")
@click.option("--top-k", "-k", default=5, show_default=True, help="Number of results.")
@click.option(
    "--tag-filter",
    default=None,
    help='RediSearch tag filter, e.g. "@tags:{python|redis}".',
)
@click.option("--json-out", is_flag=True, default=False, help="Output raw JSON.")
def query_cmd(question: str, top_k: int, tag_filter: str | None, json_out: bool) -> None:
    """Semantic search: embed QUESTION and return top matching chunks."""
    from pipeline.ingest import query

    results = query(question, top_k=top_k, tag_filter=tag_filter)

    if json_out:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n{'─' * 60}")
        click.echo(f"[{i}] {r['title']} › {r['section']}")
        click.echo(f"    Source : {r['source']}")
        click.echo(f"    Tags   : {r['tags']}")
        click.echo(f"    Score  : {r['score']:.4f}")
        click.echo(f"\n{r['content'][:400]}{'...' if len(r['content']) > 400 else ''}")


# ---------------------------------------------------------------------------
# retag command
# ---------------------------------------------------------------------------

@cli.command("retag")
@click.argument("chunk_ids", nargs=-1, required=True)
@click.option("--add",    "-a", multiple=True, help="Tag(s) to add.")
@click.option("--remove", "-r", multiple=True, help="Tag(s) to remove.")
def retag_cmd(
    chunk_ids: tuple[str, ...],
    add: tuple[str, ...],
    remove: tuple[str, ...],
) -> None:
    """Add/remove tags on existing Redis chunks by CHUNK_ID."""
    from pipeline.tagger import retag_in_redis

    if not add and not remove:
        raise click.UsageError("Provide at least --add or --remove.")

    retag_in_redis(
        chunk_ids=list(chunk_ids),
        add_tags=list(add) or None,
        remove_tags_list=list(remove) or None,
    )
    click.echo(f"Tags updated for {len(chunk_ids)} chunk(s).")


# ---------------------------------------------------------------------------
# index management
# ---------------------------------------------------------------------------

@cli.group("index")
def index_group() -> None:
    """Manage the Redis vector index."""


@index_group.command("create")
def index_create_cmd() -> None:
    """Create the RediSearch vector index (no-op if already exists)."""
    from pipeline import redis_store

    redis_store.create_index()
    click.echo("Index ready.")


@index_group.command("drop")
@click.option(
    "--delete-docs",
    is_flag=True,
    default=False,
    help="Also delete all indexed documents.",
)
def index_drop_cmd(delete_docs: bool) -> None:
    """Drop the RediSearch index."""
    from pipeline import redis_store

    redis_store.drop_index(delete_docs=delete_docs)
    click.echo("Index dropped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
