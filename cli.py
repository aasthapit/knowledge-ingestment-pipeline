"""
cli.py
Command-line interface for the knowledge ingestion pipeline.

Usage examples
--------------
# Ingest any document (PDF, DOCX, HTML, URL, Markdown) via Docling:
    python cli.py ingest doc report.pdf --tags finance --auto-push
    python cli.py ingest doc https://docs.example.com/guide

# Legacy Markdown ingestion (direct to Redis, no staging):
    python cli.py ingest file docs/setup.md --tags python --tags redis
    python cli.py ingest dir ./docs --tags internal

# Review queue (for quality-flagged documents):
    python cli.py review list
    python cli.py review show <doc_id>
    python cli.py review approve <doc_id>
    python cli.py review reject  <doc_id> --reason "duplicate"
    python cli.py review push
    python cli.py review push --doc-id <doc_id>

# Semantic search:
    python cli.py query "How do I configure Redis?"

# Tag management:
    python cli.py retag CHUNK_ID1 CHUNK_ID2 --add devops --remove draft

# Index management:
    python cli.py index create
    python cli.py index drop
"""
from __future__ import annotations

import json
import logging
import sys

import click
from rich.console import Console
from rich.table import Table
from rich import box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_console = Console()


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Knowledge Ingestment Pipeline — convert, review, and vectorise documents."""


# ---------------------------------------------------------------------------
# ingest sub-group
# ---------------------------------------------------------------------------

@cli.group("ingest")
def ingest_group() -> None:
    """Ingest documents into the pipeline."""


@ingest_group.command("jsonl")
@click.argument("path", type=click.Path(exists=True))
@click.option("--tags", "-t", multiple=True, help="Extra tags applied to every chunk.")
@click.option("--name", "-n", default=None, help="Batch name (defaults to filename).")
def ingest_jsonl_cmd(path: str, tags: tuple[str, ...], name: str | None) -> None:
    """
    Bulk-import a JSONL chunk file into the staging area.

    Supports both the crawler schema (text + page_url) and the pipeline
    export schema (content + source).  Pre-computed embeddings are reused
    automatically when present.

    After import, run 'review push' to embed (if needed) and push to the
    vector store.
    """
    from pipeline.ingest import ingest_jsonl

    total_seen = 0

    def _progress(done: int, total: int) -> None:
        nonlocal total_seen
        if done != total_seen:
            total_seen = done
            click.echo(f"  Parsed {done:,} chunks…", nl=False)
            click.echo("\r", nl=False)

    result = ingest_jsonl(
        source=path,
        batch_name=name,
        extra_tags=list(tags),
        progress_cb=_progress,
    )

    click.echo()
    click.echo(
        f"Done. {result['total_chunks']:,} chunks from {result['unique_sources']:,} "
        f"source(s) staged ({result['schema']} schema)."
    )
    embed_note = (
        "Pre-computed embeddings reused — run 'review push' to index."
        if result["has_embeddings"] else
        "No embeddings in file — run 'review push' to embed and index."
    )
    click.echo(embed_note)
    click.echo(f"Batch ID: {result['doc_id']}")


@ingest_group.command("doc")
@click.argument("source")
@click.option("--tags", "-t", multiple=True, help="Extra tags to attach to all chunks.")
@click.option(
    "--auto-push",
    is_flag=True,
    default=False,
    help="Immediately embed and push if quality passes (skip manual review step).",
)
def ingest_doc_cmd(
    source: str,
    tags: tuple[str, ...],
    auto_push: bool,
) -> None:
    """
    Ingest any document — PDF, DOCX, PPTX, HTML, URL, or Markdown.

    SOURCE can be a local file path or an HTTP/HTTPS URL.
    Uses Docling for conversion and quality-based auto-staging.
    """
    from pipeline.ingest import ingest_document

    result = ingest_document(
        source=source,
        extra_tags=list(tags),
        auto_push=auto_push,
    )

    _console.print()
    if result["quality_passed"]:
        _console.print(f"[bold green]Quality PASS[/] (score={result['quality_score']:.2f})")
        if auto_push:
            _console.print(f"[green]Pushed {result['chunk_count']} chunk(s) to vector store.[/]")
        else:
            _console.print(
                f"[green]{result['chunk_count']} chunk(s) staged and auto-approved.[/]\n"
                "Run [bold]python cli.py review push[/] to push to vector store."
            )
    else:
        _console.print(f"[bold yellow]Quality REVIEW[/] (score={result['quality_score']:.2f})")
        _console.print(f"[yellow]{result['chunk_count']} chunk(s) staged for human review.[/]")
        if result["flags"]:
            _console.print("[dim]Issues:[/]")
            for flag in result["flags"]:
                _console.print(f"  [dim]• {flag}[/]")
        _console.print(
            f"\nRun [bold]python cli.py review show {result['doc_id']}[/] to inspect."
        )

    _console.print(f"\n  doc_id : [bold]{result['doc_id']}[/]")
    _console.print(f"  title  : {result['title']}")
    _console.print(f"  tags   : {', '.join(result['tags']) or '(none)'}")


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
# review sub-group — manage the staging/review queue
# ---------------------------------------------------------------------------

@cli.group("review")
def review_group() -> None:
    """Inspect, approve, reject, and push staged documents."""


@review_group.command("list")
@click.option("--pending-only", is_flag=True, default=False, help="Show only documents awaiting review.")
def review_list_cmd(pending_only: bool) -> None:
    """List all staged documents and their quality status."""
    from pipeline.review import list_all_docs, list_pending_docs

    docs = list_pending_docs() if pending_only else list_all_docs()

    if not docs:
        _console.print("[dim]No staged documents found.[/]")
        return

    table = Table(box=box.ROUNDED, show_lines=False, header_style="bold cyan")
    table.add_column("Status",   style="bold",  width=14)
    table.add_column("Score",    justify="right", width=6)
    table.add_column("Title",    no_wrap=False,  max_width=40)
    table.add_column("Type",     width=8)
    table.add_column("Chunks",   justify="right", width=7)
    table.add_column("Doc ID",   width=36)

    STATUS_STYLE = {
        "approved":       "[green]approved[/]",
        "pending_review": "[yellow]needs review[/]",
        "rejected":       "[red]rejected[/]",
    }

    for d in sorted(docs, key=lambda x: x.get("status", "")):
        status = d.get("status", "?")
        score  = d.get("quality_score", "?")
        try:
            score_str = f"{float(score):.2f}"
        except (ValueError, TypeError):
            score_str = str(score)

        flags = d.get("quality_flags", [])
        flag_hint = f" ({len(flags)} flag{'s' if len(flags) != 1 else ''})" if flags else ""

        table.add_row(
            STATUS_STYLE.get(status, status),
            score_str,
            d.get("title", "?") + flag_hint,
            d.get("source_type", "?"),
            str(d.get("chunk_count", "?")),
            d.get("doc_id", "?"),
        )

    _console.print(table)
    _console.print(f"[dim]{len(docs)} document(s) total[/]")


@review_group.command("show")
@click.argument("doc_id")
def review_show_cmd(doc_id: str) -> None:
    """Show full details and sample chunks for a staged document."""
    from pipeline.review import get_doc_detail

    detail = get_doc_detail(doc_id)
    if not detail:
        _console.print(f"[red]Doc ID not found:[/] {doc_id}")
        sys.exit(1)

    status = detail.get("status", "?")
    STATUS_COLOUR = {"approved": "green", "pending_review": "yellow", "rejected": "red"}
    colour = STATUS_COLOUR.get(status, "white")

    _console.rule(f"[bold]{detail.get('title', 'Unknown')}[/]")
    _console.print(f"  [bold]Doc ID       :[/] {doc_id}")
    _console.print(f"  [bold]Status       :[/] [{colour}]{status}[/{colour}]")
    _console.print(f"  [bold]Source       :[/] {detail.get('source_path', '?')}")
    _console.print(f"  [bold]Type         :[/] {detail.get('source_type', '?')}")
    _console.print(f"  [bold]Author       :[/] {detail.get('author') or '—'}")
    _console.print(f"  [bold]Pages        :[/] {detail.get('page_count') or '—'}")
    _console.print(f"  [bold]Chunks       :[/] {detail.get('chunk_count', 0)}")
    _console.print(f"  [bold]Quality score:[/] {detail.get('quality_score', '?')}")

    flags = detail.get("quality_flags", [])
    if flags:
        _console.print("\n  [bold yellow]Quality flags:[/]")
        for f in flags:
            _console.print(f"    • {f}")

    samples = detail.get("sample_chunks", [])
    if samples:
        _console.print(f"\n  [bold]Sample chunks ({len(samples)} of {detail['chunk_count']}):[/]")
        for i, ch in enumerate(samples, 1):
            section = ch.get("section", "?")
            content = ch.get("content", "")[:300]
            tags = ", ".join(ch.get("tags", [])) or "(none)"
            _console.print(f"\n  [bold cyan][{i}][/] {section}")
            _console.print(f"      tags: {tags}")
            _console.print(f"      {content}{'…' if len(ch.get('content','')) > 300 else ''}")

            cit = (ch.get("metadata") or {}).get("citation", {})
            if cit.get("page_number"):
                _console.print(f"      [dim]page {cit['page_number']} of {cit.get('page_count', '?')}[/]")


@review_group.command("approve")
@click.argument("doc_id")
def review_approve_cmd(doc_id: str) -> None:
    """Approve a staged document (mark it ready to push)."""
    from pipeline.review import approve_doc

    if approve_doc(doc_id):
        _console.print(f"[green]Approved[/] {doc_id}.")
        _console.print("Run [bold]python cli.py review push[/] to push to vector store.")
    else:
        _console.print(f"[red]Not found:[/] {doc_id}")
        sys.exit(1)


@review_group.command("reject")
@click.argument("doc_id")
@click.option("--reason", "-r", default="", help="Optional reason for rejection.")
def review_reject_cmd(doc_id: str, reason: str) -> None:
    """Reject a staged document (it will not be pushed to the vector store)."""
    from pipeline.review import reject_doc

    if reject_doc(doc_id, reason=reason):
        _console.print(f"[red]Rejected[/] {doc_id}." + (f" Reason: {reason}" if reason else ""))
    else:
        _console.print(f"[red]Not found:[/] {doc_id}")
        sys.exit(1)


@review_group.command("push")
@click.option("--doc-id", default=None, help="Push only this specific document ID.")
@click.option(
    "--remove-staging",
    is_flag=True,
    default=False,
    help="Delete staging docs/chunks after a successful push (default: keep for audit).",
)
def review_push_cmd(doc_id: str | None, remove_staging: bool) -> None:
    """
    Embed all approved documents and push them to the vector store.

    Operates on ALL approved documents unless --doc-id is given.
    Staging data is retained by default for audit and JSONL export.
    """
    from pipeline.review import push_approved

    _console.print("Embedding and pushing approved documents …")
    result = push_approved(doc_id=doc_id, remove_after_push=remove_staging)

    if result["errors"]:
        for err in result["errors"]:
            _console.print(f"[red]Error:[/] {err}")

    _console.print(
        f"[bold green]Done.[/] "
        f"{result['pushed_docs']} doc(s), "
        f"{result['pushed_chunks']} chunk(s) pushed to Redis."
    )


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
