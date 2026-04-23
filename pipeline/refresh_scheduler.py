"""
refresh_scheduler.py
Background scheduler for periodic Confluence re-crawl and re-push.

A single APScheduler BackgroundScheduler polls MongoDB every 5 minutes for
Confluence source configs whose next_refresh_at is due, then re-crawls and
re-pushes them. This avoids dynamic per-source APScheduler jobs and persistent
job stores; the schedule state lives entirely in usecase_confluence_sources.

The scheduler is started once from app.py via start_scheduler(). Streamlit
reruns app.py on every page navigation, so start_scheduler() is idempotent.
"""
from __future__ import annotations

import io
import json
import logging
import threading

logger = logging.getLogger(__name__)

_scheduler = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Core refresh logic
# ---------------------------------------------------------------------------

def _do_confluence_refresh(
    usecase_id: str,
    agent_filter: str,
    kb_name: str,
    page_urls: list[str],
    max_depth: int,
    extra_tags: list[str],
    source_id: str | None = None,
    on_step: "Callable[[str], None] | None" = None,
) -> None:
    """
    Re-crawl registered Confluence pages for a usecase+agent pair, stage the
    content, and immediately push to the vector DB.

    Uses incremental refresh when a prior crawl snapshot exists: only pages
    whose version has changed (or that are new) are fully fetched.

    Reuses ConfluenceCrawler, ingest_jsonl, and push_approved — no new
    abstractions. The content is auto-pushed without human review because it
    originates from pages that were previously registered and reviewed.
    """
    from typing import Callable

    from pipeline.config import settings
    from pipeline.confluence import ConfluenceCrawler
    from pipeline.ingest import ingest_jsonl
    from pipeline.review import push_approved

    def _step(msg: str) -> None:
        logger.info(msg)
        if on_step:
            on_step(msg)

    if not settings.confluence_base_url:
        raise RuntimeError("CONFLUENCE_BASE_URL is not configured.")

    _step(
        f"Embedding: {settings.embedding_provider} / {settings.embedding_model} "
        f"({settings.embedding_dimensions}d, batch {settings.embed_batch_size})"
    )

    crawler = ConfluenceCrawler(
        base_url=settings.confluence_base_url,
        auth_type=settings.confluence_auth_type,
        email=settings.confluence_email,
        api_token=settings.confluence_api_token,
        verify_ssl=settings.confluence_verify_ssl,
    )

    # Load prior snapshot for incremental diff.
    old_snapshot: dict[str, int] = {}
    if source_id:
        try:
            from pipeline.mongo_store import get_usecase_ledger
            snap = get_usecase_ledger().get_crawl_snapshot(source_id)
            old_snapshot = {p["page_id"]: p["version"] for p in snap}
            if old_snapshot:
                _step(f"Prior snapshot: {len(old_snapshot)} page(s) — will skip unchanged.")
        except Exception as exc:
            logger.warning("Could not load crawl snapshot: %s", exc)

    all_pages = []
    full_metadata: list[dict] = []   # current page list across all root URLs

    for page_url in page_urls:
        try:
            _step(f"Discovering page tree from {page_url} …")
            meta = crawler.crawl_metadata(page_url, max_depth=max_depth)
            full_metadata.extend(meta)

            if old_snapshot:
                # Incremental: only fetch pages that are new or version-bumped.
                changed = [
                    m["page_id"] for m in meta
                    if old_snapshot.get(m["page_id"]) != m["version"]
                ]
                skipped = len(meta) - len(changed)
                if skipped:
                    _step(f"Skipping {skipped} unchanged page(s).")
                if not changed:
                    _step("No changes detected for this URL — skipping full fetch.")
                    continue
                _step(f"Fetching {len(changed)} changed/new page(s) …")
                pages = crawler.fetch_pages_by_ids(changed, extra_tags=extra_tags)
            else:
                # First run: full crawl.
                _step(f"First crawl — fetching all {len(meta)} page(s) …")
                pages = crawler.fetch_pages_by_ids(
                    [m["page_id"] for m in meta], extra_tags=extra_tags
                )

            all_pages.extend(pages)
        except Exception as exc:
            logger.warning("Could not crawl %s: %s", page_url, exc)

    # Update snapshot with current metadata (all pages, not just changed ones).
    if source_id and full_metadata:
        try:
            from pipeline.mongo_store import get_usecase_ledger
            snapshot = [
                {
                    "page_id":       m["page_id"],
                    "title":         m["title"],
                    "version":       m["version"],
                    "last_modified": m["last_modified"],
                }
                for m in full_metadata
            ]
            get_usecase_ledger().record_crawl_snapshot(source_id, snapshot)
        except Exception as exc:
            logger.warning("Could not store page snapshot: %s", exc)

    if not all_pages:
        _step("No new or changed pages to ingest.")
        logger.warning(
            "No pages retrieved for usecase=%s agent=%s", usecase_id, agent_filter
        )
        return

    _step(f"Staging {len(all_pages)} page(s) to MongoDB …")

    # Convert pages to JSONL bytes using the existing pipeline-schema format.
    jsonl_lines = [
        json.dumps(crawler.to_record(p), ensure_ascii=False)
        for p in all_pages
    ]
    buf = io.BytesIO(("\n".join(jsonl_lines) + "\n").encode("utf-8"))
    buf.name = f"confluence_refresh_{usecase_id}_{agent_filter}.jsonl"

    result = ingest_jsonl(
        source=buf,
        batch_name=f"confluence_refresh_{usecase_id}_{agent_filter}",
        extra_tags=extra_tags,
        kb_name=kb_name,
        usecase_id=usecase_id,
        agent_filter=agent_filter,
    )

    _step(
        f"Embedding {result['total_chunks']} chunk(s) → "
        f"pushing to Redis index '{settings.redis_index_name}' …"
    )
    push_result = push_approved(doc_id=result["doc_id"], remove_after_push=False)
    _step(
        f"Done — {push_result.get('pushed_chunks', 0)} chunk(s) upserted into Redis."
    )
    logger.info(
        "Refresh complete for usecase=%s agent=%s: %d chunks pushed (%s)",
        usecase_id, agent_filter, result["total_chunks"], push_result,
    )


def _run_due_refreshes() -> None:
    """
    Poll MongoDB for source configs that are due for refresh and process each.
    Called by APScheduler every 5 minutes.
    """
    try:
        from pipeline.mongo_store import get_usecase_ledger
        uc_ledger = get_usecase_ledger()
        due = uc_ledger.get_sources_due_for_refresh()
    except Exception as exc:
        logger.error("Could not fetch due refreshes: %s", exc)
        return

    for source in due:
        source_id    = source["source_id"]
        usecase_id   = source["usecase_id"]
        agent_filter = source["agent_filter"]
        kb_name      = source.get("kb_name", "default")
        page_urls    = source.get("page_urls") or []
        max_depth    = source.get("max_depth", -1)
        extra_tags   = source.get("extra_tags") or []
        cron_expr    = source.get("refresh_cron")

        logger.info(
            "Starting scheduled Confluence refresh for usecase=%s agent=%s",
            usecase_id, agent_filter,
        )
        uc_ledger.mark_refresh_running(source_id)

        try:
            _do_confluence_refresh(
                usecase_id=usecase_id,
                agent_filter=agent_filter,
                kb_name=kb_name,
                page_urls=page_urls,
                max_depth=max_depth,
                extra_tags=extra_tags,
                source_id=source_id,
            )
            uc_ledger.mark_refresh_done(source_id)
            if cron_expr:
                uc_ledger.update_next_refresh(source_id, cron_expr)
        except Exception as exc:
            logger.error(
                "Refresh failed for usecase=%s agent=%s: %s",
                usecase_id, agent_filter, exc,
            )
            uc_ledger.mark_refresh_failed(source_id, str(exc))


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------

def start_scheduler() -> None:
    """
    Start the APScheduler BackgroundScheduler (idempotent).

    The poll job fires every 5 minutes. Because Streamlit reruns app.py on
    every page navigation, this function is guarded by a threading.Lock so
    only one scheduler instance is ever created per process.
    """
    global _scheduler
    with _lock:
        if _scheduler is not None and _scheduler.running:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            _scheduler = BackgroundScheduler(daemon=True)
            _scheduler.add_job(
                _run_due_refreshes,
                trigger="interval",
                minutes=5,
                id="confluence_refresh_poll",
                replace_existing=True,
            )
            _scheduler.start()
            logger.info("Confluence refresh scheduler started (poll interval: 5 min).")
        except Exception as exc:
            logger.error("Could not start refresh scheduler: %s", exc)


def stop_scheduler() -> None:
    """Gracefully stop the scheduler (e.g. on process exit)."""
    global _scheduler
    with _lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            logger.info("Confluence refresh scheduler stopped.")


def trigger_refresh_now(
    usecase_id: str,
    agent_filter: str,
    on_step: "Callable[[str], None] | None" = None,
) -> None:
    """
    Manually trigger an immediate refresh for a specific usecase+agent pair.
    Runs synchronously in the calling thread (used by the UI 'Refresh now' button).

    on_step: optional callback invoked with a human-readable status string at
             each major step — useful for surfacing progress in the UI.
    """
    from typing import Callable  # noqa: F401 (used in annotation above)

    from pipeline.mongo_store import get_usecase_ledger

    uc_ledger = get_usecase_ledger()
    source = uc_ledger.get_confluence_source(usecase_id, agent_filter)
    if not source:
        raise ValueError(
            f"No Confluence source registered for usecase_id={usecase_id!r}, "
            f"agent_filter={agent_filter!r}."
        )

    source_id  = source["source_id"]
    kb_name    = source.get("kb_name", "default")
    page_urls  = source.get("page_urls") or []
    max_depth  = source.get("max_depth", -1)
    extra_tags = source.get("extra_tags") or []
    cron_expr  = source.get("refresh_cron")

    uc_ledger.mark_refresh_running(source_id)
    try:
        _do_confluence_refresh(
            usecase_id=usecase_id,
            agent_filter=agent_filter,
            kb_name=kb_name,
            page_urls=page_urls,
            max_depth=max_depth,
            extra_tags=extra_tags,
            source_id=source_id,
            on_step=on_step,
        )
        uc_ledger.mark_refresh_done(source_id)
        if cron_expr:
            uc_ledger.update_next_refresh(source_id, cron_expr)
    except Exception as exc:
        uc_ledger.mark_refresh_failed(source_id, str(exc))
        raise
