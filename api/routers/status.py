from fastapi import APIRouter
from api.models import StatusResponse, StatsResponse, ServiceStatus
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    indexed_chunks = 0
    pending_review = 0
    approved = 0
    recent_pushes = []

    try:
        from pipeline.redis_store import get_client
        client = get_client()
        info = client.ft("knowledge_index").info()
        indexed_chunks = int(info.get("num_docs", 0))
    except Exception:
        pass

    try:
        from pipeline.mongo_store import get_staging
        staging = get_staging()
        docs = staging.list_all()
        pending_review = sum(1 for d in docs if d.get("status") == "pending_review")
        approved = sum(1 for d in docs if d.get("status") == "approved")
    except Exception:
        pass

    pushed_today = 0
    try:
        from pipeline.mongo_store import get_ledger
        from datetime import datetime, timezone
        ledger = get_ledger()
        snapshots = ledger.list_snapshots()
        today = datetime.now(timezone.utc).date()
        for s in snapshots:
            pushed_at = s.get("pushed_at", "")
            if pushed_at and str(pushed_at)[:10] == str(today):
                pushed_today += s.get("docs_pushed", 0)
        recent_pushes = snapshots[:5]
    except Exception:
        pass

    return StatsResponse(
        indexed_chunks=indexed_chunks,
        pending_review=pending_review,
        approved=approved,
        pushed_today=pushed_today,
        recent_pushes=recent_pushes,
    )


@router.get("/status", response_model=StatusResponse)
def get_status():
    from pipeline.config import settings

    # Redis
    redis_status = ServiceStatus(ok=False, url=settings.REDIS_URL)
    try:
        from pipeline.redis_store import get_client
        get_client().ping()
        redis_status.ok = True
    except Exception as e:
        redis_status.detail = str(e)

    # MongoDB
    mongo_uri = settings.MONGODB_URI or f"{settings.MONGODB_HOST}:{settings.MONGODB_PORT}"
    mongo_status = ServiceStatus(ok=False, url=mongo_uri)
    try:
        from pipeline.mongo_store import get_staging
        get_staging().list_all()
        mongo_status.ok = True
    except Exception as e:
        mongo_status.detail = str(e)

    # Embeddings
    embed_status = ServiceStatus(ok=False, url=settings.EMBEDDING_PROVIDER)
    try:
        from pipeline.embedder import embed_texts
        embed_texts(["ping"])
        embed_status.ok = True
    except Exception as e:
        embed_status.detail = str(e)

    config = {
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "embed_batch_size": settings.EMBED_BATCH_SIZE,
        "chunk_max_chars": settings.CHUNK_MAX_CHARS,
        "chunk_overlap_chars": settings.CHUNK_OVERLAP_CHARS,
        "docling_max_tokens": settings.DOCLING_MAX_TOKENS,
        "redis_index_name": settings.REDIS_INDEX_NAME,
    }

    kb_stats: dict = {}
    try:
        from pipeline.mongo_store import get_staging
        docs = get_staging().list_all()
        kb_stats["total_staged"] = len(docs)
        kb_stats["pending"] = sum(1 for d in docs if d.get("status") == "pending_review")
        kb_stats["approved"] = sum(1 for d in docs if d.get("status") == "approved")
        kb_stats["pushed"] = sum(1 for d in docs if d.get("status") == "pushed")
    except Exception:
        pass

    return StatusResponse(
        redis=redis_status,
        mongodb=mongo_status,
        embeddings=embed_status,
        config=config,
        kb_stats=kb_stats,
    )
