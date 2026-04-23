from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.models import UpsertConfluenceSourceRequest, MessageResponse
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


def _ledger():
    from pipeline.mongo_store import get_usecase_ledger
    return get_usecase_ledger()


@router.get("/")
def list_usecases():
    try:
        usecases = _ledger().get_distinct_usecases()
        return {"usecases": usecases}
    except Exception:
        return {"usecases": []}


@router.get("/{usecase_id}/agents")
def list_agents(usecase_id: str):
    agents = _ledger().get_agent_filters_for_usecase(usecase_id)
    return {"agents": agents}


@router.get("/{usecase_id}/{agent_filter}/stats")
def get_stats(usecase_id: str, agent_filter: str):
    try:
        chunk_ids = _ledger().get_chunk_ids(usecase_id, agent_filter)
        entries = _ledger().list_entries(usecase_id=usecase_id, agent_filter=agent_filter)
        entry = entries[0] if entries else {}
        return {
            "chunk_count": len(chunk_ids),
            "doc_count": len(entry.get("doc_ids", [])),
            "last_pushed_at": entry.get("last_pushed_at"),
            "kb_name": entry.get("kb_name", "default"),
        }
    except Exception:
        return {"chunk_count": 0, "doc_count": 0, "last_pushed_at": None}


@router.get("/{usecase_id}/{agent_filter}/docs")
def get_docs(usecase_id: str, agent_filter: str):
    from pipeline.mongo_store import get_ledger
    try:
        chunk_ids = set(_ledger().get_chunk_ids(usecase_id, agent_filter))
        all_docs = get_ledger().list_docs()
        docs = [
            d for d in all_docs
            if d.get("usecase_id") == usecase_id and d.get("agent_filter") == agent_filter
        ]
        return {"docs": docs}
    except Exception:
        return {"docs": []}


@router.get("/{usecase_id}/{agent_filter}/sources")
def get_sources(usecase_id: str, agent_filter: str):
    sources = _ledger().list_confluence_sources(
        usecase_id=usecase_id, agent_filter=agent_filter
    )
    return {"sources": sources}


@router.post("/{usecase_id}/{agent_filter}/sources", response_model=MessageResponse)
def upsert_source(usecase_id: str, agent_filter: str, req: UpsertConfluenceSourceRequest):
    _ledger().upsert_confluence_source(
        usecase_id=usecase_id,
        agent_filter=agent_filter,
        page_urls=req.page_urls,
        refresh_cron=req.refresh_cron,
        max_depth=req.max_depth,
        kb_name=req.kb_name,
        extra_tags=req.extra_tags,
    )
    return MessageResponse(message="Source registered")


@router.post("/{usecase_id}/{agent_filter}/sources/{source_id}/refresh")
def refresh_source(usecase_id: str, agent_filter: str, source_id: str):
    def event_stream():
        try:
            from pipeline.refresh_scheduler import trigger_refresh_now

            messages = []

            def on_step(msg: str):
                messages.append(msg)

            yield f"data: {json.dumps({'type': 'progress', 'message': 'Starting refresh...'})}\n\n"
            trigger_refresh_now(
                usecase_id=usecase_id,
                agent_filter=agent_filter,
                on_step=on_step,
            )
            yield f"data: {json.dumps({'type': 'done', 'result': {'messages': messages}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{usecase_id}/{agent_filter}/export")
def export_jsonl(usecase_id: str, agent_filter: str, status: str = "pushed"):
    from fastapi.responses import Response
    import io
    from pipeline.mongo_store import get_staging, get_usecase_ledger
    from pipeline.exporter import export_chunks_as_jsonl

    try:
        ledger = get_usecase_ledger()
        entries = ledger.list_entries(usecase_id=usecase_id, agent_filter=agent_filter)
        if not entries:
            return Response(content="", media_type="application/x-ndjson")

        doc_ids = entries[0].get("doc_ids", [])
        staging = get_staging()
        chunks = []
        for doc_id in doc_ids:
            doc_chunks = staging.get_chunks(doc_id)
            chunks.extend(doc_chunks)

        buf = io.StringIO()
        for chunk in chunks:
            buf.write(json.dumps(chunk) + "\n")

        filename = f"export_{usecase_id}_{agent_filter}.jsonl"
        return Response(
            content=buf.getvalue(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
