from fastapi import APIRouter, HTTPException
from api.models import SearchRequest, SearchResult
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


@router.post("/", response_model=list[SearchResult])
def search(req: SearchRequest):
    from pipeline import embedder
    from pipeline.mongo_store import get_vs_config_store
    from pipeline.vector_store import get_vector_store_client

    vectors = embedder.embed_texts([req.query])
    vec = vectors[0]

    # Resolve the target vector store
    vs_store = get_vs_config_store()
    if req.vs_id:
        vs_config = vs_store.get(req.vs_id)
        if not vs_config:
            raise HTTPException(404, f"Vector store {req.vs_id!r} not found.")
    else:
        # Legacy fallback: use default Redis
        vs_config = vs_store.get("default") or {}

    client = get_vector_store_client(vs_config)

    tag_filter = list(req.tag_filter) if req.tag_filter else []
    if req.source_type:
        tag_filter.append(req.source_type)

    redis_tag_filter: str | None = None
    if tag_filter:
        redis_tag_filter = "@tags:{" + "|".join(tag_filter) + "}"

    fetch_k = req.top_k * 5 if req.usecase_id else req.top_k
    results = client.search(
        vec,
        top_k=fetch_k,
        tag_filter=redis_tag_filter,
        agent_filter=req.agent_filter,
    )

    # Legacy usecase post-filter (backwards compat)
    if req.usecase_id:
        try:
            from pipeline.mongo_store import get_usecase_ledger
            allowed = set(get_usecase_ledger().get_chunk_ids(req.usecase_id, req.agent_filter or ""))
            results = [r for r in results if r.get("chunk_id") in allowed]
        except Exception:
            pass

    results = results[: req.top_k]

    return [
        SearchResult(
            chunk_id=r.get("chunk_id", ""),
            content=r.get("content", ""),
            source=r.get("source", ""),
            title=r.get("title", ""),
            section=r.get("section", ""),
            tags=[t for t in r.get("tags", "").split(",") if t] if isinstance(r.get("tags"), str) else r.get("tags", []),
            score=r.get("score", 0.0),
            page_number=r.get("page_number"),
        )
        for r in results
    ]


@router.get("/vector-stores")
def list_searchable_stores():
    """List all configured vector stores for use in search."""
    from pipeline.mongo_store import get_vs_config_store
    return get_vs_config_store().list_all()


@router.get("/usecases")
def list_usecases():
    try:
        from pipeline.mongo_store import get_usecase_ledger
        usecases = get_usecase_ledger().get_distinct_usecases()
        return {"usecases": usecases}
    except Exception:
        return {"usecases": []}


@router.get("/usecases/{usecase_id}/agents")
def list_agents(usecase_id: str):
    try:
        from pipeline.mongo_store import get_usecase_ledger
        agents = get_usecase_ledger().get_agent_filters_for_usecase(usecase_id)
        return {"agents": agents}
    except Exception:
        return {"agents": []}
