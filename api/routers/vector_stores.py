"""Vector Store configuration CRUD router."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models import CreateVectorStoreRequest, MessageResponse, UpdateVectorStoreRequest
from pipeline.mongo_store import get_vs_config_store

router = APIRouter(prefix="/vector-stores", tags=["Vector Stores"])


@router.get("/")
def list_vector_stores():
    return get_vs_config_store().list_all()


@router.post("/", status_code=201)
def create_vector_store(req: CreateVectorStoreRequest):
    vs_id = get_vs_config_store().create(
        name=req.name,
        vs_type=req.vs_type,
        endpoint=req.endpoint,
        api_key=req.api_key,
        collection=req.collection,
        extra=req.extra,
    )
    return {"vs_id": vs_id, "message": "Vector store config created."}


@router.get("/{vs_id}")
def get_vector_store(vs_id: str):
    vs = get_vs_config_store().get(vs_id)
    if not vs:
        raise HTTPException(404, f"Vector store {vs_id!r} not found.")
    return vs


@router.patch("/{vs_id}", response_model=MessageResponse)
def update_vector_store(vs_id: str, req: UpdateVectorStoreRequest):
    store = get_vs_config_store()
    if not store.get(vs_id):
        raise HTTPException(404, f"Vector store {vs_id!r} not found.")
    try:
        store.update(
            vs_id,
            name=req.name,
            endpoint=req.endpoint,
            api_key=req.api_key,
            collection=req.collection,
            extra=req.extra,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"message": "Vector store config updated."}


@router.delete("/{vs_id}", response_model=MessageResponse)
def delete_vector_store(vs_id: str):
    try:
        get_vs_config_store().delete(vs_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"message": "Vector store config deleted."}


@router.post("/{vs_id}/test")
def test_connection(vs_id: str):
    """Ping the vector store to verify connectivity."""
    from pipeline.vector_store import get_vector_store_client
    vs = get_vs_config_store().get(vs_id)
    if not vs:
        raise HTTPException(404, f"Vector store {vs_id!r} not found.")
    try:
        client = get_vector_store_client(vs)
        client.ensure_index()
        return {"ok": True, "message": "Connection successful."}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}
