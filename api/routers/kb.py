"""Knowledge Base CRUD router."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models import CreateKBRequest, MessageResponse, UpdateKBRequest
from pipeline.mongo_store import get_kb_store

router = APIRouter(prefix="/kb", tags=["Knowledge Bases"])


@router.get("/")
def list_kbs(source_type: str | None = None):
    return get_kb_store().list_all(source_type=source_type)


@router.post("/", status_code=201)
def create_kb(req: CreateKBRequest):
    store = get_kb_store()
    if store.get_by_name(req.name):
        raise HTTPException(400, f"Knowledge Base '{req.name}' already exists.")
    kb_id = store.create(
        name=req.name,
        source_type=req.source_type,
        description=req.description,
        confluence_urls=req.confluence_urls or [],
        max_depth=req.max_depth,
        refresh_cron=req.refresh_cron,
        file_name=req.file_name,
        file_ref=req.file_ref,
        chunk_strategy=req.chunk_strategy,
        chunk_max_chars=req.chunk_max_chars,
        chunk_overlap_chars=req.chunk_overlap_chars,
    )
    return {"kb_id": kb_id, "message": "Knowledge Base created."}


@router.get("/{kb_id}")
def get_kb(kb_id: str):
    kb = get_kb_store().get(kb_id)
    if not kb:
        raise HTTPException(404, f"Knowledge Base {kb_id!r} not found.")
    return kb


@router.patch("/{kb_id}", response_model=MessageResponse)
def update_kb(kb_id: str, req: UpdateKBRequest):
    store = get_kb_store()
    if not store.get(kb_id):
        raise HTTPException(404, f"Knowledge Base {kb_id!r} not found.")
    store.update(
        kb_id,
        name=req.name,
        description=req.description,
        confluence_urls=req.confluence_urls,
        max_depth=req.max_depth,
        refresh_cron=req.refresh_cron,
        file_name=req.file_name,
        file_ref=req.file_ref,
        chunk_strategy=req.chunk_strategy,
        chunk_max_chars=req.chunk_max_chars,
        chunk_overlap_chars=req.chunk_overlap_chars,
    )
    return {"message": "Knowledge Base updated."}


@router.delete("/{kb_id}", response_model=MessageResponse)
def delete_kb(kb_id: str):
    store = get_kb_store()
    if not store.get(kb_id):
        raise HTTPException(404, f"Knowledge Base {kb_id!r} not found.")
    store.delete(kb_id)
    return {"message": "Knowledge Base deleted."}


@router.get("/{kb_id}/staging")
def get_kb_staging(kb_id: str):
    """Return all staging docs for this KB."""
    from pipeline.mongo_store import get_staging
    return get_staging().list_all(kb_id=kb_id)
