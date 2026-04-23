from fastapi import APIRouter, HTTPException
from api.models import (
    RejectRequest, UpdateChunkRequest, SplitDocRequest,
    SplitChunkRequest, PushRequest, PushResult, MessageResponse,
)
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


@router.get("/")
def list_docs(status: str | None = None, kb_id: str | None = None):
    from pipeline.review import list_all_docs
    from pipeline.mongo_store import get_staging
    docs = get_staging().list_all(kb_id=kb_id)
    if status:
        docs = [d for d in docs if d.get("status") == status]
    return {"docs": docs}


@router.get("/{doc_id}")
def get_doc(doc_id: str):
    from pipeline.review import get_doc_detail
    doc = get_doc_detail(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post("/{doc_id}/approve", response_model=MessageResponse)
def approve_doc(doc_id: str):
    from pipeline.review import approve_doc as _approve
    ok = _approve(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found or already processed")
    return MessageResponse(message="Approved")


@router.post("/{doc_id}/reject", response_model=MessageResponse)
def reject_doc(doc_id: str, req: RejectRequest):
    from pipeline.review import reject_doc as _reject
    ok = _reject(doc_id, req.reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return MessageResponse(message="Rejected")


@router.patch("/{doc_id}/chunks/{chunk_id}", response_model=MessageResponse)
def update_chunk(doc_id: str, chunk_id: str, req: UpdateChunkRequest):
    from pipeline.review import update_chunk as _update
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    ok = _update(doc_id, chunk_id, updates)
    if not ok:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return MessageResponse(message="Updated")


@router.post("/{doc_id}/split")
def split_doc(doc_id: str, req: SplitDocRequest):
    from pipeline.review import split_doc as _split
    new_id = _split(doc_id, req.chunk_ids, req.new_title)
    if not new_id:
        raise HTTPException(status_code=400, detail="Split failed")
    return {"new_doc_id": new_id}


@router.post("/{doc_id}/chunks/{chunk_id}/split")
def split_chunk(doc_id: str, chunk_id: str, req: SplitChunkRequest):
    from pipeline.review import split_chunk as _split_chunk
    new_ids = _split_chunk(doc_id, chunk_id, req.content_parts)
    return {"new_chunk_ids": new_ids}


@router.post("/push", response_model=PushResult)
def push(req: PushRequest):
    from pipeline.review import push_approved
    result = push_approved(
        corpus_id=req.corpus_id,
        doc_id=req.doc_id,
        remove_after_push=req.remove_after_push,
    )
    return PushResult(
        pushed_docs=result.get("pushed_docs", 0),
        pushed_chunks=result.get("pushed_chunks", 0),
        failed_docs=result.get("failed_docs", 0),
    )
