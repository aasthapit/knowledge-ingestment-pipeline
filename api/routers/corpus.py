"""Corpus CRUD router."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from api.models import (
    CreateCorpusRequest,
    UpdateCorpusRequest,
    CorpusKBRequest,
    MessageResponse,
)
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


def _store():
    from pipeline.mongo_store import get_corpus_store
    return get_corpus_store()


@router.get("/")
def list_corpora():
    return {"corpora": _store().list_all()}


@router.post("/", status_code=201)
def create_corpus(req: CreateCorpusRequest):
    try:
        corpus_id = _store().create(
            name=req.name,
            description=req.description,
            usecase_id=req.usecase_id,
            agent_filter=req.agent_filter,
            kb_ids=req.kb_ids,
            vector_store_id=req.vector_store_id,
        )
        return {"corpus_id": corpus_id}
    except Exception as e:
        if "duplicate" in str(e).lower() or "E11000" in str(e):
            raise HTTPException(status_code=409, detail=f"Corpus name '{req.name}' already exists")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{corpus_id}")
def get_corpus(corpus_id: str):
    corpus = _store().get(corpus_id)
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    return corpus


@router.patch("/{corpus_id}", response_model=MessageResponse)
def update_corpus(corpus_id: str, req: UpdateCorpusRequest):
    if not _store().get(corpus_id):
        raise HTTPException(status_code=404, detail="Corpus not found")
    _store().update(
        corpus_id=corpus_id,
        name=req.name,
        description=req.description,
        usecase_id=req.usecase_id,
        agent_filter=req.agent_filter,
        vector_store_id=req.vector_store_id,
    )
    return MessageResponse(message="Corpus updated")


@router.delete("/{corpus_id}", response_model=MessageResponse)
def delete_corpus(corpus_id: str):
    if not _store().get(corpus_id):
        raise HTTPException(status_code=404, detail="Corpus not found")
    _store().delete(corpus_id)
    return MessageResponse(message="Corpus deleted")


@router.post("/{corpus_id}/kbs", response_model=MessageResponse)
def add_kbs(corpus_id: str, req: CorpusKBRequest):
    if not _store().get(corpus_id):
        raise HTTPException(status_code=404, detail="Corpus not found")
    _store().add_kbs(corpus_id=corpus_id, kb_ids=req.kb_ids)
    return MessageResponse(message=f"Added {len(req.kb_ids)} Knowledge Base(s) to corpus")


@router.delete("/{corpus_id}/kbs", response_model=MessageResponse)
def remove_kbs(corpus_id: str, req: CorpusKBRequest):
    if not _store().get(corpus_id):
        raise HTTPException(status_code=404, detail="Corpus not found")
    _store().remove_kbs(corpus_id=corpus_id, kb_ids=req.kb_ids)
    return MessageResponse(message=f"Removed {len(req.kb_ids)} Knowledge Base(s) from corpus")


@router.get("/{corpus_id}/changelog")
def get_changelog(corpus_id: str, limit: int = 100):
    if not _store().get(corpus_id):
        raise HTTPException(status_code=404, detail="Corpus not found")
    return {"changelog": _store().get_changelog(corpus_id, limit=limit)}


@router.get("/{corpus_id}/export")
def export_corpus(corpus_id: str):
    """Export all staged chunks from every KB in this corpus as JSONL."""
    from pipeline.mongo_store import get_staging, get_kb_store
    corpus = _store().get(corpus_id)
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    staging  = get_staging()
    kb_store = get_kb_store()
    lines: list[str] = []
    for kb_id in corpus.get("kb_ids", []):
        kb = kb_store.get(kb_id)
        if not kb:
            continue
        for doc_id in kb.get("doc_ids", []):
            for chunk in staging.get_chunks(doc_id):
                lines.append(json.dumps(chunk))

    filename = f"corpus_{corpus.get('name', corpus_id)}.jsonl"
    return Response(
        content="\n".join(lines),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
