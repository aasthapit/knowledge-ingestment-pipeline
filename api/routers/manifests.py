"""Manifests router — corpus-scoped document snapshots."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.models import (
    CreateManifestRequest, SnapshotManifestRequest, CreateFromSourcesRequest,
    DiffManifestsRequest, RemoveManifestDocsRequest, MessageResponse,
)
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


def _mgr():
    from pipeline.manifests import get_manifest_manager
    return get_manifest_manager()


@router.get("/")
def list_manifests(corpus_id: str | None = None, status: str | None = None):
    manifests = _mgr().list_manifests(corpus_id=corpus_id, status=status)
    return {"manifests": manifests}


@router.post("/")
def create_manifest(req: CreateManifestRequest):
    manifest_id = _mgr().create_manifest(
        name=req.name,
        corpus_id=req.corpus_id,
        description=req.description,
        tags=req.tags,
    )
    return {"manifest_id": manifest_id}


@router.get("/{manifest_id}")
def get_manifest(manifest_id: str):
    m = _mgr().get_manifest(manifest_id)
    if not m:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return m


@router.post("/{manifest_id}/freeze", response_model=MessageResponse)
def freeze_manifest(manifest_id: str):
    ok = _mgr().freeze_manifest(manifest_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not freeze manifest")
    return MessageResponse(message="Frozen")


@router.post("/{manifest_id}/archive", response_model=MessageResponse)
def archive_manifest(manifest_id: str):
    ok = _mgr().archive_manifest(manifest_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not archive manifest")
    return MessageResponse(message="Archived")


@router.post("/snapshot")
def snapshot_corpus(req: SnapshotManifestRequest):
    try:
        manifest_id = _mgr().snapshot_corpus_to_manifest(
            corpus_id=req.corpus_id,
            manifest_name=req.manifest_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"manifest_id": manifest_id}


@router.post("/from-sources")
def create_from_sources(req: CreateFromSourcesRequest):
    manifest_id = _mgr().create_manifest_from_sources(
        name=req.name,
        source_refs=req.source_refs,
        source_type=req.source_type,
        corpus_id=req.corpus_id,
        kb_id=req.kb_id,
        description=req.description,
        tags=req.tags,
    )
    return {"manifest_id": manifest_id}


@router.post("/diff")
def diff_manifests(req: DiffManifestsRequest):
    return _mgr().diff_manifests(req.manifest_id_a, req.manifest_id_b)


@router.post("/{manifest_id}/ingest")
def ingest_from_manifest(manifest_id: str, kb_id: str | None = None, extra_tags: list[str] = []):
    def event_stream():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Starting re-ingest...'})}\n\n"
            result = _mgr().ingest_from_manifest(
                manifest_id=manifest_id,
                kb_id=kb_id,
                extra_tags=extra_tags,
            )
            yield f"data: {json.dumps({'type': 'done', 'result': result})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.delete("/{manifest_id}/docs")
def remove_manifest_docs(manifest_id: str, req: RemoveManifestDocsRequest):
    return _mgr().remove_manifest_docs(manifest_id, doc_ids=req.doc_ids)
