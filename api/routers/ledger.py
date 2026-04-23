from fastapi import APIRouter, HTTPException
from api.models import DriftCheckResult, MessageResponse
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


@router.get("/")
def list_docs(kb_name: str | None = None, drift_status: str | None = None):
    from pipeline.mongo_store import get_ledger
    docs = get_ledger().list_docs(kb_name=kb_name, drift_status=drift_status)
    return {"docs": docs}


@router.get("/snapshots")
def list_snapshots():
    from pipeline.mongo_store import get_ledger
    snapshots = get_ledger().list_snapshots()
    return {"snapshots": snapshots}


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str):
    from pipeline.mongo_store import get_ledger
    snapshot = get_ledger().get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snapshot


@router.post("/drift-check", response_model=DriftCheckResult)
def run_drift_check(kb_name: str | None = None):
    from pipeline.mongo_store import get_ledger
    result = get_ledger().run_drift_check(kb_name=kb_name)
    return DriftCheckResult(
        current=result.get("current", 0),
        stale=result.get("stale", 0),
        deleted=result.get("deleted", 0),
        unknown=result.get("unknown", 0),
    )


@router.post("/{doc_id}/drift-check")
def check_drift_one(doc_id: str):
    from pipeline.mongo_store import get_ledger
    status = get_ledger().check_drift_one(doc_id)
    return {"doc_id": doc_id, "drift_status": status}


@router.delete("/{doc_id}", response_model=MessageResponse)
def remove_doc(doc_id: str):
    try:
        from pipeline.mongo_store import get_ledger
        ledger = get_ledger()
        doc = ledger.get_doc(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found in ledger")

        chunk_ids = doc.get("chunk_ids", [])
        if chunk_ids:
            from pipeline.redis_store import delete_chunks
            delete_chunks(chunk_ids)

        ledger.remove_doc(doc_id)
        return MessageResponse(message=f"Removed {doc_id} and {len(chunk_ids)} chunks")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
