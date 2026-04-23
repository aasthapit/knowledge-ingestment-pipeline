from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from api.models import IngestURLRequest, IngestResult, SaveSchemaRequest, MessageResponse
import tempfile, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


@router.post("/document", response_model=IngestResult)
async def ingest_document(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    tags: str = Form(default=""),
    kb_name: str = Form(default="default"),
    usecase_id: str | None = Form(default=None),
    agent_filter: str | None = Form(default=None),
    auto_push: bool = Form(default=False),
):
    from pipeline.ingest import ingest_document as _ingest

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    if file:
        suffix = os.path.splitext(file.filename or "upload")[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            result = _ingest(
                source=tmp_path,
                extra_tags=tag_list,
                auto_push=auto_push,
                kb_name=kb_name,
                usecase_id=usecase_id or "",
                agent_filter=agent_filter or "",
            )
        finally:
            os.unlink(tmp_path)
    elif url:
        result = _ingest(
            source=url,
            extra_tags=tag_list,
            auto_push=auto_push,
            kb_name=kb_name,
            usecase_id=usecase_id or "",
            agent_filter=agent_filter or "",
        )
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or a url")

    return IngestResult(
        doc_id=result.get("doc_id", ""),
        quality_score=result.get("quality_score", 0.0),
        quality_passed=result.get("quality_passed", False),
        quality_flags=result.get("quality_flags", []),
        chunk_count=result.get("chunk_count", 0),
        tags=result.get("tags", []),
    )


@router.post("/jsonl", response_model=IngestResult)
async def ingest_jsonl(
    file: UploadFile = File(...),
    tags: str = Form(default=""),
    kb_name: str = Form(default="default"),
    usecase_id: str | None = Form(default=None),
    agent_filter: str | None = Form(default=None),
    batch_name: str | None = Form(default=None),
    field_map: str | None = Form(default=None),  # JSON string
):
    import json
    from pipeline.ingest import ingest_jsonl as _ingest_jsonl

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    fm = json.loads(field_map) if field_map else None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = _ingest_jsonl(
            source=tmp_path,
            batch_name=batch_name,
            extra_tags=tag_list,
            kb_name=kb_name,
            usecase_id=usecase_id or "",
            agent_filter=agent_filter or "",
            field_map=fm,
        )
    finally:
        os.unlink(tmp_path)

    return IngestResult(
        doc_id=result.get("doc_id", ""),
        quality_score=1.0 if result.get("quality_passed") else 0.0,
        quality_passed=result.get("quality_passed", False),
        quality_flags=[],
        chunk_count=result.get("total", 0),
        tags=tag_list,
        detected_schema=result.get("schema"),
    )


@router.post("/peek")
async def peek_jsonl(
    file: UploadFile = File(...),
    field_map: str | None = Form(default=None),
):
    import json
    from pipeline.jsonl_importer import peek_jsonl as _peek

    fm = json.loads(field_map) if field_map else None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = _peek(tmp_path, n=5, field_map=fm)
    finally:
        os.unlink(tmp_path)

    return result


@router.get("/schemas")
def list_schemas():
    try:
        from pipeline.jsonl_importer import _load_schemas
        schemas = _load_schemas()
        return {"schemas": [s.get("name") for s in schemas]}
    except Exception:
        return {"schemas": []}


@router.post("/schemas", response_model=MessageResponse)
def save_schema(req: SaveSchemaRequest):
    from pipeline.jsonl_importer import save_custom_schema
    save_custom_schema(
        name=req.name,
        field_map=req.field_map,
        required_keys=req.required_keys,
        tags_static=req.tags_static,
        section_join=req.section_join,
    )
    return MessageResponse(message=f"Schema '{req.name}' saved")
