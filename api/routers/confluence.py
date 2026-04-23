from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.models import ConfluenceCrawlRequest
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter()


@router.post("/crawl")
def crawl_confluence(req: ConfluenceCrawlRequest):
    def event_stream():
        try:
            from pipeline.confluence import ConfluenceCrawler
            from pipeline.ingest import ingest_jsonl
            import tempfile

            crawler = ConfluenceCrawler(
                base_url=req.base_url,
                auth_type=req.auth_type,
                email=req.email or "",
                api_token=req.api_token or "",
                ssl_verify=req.ssl_verify,
            )

            yield f"data: {json.dumps({'type': 'progress', 'message': 'Connecting to Confluence...'})}\n\n"

            pages = crawler.crawl(req.page_url, max_depth=req.max_depth)
            total = len(pages)

            yield f"data: {json.dumps({'type': 'progress', 'message': f'Found {total} pages. Converting...'})}\n\n"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as tmp:
                for i, page in enumerate(pages):
                    record = crawler.to_record(page)
                    tmp.write(json.dumps(record) + "\n")
                    if i % 5 == 0:
                        yield f"data: {json.dumps({'type': 'progress', 'message': f'Processing page {i+1}/{total}: {page.title}'})}\n\n"
                tmp_path = tmp.name

            yield f"data: {json.dumps({'type': 'progress', 'message': 'Staging pages for review...'})}\n\n"

            result = ingest_jsonl(
                source=tmp_path,
                extra_tags=req.tags,
                kb_name=req.kb_name,
                usecase_id=req.usecase_id or "",
                agent_filter=req.agent_filter or "",
            )
            os.unlink(tmp_path)

            if req.usecase_id and req.agent_filter:
                try:
                    from pipeline.mongo_store import get_usecase_ledger
                    get_usecase_ledger().upsert_confluence_source(
                        usecase_id=req.usecase_id,
                        agent_filter=req.agent_filter,
                        page_urls=[req.page_url],
                    )
                except Exception:
                    pass

            yield f"data: {json.dumps({'type': 'done', 'result': {'pages': total, 'doc_id': result.get('doc_id', '')}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
