import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import status, ingest, review, search, confluence, ledger, manifests, corpus
from api.routers import kb as kb_router
from api.routers import vector_stores as vs_router

app = FastAPI(title="Knowledge Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router, tags=["status"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(review.router, prefix="/api/review", tags=["review"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(confluence.router, prefix="/api/confluence", tags=["confluence"])
app.include_router(ledger.router, prefix="/api/ledger", tags=["ledger"])
app.include_router(manifests.router, prefix="/api/manifests", tags=["manifests"])
app.include_router(corpus.router, prefix="/api/corpus", tags=["corpus"])
app.include_router(kb_router.router, prefix="/api", tags=["Knowledge Bases"])
app.include_router(vs_router.router, prefix="/api", tags=["Vector Stores"])


@app.on_event("startup")
def startup():
    try:
        from pipeline.refresh_scheduler import start_scheduler
        start_scheduler()
    except Exception:
        pass


@app.get("/api/health")
def health():
    return {"status": "ok"}
