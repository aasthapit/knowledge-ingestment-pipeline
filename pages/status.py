"""Status — service connections, configuration, and knowledge base stats."""
import streamlit as st

st.title("⚙️  Status & Settings")
st.caption("Live connection status and current configuration for the knowledge base pipeline.")

# ── Connection checks ─────────────────────────────────────────────────────────

from pipeline.config import settings

def _check_redis() -> tuple[bool, str]:
    try:
        from pipeline import redis_store
        client = redis_store.get_client()
        client.ping()
        return True, settings.redis_url
    except Exception as exc:
        return False, str(exc)


def _check_mongo() -> tuple[bool, str]:
    try:
        from pipeline.mongo_store import get_staging
        get_staging().list_all()
        return True, settings.mongodb_uri or f"{settings.mongodb_host}:{settings.mongodb_port}"
    except Exception as exc:
        return False, str(exc)


def _check_openai() -> tuple[bool, str]:
    if settings.embedding_provider != "openai":
        return None, f"Using {settings.embedding_provider}"
    if not settings.openai_api_key:
        return False, "OPENAI_API_KEY not set"
    masked = settings.openai_api_key[:7] + "…" + settings.openai_api_key[-4:]
    return True, f"Key configured ({masked})"


st.subheader("Service Connections")
svc_cols = st.columns(3)

redis_ok, redis_msg = _check_redis()
mongo_ok, mongo_msg = _check_mongo()
openai_ok, openai_msg = _check_openai()

with svc_cols[0]:
    with st.container(border=True):
        if redis_ok:
            st.markdown("🟢  **Redis**")
        else:
            st.markdown("🔴  **Redis**")
        st.caption(redis_msg)
        if not redis_ok:
            st.caption("Start Redis: `docker run -p 6379:6379 redis/redis-stack-server`")

with svc_cols[1]:
    with st.container(border=True):
        if mongo_ok:
            st.markdown("🟢  **MongoDB**")
        else:
            st.markdown("🔴  **MongoDB**")
        st.caption(mongo_msg)
        if not mongo_ok:
            st.caption("Set `MONGODB_URI` in your `.env` file.")

with svc_cols[2]:
    with st.container(border=True):
        if openai_ok is None:
            st.markdown("⚪  **Embeddings**")
        elif openai_ok:
            st.markdown("🟢  **Embeddings**")
        else:
            st.markdown("🔴  **Embeddings**")
        st.caption(openai_msg)

# ── Knowledge base stats ──────────────────────────────────────────────────────
st.divider()
st.subheader("Knowledge Base Stats")

stat_cols = st.columns(4)

vector_count = 0
try:
    if redis_ok:
        from pipeline import redis_store
        client = redis_store.get_client()
        try:
            info = client.ft(settings.redis_index_name).info()
            vector_count = int(info.get("num_docs", 0))
        except Exception:
            vector_count = 0
except Exception:
    pass

pending = approved = rejected = total_staged = 0
try:
    from pipeline.mongo_store import get_staging
    docs = get_staging().list_all()
    pending      = sum(1 for d in docs if d.get("status") == "pending_review")
    approved     = sum(1 for d in docs if d.get("status") == "approved")
    rejected     = sum(1 for d in docs if d.get("status") == "rejected")
    total_staged = len(docs)
except Exception:
    pass

stat_cols[0].metric("Searchable sections",  f"{vector_count:,}", help="Sections indexed in the vector store")
stat_cols[1].metric("Awaiting review",       pending)
stat_cols[2].metric("Approved (not pushed)", approved)
stat_cols[3].metric("Total staged",          total_staged)

# ── Configuration ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Configuration")

cfg_left, cfg_right = st.columns(2)

with cfg_left:
    st.markdown("**Pipeline**")
    cfg_rows = [
        ("Chunk size (tokens)", str(settings.docling_max_tokens)),
        ("Chunk size (chars)",  str(settings.chunk_max_chars)),
        ("Chunk overlap (chars)", str(settings.chunk_overlap_chars)),
    ]
    for label, value in cfg_rows:
        r1, r2 = st.columns([2, 3])
        r1.caption(label)
        r2.write(value)

with cfg_right:
    st.markdown("**Embeddings**")
    emb_rows = [
        ("Provider",    settings.embedding_provider),
        ("Model",       settings.embedding_model),
        ("Dimensions",  str(settings.embedding_dimensions)),
        ("Batch size",  str(settings.embed_batch_size)),
    ]
    for label, value in emb_rows:
        r1, r2 = st.columns([2, 3])
        r1.caption(label)
        r2.write(value)

# ── .env hint ─────────────────────────────────────────────────────────────────
with st.expander("How to change these settings"):
    st.markdown(
        "Copy `.env.example` to `.env` in the project folder and edit the values. "
        "Restart the app after making changes.\n\n"
        "```bash\n"
        "cp .env.example .env\n"
        "# edit .env, then:\n"
        "streamlit run app.py\n"
        "```"
    )
