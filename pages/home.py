"""Dashboard — live stats and quick-action cards."""
import streamlit as st

st.title("📚 Knowledge Base")
st.caption("Add documents, review flagged content, and search your organisation's knowledge.")

# ── Live stats ────────────────────────────────────────────────────────────────
pending_count = approved_count = rejected_count = total_vectors = 0
redis_ok = False

try:
    from pipeline import redis_store
    staging = redis_store.get_staging()
    docs = staging.list_all()
    pending_count  = sum(1 for d in docs if d.get("status") == "pending_review")
    approved_count = sum(1 for d in docs if d.get("status") == "approved")
    rejected_count = sum(1 for d in docs if d.get("status") == "rejected")
    redis_ok = True
except Exception:
    pass

try:
    from pipeline.config import settings
    if settings.vector_backend == "qdrant":
        from pipeline import qdrant_store
        total_vectors = qdrant_store.count()
    else:
        from pipeline import redis_store as rs
        client = rs.get_client()
        info = client.ft(settings.redis_index_name).info()
        total_vectors = int(info.get("num_docs", 0))
except Exception:
    pass

c1, c2, c3, c4 = st.columns(4)
c1.metric("In Knowledge Base", f"{total_vectors:,}", help="Total document sections indexed and searchable")
c2.metric(
    "Needs Your Review",
    pending_count,
    delta=f"{pending_count} waiting" if pending_count else None,
    delta_color="inverse",
    help="Documents flagged for quality review before being added",
)
c3.metric("Ready to Push",  approved_count,  help="Approved and waiting to be pushed to the knowledge base")
c4.metric("Rejected",       rejected_count,  help="Documents you have rejected")

if not redis_ok:
    st.warning(
        "⚠️ Could not connect to Redis. "
        "Make sure Redis Stack is running: `docker run -p 6379:6379 redis/redis-stack-server`",
        icon="⚠️",
    )

st.divider()

# ── Quick action cards ────────────────────────────────────────────────────────
st.subheader("What would you like to do?")

col_add, col_review, col_search = st.columns(3, gap="large")

with col_add:
    with st.container(border=True):
        st.markdown("### ➕ Add a Document")
        st.markdown(
            "Upload a PDF, Word document, PowerPoint, or paste a web link. "
            "The system will read it, assess quality, and add it to the review queue."
        )
        st.page_link("pages/ingest.py", label="Add Document →", icon="➕")

with col_review:
    with st.container(border=True):
        st.markdown("### 📋 Review Queue")
        if pending_count:
            st.markdown(
                f"**{pending_count} document{'s' if pending_count != 1 else ''} "
                f"need{'s' if pending_count == 1 else ''} your attention.** "
                "Review quality flags, approve or reject, then push to the knowledge base."
            )
        else:
            st.markdown(
                "No documents are waiting for review right now. "
                "Approve queued documents and push them to make them searchable."
            )
        st.page_link("pages/review.py", label="Open Review Queue →", icon="📋")

with col_search:
    with st.container(border=True):
        st.markdown("### 🔎 Search")
        st.markdown(
            "Ask a question in plain language and the knowledge base will find the "
            "most relevant sections from your documents, with links back to the source."
        )
        st.page_link("pages/search.py", label="Search Knowledge Base →", icon="🔎")
