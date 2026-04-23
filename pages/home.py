"""Dashboard — live stats and quick-action cards."""
import streamlit as st

st.title("📚 Knowledge Base")
st.caption("Add documents, review flagged content, and search your organisation's knowledge.")

# ── Live stats ────────────────────────────────────────────────────────────────
pending_count = approved_count = pushed_count = total_vectors = kb_count = 0
mongo_ok = False

try:
    from pipeline.mongo_store import get_staging, get_kb_store
    docs = get_staging().list_all()
    pending_count  = sum(1 for d in docs if d.get("status") == "pending_review")
    approved_count = sum(1 for d in docs if d.get("status") == "approved")
    pushed_count   = sum(1 for d in docs if d.get("status") == "pushed")
    kb_count = len(get_kb_store().list_all())
    mongo_ok = True
except Exception:
    pass

try:
    from pipeline.config import settings
    from pipeline import redis_store as rs
    client = rs.get_client()
    info = client.ft(settings.redis_index_name).info()
    total_vectors = int(info.get("num_docs", 0))
except Exception:
    pass

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Knowledge Bases",
    kb_count,
    help="Named document sources (Confluence or JSONL)",
)
c2.metric(
    "In Knowledge Base",
    f"{total_vectors:,}",
    help="Total document sections indexed and searchable",
)
c3.metric(
    "Needs Your Review",
    pending_count,
    delta=f"{pending_count} waiting" if pending_count else None,
    delta_color="inverse",
    help="Documents with quality flags waiting for a human decision",
)
c4.metric(
    "Ready to Push",
    approved_count,
    help="Approved and waiting to be embedded and pushed to the knowledge base",
)
c5.metric(
    "Pushed",
    pushed_count,
    help="Documents that have been embedded and are live in the knowledge base",
)

if not mongo_ok:
    st.warning(
        "Could not connect to MongoDB. "
        "Make sure MongoDB is reachable and `MONGODB_URI` is set in your `.env` file.",
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
            "The system checks chunk size and recency, then routes it to the "
            "review queue or auto-approves it."
        )
        st.page_link("pages/ingest.py", label="Add Document →", icon="➕")

with col_review:
    with st.container(border=True):
        st.markdown("### 📋 Review Queue")
        if pending_count:
            st.markdown(
                f"**{pending_count} document{'s' if pending_count != 1 else ''} "
                f"need{'s' if pending_count == 1 else ''} your attention.** "
                "Review flagged chunks, approve or reject, then push to the knowledge base."
            )
        else:
            st.markdown(
                "No documents are waiting for review. "
                "Approved documents are ready to push to make them searchable."
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
