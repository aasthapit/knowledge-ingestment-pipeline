"""Dashboard — live stats and quick-action cards."""
import streamlit as st

st.title("📚 Knowledge Ingestion Pipeline")
st.caption("Crawl Confluence, stage and review documents, then export clean JSONL for your AI knowledge bases.")

# ── Live stats ────────────────────────────────────────────────────────────────
pending_count = approved_count = pushed_count = kb_count = corpus_count = 0
mongo_ok = False

try:
    from pipeline.mongo_store import get_staging, get_kb_store, get_corpus_store
    docs = get_staging().list_all()
    pending_count  = sum(1 for d in docs if d.get("status") == "pending_review")
    approved_count = sum(1 for d in docs if d.get("status") == "approved")
    pushed_count   = sum(1 for d in docs if d.get("status") == "pushed")
    kb_count = len(get_kb_store().list_all())
    corpus_count = len(get_corpus_store().list_all())
    mongo_ok = True
except Exception:
    pass

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Knowledge Bases",
    kb_count,
    help="Named document sources (Confluence, file, or web)",
)
c2.metric(
    "Corpora",
    corpus_count,
    help="Named groupings of Knowledge Bases for export",
)
c3.metric(
    "Needs Your Review",
    pending_count,
    delta=f"{pending_count} waiting" if pending_count else None,
    delta_color="inverse",
    help="Documents with quality flags waiting for a human decision",
)
c4.metric(
    "Approved",
    approved_count,
    help="Approved staged documents ready for JSONL export",
)
c5.metric(
    "Pushed",
    pushed_count,
    help="Documents marked as pushed after export",
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

col_add, col_review, col_export = st.columns(3, gap="large")

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
                "Review flagged chunks, edit content, and approve or reject."
            )
        else:
            st.markdown(
                "No documents are waiting for review. "
                "Approved documents are ready to export as JSONL."
            )
        st.page_link("pages/review.py", label="Open Review Queue →", icon="📋")

with col_export:
    with st.container(border=True):
        st.markdown("### 📦 Export JSONL")
        st.markdown(
            "Group Knowledge Bases into a Corpus and download clean JSONL — "
            "ready to feed into any embedding pipeline or vector store."
        )
        st.page_link("pages/corpus.py", label="Manage Corpora →", icon="📦")
