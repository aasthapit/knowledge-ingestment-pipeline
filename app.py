"""
app.py
Streamlit entry point.  Run with:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Start background Confluence refresh scheduler (idempotent — guarded by a lock).
# Streamlit reruns app.py on every page navigation; the scheduler is only created once.
try:
    from pipeline.refresh_scheduler import start_scheduler
    start_scheduler()
except Exception:
    pass  # never crash the UI due to scheduler issues

pg = st.navigation(
    {
        "": [
            st.Page("pages/home.py",    title="Dashboard",    icon="🏠", default=True),
        ],
        "Sources": [
            st.Page("pages/kb.py",           title="Knowledge Bases",  icon="📂"),
            st.Page("pages/ingest.py",       title="Add Document",     icon="➕"),
            st.Page("pages/confluence.py",   title="Confluence",       icon="🔗"),
        ],
        "Review": [
            st.Page("pages/review.py",       title="Review Queue",     icon="📋"),
        ],
        "Knowledge Base": [
            st.Page("pages/corpus.py",           title="Corpus",           icon="📦"),
            st.Page("pages/vector_stores.py",    title="Vector Stores",    icon="🗄️"),
            st.Page("pages/search.py",           title="Search",           icon="🔎"),
            st.Page("pages/drift.py",            title="KB Health",        icon="🔍"),
            st.Page("pages/ledger.py",           title="Ledger",           icon="📒"),
            st.Page("pages/manifests.py",        title="Manifests",        icon="📑"),
            st.Page("pages/status.py",           title="Status",           icon="⚙️"),
        ],
    }
)

pg.run()
