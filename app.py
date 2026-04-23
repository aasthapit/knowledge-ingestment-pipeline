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
        "Documents": [
            st.Page("pages/ingest.py",      title="Add Document", icon="➕"),
            st.Page("pages/confluence.py",  title="Confluence",   icon="🔗"),
            st.Page("pages/review.py",      title="Review Queue", icon="📋"),
        ],
        "Knowledge Base": [
            st.Page("pages/search.py",          title="Search",           icon="🔎"),
            st.Page("pages/drift.py",            title="KB Health",        icon="🔍"),
            st.Page("pages/ledger.py",           title="Ledger",           icon="📒"),
            st.Page("pages/usecase_ledger.py",   title="Use Case Ledger",  icon="🗂️"),
            st.Page("pages/status.py",           title="Status",           icon="⚙️"),
        ],
    }
)

pg.run()
