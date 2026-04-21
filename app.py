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

pg = st.navigation(
    {
        "": [
            st.Page("pages/home.py",    title="Dashboard",    icon="🏠", default=True),
        ],
        "Documents": [
            st.Page("pages/ingest.py",  title="Add Document", icon="➕"),
            st.Page("pages/review.py",  title="Review Queue", icon="📋"),
        ],
        "Knowledge Base": [
            st.Page("pages/search.py",  title="Search",       icon="🔎"),
            st.Page("pages/status.py",  title="Status",       icon="⚙️"),
        ],
    }
)

pg.run()
