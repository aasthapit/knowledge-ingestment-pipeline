"""Vector Stores — read-only view of vector_stores.yaml configuration."""
from __future__ import annotations

import streamlit as st

st.title("🗄️ Vector Stores")
st.caption(
    "Vector store configs are defined in **vector_stores.yaml** at the project root. "
    "Edit that file and restart the app to add, remove, or change stores."
)

TYPE_ICON = {"redis": "⚡", "custom": "🌐", "tachyon": "🚀"}


@st.cache_data(ttl=60)
def _load_stores() -> list[dict]:
    from pipeline.vs_config import get_vs_config_store
    return get_vs_config_store().list_all()


@st.cache_data(ttl=60)
def _config_path() -> str:
    from pipeline.vs_config import get_vs_config_store
    return str(get_vs_config_store().config_path)


try:
    stores = _load_stores()
except Exception as exc:
    st.error(f"Could not load vector_stores.yaml: {exc}")
    st.stop()

if not stores:
    st.warning(
        "No vector stores defined. Add entries to **vector_stores.yaml** and restart."
    )
    st.code("# vector_stores.yaml\nvector_stores:\n  - id: default\n    name: Default Redis\n    type: redis\n    ...", language="yaml")
    st.stop()

st.caption(f"Config file: `{_config_path()}`")
st.divider()

for vs in stores:
    vs_type = vs.get("type", "")
    icon = TYPE_ICON.get(vs_type, "🗄️")
    with st.expander(f"{icon} **{vs.get('name', vs.get('vs_id', '?'))}**  —  `{vs_type}`", expanded=True):
        col_id, col_type, col_coll = st.columns(3)
        col_id.metric("ID", vs.get("vs_id", "—"))
        col_type.metric("Type", vs_type)
        col_coll.metric("Collection / Index", vs.get("collection") or "—")

        if vs_type == "redis":
            extra = vs.get("extra") or {}
            st.text_input(
                "Redis URL",
                value=extra.get("redis_url") or vs.get("endpoint") or "—",
                disabled=True,
                key=f"redis_url_{vs.get('vs_id')}",
            )
        elif vs_type in ("custom", "tachyon"):
            st.text_input(
                "Endpoint",
                value=vs.get("endpoint") or "—",
                disabled=True,
                key=f"endpoint_{vs.get('vs_id')}",
            )
            has_key = bool(vs.get("api_key"))
            st.markdown(f"API key: {'✅ set' if has_key else '—'}")

        # Connection test
        if st.button("Test connection", key=f"test_{vs.get('vs_id')}"):
            with st.spinner("Testing…"):
                try:
                    from pipeline.vector_store import get_vector_store_client
                    client = get_vector_store_client(vs)
                    client.ensure_index()
                    st.success("Connection OK — index ready.")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")
