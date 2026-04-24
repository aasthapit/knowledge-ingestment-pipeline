"""Vector Stores — register and manage vector DB targets for corpora."""
from __future__ import annotations

import streamlit as st

st.title("🗄️ Vector Stores")
st.caption(
    "Register vector DB targets. The built-in Redis instance is always available. "
    "Add custom entries for any HTTP-compatible vector DB, additional Redis instances, "
    "or Tachyon — each corpus can push to a different store."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_stores() -> list[dict]:
    from pipeline.mongo_store import get_vs_config_store
    return get_vs_config_store().list_all()


@st.cache_data(ttl=30)
def _load_store(vs_id: str) -> dict | None:
    from pipeline.mongo_store import get_vs_config_store
    return get_vs_config_store().get(vs_id)


def _invalidate() -> None:
    _load_stores.clear()
    _load_store.clear()


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16] if iso else "—"


TYPE_ICON = {"redis": "⚡", "custom": "🌐", "tachyon": "🚀"}

# ── Connection guard ──────────────────────────────────────────────────────────

try:
    all_stores = _load_stores()
except Exception as _exc:
    st.error(f"Could not connect to MongoDB: {_exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────

if "vs_selected" not in st.session_state:
    st.session_state.vs_selected = None
if "vs_create_open" not in st.session_state:
    st.session_state.vs_create_open = False

# ── Layout ────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 3], gap="large")

# =============================================================================
# Left — store list + create button
# =============================================================================

with left:
    col_hd, col_btn = st.columns([2, 1])
    with col_hd:
        st.subheader("Stores")
    with col_btn:
        if st.button("＋ New", use_container_width=True):
            st.session_state.vs_create_open = not st.session_state.vs_create_open

    # ── Create form ───────────────────────────────────────────────────────────
    if st.session_state.vs_create_open:
        with st.form("create_vs_form", border=True):
            st.markdown("**Add Vector Store**")
            new_name = st.text_input("Name *", placeholder="e.g. prod-redis-2")
            new_type = st.radio(
                "Type",
                ["custom", "redis", "tachyon"],
                horizontal=True,
                format_func=lambda t: {"custom": "Custom HTTP", "redis": "Redis", "tachyon": "Tachyon"}[t],
            )

            # Fields shown for all types
            new_endpoint = st.text_input(
                "Base URL *" if new_type != "redis" else "Redis URL *",
                placeholder=(
                    "https://my-vector-db.example.com" if new_type == "custom"
                    else "redis://localhost:6379" if new_type == "redis"
                    else "https://tachyon.internal/api"
                ),
                help=(
                    "The pipeline will POST to {base_url}/upsert, /delete, /search."
                    if new_type == "custom" else
                    "Redis connection URL for this instance."
                    if new_type == "redis" else
                    "Tachyon service base URL."
                ),
            )
            new_api_key = st.text_input("API key / password", type="password")
            new_collection = st.text_input(
                "Collection / index name",
                placeholder="knowledge_index",
            )

            # Redis-specific extra fields
            new_key_prefix = ""
            new_embedding_dims = 1536
            if new_type == "redis":
                new_key_prefix = st.text_input(
                    "Key prefix",
                    value="chunk:",
                    help="Redis key prefix for stored chunk documents.",
                )
                new_embedding_dims = st.number_input(
                    "Embedding dimensions",
                    min_value=1,
                    value=1536,
                    help="Must match your embedding model output size.",
                )

            submitted = st.form_submit_button("Add", type="primary")
            if submitted:
                errors = []
                if not new_name.strip():
                    errors.append("Name is required.")
                if not new_endpoint.strip():
                    errors.append(
                        "Redis URL is required." if new_type == "redis"
                        else "Base URL is required."
                    )
                for e in errors:
                    st.error(e)
                if not errors:
                    try:
                        from pipeline.mongo_store import get_vs_config_store
                        extra = {}
                        if new_type == "redis":
                            extra = {
                                "redis_url": new_endpoint.strip(),
                                "index_name": new_collection.strip() or "knowledge_index",
                                "key_prefix": new_key_prefix.strip() or "chunk:",
                                "embedding_dims": int(new_embedding_dims),
                            }
                        vs_id = get_vs_config_store().create(
                            name=new_name.strip(),
                            vs_type=new_type,
                            endpoint=new_endpoint.strip() if new_type != "redis" else "",
                            api_key=new_api_key.strip() or "",
                            collection=new_collection.strip() or "",
                            extra=extra or None,
                        )
                        _invalidate()
                        st.session_state.vs_selected = vs_id
                        st.session_state.vs_create_open = False
                        st.success(f"Added **{new_name.strip()}**")
                        st.rerun()
                    except Exception as exc:
                        if "duplicate" in str(exc).lower() or "E11000" in str(exc):
                            st.error(f"Name **{new_name.strip()}** already exists.")
                        else:
                            st.error(str(exc))

    # ── Store list ────────────────────────────────────────────────────────────
    if not all_stores:
        st.info("No vector stores yet.")
    else:
        for vs in all_stores:
            vid   = vs["vs_id"]
            icon  = TYPE_ICON.get(vs.get("type", ""), "🗄️")
            is_def = vs.get("is_default", False)
            label = f"{icon} **{vs['name']}**"
            if is_def:
                label += "  \n*built-in*"
            btn_type = "primary" if st.session_state.vs_selected == vid else "secondary"
            if st.button(label, key=f"sel_vs_{vid}", use_container_width=True, type=btn_type):
                st.session_state.vs_selected = vid
                st.rerun()

# =============================================================================
# Right — store detail
# =============================================================================

with right:
    sel_id = st.session_state.vs_selected

    if sel_id is None:
        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:240px;border:2px dashed #ccc;border-radius:8px;"
            "color:#888;font-size:0.9rem'>Select a vector store to view details</div>",
            unsafe_allow_html=True,
        )
    else:
        vs = _load_store(sel_id)
        if vs is None:
            st.warning("Vector store not found.")
            st.session_state.vs_selected = None
        else:
            icon    = TYPE_ICON.get(vs.get("type", ""), "🗄️")
            is_def  = vs.get("is_default", False)
            vs_type = vs.get("type", "")

            # ── Header ────────────────────────────────────────────────────────
            hd1, hd2 = st.columns([3, 1])
            with hd1:
                st.subheader(f"{icon} {vs['name']}")
                if is_def:
                    st.caption("Built-in Redis instance (read-only)")
            with hd2:
                if not is_def:
                    if st.button("Delete", use_container_width=True, type="secondary"):
                        st.session_state[f"confirm_del_vs_{sel_id}"] = True

            if st.session_state.get(f"confirm_del_vs_{sel_id}"):
                st.warning(
                    f"Delete **{vs['name']}**? Corpora pointing to this store will need "
                    "to be reassigned before pushing."
                )
                dc1, dc2, _ = st.columns([1, 1, 4])
                with dc1:
                    if st.button("Yes, delete", type="primary", key="confirm_del_vs_yes"):
                        try:
                            from pipeline.mongo_store import get_vs_config_store
                            get_vs_config_store().delete(sel_id)
                            _invalidate()
                            st.session_state.vs_selected = None
                            st.session_state[f"confirm_del_vs_{sel_id}"] = False
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                with dc2:
                    if st.button("Cancel", key="confirm_del_vs_no"):
                        st.session_state[f"confirm_del_vs_{sel_id}"] = False
                        st.rerun()

            # ── Metrics ───────────────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Type", vs_type)
            if vs.get("collection"):
                m2.metric("Collection", vs["collection"])
            m3.metric("Created", _fmt_date(vs.get("created_at")))

            # ── Type-specific info ────────────────────────────────────────────
            if vs_type == "redis" and is_def:
                from pipeline.config import settings
                st.info(
                    f"Connects to: `{settings.redis_url}` · index `{settings.redis_index_name}`\n\n"
                    "Configure via `REDIS_URL` and `REDIS_INDEX_NAME` in your `.env` file."
                )
            elif vs_type == "redis" and not is_def:
                extra = vs.get("extra") or {}
                st.info(
                    f"Redis URL: `{extra.get('redis_url', '—')}`  \n"
                    f"Index: `{extra.get('index_name', '—')}`  \n"
                    f"Key prefix: `{extra.get('key_prefix', '—')}`  \n"
                    f"Embedding dims: `{extra.get('embedding_dims', '—')}`"
                )
            elif vs_type == "tachyon":
                st.info(
                    "Tachyon handles its own embedding and indexing — the embedding step is "
                    "skipped when pushing to this store. Implementation is pending Tachyon API finalisation."
                )
                if vs.get("endpoint"):
                    st.code(vs["endpoint"], language=None)
            elif vs.get("endpoint"):
                st.code(vs["endpoint"], language=None)

            st.divider()

            # ── Test connection ────────────────────────────────────────────────
            if st.button("🔌 Test connection", key="test_vs_conn"):
                with st.spinner("Connecting…"):
                    try:
                        from pipeline.vector_store import get_vector_store_client
                        client = get_vector_store_client(vs)
                        client.ensure_index()
                        st.success("Connection successful.")
                    except Exception as exc:
                        st.error(f"Connection failed: {exc}")

            # ── Edit form (non-default stores) ────────────────────────────────
            if not is_def:
                st.divider()
                with st.expander("Edit"):
                    with st.form(f"edit_vs_{sel_id}", border=False):
                        if vs_type == "redis":
                            extra = vs.get("extra") or {}
                            e_redis_url = st.text_input("Redis URL", value=extra.get("redis_url", ""))
                            e_index     = st.text_input("Index name", value=extra.get("index_name", ""))
                            e_prefix    = st.text_input("Key prefix", value=extra.get("key_prefix", "chunk:"))
                            e_dims      = st.number_input(
                                "Embedding dimensions",
                                min_value=1,
                                value=int(extra.get("embedding_dims", 1536)),
                            )
                            saved = st.form_submit_button("Save", type="primary")
                            if saved:
                                try:
                                    from pipeline.mongo_store import get_vs_config_store
                                    get_vs_config_store().update(
                                        sel_id,
                                        collection=e_index.strip() or None,
                                        extra={
                                            "redis_url": e_redis_url.strip(),
                                            "index_name": e_index.strip(),
                                            "key_prefix": e_prefix.strip(),
                                            "embedding_dims": int(e_dims),
                                        },
                                    )
                                    _invalidate()
                                    st.success("Updated.")
                                    st.rerun()
                                except ValueError as exc:
                                    st.error(str(exc))
                        else:
                            e_endpoint   = st.text_input("Base URL", value=vs.get("endpoint", ""))
                            e_api_key    = st.text_input("API key", value="", type="password",
                                                         help="Leave blank to keep the current key.")
                            e_collection = st.text_input("Collection", value=vs.get("collection", ""))
                            saved = st.form_submit_button("Save", type="primary")
                            if saved:
                                try:
                                    from pipeline.mongo_store import get_vs_config_store
                                    get_vs_config_store().update(
                                        sel_id,
                                        endpoint=e_endpoint.strip() or None,
                                        api_key=e_api_key.strip() or None,
                                        collection=e_collection.strip() or None,
                                    )
                                    _invalidate()
                                    st.success("Updated.")
                                    st.rerun()
                                except ValueError as exc:
                                    st.error(str(exc))
