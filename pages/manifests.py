"""Document Manifests — track, version, and operate on corpus document sets."""
from __future__ import annotations

import json

import streamlit as st

st.title("Document Manifests")
st.caption(
    "Name and version your document corpus. Track exactly which sources are in each "
    "knowledge base, diff versions, re-ingest from a saved manifest, or remove a batch "
    "of documents from the corpus in one operation."
)

# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_manifests(
    usecase_id: str | None = None,
    agent_filter: str | None = None,
    status: str | None = None,
) -> list[dict]:
    from pipeline.manifests import get_manifest_manager
    return get_manifest_manager().list_manifests(
        usecase_id=usecase_id or None,
        agent_filter=agent_filter or None,
        status=status or None,
    )


@st.cache_data(ttl=30)
def _load_usecases() -> list[str]:
    from pipeline.mongo_store import get_usecase_ledger
    return get_usecase_ledger().get_distinct_usecases()


@st.cache_data(ttl=30)
def _load_agent_filters(usecase_id: str) -> list[str]:
    from pipeline.mongo_store import get_usecase_ledger
    return get_usecase_ledger().get_agent_filters_for_usecase(usecase_id)


# ── Connection guard ──────────────────────────────────────────────────────────

try:
    all_manifests = _load_manifests()
except Exception as _conn_exc:
    st.error(f"Could not connect to MongoDB: {_conn_exc}")
    st.info("Make sure MongoDB is running and `MONGODB_URI` is set in your `.env` file.")
    st.stop()

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_browse, tab_create, tab_diff, tab_reingest = st.tabs([
    "Browse",
    "Create / Snapshot",
    "Diff",
    "Re-ingest",
])


# =============================================================================
# Tab 1 — Browse
# =============================================================================

with tab_browse:
    import pandas as pd

    # ── Filters ───────────────────────────────────────────────────────────────
    f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
    with f_col1:
        try:
            uc_opts = ["(all)"] + _load_usecases()
        except Exception:
            uc_opts = ["(all)"]
        browse_uc = st.selectbox("Use case", uc_opts, key="browse_uc")
    with f_col2:
        if browse_uc and browse_uc != "(all)":
            af_opts = ["(all)"] + _load_agent_filters(browse_uc)
        else:
            af_opts = ["(all)"]
        browse_af = st.selectbox("Agent filter", af_opts, key="browse_af")
    with f_col3:
        browse_status = st.selectbox(
            "Status", ["(all)", "open", "frozen", "archived"], key="browse_status"
        )

    # ── Manifest summary table ────────────────────────────────────────────────
    try:
        manifests = _load_manifests(
            usecase_id=browse_uc if browse_uc != "(all)" else None,
            agent_filter=browse_af if browse_af != "(all)" else None,
            status=browse_status if browse_status != "(all)" else None,
        )
    except Exception as exc:
        st.error(f"Could not load manifests: {exc}")
        manifests = []

    STATUS_ICONS = {"open": "🟢", "frozen": "🔒", "archived": "🗄️"}

    if not manifests:
        st.info("No manifests found. Use **Create / Snapshot** to create one.")
    else:
        rows = []
        for m in manifests:
            status = m.get("status", "")
            rows.append({
                "manifest_id":   m.get("manifest_id", ""),
                "Name":          m.get("name", ""),
                "Use case":      m.get("usecase_id") or "—",
                "Agent":         m.get("agent_filter") or "—",
                "Status":        STATUS_ICONS.get(status, "?") + " " + status,
                "Docs":          m.get("entry_count", 0),
                "Pushed":        m.get("pushed_count", 0),
                "Created":       (m.get("created_at") or "")[:10],
                "Tags":          ", ".join(m.get("tags") or []),
                "Created by":    m.get("created_by") or "—",
            })

        df = pd.DataFrame(rows)
        selection = st.dataframe(
            df.drop(columns=["manifest_id"]),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Docs":   st.column_config.NumberColumn("Docs",   format="%d"),
                "Pushed": st.column_config.NumberColumn("Pushed", format="%d"),
            },
        )

        selected_rows = selection.selection.rows if selection else []
        if selected_rows:
            idx = selected_rows[0]
            manifest_id_sel = rows[idx]["manifest_id"]

            from pipeline.manifests import get_manifest_manager
            mm = get_manifest_manager()
            mf = mm.get_manifest(manifest_id_sel)

            if mf:
                st.divider()
                st.subheader(f"Manifest: {mf['name']}")
                if mf.get("description"):
                    st.caption(mf["description"])

                # Manifest-level actions
                mf_status = mf.get("status", "")
                act_col1, act_col2, act_col3 = st.columns([1, 1, 4])
                with act_col1:
                    if mf_status == "open":
                        if st.button("🔒 Freeze", key="freeze_btn"):
                            mm.freeze_manifest(manifest_id_sel)
                            st.success("Manifest frozen.")
                            st.cache_data.clear()
                            st.rerun()
                with act_col2:
                    if mf_status != "archived":
                        if st.button("🗄️ Archive", key="archive_btn"):
                            mm.archive_manifest(manifest_id_sel)
                            st.success("Manifest archived.")
                            st.cache_data.clear()
                            st.rerun()
                with act_col3:
                    dl_data = json.dumps(mf, indent=2, default=str).encode()
                    st.download_button(
                        "⬇ Download JSON",
                        data=dl_data,
                        file_name=f"manifest_{manifest_id_sel[:8]}.json",
                        mime="application/json",
                        key="dl_manifest",
                    )

                # Entry table
                entries = mf.get("entries") or []
                if not entries:
                    st.info("This manifest has no entries yet.")
                else:
                    st.caption(f"{len(entries)} entries  ·  {mf.get('pushed_count', 0)} pushed")
                    ENTRY_STATUS_ICONS = {
                        "pending":  "⏳",
                        "staged":   "📥",
                        "approved": "✅",
                        "pushed":   "🚀",
                        "removed":  "🗑️",
                    }
                    entry_rows = []
                    for e in entries:
                        es = e.get("status", "")
                        entry_rows.append({
                            "doc_id":      e.get("doc_id", ""),
                            "Status":      ENTRY_STATUS_ICONS.get(es, "?") + " " + es,
                            "Title":       e.get("title", ""),
                            "Source type": e.get("source_type", ""),
                            "Source ref":  e.get("source_ref", ""),
                            "Version":     e.get("version_id", ""),
                            "Staged":      (e.get("staged_at") or "")[:10],
                            "Pushed":      (e.get("pushed_at") or "")[:10],
                        })

                    entry_df = pd.DataFrame(entry_rows)
                    entry_sel = st.dataframe(
                        entry_df.drop(columns=["doc_id"]),
                        use_container_width=True,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        key="entry_table",
                    )

                    # Remove action for selected entry
                    sel_entry_rows = entry_sel.selection.rows if entry_sel else []
                    if sel_entry_rows:
                        sel_entry = entry_rows[sel_entry_rows[0]]
                        sel_doc_id = sel_entry.get("doc_id", "")
                        sel_entry_status = sel_entry.get("Status", "")

                        if sel_doc_id and "pushed" in sel_entry_status:
                            st.markdown("---")
                            # Check cross-manifest references
                            other_manifests = [
                                m for m in mm.find_manifests_by_doc_id(sel_doc_id)
                                if m.get("manifest_id") != manifest_id_sel
                            ]
                            if other_manifests:
                                other_names = ", ".join(
                                    m.get("name", m["manifest_id"]) for m in other_manifests
                                )
                                st.warning(
                                    f"This document also appears in: **{other_names}**. "
                                    "Removing it will affect those manifests too."
                                )
                            label = (
                                f"Remove from KB (affects {1 + len(other_manifests)} manifest(s))"
                                if other_manifests
                                else "Remove from KB"
                            )
                            if st.button(f"🗑️ {label}", key="remove_entry_btn", type="primary"):
                                with st.spinner("Removing…"):
                                    result = mm.remove_manifest_docs(
                                        manifest_id_sel, doc_ids=[sel_doc_id]
                                    )
                                if result["errors"]:
                                    for err in result["errors"]:
                                        st.error(err)
                                else:
                                    st.success(
                                        f"Removed {result['removed_docs']} doc(s), "
                                        f"{result['removed_chunks']} chunk(s)."
                                    )
                                st.cache_data.clear()
                                st.rerun()


# =============================================================================
# Tab 2 — Create / Snapshot
# =============================================================================

with tab_create:

    # ── Snapshot current corpus ───────────────────────────────────────────────
    st.subheader("Snapshot current corpus")
    st.caption("Save the current state of the knowledge base as a named, frozen manifest.")

    with st.form("snapshot_form"):
        sn_col1, sn_col2 = st.columns(2)
        with sn_col1:
            try:
                sn_uc_opts = _load_usecases()
            except Exception:
                sn_uc_opts = []
            sn_usecase = st.selectbox(
                "Use case ID *",
                sn_uc_opts or ["—"],
                key="sn_usecase",
            )
        with sn_col2:
            sn_af_opts = _load_agent_filters(sn_usecase) if sn_usecase and sn_usecase != "—" else []
            sn_agent = st.selectbox(
                "Agent filter *",
                sn_af_opts or ["—"],
                key="sn_agent",
            )
        sn_name = st.text_input("Manifest name *", placeholder="SSOP v2 — April 2026")
        sn_desc = st.text_input("Description", placeholder="Stable corpus before model update")
        sn_tags = st.text_input("Tags (comma-separated)", placeholder="stable, pre-upgrade")
        sn_submitted = st.form_submit_button("Save as Frozen Manifest")

    if sn_submitted:
        sn_errors = []
        if not sn_usecase or sn_usecase == "—":
            sn_errors.append("Use case ID is required.")
        if not sn_agent or sn_agent == "—":
            sn_errors.append("Agent filter is required.")
        if not sn_name.strip():
            sn_errors.append("Manifest name is required.")
        for e in sn_errors:
            st.error(e)
        if not sn_errors:
            tags_list = [t.strip() for t in sn_tags.split(",") if t.strip()]
            with st.spinner("Creating snapshot…"):
                try:
                    from pipeline.manifests import get_manifest_manager
                    mid = get_manifest_manager().snapshot_corpus_to_manifest(
                        usecase_id=sn_usecase,
                        agent_filter=sn_agent,
                        manifest_name=sn_name.strip(),
                        description=sn_desc.strip(),
                        created_by="ui",
                        tags=tags_list,
                    )
                    st.success(f"Snapshot created: `{mid}`")
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(f"Could not create snapshot: {exc}")

    st.divider()

    # ── Create from source list ───────────────────────────────────────────────
    st.subheader("Create manifest from source list")
    st.caption(
        "Define a manifest before ingestion — list the Confluence URLs or doc IDs that "
        "should be part of this corpus version. Entries start as **pending** until ingested."
    )

    with st.form("sources_form"):
        src_name = st.text_input("Manifest name *", placeholder="SSOP OCP Docs — Q2 2026")
        src_type = st.radio(
            "Source type", ["confluence", "jsonl", "url", "pdf"], horizontal=True, key="src_type"
        )
        src_refs_raw = st.text_area(
            "Source refs (one per line) *",
            placeholder="https://confluence.example.com/spaces/OPS/pages/12345678/Page",
            height=120,
        )
        src_col1, src_col2 = st.columns(2)
        with src_col1:
            src_usecase = st.text_input("Use case ID", placeholder="GENAI1597_SSOP")
        with src_col2:
            src_agent = st.text_input("Agent filter", placeholder="ssop_cloud_operations_agent")
        src_desc = st.text_input("Description")
        src_tags = st.text_input("Tags (comma-separated)")
        src_submitted = st.form_submit_button("Create Manifest")

    if src_submitted:
        src_errors = []
        if not src_name.strip():
            src_errors.append("Manifest name is required.")
        refs = [r.strip() for r in src_refs_raw.splitlines() if r.strip()]
        if not refs:
            src_errors.append("At least one source ref is required.")
        for e in src_errors:
            st.error(e)
        if not src_errors:
            tags_list = [t.strip() for t in src_tags.split(",") if t.strip()]
            try:
                from pipeline.manifests import get_manifest_manager
                mid = get_manifest_manager().create_manifest_from_sources(
                    name=src_name.strip(),
                    source_refs=refs,
                    source_type=src_type,
                    usecase_id=src_usecase.strip() or None,
                    agent_filter=src_agent.strip() or None,
                    description=src_desc.strip(),
                    created_by="ui",
                    tags=tags_list,
                )
                st.success(f"Manifest created with {len(refs)} pending entries: `{mid}`")
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"Could not create manifest: {exc}")


# =============================================================================
# Tab 3 — Diff
# =============================================================================

with tab_diff:
    st.subheader("Diff two manifests")
    st.caption("Compare entries between two manifests by doc_id and version_id.")

    try:
        diff_manifests_list = _load_manifests()
    except Exception as exc:
        st.error(f"Could not load manifests: {exc}")
        diff_manifests_list = []

    if len(diff_manifests_list) < 2:
        st.info("You need at least two manifests to diff. Create them in the **Create / Snapshot** tab.")
    else:
        import pandas as pd

        manifest_options = {
            f"{m['name']} ({(m.get('created_at') or '')[:10]})": m["manifest_id"]
            for m in diff_manifests_list
        }
        option_labels = list(manifest_options.keys())

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            diff_a_label = st.selectbox("Manifest A (before)", option_labels, key="diff_a")
        with d_col2:
            diff_b_label = st.selectbox(
                "Manifest B (after)",
                option_labels,
                index=min(1, len(option_labels) - 1),
                key="diff_b",
            )

        if st.button("Compare", key="diff_btn", type="primary"):
            mid_a = manifest_options[diff_a_label]
            mid_b = manifest_options[diff_b_label]
            if mid_a == mid_b:
                st.warning("Select two different manifests to compare.")
            else:
                with st.spinner("Comparing…"):
                    try:
                        from pipeline.manifests import get_manifest_manager
                        result = get_manifest_manager().diff_manifests(mid_a, mid_b)
                    except Exception as exc:
                        st.error(f"Diff failed: {exc}")
                        result = None

                if result is not None:
                    n_added     = len(result["added"])
                    n_removed   = len(result["removed"])
                    n_changed   = len(result["changed"])
                    n_unchanged = len(result["unchanged"])

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Added",     n_added,
                               delta=n_added or None, delta_color="normal")
                    mc2.metric("Removed",   n_removed,
                               delta=-n_removed if n_removed else None, delta_color="inverse")
                    mc3.metric("Changed",   n_changed,
                               delta=n_changed or None, delta_color="off")
                    mc4.metric("Unchanged", n_unchanged)

                    if n_added == 0 and n_removed == 0 and n_changed == 0:
                        st.success("Manifests are identical — no differences found.")
                    else:
                        if result["added"]:
                            with st.expander(f"🆕 {n_added} added"):
                                rows = [
                                    {
                                        "Title":       e.get("title", ""),
                                        "Source type": e.get("source_type", ""),
                                        "Source ref":  e.get("source_ref", ""),
                                        "Version":     e.get("version_id", ""),
                                    }
                                    for e in result["added"]
                                ]
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                        if result["removed"]:
                            with st.expander(f"🗑️ {n_removed} removed"):
                                rows = [
                                    {
                                        "Title":       e.get("title", ""),
                                        "Source type": e.get("source_type", ""),
                                        "Source ref":  e.get("source_ref", ""),
                                        "Version":     e.get("version_id", ""),
                                    }
                                    for e in result["removed"]
                                ]
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                        if result["changed"]:
                            with st.expander(f"✏️ {n_changed} changed"):
                                rows = []
                                for c in result["changed"]:
                                    before = c["before"]
                                    after  = c["after"]
                                    rows.append({
                                        "Title":          after.get("title", ""),
                                        "Source ref":     after.get("source_ref", ""),
                                        "Version before": before.get("version_id", ""),
                                        "Version after":  after.get("version_id", ""),
                                    })
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# Tab 4 — Re-ingest
# =============================================================================

with tab_reingest:
    st.subheader("Re-ingest from manifest")
    st.caption(
        "Re-crawl Confluence sources listed in a manifest and stage them for review. "
        "File-upload entries cannot be automatically re-ingested — they require re-upload."
    )

    try:
        ri_manifests = [m for m in _load_manifests() if m.get("status") != "archived"]
    except Exception as exc:
        st.error(f"Could not load manifests: {exc}")
        ri_manifests = []

    if not ri_manifests:
        st.info("No active manifests found.")
    else:
        ri_options = {
            f"{m['name']} ({(m.get('created_at') or '')[:10]})": m["manifest_id"]
            for m in ri_manifests
        }
        ri_label = st.selectbox("Select manifest", list(ri_options.keys()), key="ri_sel")
        ri_mid   = ri_options[ri_label]

        from pipeline.manifests import get_manifest_manager as _gmm
        ri_mf = _gmm().get_manifest(ri_mid)
        ri_entries = ri_mf.get("entries") or [] if ri_mf else []

        n_confluence = sum(1 for e in ri_entries if e.get("source_type") == "confluence")
        n_other      = len(ri_entries) - n_confluence

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total entries", len(ri_entries))
        mc2.metric("Confluence (re-ingestable)", n_confluence)
        mc3.metric("Other (manual re-upload needed)", n_other)

        if n_other > 0:
            st.warning(
                f"{n_other} entry/entries are not Confluence sources (file uploads, JSONL, URLs). "
                "These will be skipped — re-upload them manually via the **Add Document** page."
            )

        ri_extra_tags = st.text_input("Extra tags to apply (comma-separated)", key="ri_tags")

        if n_confluence == 0:
            st.info("No Confluence sources in this manifest to re-ingest.")
        else:
            from pipeline.config import settings as _cfg
            if not _cfg.confluence_base_url or not _cfg.confluence_api_token:
                st.warning(
                    "Confluence credentials not configured. "
                    "Set `CONFLUENCE_BASE_URL` and `CONFLUENCE_API_TOKEN` in your `.env` file."
                )
            else:
                if st.button(
                    f"Re-ingest {n_confluence} Confluence source(s)",
                    key="ri_btn",
                    type="primary",
                ):
                    progress_bar = st.progress(0.0)
                    status_msg   = st.empty()

                    def _progress_cb(done: int, total: int) -> None:
                        progress_bar.progress(done / max(total, 1))
                        status_msg.caption(f"Re-ingesting… {done}/{total}")

                    extra_tags = [t.strip() for t in ri_extra_tags.split(",") if t.strip()]
                    with st.spinner("Re-ingesting…"):
                        try:
                            result = _gmm().ingest_from_manifest(
                                manifest_id=ri_mid,
                                extra_tags=extra_tags or None,
                                progress_cb=_progress_cb,
                            )
                        except Exception as exc:
                            st.error(f"Re-ingest failed: {exc}")
                            result = None

                    if result is not None:
                        status_msg.empty()
                        progress_bar.progress(1.0)
                        st.success(
                            f"Done — {result['ingested']} source(s) re-ingested, "
                            f"{result['skipped']} skipped."
                        )
                        for err in result.get("errors") or []:
                            st.error(err)
                        if result["ingested"] > 0:
                            st.info("Go to **Review Queue** to approve and push the re-ingested docs.")
                        st.cache_data.clear()
