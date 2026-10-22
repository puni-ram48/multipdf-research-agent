import streamlit as st


def render_layout_viewer():
    summary = st.session_state.get("ingest_summary")
    if not summary:
        return

    with st.expander("🗺️ Document Processing & Layout Map", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", summary.get("total_chunks", 0))
        col2.metric("Abstracts Found", summary.get("abstracts_indexed", 0))
        col3.metric("Tables Found", summary.get("tables_indexed", 0))

        st.caption(f"Embedding model active: `{summary.get('embedding_model', 'N/A')}`")

        for file_info in summary.get("file_summaries", []):
            st.markdown(f"**📄 {file_info['filename']}** ({file_info['total_pages']} pages)")
            st.text(f"Abstract Detected: {'Yes' if file_info['abstract_found'] else 'No'}")
            st.text(f"Tables Extracted: {file_info['tables_found']}")