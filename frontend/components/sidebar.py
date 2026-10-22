import streamlit as st
from api_client import api_client


def render_sidebar():
    with st.sidebar:
        st.title("📄 MultiPDF Research Agent")
        st.caption("Docling Parsing · BGE Embeddings · Gemini 2.5 Flash")
        st.divider()

        # Backend Connection Monitor
        is_alive = api_client.health_check()
        if is_alive:
            st.success("🟢 Backend Connected")
        else:
            st.error("🔴 Backend Offline (check FastAPI server)")

        st.divider()

        # 1. File Upload
        st.header("1️⃣ Upload Papers")
        files = st.file_uploader(
            "Upload one or more PDF research papers",
            type="pdf",
            accept_multiple_files=True,
            help="Docling adaptive engine will parse multi-column layouts and tables."
        )

        if st.button("⚙️ Process PDFs", use_container_width=True, type="primary"):
            if not files:
                st.error("Please upload at least one PDF first.")
            elif not is_alive:
                st.error("Cannot process: FastAPI backend service is offline.")
            else:
                with st.spinner("Processing documents with Docling & generating vector index..."):
                    try:
                        current_session = st.session_state.get("session_id", None)
                        result = api_client.ingest_pdfs(files, session_id=current_session)

                        st.session_state["session_id"] = result["session_id"]
                        st.session_state["indexed"] = True
                        st.session_state["ingest_summary"] = result
                        st.session_state["chat_history"] = []

                        st.success(f"✅ Indexed {result['total_documents']} file(s) across {result['total_chunks']} chunks!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.divider()

        # 2. Query Mode & Memory Controls
        st.header("2️⃣ Memory & Session")
        session_id = st.session_state.get("session_id", "Not active")
        st.caption(f"Session ID: `{session_id}`")

        if st.button("🧹 Clear Chat History", use_container_width=True):
            if session_id != "Not active":
                api_client.clear_memory(session_id)
            st.session_state["chat_history"] = []
            st.rerun()