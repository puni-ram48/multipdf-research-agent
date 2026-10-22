import streamlit as st
from api_client import api_client
from components.sidebar import render_sidebar
from components.chat_box import render_chat_box
from components.layout_viewer import render_layout_viewer

st.set_page_config(
    page_title="MultiPDF Agent",
    page_icon="📄",
    layout="wide"
)

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Render Sidebar Component
render_sidebar()

# Main Application Interface
st.title("💬 Research Paper MultiPDF Chatbot")

if not st.session_state["indexed"]:
    st.info("👈 Please upload research papers in the sidebar and click **Process PDFs** to get started.")
else:
    # Display Document Layout Processing Information
    render_layout_viewer()
    st.divider()

    # Display Chat History
    render_chat_box()

    # Query Input Area
    if prompt := st.chat_input("Ask a question about your research papers..."):
        session_id = st.session_state["session_id"]

        with st.spinner("Searching vector index & generating answer..."):
            try:
                response = api_client.send_chat_question(session_id, prompt)

                st.session_state["chat_history"].append({
                    "question": prompt,
                    "answer": response["answer"],
                    "sources": response.get("sources", []),
                    "faithfulness_score": response.get("faithfulness_score")
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error generating answer: {e}")