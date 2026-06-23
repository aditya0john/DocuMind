"""
app.py
------
DocuMind — Multi-Document RAG Agent
Streamlit UI entry point.

Run with:
  streamlit run app.py
"""

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_documents
from src.agent import create_agent, format_chat_history

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; }
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stSidebarContent"] { padding-top: 1.5rem; }
    .source-chip {
        display: inline-block;
        background: #e8f4fd;
        color: #1a6ea8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .stat-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 DocuMind")
    st.caption("Multi-document RAG agent powered by MistralAi + ChromaDB")
    st.divider()

    # API key check
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        api_key = st.text_input("MISTRAL API KEY", type="password", placeholder="sk-...")
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key

    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    ingest_btn = st.button(
        "⚡ Ingest Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if ingest_btn and uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    file_paths = []
                    for f in uploaded_files:
                        path = os.path.join(tmp_dir, f.name)
                        with open(path, "wb") as out:
                            out.write(f.getbuffer())
                        file_paths.append(path)

                    ingest_documents(file_paths)

                # Persist file names in session
                st.session_state["docs_ingested"] = True
                st.session_state["ingested_files"] = [f.name for f in uploaded_files]
                st.success(f"✅ {len(uploaded_files)} document(s) ready!")

            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    # Show loaded docs
    if st.session_state.get("ingested_files"):
        st.divider()
        st.caption("**Active documents**")
        for fname in st.session_state["ingested_files"]:
            st.markdown(f"📄 `{fname}`")

    st.divider()

    # Stats + controls
    col1, col2 = st.columns(2)
    with col1:
        msg_count = len(st.session_state.get("messages", []))
        st.markdown(
            f'<div class="stat-box"><b>{msg_count}</b><br>'
            f'<span style="font-size:.75rem;color:#888">messages</span></div>',
            unsafe_allow_html=True,
        )
    with col2:
        doc_count = len(st.session_state.get("ingested_files", []))
        st.markdown(
            f'<div class="stat-box"><b>{doc_count}</b><br>'
            f'<span style="font-size:.75rem;color:#888">documents</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if st.button("🔄 Reset all", use_container_width=True):
        for key in ["messages", "docs_ingested", "ingested_files"]:
            st.session_state.pop(key, None)
        # Remove ChromaDB
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.rerun()


# ── Main area ──────────────────────────────────────────────────────────
st.title("Ask your documents anything")

if not os.getenv("MISTRAL_API_KEY"):
    st.warning("⚠️ Add your Mistral API key in the sidebar to get started.")
    st.stop()

if not st.session_state.get("docs_ingested"):
    # Empty state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nUpload one or more PDFs in the sidebar")
    with col2:
        st.info("**Step 2**\n\nClick ⚡ Ingest Documents to build the knowledge base")
    with col3:
        st.info("**Step 3**\n\nAsk questions — the agent will search, summarize, and compare")

    st.markdown("---")
    st.subheader("Example questions you can ask")
    examples = [
        "What are the main findings in the report?",
        "Summarize the document named 'research.pdf'",
        "Compare how both documents discuss climate change",
        "What revenue figures are mentioned?",
        "What methodology was used in the study?",
    ]
    for ex in examples:
        st.markdown(f"- _{ex}_")
    st.stop()

# ── Chat history ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask anything about your documents..."):
    # Append and show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = create_agent()
                history = format_chat_history(st.session_state["messages"][:-1])

                result = agent.invoke({
                    "input": user_input,
                    "chat_history": history,
                })
                response = result["output"]

            except Exception as e:
                response = (
                    f"❌ Something went wrong: `{e}`\n\n"
                    "Check that your API key is valid and documents are ingested."
                )

        st.markdown(response)

    # Save assistant message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response,
    })
