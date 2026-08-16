"""
app.py - Streamlit Interface for Meridian Components Supply Chain RAG

Features:
- Upload one or more PDF files
- Index button with chunk count reporting
- Question input with preset test questions
- GPT-4o answers with source citations (doc name + page number)
- ChromaDB persistence across sessions
"""

import os
import shutil
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment
load_dotenv()

from ingest import ingest_documents, get_chroma_collection, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from rag import query_rag, LLM_MODEL, DEFAULT_TOP_K

# Page configuration
st.set_page_config(
    page_title="Meridian Supply Chain Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .source-tag {
        font-weight: 600;
        color: #0284C7;
        font-size: 0.9rem;
    }
    .excerpt-text {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 4px;
        font-style: italic;
    }
    .stButton>button {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def get_indexed_stats():
    """Returns the count of chunks in ChromaDB or 0 if unavailable."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return 0, False
        _, collection = get_chroma_collection(persist_directory=CHROMA_DIR)
        count = collection.count()
        return count, True
    except Exception:
        return 0, False


# Session State Management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/supply-chain.png", width=64)
    st.title("System Status")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        user_key = st.text_input("OpenAI API Key", type="password", help="Enter key if not set in .env")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            st.success("API key set for session!")

    chunk_count, is_connected = get_indexed_stats()

    if is_connected and chunk_count > 0:
        st.success(f"● Knowledge Store: Active\n\n**{chunk_count} chunks** persisted in ChromaDB")
    else:
        st.warning("● Knowledge Store: Empty / Pending Ingestion")

    st.markdown("---")
    st.subheader("Configuration")
    st.write(f"**Embedding Model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM Model:** `{LLM_MODEL}`")
    st.write(f"**Chunk Size:** `{CHUNK_SIZE}` chars")
    st.write(f"**Chunk Overlap:** `{CHUNK_OVERLAP}` chars")
    
    top_k_val = st.slider("Top K Retrieved Chunks", min_value=2, max_value=10, value=DEFAULT_TOP_K, step=1)
    
    st.markdown("---")
    st.subheader("📁 Upload & Index Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more supply chain PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload Meridian Supply Chain Review & Procurement Policy Handbook"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        index_uploaded = st.button("Index Uploaded", type="primary", use_container_width=True)
    with col_btn2:
        index_data_folder = st.button("Index data/ PDFs", use_container_width=True)

    if index_uploaded:
        if not uploaded_files:
            st.error("Please select at least one PDF file first.")
        else:
            with st.spinner("Processing and indexing uploaded documents into ChromaDB..."):
                try:
                    temp_dir = tempfile.mkdtemp()
                    temp_paths = []
                    for uf in uploaded_files:
                        fp = os.path.join(temp_dir, uf.name)
                        with open(fp, "wb") as f:
                            f.write(uf.getbuffer())
                        temp_paths.append(fp)

                    result = ingest_documents(pdf_paths=temp_paths, persist_directory=CHROMA_DIR)
                    st.success(f"✓ {result['files']} files processed, {result['chunks']} chunks stored.")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if index_data_folder:
        with st.spinner("Indexing default PDFs from data/ directory..."):
            try:
                result = ingest_documents(persist_directory=CHROMA_DIR)
                st.success(f"✓ {result['files']} files processed, {result['chunks']} chunks stored.")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")


# Main Content Area
st.markdown('<div class="main-header">Meridian Components — Supply Chain RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Internal intelligence assistant for cross-document procurement policy rules and quarterly performance metrics.</div>', unsafe_allow_html=True)

# Assignment Test Questions Quick Picker
SAMPLE_QUESTIONS = [
    "-- Select a test question or write custom --",
    "1. Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?",
    "2. How many line stoppages happened in Q1, what was the total downtime, and what caused them?",
    "3. What is the approval authority for a purchase order worth ₹1.4 crore?",
    "4. What are the four supplier classification categories, and what qualifies a supplier as Critical?",
    "5. Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?",
    "6. The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?",
    "7. Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?",
    "8. Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?",
    "9. Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?",
    "10. What is the annual salary of the Head of Procurement? (Deliberate trap question)"
]

selected_sample = st.selectbox("🎯 Quick Test Questions (Assignment Section 7):", SAMPLE_QUESTIONS)

default_text = ""
if selected_sample and selected_sample != SAMPLE_QUESTIONS[0]:
    # Strip leading number and dot if present
    default_text = selected_sample.split(". ", 1)[-1].replace(" (Deliberate trap question)", "")

# Question Input Form
with st.form(key="query_form", clear_on_submit=False):
    user_query = st.text_area("Ask a question:", value=default_text, height=90, placeholder="e.g. Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?")
    submitted = st.form_submit_button("Submit Question", type="primary")

if submitted and user_query.strip():
    if chunk_count == 0:
        st.warning("⚠️ No documents indexed yet. Please click 'Index data/ PDFs' in the sidebar or upload the PDFs first.")
    else:
        with st.spinner("Searching documents and generating verified answer with GPT-4o..."):
            try:
                response = query_rag(question=user_query.strip(), top_k=top_k_val)
                st.session_state.chat_history.insert(0, {
                    "question": user_query.strip(),
                    "answer": response["answer"],
                    "sources": response.get("sources", [])
                })
            except Exception as e:
                st.error(f"Error during query execution: {e}")

# Display Results & History
if st.session_state.chat_history:
    for idx, item in enumerate(st.session_state.chat_history):
        st.markdown(f"### 💬 Question: {item['question']}")
        
        # Answer display
        st.markdown("#### 💡 Answer:")
        st.markdown(item["answer"])

        # Sources display
        if item.get("sources"):
            st.markdown("#### 📚 Sources & Citations:")
            cols = st.columns(min(len(item["sources"]), 3))
            for i, src in enumerate(item["sources"]):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-tag">📄 {src['file']}</span><br>
                        <strong>Page {src['page']}</strong>
                        <div class="excerpt-text">"{src.get('excerpt', '')}"</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
