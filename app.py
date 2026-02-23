import os
import tempfile
import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag import RAG

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

/* Glass Dark Card */
.author-info {
    padding: 1.2rem;
    border-radius: 12px;
    margin: 1rem 0;

    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);

    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Text Colors */
.author-info h3 {
    color: white;
    margin-bottom: 0.3rem;
}

.author-info p {
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.8rem;
}

/* Hire Button */
.hire-button {
    display: inline-block;
    padding: 0.55rem 1.1rem;
    background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
    color: white !important;

    text-decoration: none;
    border-radius: 8px;
    font-weight: 600;

    transition: all 0.2s ease;
}

.hire-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(255,75,75,0.35);
}

/* Interactive Boxes */
.interactive-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;

    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
    
if "rag" not in st.session_state:
    st.session_state.rag = RAG(st.session_state.vector_store)
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_document(uploaded_file):
    """Process an uploaded document."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write the uploaded file data to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Process the document
        chunks = st.session_state.processor.process_document(tmp_path)
        
        # Add to vector store
        st.session_state.vector_store.add_documents(chunks)
        
        # Add to uploaded files list if not already there
        if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "chunks": len(chunks)
            })
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

def clear_documents():
    """Clear all uploaded documents and reset the vector store."""
    st.session_state.vector_store = VectorStore()
    st.session_state.uploaded_files = []
    st.session_state.rag = RAG(st.session_state.vector_store)

# App header with author info
st.title("🤖 Document Assistant")
st.markdown("""
<div class="author-info">
    <h3>Created by Sourabh Patil</h3>
    <p>AI/ML Engineer | Python Developer | Data Scientist</p>
    <a href="https://www.linkedin.com/in/sourabh-patil-60a0b7192/" target="_blank" class="hire-button">🚀 Hire Me!</a>
</div>
""", unsafe_allow_html=True)

st.write("Upload documents and ask questions about their content using advanced AI technology")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                process_document(uploaded_file)
    
    st.header("Settings")
    with st.expander("Advanced Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        
        if chunk_size != st.session_state.processor.chunk_size or chunk_overlap != st.session_state.processor.chunk_overlap:
            st.session_state.processor = DocumentProcessor(chunk_size, chunk_overlap)
    
    if st.button("Clear All Documents", type="secondary"):
        clear_documents()
    
    st.header("Uploaded Documents")
    if st.session_state.uploaded_files:
        for doc in st.session_state.uploaded_files:
            with st.container():
                st.markdown(f"""
                <div class="interactive-box">
                    📄 <b>{doc['name']}</b><br>
                    Chunks: {doc['chunks']} | Size: {doc['size']/1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No documents uploaded")

# Main content
col1, col2 = st.columns([2, 1])

with col1:

    st.header("💬 Chat")

    # ✅ Chat container (messages at top)
    chat_container = st.container()

    with chat_container:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])

            with st.chat_message("assistant"):
                st.write(chat["answer"])

    # ✅ Input ALWAYS stays at bottom
    prompt = st.chat_input("Ask something about your documents...")

    if prompt:

        if not st.session_state.uploaded_files:
            st.warning("Please upload and process a document first.")
            st.stop()

        # ✅ Show user message instantly
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        # ✅ Generate response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer = st.session_state.rag.generate_answer(prompt)
                    st.write(answer)

        # ✅ Save history
        st.session_state.chat_history.append({
            "question": prompt,
            "answer": answer
        })

        st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💭 Conversation History")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"""
                <div class="interactive-box" style="border-color: #2196F3;">
                    <b>You:</b> {chat['question']}<br><br>
                    <b>Assistant:</b> {chat['answer']}
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.header("🔍 RAG Architecture")
    with st.expander("How it works", expanded=True):
        st.write("""
        This application uses a Retrieval Augmented Generation (RAG) architecture:
        
        1. **Document Processing**: Documents are chunked into smaller segments
        2. **Vector Embeddings**: Chunks are converted to vector embeddings
        3. **FAISS Index**: Embeddings are stored in a FAISS vector index
        4. **Semantic Search**: User queries retrieve the most relevant chunks
        5. **Generation**: Retrieved context is sent to an LLM to generate answers
        """)
    
    # System stats
    st.header("📊 System Statistics")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        if st.session_state.uploaded_files:
            total_chunks = sum(doc["chunks"] for doc in st.session_state.uploaded_files)
            st.metric("📚 Documents", len(st.session_state.uploaded_files))
    with col_stats2:
        if st.session_state.uploaded_files:
            st.metric("🧩 Total Chunks", total_chunks)
        else:
            st.metric("📚 Documents", 0)
            st.metric("🧩 Total Chunks", 0) 