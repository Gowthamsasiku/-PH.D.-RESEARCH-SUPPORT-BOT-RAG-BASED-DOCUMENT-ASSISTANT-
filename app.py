import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

# ---------- HELPER FUNCTIONS ----------
def load_all_pdfs(folder_path):
    """Load all PDF files from the specified folder and return their text content."""
    all_docs = []
    try:
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            st.error(f"No PDF files found in {folder_path}")
            return all_docs
            
        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                all_docs.append({"filename": filename, "text": text})
                st.sidebar.success(f"✅ Loaded: {filename}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {filename}: {str(e)}")
    except Exception as e:
        st.error(f"Error accessing PDF folder: {str(e)}")
    return all_docs

def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    """Split documents into chunks for processing."""
    if not docs:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        try:
            for i, chunk in enumerate(splitter.split_text(doc["text"])):
                chunks.append({
                    "filename": doc["filename"],
                    "chunk_index": i,
                    "text": chunk
                })
        except Exception as e:
            st.error(f"Error processing {doc.get('filename', 'unknown')}: {str(e)}")
    return chunks

def store_chunks_in_chroma(chunks, persist_directory):
    """Store document chunks in Chroma vector store."""
    if not chunks:
        return None
        
    try:
        # Clear existing Chroma database if it exists
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"filename": c["filename"], "chunk_index": c["chunk_index"]} for c in chunks]

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create a new Chroma instance with explicit settings
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Force persistence
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        # Clean up if there was an error
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
        return None

# ---------- CONFIGURATION ----------
# Load environment variables
print("Current working directory:", os.getcwd())
print("Loading .env file...")
load_dotenv()

# Debug: Print all environment variables (be careful with sensitive data)
print("Environment variables loaded:")
for key, value in os.environ.items():
    if 'API' in key or 'KEY' in key:
        print(f"{key} = {'*' * 8}{value[-4:] if value else ''}")

# Constants
PDF_FOLDER = r"C:\Users\Sridevi\Downloads\-PH.D.-RESEARCH-SUPPORT-BOT-RAG-BASED-DOCUMENT-ASSISTANT--main\pdfs"
CHROMA_DB_DIR = r"C:\Users\Sridevi\Downloads\-PH.D.-RESEARCH-SUPPORT-BOT-RAG-BASED-DOCUMENT-ASSISTANT--main\chroma_db"

# Get and clean the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip('\"\'')

if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found in .env file. "
        "Please make sure you have a .env file with GOOGLE_API_KEY=your_key_here"
    )

# Set the API key in the environment for other libraries
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)

# ---------- STREAMLIT UI ----------
# Set page config
st.set_page_config(
    page_title="Research Support Bot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'docs' not in st.session_state:
    st.session_state.docs = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Research Support Bot")
    st.markdown("---")
    
    # Initialize button
    if st.button("🔄 Initialize System"):
        with st.spinner("Initializing RAG system..."):
            try:
                # Clear previous state
                st.session_state.docs = []
                st.session_state.chunks = []
                st.session_state.vectorstore = None
                
                # Load and process PDFs
                st.sidebar.info("Loading PDFs...")
                st.session_state.docs = load_all_pdfs(PDF_FOLDER)
                
                if st.session_state.docs:
                    st.sidebar.info("Processing documents...")
                    st.session_state.chunks = split_documents(st.session_state.docs)
                    
                    if st.session_state.chunks:
                        st.sidebar.info("Creating vector store...")
                        st.session_state.vectorstore = store_chunks_in_chroma(
                            st.session_state.chunks, 
                            CHROMA_DB_DIR
                        )
                        
                        if st.session_state.vectorstore:
                            st.session_state.initialized = True
                            st.sidebar.success("✅ RAG system initialized successfully!")
                            st.rerun()
                
                if not st.session_state.initialized:
                    st.sidebar.error("❌ Failed to initialize the system. Please check the error messages above.")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error initializing system: {str(e)}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This is a Research Support Bot that helps you find answers from your research documents using RAG (Retrieval-Augmented Generation).
    
    **How to use:**
    1. Place your PDFs in the 'pdfs' folder
    2. Click 'Initialize System' to process the documents
    3. Ask questions about your research
    """)

# ---------- MAIN CONTENT ----------
st.title("📚 Research Support Bot")
st.markdown("Ask questions about your research documents and get AI-powered answers!")

# Show initialization status
if not st.session_state.initialized:
    st.warning(
        "⚠️ Please click the 'Initialize System' button in the sidebar to get started. "
        "Make sure you have placed your PDFs in the 'pdfs' folder."
    )
    st.stop()

# Display document info
if st.session_state.docs:
    with st.expander("📂 Loaded Documents", expanded=False):
        for doc in st.session_state.docs:
            st.markdown(f"- {doc['filename']} ({len(doc['text'])} characters)")

# Chat interface
st.markdown("### Ask a Question")
user_question = st.text_input(
    "Enter your research question:", 
    "",
    placeholder="E.g., What are the key findings in these papers?"
)

if user_question:
    with st.spinner("Searching for answers..."):
        try:
            # Get response from RAG system
            results = st.session_state.vectorstore.similarity_search(
                user_question, 
                k=min(3, len(st.session_state.chunks))  # Ensure k is not larger than available chunks
            )
            
            if not results:
                st.warning("No relevant information found in the documents.")
            else:
                context = "\n\n".join([r.page_content for r in results])
                
                # Generate response using Gemini
                prompt = f"""You are a helpful research assistant. Use the provided context from academic documents to answer the question clearly and factually.

Context:
{context}

Question: {user_question}

Answer:"""
                
                try:
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    response = model.generate_content(prompt)
                    
                    # Display response
                    st.markdown("### Answer")
                    st.markdown(response.text)
                    
                    # Display sources
                    st.markdown("### Sources")
                    for i, r in enumerate(results, 1):
                        with st.expander(f"Source {i}: {r.metadata['filename']} (Chunk {r.metadata['chunk_index'] + 1})"):
                            st.text(r.page_content)
                            
                except Exception as genai_error:
                    st.error(f"Error generating response: {str(genai_error)}")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add some spacing at the bottom
st.markdown("\n\n---")
st.caption("Research Support Bot | Powered by Streamlit, LangChain, and Gemini")

if __name__ == "__main__":
    # The main app is already running through Streamlit
    pass
