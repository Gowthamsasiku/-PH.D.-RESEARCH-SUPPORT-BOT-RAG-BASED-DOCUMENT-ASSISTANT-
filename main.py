import os
import numpy as np
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# ---------- CONFIG ----------
PDF_FOLDER = r"C:\Users\HP\Desktop\Project\-PH.D.-RESEARCH-SUPPORT-BOT-RAG-BASED-DOCUMENT-ASSISTANT-\pdfs"
CHROMA_DB_DIR = r"C:\Users\HP\Desktop\Project\-PH.D.-RESEARCH-SUPPORT-BOT-RAG-BASED-DOCUMENT-ASSISTANT-\chroma_db"
GOOGLE_API_KEY = "Your API KEY"   # 🔑 Replace with your key

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)


# ---------- 1. LOAD PDFs ----------
def load_all_pdfs(folder_path=PDF_FOLDER):
    all_docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        all_docs.append({"filename": filename, "text": text})
    return all_docs


# ---------- 2. SPLIT DOCUMENTS ----------
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "filename": doc["filename"],
                "chunk_index": i,
                "text": chunk
            })
    return chunks


# ---------- 3. STORE CHUNKS IN CHROMA ----------
def store_chunks_in_chroma(chunks):
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"filename": c["filename"], "chunk_index": c["chunk_index"]} for c in chunks]

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print(f"✅ Stored {len(chunks)} chunks in Chroma DB at {CHROMA_DB_DIR}")
    return vectorstore


# ---------- 4. RETRIEVAL + GEMINI ----------
def retrieve_and_answer(vectorstore, query):
    results = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
You are a research assistant. Use the provided context from academic documents to answer the question clearly and factually.

Context:
{context}

Question: {query}

Answer:
"""
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    print("\n🧠 --- Gemini 2.5 Flash Answer ---")
    print(response.text)

    print("\n📄 --- Source Chunks ---")
    for r in results:
        print(f"File: {r.metadata['filename']} | Chunk: {r.metadata['chunk_index']}")
    print("-" * 80)


# ---------- MAIN ----------
if __name__ == "__main__":
    print("📚 Loading PDF documents...")
    docs = load_all_pdfs()
    print(f"✅ Total PDFs loaded: {len(docs)}")

    print("\n🔍 Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}")

    print("\n💾 Storing chunks in Chroma Vector DB...")
    vectorstore = store_chunks_in_chroma(chunks)

    print("\n✨ RAG System Ready! Ask research questions below (type 'exit' to quit).")
    while True:
        question = input("\n❓ Enter your research question: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting RAG Assistant.")
            break
        retrieve_and_answer(vectorstore, question)
