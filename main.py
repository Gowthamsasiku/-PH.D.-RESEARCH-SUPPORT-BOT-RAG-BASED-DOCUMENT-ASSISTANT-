import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- CONFIGURATION ----------
PDF_FOLDER = r"C:\Users\HP\Desktop\Project\-PH.D.-RESEARCH-SUPPORT-BOT-RAG-BASED-DOCUMENT-ASSISTANT-\pdfs"
GOOGLE_API_KEY = "AIzaSyAOn7HhN8Mfv46e6RPYuQ31BDZax_a1mXE"  # Replace with your key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Set as environment variable

# ---------- FUNCTION TO LOAD PDFs ----------
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




# ---------- MAIN ----------
if __name__ == "__main__":
    docs = load_all_pdfs()
    # docs now contains all PDFs with their text for further processing
    print(f"Total PDFs loaded: {len(docs)}\n")
    

# ---------- 2. SPLIT DOCUMENTS ----------
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    """
    Split each document's text into smaller chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        doc_chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "filename": doc["filename"],
                "chunk_index": i,
                "text": chunk
            })
    return chunks

def view_chunks(chunks, preview_chars=500, max_chunks=10):
    """
    Nicely display chunks for review.
    """
    print(f"\nTotal chunks available: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks):
        if i >= max_chunks:
            print(f"...and {len(chunks) - max_chunks} more chunks.")
            break
        print(f"Filename: {chunk['filename']} | Chunk: {chunk['chunk_index']}")
        print(chunk["text"][:preview_chars] + "...")
        print("-" * 80)
chunks = split_documents(docs)

    # Step 3: View first few chunks
view_chunks(chunks, preview_chars=500, max_chunks=50)