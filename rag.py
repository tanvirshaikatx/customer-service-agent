# rag.py
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os

def create_vector_store(pdf_path='data/menu.pdf'):
    # Verify PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
        print("Please create a 'data' directory and add 'menu.pdf'")
        return

    # Load and process PDF
    print("📖 Processing PDF...")
    reader = PdfReader(pdf_path)
    raw_text = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(raw_text)

    # Initialize embeddings
    print("🔧 Loading embeddings...")
    embeddings = OllamaEmbeddings(model="llama3.2", temperature=0.7)
    if not embeddings:
        print("❌ Error: Failed to load embeddings")
        return
    
    # Create and save vector store
    print("🧠 Creating vector store...")
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    vectorstore.save_local("faiss_index")
    print("✅ Vectorstore created at 'faiss_index'")

if __name__ == "__main__":
    create_vector_store()