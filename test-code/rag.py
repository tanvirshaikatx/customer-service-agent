# rag.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

def create_vector_store(pdf_path='data/menu.pdf'):
    load_dotenv()
    
    reader = PdfReader(pdf_path)
    raw_text = ''
    for page in reader.pages:
        raw_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ Vectorstore saved at 'faiss_index'.")

if __name__ == "__main__":
    create_vector_store()
