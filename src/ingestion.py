"""
ingestion.py
------------
Full RAG ingestion pipeline:
  PDF file -> raw text -> chunks -> Mistral embeddings -> ChromaDB vector store
"""

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "mistral-embed"


# ── Step 1: Extract text from PDF ─────────────────────────────────────
def load_pdf(file_path: str) -> list[Document]:
    """
    Open a PDF and extract page text as LangChain Document objects.
    Each page becomes one Document with source + page metadata.
    """
    doc = fitz.open(file_path)
    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:  # skip blank pages
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": page_num + 1,
                    "file_path": file_path,
                }
            ))

    doc.close()
    return documents


# ── Step 2: Chunk the documents ────────────────────────────────────────
def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split large pages into smaller overlapping chunks so that embeddings
    are focused and retrieval is precise.

    chunk_size=800   -> fits comfortably in an embedding window
    chunk_overlap=100 -> avoids cutting sentences mid-thought
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# ── Step 3: Embed + store in ChromaDB ─────────────────────────────────
def ingest_documents(file_paths: list[str]) -> Chroma:
    """
    Run the full pipeline for a list of PDF file paths:
      1. Load all PDFs
      2. Chunk every document
      3. Embed chunks with mistral-embed
      4. Persist to local ChromaDB
    Returns the populated Chroma vectorstore.
    """
    all_chunks: list[Document] = []

    for path in file_paths:
        print(f"Loading: {os.path.basename(path)}")
        docs = load_pdf(path)
        chunks = chunk_documents(docs)
        print(f"   -> {len(docs)} pages, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nEmbedding {len(all_chunks)} total chunks with mistral-embed...")
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"Stored in ChromaDB at '{CHROMA_PATH}'")
    return vectorstore


# ── Loader (for when the DB already exists) ────────────────────────────
def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )