"""
ingestion.py
------------
Full RAG ingestion pipeline:
  PDF file -> raw text -> chunks -> Mistral embeddings -> Pinecone vector store

WHY PINECONE instead of ChromaDB?
  ChromaDB writes to local disk (./chroma_db) which is wiped on every
  cloud redeploy. Pinecone is managed cloud storage — vectors persist forever.
"""

import os
import time
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIM   = 1024        # mistral-embed output dimension — fixed, do not change
INDEX_NAME      = "documind"  # Pinecone index name — auto-created on first ingest


# ── Pinecone index setup ───────────────────────────────────────────────
def _ensure_index():
    """
    Create the Pinecone index if it doesn't exist yet.
    Safe to call every time — skips creation if index already exists.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until index is ready before writing to it
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index ready ✅")
    else:
        print(f"Using existing Pinecone index '{INDEX_NAME}'")


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

    chunk_size=800    -> fits comfortably in an embedding window
    chunk_overlap=100 -> avoids cutting sentences mid-thought
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# ── Step 3: Embed + upsert to Pinecone ────────────────────────────────
def ingest_documents(file_paths: list[str]) -> PineconeVectorStore:
    """
    Run the full pipeline for a list of PDF file paths:
      1. Load all PDFs
      2. Chunk every document
      3. Embed chunks with mistral-embed
      4. Upsert to Pinecone (persists in cloud — survives redeploys)
    Returns the populated PineconeVectorStore.
    """
    all_chunks: list[Document] = []

    for path in file_paths:
        print(f"Loading: {os.path.basename(path)}")
        docs   = load_pdf(path)
        chunks = chunk_documents(docs)
        print(f"   -> {len(docs)} pages, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    _ensure_index()  # create index if this is the first run

    print(f"\nEmbedding {len(all_chunks)} total chunks with mistral-embed...")
    embeddings  = MistralAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore.from_documents(
        documents  = all_chunks,
        embedding  = embeddings,
        index_name = INDEX_NAME,
    )

    print(f"Upserted to Pinecone index '{INDEX_NAME}' ✅")
    return vectorstore


# ── Loader (for when index already has vectors) ────────────────────────
def load_vectorstore() -> PineconeVectorStore:
    """Connect to the existing Pinecone index — no re-embedding needed."""
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore(
        index_name = INDEX_NAME,
        embedding  = embeddings,
    )