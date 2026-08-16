"""
ingest.py - Document Ingestion Pipeline for Meridian Components Supply Chain RAG

Loads PDFs, chunks text with RecursiveCharacterTextSplitter (800-1200 chars),
generates embeddings using OpenAI text-embedding-3-small, and stores them in ChromaDB.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# Constants based on technical requirements
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "supplychain_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000        # Within mandatory 800-1200 chars range
CHUNK_OVERLAP = 150      # Within mandatory 100-200 chars range


def get_openai_api_key() -> str:
    """Retrieve OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Check standard fallback or alert user
        raise ValueError(
            "OPENAI_API_KEY is not set. Please add it to your .env file or environment."
        )
    return api_key


def get_chroma_collection(persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    """Initializes or connects to the persisted ChromaDB collection."""
    api_key = get_openai_api_key()
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
        metadata={"description": "Meridian Components Supply Chain Documents"}
    )
    return client, collection


def load_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page by page from a PDF file using pypdf.
    Returns a list of page dictionaries with content and metadata.
    """
    path_obj = Path(file_path)
    file_name = path_obj.name
    pages_data = []

    reader = pypdf.PdfReader(str(file_path))
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_data.append({
                "text": text.strip(),
                "file_name": file_name,
                "file_path": str(file_path),
                "page": page_idx + 1  # 1-indexed page numbering
            })

    return pages_data


def chunk_documents(pages_data: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Splits page texts into chunks using RecursiveCharacterTextSplitter.
    Preserves document and page metadata on each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_counter = 0

    for page_info in pages_data:
        text = page_info["text"]
        split_texts = splitter.split_text(text)

        for i, split_text in enumerate(split_texts):
            chunk_counter += 1
            chunks.append({
                "id": f"{page_info['file_name']}_p{page_info['page']}_c{i+1}",
                "text": split_text,
                "metadata": {
                    "file_name": page_info["file_name"],
                    "source": page_info["file_name"],
                    "page": page_info["page"],
                    "chunk_index": i + 1
                }
            })

    return chunks


def ingest_documents(pdf_paths: Optional[List[str]] = None, persist_directory: str = CHROMA_DIR, reset_collection: bool = True) -> Dict[str, Any]:
    """
    End-to-end ingestion pipeline:
    1. Discovers PDF files if not provided (defaulting to ./data/*.pdf)
    2. Reads and extracts page texts
    3. Chunks text using recursive character splitting
    4. Stores and persists embeddings in ChromaDB
    5. Returns summary metrics: {"files": N, "chunks": M}
    """
    if pdf_paths is None:
        pdf_paths = glob.glob("./data/*.pdf")
        if not pdf_paths:
            # Check fallback directory
            pdf_paths = glob.glob("../data/*.pdf")

    if not pdf_paths:
        raise FileNotFoundError("No PDF files found to ingest. Please place PDF files in the data/ directory.")

    print(f"[*] Found {len(pdf_paths)} PDF(s) to ingest: {[Path(p).name for p in pdf_paths]}")

    # Load all pages
    all_pages = []
    for pdf_path in pdf_paths:
        pages = load_pdf_pages(pdf_path)
        all_pages.extend(pages)
        print(f"    - Loaded {len(pages)} pages from {Path(pdf_path).name}")

    # Chunk all pages
    chunks = chunk_documents(all_pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"[*] Generated {len(chunks)} text chunks (Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")

    # Connect to ChromaDB
    client, collection = get_chroma_collection(persist_directory=persist_directory)

    if reset_collection:
        # Delete existing collection items to ensure clean state
        try:
            client.delete_collection(name=COLLECTION_NAME)
            # Recreate with embedding function
            client, collection = get_chroma_collection(persist_directory=persist_directory)
        except Exception:
            pass

    # Batch upsert to ChromaDB
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    result = {
        "files": len(pdf_paths),
        "chunks": len(chunks),
        "file_names": [Path(p).name for p in pdf_paths],
        "persist_dir": persist_directory
    }

    print(f"[✓] Ingestion complete: {result['files']} files processed, {result['chunks']} chunks stored in ChromaDB.")
    return result


if __name__ == "__main__":
    try:
        summary = ingest_documents()
        print(f"\nSummary: {summary['files']} files processed, {summary['chunks']} chunks stored.")
    except Exception as e:
        print(f"[!] Ingestion failed: {e}")
