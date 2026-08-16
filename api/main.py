"""
api/main.py - FastAPI Backend Service for Meridian Components Supply Chain RAG

Exposes endpoints required by Section 6 (Optional Bonus):
- POST /ingest : Upload and index one or more PDF files (or index default data/ PDFs)
- POST /ask    : Query the RAG engine with question and top_k
- GET  /stats  : Retrieve collection status, chunk count, and model details
"""

import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_documents, get_chroma_collection, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from rag import query_rag, LLM_MODEL, DEFAULT_TOP_K

app = FastAPI(
    title="Meridian Supply Chain RAG API",
    description="FastAPI service for querying Meridian Components supply chain documents and procurement policies.",
    version="2.0.0",
    contact={
        "name": "Shreyy8",
        "email": "shreyansh.24scse1420096@galgotiasuniversity.ac.in",
    }
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question about supply chain documents")
    top_k: Optional[int] = Field(default=DEFAULT_TOP_K, description="Number of context chunks to retrieve")

class SourceInfo(BaseModel):
    file: str
    page: int
    excerpt: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    question: Optional[str] = None

class IngestResponse(BaseModel):
    files: int
    chunks: int
    file_names: Optional[List[str]] = None

class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int


@app.get("/", tags=["Root"])
def root():
    """Root health check endpoint."""
    return {
        "status": "online",
        "service": "Meridian Components Supply Chain RAG API",
        "docs_url": "http://localhost:8000/docs"
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_endpoint(files: Optional[List[UploadFile]] = File(default=None)):
    """
    Ingests PDF files into ChromaDB.
    If files are uploaded via multipart form, saves and indexes them.
    If no files are passed, indexes default PDFs from the data/ directory.
    """
    try:
        if files and len(files) > 0:
            temp_dir = tempfile.mkdtemp()
            saved_paths = []
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                saved_paths.append(file_path)

            result = ingest_documents(pdf_paths=saved_paths, persist_directory=CHROMA_DIR)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return IngestResponse(files=result["files"], chunks=result["chunks"], file_names=result["file_names"])
        else:
            # Default to data/ directory
            result = ingest_documents(persist_directory=CHROMA_DIR)
            return IngestResponse(files=result["files"], chunks=result["chunks"], file_names=result["file_names"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse, tags=["RAG Query"])
def ask_endpoint(payload: AskRequest):
    """
    Asks a question against the indexed documents and returns GPT-4o answer with sources.
    """
    try:
        res = query_rag(
            question=payload.question,
            top_k=payload.top_k or DEFAULT_TOP_K,
            persist_directory=CHROMA_DIR
        )
        return AskResponse(
            question=res["question"],
            answer=res["answer"],
            sources=[SourceInfo(file=s["file"], page=s["page"], excerpt=s.get("excerpt")) for s in res.get("sources", [])]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["Metadata & Stats"])
def stats_endpoint():
    """
    Returns ChromaDB collection name, total chunks, embedding model, and LLM model.
    """
    try:
        _, collection = get_chroma_collection(persist_directory=CHROMA_DIR)
        count = collection.count()
        return StatsResponse(
            collection_name=COLLECTION_NAME,
            total_chunks=count,
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)