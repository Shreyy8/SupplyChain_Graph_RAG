"""
rag.py - Retrieval Augmented Generation Query Engine

Retrieves relevant chunks from ChromaDB and generates grounded answers using OpenAI GPT-4o.
Adheres strictly to the honest refusal policy and returns source document names and page numbers.
"""

import os
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "supplychain_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
DEFAULT_TOP_K = 6  # Recommended 5-6 in assignment to ensure cross-document recall
TEMPERATURE = 0.0  # Mandatory range: 0.0 - 0.2


def get_rag_components(persist_directory: str = CHROMA_DIR):
    """Initializes ChromaDB collection and OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL
    )
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )
    openai_client = OpenAI(api_key=api_key)
    return chroma_client, collection, openai_client


def query_rag(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    persist_directory: str = CHROMA_DIR,
    model_name: str = LLM_MODEL
) -> Dict[str, Any]:
    """
    Executes a RAG query:
    1. Retrieves top_k matching chunks from ChromaDB
    2. Constructs prompt with retrieved context
    3. Calls GPT-4o with strict refusal system prompt
    4. Returns answer and source citations (document name + page number)
    """
    _, collection, openai_client = get_rag_components(persist_directory=persist_directory)

    # Check if collection is empty
    count = collection.count()
    if count == 0:
        return {
            "answer": "No documents have been indexed yet. Please upload and index PDF documents first.",
            "sources": [],
            "question": question
        }

    # Query ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, count)
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    # Build context string and structured sources list
    context_blocks = []
    sources = []
    seen_sources = set()

    for doc_text, meta in zip(documents, metadatas):
        file_name = meta.get("file_name", "Unknown Document")
        page = meta.get("page", 1)
        source_key = (file_name, page)

        context_blocks.append(f"[Document: {file_name} | Page: {page}]\n{doc_text}")

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "file": file_name,
                "page": page,
                "excerpt": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            })

    context_str = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are an internal supply chain assistant for Meridian Components Pvt. Ltd.\n"
        "Answer only from the context provided below. If the context does not contain the answer, "
        "say the information is not available in the uploaded documents.\n"
        "Do not invent facts, extrapolate, or assume information not present in the context.\n"
        "Always cite the exact document name and page number for every piece of information used in your answer."
    )

    user_prompt = f"""Context:
{context_str}

Question:
{question}

Answer (with document and page citations):"""

    response = openai_client.chat.completions.create(
        model=model_name,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer_text = response.choices[0].message.content.strip()

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "total_chunks_retrieved": len(documents)
    }


if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?"
    print(f"Asking: {test_q}\n")
    try:
        res = query_rag(test_q)
        print("ANSWER:\n" + res["answer"])
        print("\nSOURCES:")
        for s in res["sources"]:
            print(f"- {s['file']} (Page {s['page']})")
    except Exception as e:
        print(f"Error running query: {e}")
