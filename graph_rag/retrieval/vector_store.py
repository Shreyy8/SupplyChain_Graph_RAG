# graph_rag/retrieval/vector_store.py
#
# PURPOSE:
#   Builds and manages the ChromaDB vector index that stores embeddings
#   for all text chunks produced by text_chunker.py.
#
#   This is the vector retrieval half of our hybrid retrieval system.
#   The other half is Neo4j graph traversal (graph_query_agent.py).
#   Together they give us both semantic similarity AND structural reasoning.
#
# WHAT IS AN EMBEDDING?
#   An embedding is a list of numbers (a vector) that represents the
#   meaning of a piece of text. Texts with similar meanings have vectors
#   that are mathematically close to each other.
#
#   Example:
#     "TSMC manufactures chips for Apple"  -> [0.12, -0.34, 0.87, ...]
#     "Apple relies on TSMC for silicon"   -> [0.11, -0.31, 0.85, ...]  <- similar
#     "Samsung sells OLED display panels"  -> [0.67,  0.21, -0.43, ...] <- different
#
#   When you ask a question, we convert it to an embedding and find the
#   chunks whose embeddings are closest. Those chunks likely contain
#   relevant information even if they do not share exact keywords.
#
# WHY CHROMADB?
#   ChromaDB stores the vector, the original text, and metadata together.
#   One query returns all three. It persists to disk so we do not need to
#   re-embed every time the application restarts. Zero external setup needed.
#
# HOW THIS FITS IN THE PIPELINE:
#   text_chunker.py --> [chunk Documents]
#                           |
#                           v
#                     vector_store.py  (builds index during ingestion)
#                           |
#                           v
#                     ChromaDB on disk (./data/chroma_db)
#                           |
#                           v
#                     vector_agent.py  (queries index at runtime)
#
# DO YOU RUN THIS FILE?
#   No. Imported by scripts/run_ingestion.py (to build) and
#   vector_agent.py (to query).


from typing import List, Dict, Any         # type hints
from langchain_core.documents import Document     # LangChain text container
from langchain_community.vectorstores import Chroma
# Chroma is LangChain's wrapper around ChromaDB.
# It handles embedding, storing, and querying in one clean interface.

from langchain_openai import OpenAIEmbeddings
# OpenAIEmbeddings calls the OpenAI embeddings API to convert text to vectors.
# We use text-embedding-3-small — configured in .env as EMBEDDING_MODEL.

from loguru import logger                  # structured logging
from graph_rag.utils.config import settings  # validated config values


class VectorStore:
    """
    Builds and queries a ChromaDB vector index for our document chunks.

    Two main use cases:

    1. During ingestion (run_ingestion.py):
       vs = VectorStore()
       vs.build_index(chunks)          # embeds all chunks, saves to disk

    2. During query time (vector_agent.py):
       vs = VectorStore()
       vs.load_index()                 # loads the existing index from disk
       results = vs.query("question")  # returns similar chunks
    """

    def __init__(self):
        """
        Initialises the embedding model and ChromaDB configuration.
        Does not build or load the index yet.
        """

        # OpenAIEmbeddings calls the OpenAI API to convert text to vectors
        # model is read from .env — default text-embedding-3-small
        # text-embedding-3-small produces 1536-dimensional vectors
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Path where ChromaDB stores its index files on disk
        # ChromaDB creates this directory automatically if it does not exist
        self.persist_dir = str(settings.chroma_persist_dir)

        # Name of the ChromaDB collection (like a table name)
        # All five PDFs get stored in this single collection
        self.collection_name = settings.chroma_collection_name

        # The Chroma vector store object — None until build_index() or
        # load_index() is called
        self.vector_store = None

        logger.info(
            f"VectorStore initialised — "
            f"collection: {self.collection_name}, "
            f"persist_dir: {self.persist_dir}"
        )

    def build_index(self, chunks: List[Document]) -> None:
        """
        Embeds all text chunks and stores them in ChromaDB.

        This is called once during ingestion. For each chunk it:
          1. Calls OpenAI's embedding API to convert the text to a vector
          2. Stores the vector + original text + metadata in ChromaDB
          3. Persists everything to disk at chroma_persist_dir

        After this method completes, the index is saved to disk and can
        be loaded instantly on future runs without re-embedding.

        Args:
            chunks: List of chunk Documents from text_chunker.py.
                    Each chunk has page_content (text) and metadata
                    (source, page, chunk_id, etc.)

        Note:
            This method makes one OpenAI API call per batch of chunks.
            ChromaDB batches embeddings internally for efficiency.
            For ~100 chunks using text-embedding-3-small, this costs
            less than USD 0.01 in API fees.
        """

        if not chunks:
            logger.error("No chunks provided to build_index(). Aborting.")
            return

        logger.info(
            f"Building ChromaDB index from {len(chunks)} chunks. "
            f"This calls the OpenAI embeddings API — may take 20-30 seconds."
        )

        # Extract the unique chunk_id from each chunk's metadata to use
        # as the ChromaDB document ID. ChromaDB requires unique IDs.
        # If chunk_id is missing in metadata, fall back to the loop index.
        ids = [
            chunk.metadata.get("chunk_id", f"chunk_{i}")
            for i, chunk in enumerate(chunks)
        ]

        # Chroma.from_documents() does three things in one call:
        #   1. Calls OpenAI embeddings API to embed all chunks
        #   2. Creates a ChromaDB collection with the given name
        #   3. Persists the collection to disk at persist_directory
        #
        # If a collection with this name already exists at persist_directory,
        # ChromaDB will overwrite it. This is safe for re-running ingestion.
        self.vector_store = Chroma.from_documents(
            documents=chunks,                        # the chunk Documents
            embedding=self.embeddings,               # the embedding function
            collection_name=self.collection_name,    # collection name in ChromaDB
            persist_directory=self.persist_dir,      # where to save on disk
            ids=ids,                                 # unique ID for each chunk
        )

        logger.success(
            f"ChromaDB index built and saved to {self.persist_dir}. "
            f"{len(chunks)} chunks indexed."
        )

    def load_index(self) -> None:
        """
        Loads an existing ChromaDB index from disk.

        Called at query time (by vector_agent.py) to load the index that
        was built during ingestion. Much faster than rebuild_index() because
        it reads from disk rather than calling the OpenAI embeddings API.

        Raises:
            RuntimeError: if no index exists at persist_dir yet.
                          (This means run_ingestion.py has not been run.)
        """

        import os
        # Check that the persist directory exists and is not empty
        # An empty directory means ingestion has not been run yet
        if not os.path.exists(self.persist_dir) or not os.listdir(self.persist_dir):
            raise RuntimeError(
                f"No ChromaDB index found at {self.persist_dir}. "
                f"Run scripts/run_ingestion.py first to build the index."
            )

        logger.info(f"Loading ChromaDB index from {self.persist_dir}")

        # Chroma() loads an existing collection from disk
        # This does NOT call the OpenAI embeddings API — it just reads files
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        # Verify the collection loaded correctly by checking document count
        doc_count = self.vector_store._collection.count()
        logger.success(
            f"ChromaDB index loaded — {doc_count} chunks in collection"
        )

    def query(
        self,
        query_text: str,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds the most semantically similar chunks to a query string.

        Converts the query text to an embedding vector using OpenAI, then
        searches ChromaDB for the chunks whose embeddings are closest
        (by cosine similarity). Returns the top_k most similar chunks.

        Args:
            query_text: The user's question or search text.
                        Example: "How does Taiwan risk affect Apple revenue?"
            top_k:      Number of chunks to return.
                        Defaults to settings.top_k_vector (from .env, default 5).

        Returns:
            A list of dictionaries, one per retrieved chunk:
            [
                {
                    "text":       "Apple relies on TSMC as its primary...",
                    "source":     "apple_annual_overview_FY2024.pdf",
                    "page":       2,
                    "chunk_id":   "apple_annual_overview_FY2024.pdf__page_2__chunk_1",
                    "score":      0.87,   <- cosine similarity (higher = more similar)
                    "company":    "APPLE",
                },
                ...
            ]

        Raises:
            RuntimeError: if load_index() or build_index() has not been called.
        """

        # Ensure the index is loaded before querying
        if self.vector_store is None:
            raise RuntimeError(
                "Vector store is not loaded. "
                "Call load_index() before querying."
            )

        # Use the configured default if top_k is not specified
        top_k = top_k or settings.top_k_vector

        logger.debug(f"Querying ChromaDB for top {top_k} chunks: '{query_text[:80]}...'")

        # similarity_search_with_score() returns a list of (Document, score) tuples
        # score is cosine similarity — higher means more similar (range 0 to 1)
        results_with_scores = self.vector_store.similarity_search_with_score(
            query=query_text,
            k=top_k,
        )

        # Convert the (Document, score) tuples into plain dictionaries
        # so the calling agent does not need to know about ChromaDB internals
        formatted_results = []
        for document, score in results_with_scores:
            formatted_results.append({
                # The actual text of the chunk
                "text":     document.page_content,
                # Source PDF filename — from chunk metadata
                "source":   document.metadata.get("source", "unknown"),
                # Page number — from chunk metadata
                "page":     document.metadata.get("page", 0),
                # Unique chunk identifier
                "chunk_id": document.metadata.get("chunk_id", "unknown"),
                # Cosine similarity score — higher is better
                "score":    float(score),
                # Company name derived from the source filename
                "company":  document.metadata.get("company", "unknown"),
            })

        logger.debug(
            f"ChromaDB returned {len(formatted_results)} chunks. "
            f"Top score: {formatted_results[0]['score']:.3f}"
            if formatted_results else "ChromaDB returned 0 results."
        )

        return formatted_results

    def health_check(self) -> Dict[str, Any]:
        """
        Returns the health status of the ChromaDB vector store.
        Called by the FastAPI /health endpoint.

        Returns:
            Dict with status, document count, and a message.
        """
        try:
            # Try to load the index if not already loaded
            if self.vector_store is None:
                self.load_index()

            doc_count = self.vector_store._collection.count()

            return {
                "status":         "healthy",
                "document_count": doc_count,
                "persist_dir":    self.persist_dir,
                "message":        f"ChromaDB is healthy. {doc_count} chunks indexed.",
            }

        except Exception as e:
            return {
                "status":         "unhealthy",
                "document_count": 0,
                "persist_dir":    self.persist_dir,
                "message":        f"ChromaDB health check failed: {e}",
            }