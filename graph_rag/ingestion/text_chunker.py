# graph_rag/ingestion/text_chunker.py
#
# PURPOSE:
#   Takes the page-level Document objects produced by pdf_loader.py and
#   splits them into smaller, overlapping chunks of text.
#
#   Why do we need to chunk at all?
#   Our PDFs have pages with 400-800 words each. If we embed an entire page
#   as one vector, the embedding averages out all the concepts on that page
#   into one blurry representation. A query about "TSMC revenue" would match
#   poorly against a page that also talks about geopolitical risk, equipment
#   suppliers, and customer relationships.
#
#   Smaller chunks mean each vector represents a focused, specific idea.
#   This makes retrieval much more precise — the right chunk surfaces for
#   the right question.
#
#   Why overlapping chunks?
#   If we split text at hard boundaries, a sentence like:
#     "...TSMC supplies Apple with M3 chips. This relationship accounts for
#      25% of TSMC revenue..."
#   might get cut in half. The first part goes into chunk N, the second into
#   chunk N+1. With overlap, both chunks contain the boundary sentence, so
#   no information is lost at the split point.
#
#   FLOW:
#   pdf_loader.py --> [page Documents] --> text_chunker.py --> [chunk Documents]
#   --> (next: entity_extractor.py and vector_store.py consume these chunks)
#
# DO YOU RUN THIS FILE?
#   No. Imported and called by scripts/run_ingestion.py.


from typing import List                                      # type hint for list
from langchain_core.documents import Document                      # LangChain text container
from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter is LangChain's most intelligent splitter.
# It tries to split on paragraph breaks first (\n\n), then on single newlines
# (\n), then on sentences (". "), then on words (" "), and only splits mid-word
# as a last resort. This preserves sentence and paragraph structure much better
# than a naive fixed-size character splitter.

from loguru import logger                                    # structured logging


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Splits a list of page-level Documents into smaller overlapping chunks.

    Each input Document (one page of a PDF) may produce multiple output
    Documents (chunks). Every chunk inherits the metadata from its parent
    page — source filename, page number, company — and gets two additional
    metadata fields: chunk_index (position within the page) and chunk_id
    (a unique string identifier used by ChromaDB).

    Args:
        documents:     List of page-level Documents from pdf_loader.py.
        chunk_size:    Maximum number of characters per chunk.
                       Default 800 = roughly 120-150 words = half a paragraph.
                       Loaded from settings.chunk_size in run_ingestion.py.
        chunk_overlap: Number of characters shared between consecutive chunks.
                       Default 150 = roughly 1-2 sentences of overlap.
                       Loaded from settings.chunk_overlap in run_ingestion.py.

    Returns:
        A flat list of chunk-level Document objects ready for:
          - Embedding and storage in ChromaDB (vector_store.py)
          - Entity extraction for Neo4j (entity_extractor.py)

    Example:
        Input:  20 page Documents (5 PDFs x ~4 pages each)
        Output: ~80-120 chunk Documents (each page splits into ~4-6 chunks)
    """

    # Initialise the text splitter with our chunking parameters
    # RecursiveCharacterTextSplitter tries these separators in order:
    #   1. "\n\n"  — blank line between paragraphs (preferred split point)
    #   2. "\n"    — single newline
    #   3. ". "    — end of sentence
    #   4. " "     — word boundary
    #   5. ""      — character boundary (last resort, avoids this if possible)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # length_function tells the splitter how to measure chunk size
        # len() counts characters, which is what chunk_size refers to
        length_function=len,
        # These are the separators tried in order — paragraph breaks first
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    logger.info(
        f"Chunking {len(documents)} pages with "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    # This list collects all chunk Documents across all input pages
    all_chunks: List[Document] = []

    # Process each page Document one at a time
    for doc_index, document in enumerate(documents):

        # Split this page's text into a list of smaller text strings
        # split_text() returns plain strings, not Documents yet
        text_splits = splitter.split_text(document.page_content)

        # Convert each text string into a Document with proper metadata
        for chunk_index, chunk_text in enumerate(text_splits):

            # Build metadata for this chunk by copying the parent page's
            # metadata and adding chunk-specific fields
            chunk_metadata = {
                # Carry forward all metadata from the parent page Document
                # This preserves source filename, page number, company, etc.
                **document.metadata,

                # Position of this chunk within its parent page (0-indexed)
                # Useful for understanding where in the page this text appears
                "chunk_index": chunk_index,

                # Total number of chunks produced from this parent page
                "total_chunks_in_page": len(text_splits),

                # A unique string ID for this chunk — used by ChromaDB as
                # the document ID so we can identify and retrieve specific chunks
                # Format: "source_filename__page_N__chunk_M"
                # Example: "apple_annual_overview_FY2024.pdf__page_1__chunk_0"
                "chunk_id": (
                    f"{document.metadata.get('source', 'unknown')}"
                    f"__page_{document.metadata.get('page', 0)}"
                    f"__chunk_{chunk_index}"
                ),
            }

            # Create the chunk Document
            chunk_document = Document(
                page_content=chunk_text,
                metadata=chunk_metadata,
            )

            all_chunks.append(chunk_document)

    # Log a summary so we can verify the chunking worked as expected
    logger.info(
        f"Chunking complete — "
        f"{len(documents)} pages --> {len(all_chunks)} chunks produced"
    )

    # Log a per-source breakdown so we can see how many chunks came from each PDF
    _log_chunking_summary(all_chunks)

    return all_chunks


def _log_chunking_summary(chunks: List[Document]) -> None:
    """
    Logs a breakdown of how many chunks were produced from each source PDF.

    This is a private helper function (indicated by the leading underscore).
    Private means it is only used inside this module — not imported elsewhere.

    Args:
        chunks: The full list of chunk Documents produced by chunk_documents().
    """

    # Build a dictionary: { "apple_annual_overview_FY2024.pdf": 24, ... }
    summary = {}
    for chunk in chunks:
        # Get the source filename from this chunk's metadata
        source = chunk.metadata.get("source", "unknown")
        # Increment the count for this source
        summary[source] = summary.get(source, 0) + 1

    # Log each source and its chunk count
    logger.info("Chunks per source document:")
    for source, count in sorted(summary.items()):
        logger.info(f"  {source}: {count} chunks")


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Returns basic statistics about the produced chunks.

    Useful for understanding the chunk size distribution before
    running the expensive embedding step. Call this after chunk_documents()
    to verify the chunks look reasonable before proceeding.

    Args:
        chunks: The list of chunk Documents from chunk_documents().

    Returns:
        A dictionary with these keys:
          total_chunks    : total number of chunks across all documents
          avg_char_length : average number of characters per chunk
          min_char_length : shortest chunk in characters
          max_char_length : longest chunk in characters
          sources         : number of unique source PDFs represented

    Example output:
        {
            "total_chunks": 94,
            "avg_char_length": 712,
            "min_char_length": 143,
            "max_char_length": 800,
            "sources": 5
        }
    """

    if not chunks:
        # Return zeroed stats if the chunks list is empty
        return {
            "total_chunks": 0,
            "avg_char_length": 0,
            "min_char_length": 0,
            "max_char_length": 0,
            "sources": 0,
        }

    # Compute the character length of every chunk's text
    lengths = [len(chunk.page_content) for chunk in chunks]

    # Count unique source filenames
    unique_sources = len(set(
        chunk.metadata.get("source", "unknown") for chunk in chunks
    ))

    return {
        "total_chunks":    len(chunks),
        "avg_char_length": int(sum(lengths) / len(lengths)),
        "min_char_length": min(lengths),
        "max_char_length": max(lengths),
        "sources":         unique_sources,
    }