# api/routes/ingest.py
#
# PURPOSE:
#   Defines the POST /ingest endpoint.
#   Triggers the full ingestion pipeline end to end:
#     PDF files -> text chunks -> entity extraction -> Neo4j + ChromaDB
#
# WHEN TO CALL THIS ENDPOINT:
#   Call POST /ingest once before running any queries.
#   This is the one-time setup step that builds the knowledge graph
#   and the vector index from the five PDF documents.
#   After ingestion, call GET /health to confirm pipeline_ready is True.
#
# HOW LONG DOES IT TAKE?
#   Entity extraction calls GPT-4o once per chunk (~90-120 chunks).
#   With a 0.5s delay between calls this takes roughly 60-90 seconds.
#   Embedding all chunks in ChromaDB takes about 20-30 seconds.
#   Total: expect 2-3 minutes for a full ingestion run.
#
# DO YOU RUN THIS FILE?
#   No. Mounted onto the FastAPI app in api/main.py.


from fastapi import APIRouter, HTTPException   # FastAPI router and error handling
from loguru import logger                      # structured logging
from pathlib import Path                       # file path handling

from api.schemas.models import IngestRequest, IngestResponse
from graph_rag.utils.config import settings

from graph_rag.ingestion.pdf_loader      import load_pdfs_from_directory
from graph_rag.ingestion.text_chunker    import chunk_documents, get_chunk_stats
from graph_rag.extraction.entity_extractor import EntityExtractor
from graph_rag.graph.graph_builder       import GraphBuilder
from graph_rag.retrieval.vector_store    import VectorStore


ingest_router = APIRouter(prefix="/ingest", tags=["ingestion"])


@ingest_router.post(
    "",
    response_model=IngestResponse,
    summary="Run the ingestion pipeline",
    description=(
        "Ingests all PDF files from the configured PDF directory. "
        "Extracts text chunks, runs GPT-4o entity extraction, "
        "writes the knowledge graph to Neo4j AuraDB, "
        "and builds the ChromaDB vector index. "
        "This endpoint must be called once before any /query requests."
    ),
)
def run_ingestion(request: IngestRequest) -> IngestResponse:
    """
    Runs the full ingestion pipeline end to end.

    Steps:
      1. Load PDFs from the configured directory
      2. Split pages into overlapping text chunks
      3. Extract entities and relationships with GPT-4o
      4. Write entities and relationships to Neo4j AuraDB
      5. Embed chunks and store in ChromaDB

    Args:
        request: IngestRequest with optional pdf_dir override
                 and clear_existing flag.

    Returns:
        IngestResponse with counts for each pipeline stage.

    Raises:
        HTTPException 500 if any pipeline stage fails critically.
    """

    logger.info(
        f"Ingestion requested — "
        f"pdf_dir: {request.pdf_dir or settings.pdf_dir}, "
        f"clear_existing: {request.clear_existing}"
    )

    # Use the pdf_dir from the request if provided, otherwise use .env default
    pdf_dir = Path(request.pdf_dir) if request.pdf_dir else settings.pdf_dir

    # Counters for the response summary
    pdfs_processed              = 0
    chunks_created              = 0
    entities_extracted          = 0
    relationships_extracted     = 0
    nodes_written               = 0
    relationships_written       = 0
    chunks_indexed              = 0

    try:
        # ------------------------------------------------------------------
        # STAGE 1: Load PDFs
        # ------------------------------------------------------------------
        logger.info("Stage 1: Loading PDFs...")
        documents = load_pdfs_from_directory(pdf_dir)

        # Count unique source files
        pdfs_processed = len(set(
            doc.metadata.get("source") for doc in documents
        ))
        logger.info(f"Stage 1 complete — {pdfs_processed} PDFs, {len(documents)} pages")

        # ------------------------------------------------------------------
        # STAGE 2: Chunk documents
        # ------------------------------------------------------------------
        logger.info("Stage 2: Chunking documents...")
        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks_created = len(chunks)

        stats = get_chunk_stats(chunks)
        logger.info(
            f"Stage 2 complete — {chunks_created} chunks. "
            f"Avg length: {stats['avg_char_length']} chars"
        )

        # ------------------------------------------------------------------
        # STAGE 3: Extract entities and relationships
        # ------------------------------------------------------------------
        logger.info("Stage 3: Extracting entities and relationships with GPT-4o...")
        logger.info(
            f"This calls GPT-4o {chunks_created} times. "
            f"Expected duration: {chunks_created * 0.7:.0f}-{chunks_created * 1.2:.0f} seconds."
        )

        extractor         = EntityExtractor()
        extraction_result = extractor.extract_from_chunks(chunks)

        entities_extracted      = len(extraction_result["entities"])
        relationships_extracted = len(extraction_result["relationships"])

        logger.info(
            f"Stage 3 complete — "
            f"{entities_extracted} entities, "
            f"{relationships_extracted} relationships"
        )

        # ------------------------------------------------------------------
        # STAGE 4: Build Neo4j graph
        # ------------------------------------------------------------------
        logger.info("Stage 4: Writing graph to Neo4j AuraDB...")

        builder      = GraphBuilder()
        build_result = builder.build_graph(
            extraction_result,
            clear_existing=request.clear_existing,
        )

        nodes_written         = build_result["nodes_written"]
        relationships_written = build_result["relationships_written"]

        logger.info(
            f"Stage 4 complete — "
            f"{nodes_written} nodes, "
            f"{relationships_written} relationships written to Neo4j"
        )

        # ------------------------------------------------------------------
        # STAGE 5: Build ChromaDB vector index
        # ------------------------------------------------------------------
        logger.info("Stage 5: Building ChromaDB vector index...")

        vs = VectorStore()
        vs.build_index(chunks)
        chunks_indexed = chunks_created

        logger.success(
            f"Stage 5 complete — {chunks_indexed} chunks indexed in ChromaDB"
        )

        # ------------------------------------------------------------------
        # All stages complete
        # ------------------------------------------------------------------
        logger.success(
            f"Ingestion pipeline complete. "
            f"PDFs: {pdfs_processed}, Chunks: {chunks_created}, "
            f"Entities: {entities_extracted}, "
            f"Neo4j nodes: {nodes_written}, "
            f"ChromaDB chunks: {chunks_indexed}"
        )

        return IngestResponse(
            success                         = True,
            message                         = "Ingestion complete. Pipeline is ready for queries.",
            pdfs_processed                  = pdfs_processed,
            chunks_created                  = chunks_created,
            entities_extracted              = entities_extracted,
            relationships_extracted         = relationships_extracted,
            nodes_written_to_neo4j          = nodes_written,
            relationships_written_to_neo4j  = relationships_written,
            chunks_indexed_in_chromadb      = chunks_indexed,
        )

    except FileNotFoundError as e:
        # PDF directory does not exist
        logger.error(f"Ingestion failed — PDF directory not found: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"PDF directory not found: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        return IngestResponse(
            success  = False,
            message  = f"Ingestion failed at an intermediate stage: {str(e)}",
            error    = str(e),
            pdfs_processed              = pdfs_processed,
            chunks_created              = chunks_created,
            entities_extracted          = entities_extracted,
            relationships_extracted     = relationships_extracted,
            nodes_written_to_neo4j      = nodes_written,
            relationships_written_to_neo4j = relationships_written,
            chunks_indexed_in_chromadb  = chunks_indexed,
        )