# scripts/run_ingestion.py
#
# PURPOSE:
#   One-time script that runs the complete ingestion pipeline.
#   Run this ONCE before starting the API server or running test queries.
#
#   What it does end to end:
#     1. Loads all PDFs from data/pdfs/
#     2. Splits pages into overlapping text chunks
#     3. Calls GPT-4o on each chunk to extract entities + relationships
#     4. Writes all entities as nodes to Neo4j AuraDB
#     5. Writes all relationships as edges to Neo4j AuraDB
#     6. Embeds all chunks and stores them in ChromaDB
#     7. Runs verification queries to confirm key relationships exist
#
# DO YOU RUN THIS FILE?
#   YES. Run it once from the project root with venv activated:
#
#     python scripts/run_ingestion.py
#
#   Expected duration: 2-4 minutes
#   (GPT-4o entity extraction is the slow step — ~90 API calls)
#
# WHEN TO RE-RUN:
#   Re-run with --clear if you want to wipe and rebuild from scratch:
#
#     python scripts/run_ingestion.py --clear
#
# PREREQUISITES:
#   1. venv activated and pip install -r requirements.txt done
#   2. .env file filled in with real credentials
#   3. Five PDFs in data/pdfs/
#   4. Neo4j AuraDB instance running (green in console)


import sys
import argparse
from pathlib import Path
from loguru import logger

# Add the project root to sys.path so imports work correctly
# when running this script directly from the command line
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_rag.utils.config import settings
from graph_rag.ingestion.pdf_loader      import load_pdfs_from_directory, get_pdf_summary
from graph_rag.ingestion.text_chunker    import chunk_documents, get_chunk_stats
from graph_rag.extraction.entity_extractor import EntityExtractor
from graph_rag.graph.graph_builder       import GraphBuilder
from graph_rag.retrieval.vector_store    import VectorStore


def run_ingestion(clear_existing: bool = False):
    """
    Runs the full ingestion pipeline from PDF files to Neo4j + ChromaDB.

    Args:
        clear_existing: If True, wipes all existing Neo4j and ChromaDB
                        data before ingesting. Use when re-running from scratch.
    """

    logger.info("=" * 60)
    logger.info("FINSIGHT_GraphRAG — Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"PDF directory:  {settings.pdf_dir}")
    logger.info(f"Neo4j URI:      {settings.neo4j_uri}")
    logger.info(f"ChromaDB dir:   {settings.chroma_persist_dir}")
    logger.info(f"Clear existing: {clear_existing}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # STAGE 1: Load PDFs
    # ------------------------------------------------------------------
    logger.info("STAGE 1 — Loading PDFs")

    documents = load_pdfs_from_directory(settings.pdf_dir)

    summary = get_pdf_summary(documents)
    logger.info("Pages loaded per document:")
    for source, page_count in summary.items():
        logger.info(f"  {source}: {page_count} pages")
    logger.info(f"Total pages loaded: {len(documents)}")

    # ------------------------------------------------------------------
    # STAGE 2: Chunk documents
    # ------------------------------------------------------------------
    logger.info("STAGE 2 — Chunking documents")

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    stats = get_chunk_stats(chunks)
    logger.info(f"Chunks created:      {stats['total_chunks']}")
    logger.info(f"Avg chunk length:    {stats['avg_char_length']} chars")
    logger.info(f"Min chunk length:    {stats['min_char_length']} chars")
    logger.info(f"Max chunk length:    {stats['max_char_length']} chars")

    # ------------------------------------------------------------------
    # STAGE 3: Entity extraction
    # ------------------------------------------------------------------
    logger.info("STAGE 3 — Extracting entities and relationships with GPT-4o")
    logger.info(
        f"This will make approximately {len(chunks)} API calls to GPT-4o. "
        f"Estimated time: {len(chunks) * 0.8:.0f}-{len(chunks) * 1.5:.0f} seconds. "
        f"Please wait..."
    )

    extractor         = EntityExtractor()
    extraction_result = extractor.extract_from_chunks(chunks)

    logger.info(f"Unique entities extracted:      {len(extraction_result['entities'])}")
    logger.info(f"Unique relationships extracted: {len(extraction_result['relationships'])}")

    # Log a sample of extracted entities so you can verify quality
    logger.info("Sample entities (first 10):")
    for entity in extraction_result["entities"][:10]:
        logger.info(f"  ({entity['type']}) {entity['name']}")

    # Log a sample of extracted relationships
    logger.info("Sample relationships (first 10):")
    for rel in extraction_result["relationships"][:10]:
        props = rel.get("properties", {})
        logger.info(
            f"  {rel['source']} -[{rel['type']}]-> {rel['target']}"
            + (f" {props}" if props else "")
        )

    # ------------------------------------------------------------------
    # STAGE 4: Build Neo4j graph
    # ------------------------------------------------------------------
    logger.info("STAGE 4 — Writing knowledge graph to Neo4j AuraDB")

    builder      = GraphBuilder()
    build_result = builder.build_graph(
        extraction_result,
        clear_existing=clear_existing,
    )

    logger.info(f"Nodes written:         {build_result['nodes_written']}")
    logger.info(f"Relationships written: {build_result['relationships_written']}")
    logger.info(f"Total nodes in graph:  {build_result['total_nodes_in_graph']}")
    logger.info(f"Total rels in graph:   {build_result['total_rels_in_graph']}")

    # Run verification checks to confirm key relationships exist
    logger.info("Running verification checks on Neo4j graph...")
    verification = builder.verify_key_relationships()
    passed = sum(1 for v in verification if v["result"] == "PASS")
    logger.info(f"Verification: {passed}/{len(verification)} checks passed")

    # ------------------------------------------------------------------
    # STAGE 5: Build ChromaDB vector index
    # ------------------------------------------------------------------
    logger.info("STAGE 5 — Building ChromaDB vector index")
    logger.info(
        "This calls OpenAI embeddings API once for all chunks. "
        "Expected time: 20-30 seconds..."
    )

    vs = VectorStore()
    vs.build_index(chunks)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.success("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"PDFs processed:        {len(summary)}")
    logger.info(f"Pages loaded:          {len(documents)}")
    logger.info(f"Chunks created:        {len(chunks)}")
    logger.info(f"Entities extracted:    {len(extraction_result['entities'])}")
    logger.info(f"Relationships extracted:{len(extraction_result['relationships'])}")
    logger.info(f"Nodes in Neo4j:        {build_result['total_nodes_in_graph']}")
    logger.info(f"Rels in Neo4j:         {build_result['total_rels_in_graph']}")
    logger.info(f"Chunks in ChromaDB:    {len(chunks)}")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Start the API:    uvicorn api.main:app --reload")
    logger.info("  2. Test queries:     python scripts/test_queries.py")
    logger.info("  3. API docs:         http://localhost:8000/docs")
    logger.info("=" * 60)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the FINSIGHT_GraphRAG ingestion pipeline"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing Neo4j graph and ChromaDB index before ingesting",
    )
    args = parser.parse_args()

    run_ingestion(clear_existing=args.clear)