# api/routes/health.py
#
# PURPOSE:
#   Defines the GET /health endpoint.
#   Checks the health of Neo4j AuraDB and ChromaDB and returns
#   a structured response showing whether the pipeline is ready
#   to accept queries.
#
# WHEN TO USE THIS ENDPOINT:
#   Call GET /health before running demo queries to confirm:
#     1. Neo4j is reachable and contains nodes and relationships
#     2. ChromaDB index exists and contains chunks
#     3. pipeline_ready is True
#   If pipeline_ready is False, run the ingestion first.
#
# DO YOU RUN THIS FILE?
#   No. Mounted onto the FastAPI app in api/main.py with:
#       app.include_router(health_router)


from fastapi import APIRouter               # FastAPI router for grouping endpoints
from loguru import logger                  # structured logging

from api.schemas.models import (
    HealthResponse,
    Neo4jHealth,
    ChromaDBHealth,
)
from graph_rag.graph.neo4j_client import Neo4jClient
from graph_rag.retrieval.vector_store import VectorStore


# APIRouter groups related endpoints together
# prefix="/health" means the full path is GET /health
# tags=["health"] groups this endpoint in the auto-generated API docs
health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get(
    "",
    response_model=HealthResponse,
    summary="Check pipeline health",
    description=(
        "Returns the health status of Neo4j AuraDB and ChromaDB. "
        "pipeline_ready is True only when both databases are healthy "
        "and contain ingested data."
    ),
)
def health_check() -> HealthResponse:
    """
    Checks the health of all pipeline components.

    Connects to Neo4j and queries node/relationship counts.
    Loads the ChromaDB index and checks chunk count.
    Returns a structured response with individual component
    health and an overall pipeline_ready flag.

    Returns:
        HealthResponse with neo4j health, chromadb health,
        overall status, pipeline_ready flag, and timestamp.
    """

    logger.info("Health check requested")

    # ------------------------------------------------------------------
    # Check Neo4j health
    # ------------------------------------------------------------------
    try:
        # Neo4jClient.health_check() opens a connection, queries counts,
        # and returns a dict with status, node_count, relationship_count
        with Neo4jClient() as client:
            neo4j_data = client.health_check()

        neo4j_health = Neo4jHealth(
            status             = neo4j_data["status"],
            node_count         = neo4j_data["node_count"],
            relationship_count = neo4j_data["relationship_count"],
            message            = neo4j_data["message"],
        )

    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        neo4j_health = Neo4jHealth(
            status             = "unhealthy",
            node_count         = 0,
            relationship_count = 0,
            message            = f"Neo4j connection failed: {str(e)}",
        )

    # ------------------------------------------------------------------
    # Check ChromaDB health
    # ------------------------------------------------------------------
    try:
        vs           = VectorStore()
        chroma_data  = vs.health_check()

        chroma_health = ChromaDBHealth(
            status         = chroma_data["status"],
            document_count = chroma_data["document_count"],
            persist_dir    = chroma_data["persist_dir"],
            message        = chroma_data["message"],
        )

    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        chroma_health = ChromaDBHealth(
            status         = "unhealthy",
            document_count = 0,
            persist_dir    = "unknown",
            message        = f"ChromaDB check failed: {str(e)}",
        )

    # ------------------------------------------------------------------
    # Determine overall health and pipeline_ready
    # ------------------------------------------------------------------
    # pipeline_ready is True only when:
    #   - Neo4j is healthy AND has at least one node
    #   - ChromaDB is healthy AND has at least one chunk
    neo4j_ok  = (
        neo4j_health.status == "healthy"
        and neo4j_health.node_count > 0
    )
    chroma_ok = (
        chroma_health.status == "healthy"
        and chroma_health.document_count > 0
    )

    pipeline_ready = neo4j_ok and chroma_ok
    overall_status = "healthy" if pipeline_ready else "degraded"

    logger.info(
        f"Health check complete — "
        f"Neo4j: {neo4j_health.status} ({neo4j_health.node_count} nodes), "
        f"ChromaDB: {chroma_health.status} ({chroma_health.document_count} chunks), "
        f"pipeline_ready: {pipeline_ready}"
    )

    return HealthResponse(
        status         = overall_status,
        neo4j          = neo4j_health,
        chromadb       = chroma_health,
        pipeline_ready = pipeline_ready,
    )