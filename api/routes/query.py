# api/routes/query.py
#
# PURPOSE:
#   Defines the POST /query endpoint.
#   Receives a question, runs it through the full LangGraph pipeline,
#   and returns the answer with citations and graph paths.
#
# THIS IS THE MAIN ENDPOINT OF THE ENTIRE PIPELINE.
#   Everything we have built — entity extraction, graph traversal,
#   vector search, reranking, synthesis — is triggered by a call
#   to this single endpoint.
#
# DO YOU RUN THIS FILE?
#   No. Mounted onto the FastAPI app in api/main.py.


from fastapi import APIRouter, HTTPException   # FastAPI router and error handling
from loguru import logger                      # structured logging

from api.schemas.models import QueryRequest, QueryResponse
from graph_rag.pipeline.graph_pipeline import create_pipeline


query_router = APIRouter(prefix="/query", tags=["query"])

# Module-level pipeline instance
# Created once when the first request arrives (lazy initialisation)
# Reused for all subsequent requests — avoids rebuilding the StateGraph
# on every call which would be slow
_pipeline = None


def _get_pipeline():
    """
    Returns the module-level compiled LangGraph pipeline.
    Creates it on first call, reuses it on subsequent calls.
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising LangGraph pipeline (first request)...")
        _pipeline = create_pipeline()
        logger.success("LangGraph pipeline ready")
    return _pipeline


@query_router.post(
    "",
    response_model=QueryResponse,
    summary="Run a Graph RAG query",
    description=(
        "Runs the full Graph RAG pipeline for the given question. "
        "Extracts entities, traverses Neo4j, queries ChromaDB, "
        "reranks results, and generates an answer with GPT-4o. "
        "Requires ingestion to have been run first (POST /ingest)."
    ),
)
def run_query(request: QueryRequest) -> QueryResponse:
    """
    Runs the complete Graph RAG pipeline for a user question.

    Pipeline flow:
      entity_agent -> [graph_query_agent || vector_agent]
      -> rerank_agent -> synthesis_agent

    Args:
        request: QueryRequest containing the question and optional
                 parameter overrides.

    Returns:
        QueryResponse with the answer, sources, graph paths,
        detected intent, and extracted entities.

    Raises:
        HTTPException 500 if the pipeline fails completely.
    """

    logger.info(f"Query received: '{request.question[:80]}...'")

    try:
        # Get the compiled pipeline (creates it on first call)
        pipeline = _get_pipeline()

        # Build the initial state with just the query
        # All other state fields start as None and are filled by agents
        initial_state = {
            "query": request.question,
        }

        # Override retrieval settings if the caller specified them
        # We store these in state so agents can read them
        # Note: top_k_vector is used by vector_agent via settings,
        # but if the caller passes an override we update settings temporarily
        # For a teaching demo, passing through settings is sufficient
        if request.top_k_graph_hops:
            # Store caller override in initial state
            # graph_query_agent reads recommended_hops from entity_agent
            # but we can pre-set it here as a hint
            initial_state["recommended_hops"] = request.top_k_graph_hops

        # Run the full pipeline synchronously
        # .invoke() blocks until synthesis_agent completes and returns
        # the fully populated final state
        final_state = pipeline.invoke(initial_state)

        # Check if the pipeline encountered an error
        if final_state.get("error") and not final_state.get("final_answer"):
            logger.error(
                f"Pipeline error: {final_state.get('error')} "
                f"in {final_state.get('error_source')}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {final_state.get('error')}"
            )

        # Build the response from the final state
        response = QueryResponse(
            question        = request.question,
            answer          = final_state.get("final_answer", "No answer generated."),
            sources         = final_state.get("sources", []),
            graph_paths_used= final_state.get("graph_paths_used", []),
            intent_detected = final_state.get("intent"),
            entities_found  = final_state.get("extracted_entities", []),
            success         = not bool(final_state.get("error")),
            error           = final_state.get("error"),
        )

        logger.success(
            f"Query complete — "
            f"answer length: {len(response.answer)} chars, "
            f"sources: {len(response.sources)}, "
            f"graph paths: {len(response.graph_paths_used)}"
        )

        return response

    except HTTPException:
        # Re-raise HTTPExceptions without wrapping them
        raise

    except Exception as e:
        logger.error(f"Query endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )