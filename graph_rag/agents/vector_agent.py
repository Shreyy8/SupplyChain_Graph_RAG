# graph_rag/agents/vector_agent.py
#
# PURPOSE:
#   This is the third agent in the LangGraph pipeline.
#   It queries ChromaDB to find the text chunks most semantically
#   similar to the user's question.
#
#   This agent runs IN PARALLEL with graph_query_agent.py.
#   Both agents read from state (query and extracted_entities) but
#   write to different state fields (vector_results vs graph_results),
#   so there is no conflict. LangGraph handles the parallel execution
#   automatically when both are registered as parallel nodes in
#   graph_pipeline.py.
#
# WHAT THIS AGENT ADDS THAT GRAPH TRAVERSAL CANNOT:
#   The graph traversal in graph_query_agent returns clean structured
#   triples like:
#     (TSMC)-[SUPPLIES_CHIPS_TO]->(Apple) [component: M3 chip]
#
#   But it has no financial figures, no risk language, no narrative
#   context. A triple tells you WHAT the relationship is but not WHY
#   it matters or HOW significant it is.
#
#   Vector retrieval fills this gap. It finds the actual paragraphs
#   from the PDFs that contain the rich evidence:
#     "Apple relies on TSMC as its primary and sole foundry partner.
#      Any disruption to TSMC's manufacturing operations would directly
#      impact Apple's ability to manufacture iPhone, generating USD 201B
#      in annual revenue..."
#
#   The synthesis agent needs both — structure from the graph,
#   detail from the vector store.
#
# ENHANCED QUERY CONSTRUCTION:
#   We do not just send the raw user question to ChromaDB.
#   We build an enhanced query that combines the original question
#   with the entity names extracted by entity_agent. This makes the
#   vector search more targeted — if entities like "TSMC" and "Apple"
#   were extracted, appending them to the query ensures ChromaDB
#   finds chunks that mention those specific companies even if the
#   similarity to the raw question alone was borderline.
#
# DO YOU RUN THIS FILE?
#   No. Registered as a LangGraph node in graph_pipeline.py.
#   Runs in parallel with graph_query_agent automatically.


from typing import Dict, Any, List         # type hints
from loguru import logger                  # structured logging

from graph_rag.agents.state import GraphRAGState    # shared state schema
from graph_rag.retrieval.vector_store import VectorStore  # ChromaDB wrapper
from graph_rag.utils.config import settings              # validated config


# Module-level VectorStore instance
# We use a module-level instance rather than creating a new one on every
# call because loading the ChromaDB index from disk is expensive.
# The index is loaded once when this module is first imported and reused
# for every subsequent query.
_vector_store: VectorStore = None


def _get_vector_store() -> VectorStore:
    """
    Returns the module-level VectorStore instance, loading it if needed.

    This is a lazy initialisation pattern — the index is only loaded
    when the first query arrives, not at import time. This avoids
    slowing down application startup when the vector store is not
    immediately needed.

    Returns:
        A loaded VectorStore instance ready to query.
    """
    global _vector_store

    # If the store has not been initialised yet, create and load it
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.load_index()
        logger.info("VectorStore loaded into module-level cache")

    return _vector_store


def vector_agent(state: GraphRAGState) -> Dict[str, Any]:
    """
    LangGraph node — finds semantically similar text chunks in ChromaDB.

    Reads the user's query and extracted entities from state, builds
    an enhanced search query, queries ChromaDB, and returns a partial
    state update with the retrieved chunks.

    Runs in parallel with graph_query_agent — both start at the same
    time after entity_agent completes.

    Args:
        state: Current pipeline state. Must have:
               - query              : the user's original question
               - extracted_entities : entity names from entity_agent
                                      (used to enhance the search query)

    Returns:
        Partial state dict with:
          - vector_results : list of chunk result dicts from ChromaDB
        Or on error:
          - error, error_source
    """

    query    = state.get("query", "")
    entities = state.get("extracted_entities", [])

    logger.info(
        f"Vector agent running — "
        f"query: '{query[:60]}...', "
        f"entities: {entities}"
    )

    if not query:
        logger.error("Vector agent received empty query")
        return {
            "error":          "Empty query received by vector agent",
            "error_source":   "vector_agent",
            "vector_results": [],
        }

    try:
        # Build an enhanced query that combines the original question
        # with the extracted entity names.
        # This makes ChromaDB more likely to surface chunks about the
        # specific companies mentioned in the question.
        #
        # Example:
        #   Original query: "How does Taiwan risk affect revenue?"
        #   Entities:       ["Apple", "TSMC", "Taiwan"]
        #   Enhanced query: "How does Taiwan risk affect revenue? Apple TSMC Taiwan"
        enhanced_query = _build_enhanced_query(query, entities)

        logger.debug(f"Enhanced query: '{enhanced_query[:120]}'")

        # Get the loaded VectorStore and run the query
        vs      = _get_vector_store()
        results = vs.query(
            query_text=enhanced_query,
            top_k=settings.top_k_vector,
        )

        logger.success(
            f"Vector agent complete — "
            f"{len(results)} chunks retrieved. "
            f"Top score: {results[0]['score']:.3f}"
            if results else
            f"Vector agent complete — 0 chunks retrieved."
        )

        return {
            "vector_results": results,
        }

    except RuntimeError as e:
        # RuntimeError is raised by load_index() if the ChromaDB index
        # does not exist yet — means run_ingestion.py has not been run
        logger.error(
            f"Vector agent failed — ChromaDB index not found.\n"
            f"Fix: run  python scripts/run_ingestion.py  first.\n"
            f"Error: {e}"
        )
        return {
            "error":          f"ChromaDB index not found. Run ingestion first.",
            "error_source":   "vector_agent",
            "vector_results": [],
        }

    except Exception as e:
        logger.error(f"Vector agent failed: {e}")
        return {
            "error":          f"Vector agent failed: {str(e)}",
            "error_source":   "vector_agent",
            "vector_results": [],
        }


def _build_enhanced_query(query: str, entities: List[str]) -> str:
    """
    Builds an enhanced search query by appending entity names to the
    original question.

    Why this improves retrieval:
      ChromaDB finds chunks by cosine similarity between vectors.
      If the question is "How does geopolitical risk affect margins?",
      chunks that are semantically close but do not mention specific
      companies may rank higher than chunks that directly discuss
      TSMC and Apple margins.
      Appending the entity names shifts the query vector toward chunks
      that contain those specific company names.

    Args:
        query:    The user's original question string.
        entities: Entity names extracted by entity_agent.

    Returns:
        Enhanced query string with entity names appended.
        If entities is empty, returns the original query unchanged.

    Example:
        query    = "What happens if ASML stops shipping EUV machines?"
        entities = ["ASML", "TSMC", "Apple", "Nvidia"]
        returns  = "What happens if ASML stops shipping EUV machines?
                    Context entities: ASML TSMC Apple Nvidia"
    """

    if not entities:
        # No entities to append — return original query unchanged
        return query

    # Join entity names into a space-separated string
    entities_str = " ".join(entities)

    # Append to the query with a clear label so the embedding model
    # understands these are contextual entity hints, not part of the question
    enhanced = f"{query}\nContext entities: {entities_str}"

    return enhanced