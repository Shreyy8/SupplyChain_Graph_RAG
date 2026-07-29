# graph_rag/agents/state.py
#
# PURPOSE:
#   Defines the shared state object that flows through all five agents
#   in the LangGraph pipeline.
#
# WHAT IS STATE IN LANGGRAPH?
# ----------------------------
# In LangGraph, a pipeline is a directed graph where nodes are agents
# and edges are the connections between them. Each agent receives the
# current state, does its work, updates the state, and passes it forward.
#
# Think of state as a shared whiteboard that every agent in the pipeline
# can read from and write to. The whiteboard starts mostly empty when a
# question arrives. Each agent fills in its section:
#
#   User question arrives
#         |
#         v
#   [Entity Agent]         reads: query
#                          writes: extracted_entities, entity_types, intent
#         |
#         v
#   [Graph Query Agent]    reads: extracted_entities
#                          writes: graph_results
#         |
#   [Vector Agent]         reads: query
#   (runs in parallel)     writes: vector_results
#         |
#         v
#   [Rerank Agent]         reads: graph_results, vector_results
#                          writes: merged_results
#         |
#         v
#   [Synthesis Agent]      reads: query, merged_results, extracted_entities
#                          writes: final_answer, sources, graph_paths_used
#         |
#         v
#   Response returned to user
#
# WHY A TYPED STATE CLASS?
# -------------------------
# We use Python's TypedDict to define the state schema. TypedDict is a
# dictionary subclass where every key has an explicit type annotation.
# This gives us:
#   1. Autocomplete in VS Code — you know exactly what fields exist
#   2. Type checking — if an agent writes the wrong type, it is caught early
#   3. Documentation — the state definition IS the documentation of what
#      data flows through the pipeline
#   4. LangGraph compatibility — LangGraph requires TypedDict or Pydantic
#      models as its state schema
#
# LANGGRAPH STATE MANAGEMENT:
# ----------------------------
# LangGraph merges state updates automatically. Each agent returns a
# partial state dictionary containing only the fields it changed.
# LangGraph merges that partial update into the full state and passes
# the updated full state to the next agent. Agents do not need to copy
# the entire state — they only return what they changed.
#
# DO YOU RUN THIS FILE?
#   No. Imported by all five agents and by graph_pipeline.py.


from typing import TypedDict, List, Dict, Any, Optional
# TypedDict: defines a dictionary with typed keys — LangGraph's required format
# List, Dict, Any, Optional: standard type hints for the state fields


class GraphRAGState(TypedDict):
    """
    The shared state dictionary passed between all five agents in the pipeline.

    Every field is Optional because the state starts nearly empty —
    only 'query' is populated when a request first arrives. Each agent
    fills in its own fields as the pipeline progresses.

    LangGraph reads this TypedDict definition to understand what fields
    exist in the state and validates agent outputs against it.
    """

    # -------------------------------------------------------------------------
    # INPUT — set when the request arrives, never modified after that
    # -------------------------------------------------------------------------

    # The user's original question exactly as they typed it
    # Example: "How does Taiwan geopolitical risk affect Apple's revenue?"
    # Set by: graph_pipeline.py when the request arrives
    # Read by: entity_agent, vector_agent, synthesis_agent
    query: str

    # -------------------------------------------------------------------------
    # ENTITY AGENT OUTPUT
    # Set by: entity_agent.py after analysing the query
    # -------------------------------------------------------------------------

    # List of entity names extracted from the query
    # These are the starting points for the Neo4j graph traversal
    # Example: ["Apple", "Taiwan", "TSMC"]
    extracted_entities: Optional[List[str]]

    # The type of each extracted entity
    # Parallel list to extracted_entities — same index = same entity
    # Example: ["Company", "Country", "Company"]
    entity_types: Optional[List[str]]

    # The detected intent of the query — what kind of answer is expected
    # One of: "risk_cascade", "supply_chain", "financial", "competitive", "general"
    # Used by synthesis_agent to frame the answer appropriately
    # Example: "risk_cascade" for "what happens if TSMC shuts down?"
    intent: Optional[str]

    # How many graph hops the entity agent recommends for this query
    # Simple queries (one entity, direct relationship) -> 1 hop
    # Complex queries (multi-company cascade) -> 3 hops
    # The graph_query_agent uses this to decide traversal depth
    recommended_hops: Optional[int]

    # -------------------------------------------------------------------------
    # GRAPH QUERY AGENT OUTPUT
    # Set by: graph_query_agent.py after traversing Neo4j
    # -------------------------------------------------------------------------

    # List of graph traversal results from Neo4j
    # Each item is a dict representing one relationship triple:
    # {
    #   "source":       "TSMC",
    #   "relationship": "SUPPLIES_CHIPS_TO",
    #   "target":       "Apple",
    #   "properties":   {"component": "M3 chip"},
    #   "hop_distance": 1,
    #   "text":         "TSMC -[SUPPLIES_CHIPS_TO]-> Apple (component: M3 chip)"
    # }
    graph_results: Optional[List[Dict[str, Any]]]

    # Human-readable summary of the graph paths found
    # Example: "Found path: ASML -> TSMC -> Apple (3 hops)"
    # Included in the final response for explainability
    graph_paths_summary: Optional[str]

    # -------------------------------------------------------------------------
    # VECTOR AGENT OUTPUT
    # Set by: vector_agent.py after querying ChromaDB
    # -------------------------------------------------------------------------

    # List of text chunk results from ChromaDB vector search
    # Each item is a dict:
    # {
    #   "text":     "Apple relies on TSMC as its sole chip manufacturer...",
    #   "source":   "apple_annual_overview_FY2024.pdf",
    #   "page":     2,
    #   "chunk_id": "apple_annual_overview_FY2024.pdf__page_2__chunk_1",
    #   "score":    0.87
    # }
    vector_results: Optional[List[Dict[str, Any]]]

    # -------------------------------------------------------------------------
    # RERANK AGENT OUTPUT
    # Set by: rerank_agent.py after RRF fusion
    # -------------------------------------------------------------------------

    # The merged and re-ranked list of evidence from both retrievers
    # Items from both graph_results and vector_results appear here,
    # sorted by their final RRF score descending
    # Each item has a "rrf_score" and "retrieval_source" field added
    merged_results: Optional[List[Dict[str, Any]]]

    # The formatted context string passed to GPT-4o in synthesis
    # Built from merged_results by rerank_agent — ready to insert into prompt
    # Includes both graph triples and text chunks in a structured format
    formatted_context: Optional[str]

    # -------------------------------------------------------------------------
    # SYNTHESIS AGENT OUTPUT
    # Set by: synthesis_agent.py — the final step
    # -------------------------------------------------------------------------

    # The complete answer to the user's question
    # Generated by GPT-4o using the formatted_context as evidence
    # This is what gets returned to the user via the FastAPI /query endpoint
    final_answer: Optional[str]

    # List of source documents cited in the final answer
    # Example: ["apple_annual_overview_FY2024.pdf", "asml_technology_report_FY2023.pdf"]
    # Returned to the user so they can verify the answer against source documents
    sources: Optional[List[str]]

    # The graph relationship paths that contributed to the answer
    # Example: ["ASML -[SUPPLIES_TO]-> TSMC -[SUPPLIES_TO]-> Apple"]
    # Returned to the user to show the multi-hop reasoning chain
    # This is what makes Graph RAG explainable — you can see the chain
    graph_paths_used: Optional[List[str]]

    # -------------------------------------------------------------------------
    # ERROR TRACKING
    # -------------------------------------------------------------------------

    # If any agent encounters an error, it sets this field
    # The pipeline checks this field after each agent and can short-circuit
    # to return a graceful error response instead of crashing
    error: Optional[str]

    # Which agent set the error (for debugging)
    # Example: "entity_agent", "graph_query_agent"
    error_source: Optional[str]