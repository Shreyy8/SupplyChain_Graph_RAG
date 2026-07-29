# graph_rag/pipeline/graph_pipeline.py
#
# PURPOSE:
#   This file wires all five agents into a LangGraph StateGraph and
#   compiles it into a runnable pipeline.
#
#   This is the most important structural file in the project.
#   Everything else we have built — entities, graph traversal, vector
#   search, reranking, synthesis — comes together here.
#
# WHAT IS A LANGGRAPH STATEGRAPH?
# ---------------------------------
# A StateGraph is a directed graph where:
#   - NODES are the agents (Python functions)
#   - EDGES are the connections between agents (data flow)
#   - STATE is the shared dictionary passed from node to node
#
# Think of it as a flowchart where each box is an agent function
# and each arrow is "when this agent finishes, run that agent next."
#
# THE PIPELINE STRUCTURE WE BUILD HERE:
#
#   START
#     |
#     v
#   [entity_agent]              -- extracts entities + intent from query
#     |
#     |-----> [graph_query_agent]   \
#     |                              > run in parallel
#     |-----> [vector_agent]        /
#                    |
#                    v
#             [rerank_agent]    -- merges + ranks results from both
#                    |
#                    v
#             [synthesis_agent] -- generates final answer
#                    |
#                    v
#                   END
#
# PARALLEL EXECUTION:
#   LangGraph supports parallel node execution natively.
#   We wire both graph_query_agent and vector_agent to run after
#   entity_agent completes. LangGraph fires both simultaneously and
#   waits for both to finish before passing state to rerank_agent.
#   This is defined using a branching edge from entity_agent to both
#   parallel agents, then a joining edge from both to rerank_agent.
#
# ERROR HANDLING:
#   After entity_agent runs, we check the state for an error field.
#   If entity_agent failed, we route directly to synthesis_agent
#   which returns a graceful error message. This conditional routing
#   is done using LangGraph's add_conditional_edges() method.
#
# HOW TO USE THE COMPILED PIPELINE:
#   from graph_rag.pipeline.graph_pipeline import create_pipeline
#   pipeline = create_pipeline()
#   result   = pipeline.invoke({"query": "Your question here"})
#   print(result["final_answer"])
#
# DO YOU RUN THIS FILE?
#   No. Imported by api/routes/query.py and scripts/test_queries.py.


from langgraph.graph import StateGraph, END
# StateGraph : the main LangGraph class for building agent pipelines
# END        : a special constant marking the terminal node of the graph

from loguru import logger

from graph_rag.agents.state import GraphRAGState
# GraphRAGState : the TypedDict schema for shared state

from graph_rag.agents.entity_agent       import entity_agent
from graph_rag.agents.graph_query_agent  import graph_query_agent
from graph_rag.agents.vector_agent       import vector_agent
from graph_rag.agents.rerank_agent       import rerank_agent
from graph_rag.agents.synthesis_agent    import synthesis_agent
# All five agent functions imported so we can register them as nodes


def _should_continue_after_entity(state: GraphRAGState) -> str:
    """
    Conditional routing function called after entity_agent completes.

    LangGraph calls this function with the current state after entity_agent
    runs. The function returns a string that tells LangGraph which node
    to go to next.

    If entity_agent encountered an error (e.g. GPT-4o API failure),
    we skip straight to synthesis_agent which returns a graceful error
    message to the user rather than crashing the pipeline.

    If entity_agent succeeded, we return "parallel_retrieval" which
    routes to both graph_query_agent and vector_agent simultaneously.

    Args:
        state: The current pipeline state after entity_agent ran.

    Returns:
        "parallel_retrieval" : if entity_agent succeeded
        "synthesis"          : if entity_agent failed (skip to error response)
    """

    if state.get("error"):
        logger.warning(
            f"Error detected after entity_agent: {state.get('error')}. "
            f"Routing directly to synthesis for graceful error response."
        )
        return "synthesis"

    return "parallel_retrieval"


def create_pipeline():
    """
    Builds, wires, and compiles the LangGraph StateGraph pipeline.

    This function is called once at application startup (in FastAPI's
    lifespan handler) to create the compiled pipeline object.
    The compiled pipeline is then reused for every incoming query.

    Returns:
        A compiled LangGraph pipeline (CompiledGraph object).
        Call it with: pipeline.invoke({"query": "your question"})
        It returns the final state dict with "final_answer" populated.

    Pipeline structure:
        START -> entity_agent -> [graph_query_agent || vector_agent]
              -> rerank_agent -> synthesis_agent -> END
    """

    logger.info("Building LangGraph pipeline...")

    # ------------------------------------------------------------------
    # STEP 1: Create the StateGraph
    # ------------------------------------------------------------------
    # StateGraph(GraphRAGState) tells LangGraph the schema of our state.
    # LangGraph uses this to validate that agent outputs match the schema.
    workflow = StateGraph(GraphRAGState)

    # ------------------------------------------------------------------
    # STEP 2: Register all five agents as nodes
    # ------------------------------------------------------------------
    # add_node(name, function) registers a Python function as a node.
    # The name string is how we refer to this node when adding edges.
    # The function must accept state dict and return a partial state dict.

    workflow.add_node("entity_agent",      entity_agent)
    workflow.add_node("graph_query_agent", graph_query_agent)
    workflow.add_node("vector_agent",      vector_agent)
    workflow.add_node("rerank_agent",      rerank_agent)
    workflow.add_node("synthesis_agent",   synthesis_agent)

    # ------------------------------------------------------------------
    # STEP 3: Set the entry point
    # ------------------------------------------------------------------
    # set_entry_point() tells LangGraph which node to run first when
    # pipeline.invoke() is called. Always entity_agent — it reads
    # the query and extracts entities before anything else can run.
    workflow.set_entry_point("entity_agent")

    # ------------------------------------------------------------------
    # STEP 4: Add conditional edge from entity_agent
    # ------------------------------------------------------------------
    # add_conditional_edges() adds a routing function that runs after
    # entity_agent completes. The routing function (_should_continue_after_entity)
    # returns a string key. The path_map dict maps string keys to node names.
    #
    # "parallel_retrieval" -> runs both graph_query_agent AND vector_agent
    # "synthesis"          -> skips retrieval, goes straight to synthesis
    #                         (used when entity_agent fails)
    workflow.add_conditional_edges(
        "entity_agent",                     # source node
        _should_continue_after_entity,      # routing function
        {
            # When routing function returns "parallel_retrieval",
            # LangGraph fans out to BOTH graph_query_agent and vector_agent
            # simultaneously. Both run in parallel until both finish.
            "parallel_retrieval": "graph_query_agent",

            # When routing function returns "synthesis", skip retrieval
            "synthesis":          "synthesis_agent",
        }
    )

    # ------------------------------------------------------------------
    # STEP 5: Add the second parallel branch
    # ------------------------------------------------------------------
    # LangGraph handles parallelism by having multiple nodes with edges
    # pointing to the same downstream node. Both graph_query_agent and
    # vector_agent feed into rerank_agent.
    #
    # We need to explicitly add the edge from entity_agent to vector_agent
    # as well, to enable the parallel fan-out.
    workflow.add_edge("entity_agent", "vector_agent")

    # ------------------------------------------------------------------
    # STEP 6: Add edges from parallel agents to rerank_agent
    # ------------------------------------------------------------------
    # Both graph_query_agent and vector_agent connect to rerank_agent.
    # LangGraph waits for BOTH to finish before running rerank_agent.
    # This is the "join" step after the parallel "fan-out".
    workflow.add_edge("graph_query_agent", "rerank_agent")
    workflow.add_edge("vector_agent",      "rerank_agent")

    # ------------------------------------------------------------------
    # STEP 7: Add final sequential edges
    # ------------------------------------------------------------------
    # rerank_agent -> synthesis_agent -> END
    # After reranking, synthesis generates the answer and the pipeline ends.
    workflow.add_edge("rerank_agent",    "synthesis_agent")
    workflow.add_edge("synthesis_agent", END)

    # ------------------------------------------------------------------
    # STEP 8: Compile the workflow
    # ------------------------------------------------------------------
    # compile() validates the graph structure (checks for disconnected nodes,
    # missing edges, etc.) and returns a CompiledGraph object that can be
    # invoked with .invoke() or called asynchronously with .ainvoke().
    pipeline = workflow.compile()

    logger.success(
        "LangGraph pipeline compiled successfully. "
        "Nodes: entity_agent -> [graph_query_agent || vector_agent] "
        "-> rerank_agent -> synthesis_agent"
    )

    return pipeline


def run_query(query: str) -> dict:
    """
    Convenience function to run a single query through the pipeline.

    Creates the pipeline, invokes it with the query, and returns the
    final state dictionary. The most important field in the returned
    dict is "final_answer".

    This function is used by scripts/test_queries.py for running
    the demo questions from the command line.

    Args:
        query: The user's question string.

    Returns:
        The final pipeline state dictionary containing:
          - final_answer     : the complete answer
          - sources          : list of cited source documents
          - graph_paths_used : list of graph relationship paths used
          - extracted_entities: entities found in the question
          - intent           : detected query intent
          (plus all other state fields populated by each agent)

    Example:
        result = run_query("How does Taiwan risk affect Apple revenue?")
        print(result["final_answer"])
        print("Sources:", result["sources"])
        print("Graph paths:", result["graph_paths_used"])
    """

    logger.info(f"Running query: '{query}'")

    # Create a fresh pipeline instance
    pipeline = create_pipeline()

    # invoke() runs the full pipeline synchronously and returns final state
    # The initial state has only "query" populated — agents fill the rest
    final_state = pipeline.invoke({"query": query})

    logger.info("Query complete.")

    return final_state