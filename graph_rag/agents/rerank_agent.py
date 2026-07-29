# graph_rag/agents/rerank_agent.py
#
# PURPOSE:
#   This is the fourth agent in the LangGraph pipeline.
#   It runs after both graph_query_agent and vector_agent have completed.
#   Its job is to take the raw results from both retrievers and merge
#   them into a single ranked list of evidence, then format that list
#   into a clean context string that GPT-4o can reason over.
#
# WHY DO WE NEED A RERANK AGENT?
#   After the parallel retrieval step we have two separate lists:
#     - graph_results  : structured triples from Neo4j
#                        e.g. TSMC -[SUPPLIES_CHIPS_TO]-> Apple
#     - vector_results : text chunks from ChromaDB
#                        e.g. "Apple relies on TSMC as its sole foundry..."
#
#   These are completely different formats with incompatible scores.
#   ChromaDB scores are cosine similarity (0 to 1, higher is better).
#   Graph scores are hop-distance based (1 hop = 1.0, 3 hops = 0.33).
#
#   Simply concatenating them and sending everything to GPT-4o would:
#     1. Overwhelm the context window with irrelevant evidence
#     2. Put the most relevant evidence last (bad — LLMs attend more
#        to early context)
#     3. Mix formats in a way GPT-4o finds hard to reason over
#
#   The rerank agent solves all three problems:
#     1. Applies RRF to score and rank everything together
#     2. Keeps only the top N results (fits context window)
#     3. Formats everything into a clean, structured context string
#
# =============================================================================
# WHAT IS RRF (RECIPROCAL RANK FUSION)? — READ THIS, IT IS THE HEART OF THIS FILE
# =============================================================================
#
# THE PROBLEM RRF SOLVES:
#   We have two ranked lists from two different retrievers, and their
#   scores are NOT comparable. A ChromaDB cosine score of 0.87 and a graph
#   hop-distance score of 1.0 measure completely different things — you
#   cannot just put them side by side and sort. We need one fair way to
#   combine two rankings into one.
#
# THE KEY INSIGHT:
#   RRF throws away the raw scores entirely and uses only the RANK — the
#   POSITION of a result in its list (1st, 2nd, 3rd...). Rank is something
#   both lists have in common: "1st place in the graph list" and "1st place
#   in the vector list" are directly comparable, even though their original
#   scores are not. This is why RRF is so robust: it never has to reconcile
#   two incompatible scoring scales.
#
# THE FORMULA:
#
#       score(result) = sum over each list it appears in of   1 / (k + rank)
#
#   where:
#     - rank = the result's position in that list, starting at 1 (best)
#     - k    = a smoothing constant, here 60 (see "WHY k = 60" below)
#
#   A result gets a contribution from EVERY list it shows up in, and those
#   contributions are ADDED together. So a result that ranks well in BOTH
#   the graph list and the vector list ends up with a higher total than a
#   result that only appears in one. Agreement between the two independent
#   retrievers is treated as a strong signal of relevance — that boosting
#   is the whole point of "fusion".
#
# A WORKED EXAMPLE (with k = 60):
#
#   Suppose one particular fact — "TSMC supplies Apple" — appears as:
#     - rank 1 in the graph list   -> 1 / (60 + 1) = 1/61 = 0.01639
#     - rank 3 in the vector list  -> 1 / (60 + 3) = 1/63 = 0.01587
#   Because it appeared in BOTH, we add them:
#     total = 0.01639 + 0.01587 = 0.03226
#
#   Now suppose another fact appears only once:
#     - rank 1 in the vector list  -> 1 / (60 + 1) = 0.01639   (that's all)
#
#   Even though both were "1st place somewhere", the first fact scores
#   ~0.032 vs ~0.016 — roughly double — purely because both retrievers
#   independently surfaced it. That is RRF rewarding agreement.
#
# WHY DIVIDE BY (k + rank) INSTEAD OF JUST rank?
#   Two reasons, both about the constant k:
#     1. It DAMPENS the gap between top positions. Without k, rank 1 scores
#        1/1 = 1.0 and rank 2 scores 1/2 = 0.5 — a huge, brutal drop that
#        lets a single list's top result dominate everything. With k = 60,
#        rank 1 is 1/61 and rank 2 is 1/62 — very close. So no single result
#        from one list can steamroll the fused ranking; the results have to
#        earn their place by appearing across lists.
#     2. It keeps every score positive and well-behaved (never divides by
#        zero, never explodes).
#
# WHY k = 60?
#   60 is the value from the original RRF paper (Cormack et al., 2009) and
#   has become the community default because it works well across many
#   retrieval tasks. It is a knob, not a law: a smaller k makes top ranks
#   matter more (sharper), a larger k flattens the differences (gentler).
#   We keep 60 to stay consistent with the wider ecosystem and with the
#   ingestion-side hybrid retriever.
#
# WHY RRF INSTEAD OF, SAY, AVERAGING THE SCORES?
#   Averaging requires the two score scales to mean the same thing — they
#   do not (cosine similarity vs hop distance). Normalising them is fiddly
#   and fragile. RRF sidesteps all of that by using rank only. It is simple,
#   has no per-retriever tuning, and is hard to break — ideal for fusing a
#   graph retriever and a vector retriever that speak different languages.
#
# THE ONE THING RRF IGNORES (be honest about this):
#   By using rank only, RRF discards HOW MUCH better rank 1 was than rank 2.
#   If the top vector chunk was a near-perfect match (0.95) and the second
#   was mediocre (0.55), RRF treats them as just "1st and 2nd" — it loses
#   that margin. In exchange you get robustness and no score-scale
#   reconciliation. For fusing heterogeneous retrievers, that trade is
#   almost always worth it.
# =============================================================================
#
# THE FORMATTED CONTEXT:
#   After ranking, the rerank agent produces a formatted_context string
#   that looks like:
#
#   GRAPH EVIDENCE (relationship chains from knowledge graph):
#   1. (Company)TSMC -[SUPPLIES_CHIPS_TO]-> (Company)Apple [component: M3 chip]
#   2. (Company)ASML -[SUPPLIES_EQUIPMENT_TO]-> (Company)TSMC [equipment: EUV machine]
#   3. (Company)TSMC -[OPERATES_IN]-> (Country)Taiwan
#
#   TEXT EVIDENCE (relevant passages from source documents):
#   1. [apple_annual_overview_FY2024.pdf, page 2]
#      Apple relies on TSMC as its primary and sole foundry partner for the
#      manufacture of all Apple-designed chips...
#   2. [tsmc_manufacturing_report_FY2023.pdf, page 1]
#      TSMC's primary manufacturing operations are concentrated in Taiwan...
#
#   This structured format makes it easy for GPT-4o to distinguish between
#   structural graph evidence and detailed text evidence.
#
# DO YOU RUN THIS FILE?
#   No. Registered as a LangGraph node in graph_pipeline.py.
#   Called automatically after both parallel retrieval agents finish.


from typing import Dict, Any, List         # type hints
from loguru import logger                  # structured logging

from graph_rag.agents.state import GraphRAGState  # shared state schema
from graph_rag.utils.config import settings        # validated config


# RRF smoothing constant — the k in  score = 1 / (k + rank).
# 60 is the original-paper default; see the RRF explanation above for why
# a larger k flattens the gap between top ranks. Kept consistent with the
# ingestion-side hybrid retriever so scoring is comparable across the system.
RRF_K = 60

# Maximum number of results to include in the formatted context
# sent to GPT-4o. Too many results = context window overflow.
# Too few = insufficient evidence for complex multi-hop questions.
# 10 is a good balance for our five-document corpus.
MAX_CONTEXT_RESULTS = 10


def rerank_agent(state: GraphRAGState) -> Dict[str, Any]:
    """
    LangGraph node — merges graph and vector results using RRF,
    then formats the top results into a context string for GPT-4o.

    Reads graph_results and vector_results from state (written by the
    two parallel agents). Applies RRF scoring to rank everything
    together. Formats the top MAX_CONTEXT_RESULTS into a structured
    context string stored as formatted_context in state.

    Args:
        state: Current pipeline state. Must have:
               - graph_results  : list of triple dicts from graph_query_agent
               - vector_results : list of chunk dicts from vector_agent

    Returns:
        Partial state dict with:
          - merged_results    : RRF-ranked combined list
          - formatted_context : structured string ready for GPT-4o prompt
        Or on error:
          - error, error_source
    """

    graph_results  = state.get("graph_results",  []) or []
    vector_results = state.get("vector_results", []) or []

    logger.info(
        f"Rerank agent running — "
        f"{len(graph_results)} graph triples, "
        f"{len(vector_results)} vector chunks"
    )

    # Handle the case where both retrievers returned nothing
    # This should not happen in normal operation but we handle it gracefully
    if not graph_results and not vector_results:
        logger.warning(
            "Rerank agent received no results from either retriever. "
            "Both graph and vector results are empty."
        )
        return {
            "merged_results":    [],
            "formatted_context": "No evidence found from either retrieval system.",
        }

    try:
        # Step 1: Apply RRF to merge and rank both result lists
        merged_results = _apply_rrf(graph_results, vector_results)

        # Step 2: Take only the top MAX_CONTEXT_RESULTS for the context window
        top_results = merged_results[:MAX_CONTEXT_RESULTS]

        # Step 3: Format the top results into a structured context string
        formatted_context = _format_context(top_results, state)

        logger.success(
            f"Rerank agent complete — "
            f"{len(merged_results)} total merged results, "
            f"top {len(top_results)} selected for context. "
            f"Graph: {sum(1 for r in top_results if r.get('result_type') == 'graph_triple')}, "
            f"Vector: {sum(1 for r in top_results if r.get('result_type') == 'chunk')}"
        )

        return {
            "merged_results":    merged_results,
            "formatted_context": formatted_context,
        }

    except Exception as e:
        logger.error(f"Rerank agent failed: {e}")
        return {
            "error":          f"Rerank agent failed: {str(e)}",
            "error_source":   "rerank_agent",
            "merged_results": [],
            "formatted_context": "",
        }


def _apply_rrf(
    graph_results:  List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Applies Reciprocal Rank Fusion to merge graph and vector results.

    RRF formula:  score = 1 / (k + rank)   with k = 60 and rank starting
    at 1 for the best result in each list. A result that appears in BOTH
    lists gets a contribution from each, and the two are ADDED — so
    agreement between the retrievers boosts a result up the ranking. See
    the long RRF explanation in this file's header for a worked example
    and the reasoning behind k.

    Args:
        graph_results:  List of triple dicts from graph_query_agent.
        vector_results: List of chunk dicts from vector_agent.

    Returns:
        Single merged list sorted by RRF score descending.
        Each item has "rrf_score" and "retrieval_source" fields added.
    """

    # Dictionary to accumulate RRF scores keyed by a unique result ID.
    # The key is what lets us detect "this same fact came from both lists"
    # and add their contributions together.
    rrf_scores: Dict[str, Dict[str, Any]] = {}

    # --- Process graph results ---
    # graph_results are already sorted by hop_distance (closest first)
    # so rank 1 = most directly relevant relationship
    for rank, result in enumerate(graph_results, start=1):

        # Build a unique key from source + relationship + target
        key = (
            f"graph__{result.get('source', '').lower()}"
            f"__{result.get('relationship', '')}"
            f"__{result.get('target', '').lower()}"
        )

        # This is the RRF contribution for this result's position:
        #   1 / (60 + rank)
        rrf_score = 1.0 / (RRF_K + rank)

        rrf_scores[key] = {
            **result,
            "rrf_score":        rrf_score,
            "retrieval_source": "graph",
            "result_type":      "graph_triple",
        }

    # --- Process vector results ---
    # vector_results are sorted by cosine similarity (highest first)
    # so rank 1 = most semantically similar chunk
    for rank, result in enumerate(vector_results, start=1):

        # Use chunk_id as the unique key for vector results
        key = f"vector__{result.get('chunk_id', rank)}"

        rrf_score = 1.0 / (RRF_K + rank)

        if key in rrf_scores:
            # This result appeared in both systems — ADD the scores.
            # This is the "fusion" step: cross-retriever agreement is
            # rewarded, pushing the result higher in the final ranking.
            rrf_scores[key]["rrf_score"]        += rrf_score
            rrf_scores[key]["retrieval_source"]  = "both"
        else:
            rrf_scores[key] = {
                **result,
                "rrf_score":        rrf_score,
                "retrieval_source": "vector",
                "result_type":      "chunk",
            }

    # Sort by RRF score descending — highest score = most relevant = first
    merged = sorted(
        rrf_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return merged


def _format_context(
    top_results: List[Dict[str, Any]],
    state: GraphRAGState,
) -> str:
    """
    Formats the top-ranked results into a structured context string
    for the GPT-4o synthesis prompt.

    Separates results into two sections:
      1. GRAPH EVIDENCE   — the relationship triples from Neo4j
      2. TEXT EVIDENCE    — the text chunks from ChromaDB

    This separation helps GPT-4o understand which evidence is structural
    (graph) and which is detailed narrative (text).

    Args:
        top_results: The top MAX_CONTEXT_RESULTS from _apply_rrf().
        state:       The full pipeline state (used to add query context).

    Returns:
        A formatted multi-line string ready to insert into the synthesis prompt.
    """

    query  = state.get("query", "")
    intent = state.get("intent", "general")

    # Separate results by type
    graph_items  = [r for r in top_results if r.get("result_type") == "graph_triple"]
    vector_items = [r for r in top_results if r.get("result_type") == "chunk"]

    lines = []

    # --- Header ---
    lines.append(f"QUERY: {query}")
    lines.append(f"DETECTED INTENT: {intent}")
    lines.append("")

    # --- Graph Evidence Section ---
    if graph_items:
        lines.append("=" * 60)
        lines.append("GRAPH EVIDENCE (relationship chains from knowledge graph):")
        lines.append("=" * 60)

        for i, item in enumerate(graph_items, start=1):
            # The text field already contains a formatted triple string
            # e.g. "(Company)TSMC -[SUPPLIES_CHIPS_TO]-> (Company)Apple [component: M3 chip]"
            text         = item.get("text", "")
            hop_distance = item.get("hop_distance", "?")
            rrf_score    = item.get("rrf_score", 0)
            source_label = item.get("retrieval_source", "graph")

            lines.append(
                f"{i}. [hop {hop_distance}, score {rrf_score:.4f}, source: {source_label}]"
            )
            lines.append(f"   {text}")
            lines.append("")

    else:
        lines.append("GRAPH EVIDENCE: No graph relationships found.")
        lines.append("")

    # --- Text Evidence Section ---
    if vector_items:
        lines.append("=" * 60)
        lines.append("TEXT EVIDENCE (relevant passages from source documents):")
        lines.append("=" * 60)

        for i, item in enumerate(vector_items, start=1):
            source    = item.get("source", "unknown")
            page      = item.get("page", "?")
            text      = item.get("text", "")
            rrf_score = item.get("rrf_score", 0)
            company   = item.get("company", "")

            # Truncate very long chunks to keep context window manageable
            # 600 characters is enough to preserve the key sentences
            if len(text) > 600:
                text = text[:600] + "..."

            lines.append(
                f"{i}. [{source}, page {page}, company: {company}, "
                f"score {rrf_score:.4f}]"
            )
            lines.append(f"   {text}")
            lines.append("")

    else:
        lines.append("TEXT EVIDENCE: No text chunks found.")
        lines.append("")

    return "\n".join(lines)