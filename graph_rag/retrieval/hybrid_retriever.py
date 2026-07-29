# graph_rag/retrieval/hybrid_retriever.py
#
# =============================================================================
# WHAT IS HYBRID RETRIEVAL AND WHAT TYPE ARE WE USING
# =============================================================================
#
# HYBRID RETRIEVAL — THE CONCEPT
# --------------------------------
# Standard retrieval systems use one method to find relevant information.
# Hybrid retrieval combines two or more different retrieval methods and merges
# their results. The word "hybrid" means exactly what it sounds like — two
# engines running in parallel, each finding things the other would miss,
# their results combined into one ranked list.
#
# THE TWO METHODS WE COMBINE
# ---------------------------
# Method 1: DENSE VECTOR RETRIEVAL (ChromaDB)
#   How it works:
#     Every text chunk is converted to a 1536-dimensional vector (embedding)
#     using OpenAI's text-embedding-3-small model. When a question arrives,
#     the question is also converted to a vector. ChromaDB then finds the
#     chunks whose vectors are mathematically closest (cosine similarity).
#
#   What it is good at:
#     Semantic similarity — finds relevant text even when exact words differ.
#     "chip manufacturing vulnerability" matches "semiconductor production risk"
#     because their vector representations are close, even with zero word overlap.
#
#   What it misses:
#     Relationships and structure. ChromaDB returns isolated chunks. It has no
#     awareness of how TSMC connects to Apple, or how ASML connects to TSMC.
#     Each chunk is an island.
#
# Method 2: GRAPH TRAVERSAL RETRIEVAL (Neo4j)
#   How it works:
#     Entities mentioned in the question (e.g. "ASML", "revenue risk") are
#     identified, then Neo4j traverses the knowledge graph starting from those
#     entities, following relationship edges hop by hop up to TOP_K_GRAPH_HOPS
#     deep. Returns structured triples: (ASML)-[SUPPLIES_TO]->(TSMC).
#
#   What it is good at:
#     Multi-hop reasoning — follows chains of cause and effect across multiple
#     documents. Finds ASML -> TSMC -> Apple -> revenue in one traversal even
#     though no single document contains the complete chain.
#
#   What it misses:
#     Semantic similarity. If the question uses different words than what was
#     extracted into the graph, graph traversal finds nothing. It also returns
#     structural triples, not the rich narrative text with financial figures.
#
# WHY YOU NEED BOTH
# ------------------
# The graph tells you the STRUCTURE (who connects to whom and how).
# The vector store fills in the DETAIL (the actual text with figures and context).
#
# Graph traversal returns: (ASML)-[SUPPLIES_TO]->(TSMC)-[SUPPLIES_TO]->(Apple)
# That is clean structure but has no financial figures, no risk language.
#
# Vector search returns: paragraphs about ASML export restrictions and Apple
# revenue — rich detail but no explicit chain of cause and effect.
#
# Combined: the synthesis agent (GPT-4o) sees both the structural chain AND
# the supporting evidence text, and produces a complete, accurate answer.
#
# THE SPECIFIC TYPE: PARALLEL HYBRID RETRIEVAL WITH RRF
# -------------------------------------------------------
# We use Parallel Hybrid Retrieval with Score-Normalised
# Reciprocal Rank Fusion (RRF).
#
# PARALLEL means both retrievers run simultaneously (via Python threading),
# not sequentially. This halves the retrieval latency compared to running
# them one after the other.
#
# SCORE-NORMALISED means we normalise scores from both systems to a 0-1
# range before combining. ChromaDB returns cosine similarity (0 to 1).
# Neo4j graph results are scored by hop distance (1 hop = most relevant,
# 3 hops = least relevant). These are incompatible scales. Normalisation
# puts them on the same footing so neither system dominates purely because
# its raw numbers happen to be larger.
#
# RECIPROCAL RANK FUSION (RRF) works as follows:
#   Instead of adding raw scores directly, we use the RANK POSITION of each
#   result within its own retriever's output list. The RRF formula is:
#
#       RRF_score = 1 / (k + rank)
#
#   where k = 60 (the standard smoothing constant, prevents top-ranked items
#   from dominating too strongly) and rank starts at 1.
#
#   A result ranked 1st by ChromaDB gets:  1 / (60 + 1) = 0.01639
#   A result ranked 5th by ChromaDB gets:  1 / (60 + 5) = 0.01538
#   The difference is intentionally small — RRF values diversity of evidence
#   over dominance by a single high-scoring result.
#
#   When a result appears in BOTH retrievers' outputs (e.g. a chunk about
#   TSMC-Apple relationship is both semantically similar AND connected in the
#   graph), its RRF scores from both systems are summed. This boosting effect
#   is the core insight of RRF — agreement between different retrieval systems
#   is a strong signal of relevance.
#
# PRACTICAL EFFECT:
#   A chunk that ranks 1st in ChromaDB AND appears as a graph triple gets
#   a higher final score than a chunk that only appears in one system.
#   This rewards evidence that multiple retrieval methods agree on.
#
# =============================================================================
#
# DO YOU RUN THIS FILE?
#   No. Imported and used by the rerank_agent.py inside the LangGraph pipeline.


import concurrent.futures                  # runs both retrievers in parallel threads
from typing import List, Dict, Any, Tuple  # type hints
from loguru import logger                  # structured logging

from graph_rag.retrieval.vector_store import VectorStore
# VectorStore.query() returns semantically similar text chunks from ChromaDB

from graph_rag.graph.neo4j_client import Neo4jClient
# Neo4jClient.run_query() executes Cypher traversals on Neo4j AuraDB

from graph_rag.utils.config import settings  # validated config values


# RRF smoothing constant — standard value used in the original RRF paper
# (Cormack, Clarke, Buettcher 2009). Higher k reduces the score gap between
# top and bottom ranked results, promoting diversity in the final merged list.
RRF_K = 60


class HybridRetriever:
    """
    Combines ChromaDB vector search and Neo4j graph traversal using
    Reciprocal Rank Fusion to produce a single ranked list of evidence.

    The two retrievers run in parallel. Their results are score-normalised,
    ranked, and merged using RRF into one unified context list that the
    synthesis agent uses to generate the final answer.

    Usage:
        retriever = HybridRetriever()
        retriever.load()                       # loads ChromaDB index from disk
        results = retriever.retrieve(
            query="How does Taiwan risk affect Apple?",
            entities=["Apple", "Taiwan", "TSMC"]
        )
    """

    def __init__(self):
        """
        Initialises the vector store and prepares for Neo4j queries.
        Does not load the ChromaDB index yet — call load() for that.
        """
        # VectorStore manages the ChromaDB index
        self.vector_store = VectorStore()

        # Flag to track whether the ChromaDB index has been loaded
        self.is_loaded = False

        logger.info("HybridRetriever initialised")

    def load(self) -> None:
        """
        Loads the ChromaDB index from disk.
        Must be called before retrieve(). Called once at application startup.
        """
        self.vector_store.load_index()
        self.is_loaded = True
        logger.info("HybridRetriever loaded and ready")

    def retrieve(
        self,
        query: str,
        entities: List[str],
        top_k_vector: int = None,
        top_k_graph_hops: int = None,
    ) -> Dict[str, Any]:
        """
        Runs both retrievers in parallel and merges results using RRF.

        This is the main method called by the rerank_agent during a query.
        It runs ChromaDB and Neo4j simultaneously, then combines their outputs
        into a single ranked evidence list.

        Args:
            query:           The user's original question string.
            entities:        List of entity names extracted from the question
                             by entity_agent.py. Used as starting points for
                             the Neo4j graph traversal.
                             Example: ["Apple", "TSMC", "Taiwan"]
            top_k_vector:    Number of chunks to retrieve from ChromaDB.
                             Defaults to settings.top_k_vector (from .env).
            top_k_graph_hops: Depth of Neo4j graph traversal.
                             Defaults to settings.top_k_graph_hops (from .env).

        Returns:
            A dictionary with:
              "vector_results"  : raw results from ChromaDB
              "graph_results"   : raw results from Neo4j
              "merged_results"  : RRF-merged and ranked combined results
              "query"           : the original query string
              "entities_used"   : entities used for graph traversal
        """

        if not self.is_loaded:
            raise RuntimeError(
                "HybridRetriever is not loaded. Call load() first."
            )

        # Use configured defaults if not overridden by the caller
        top_k_vector     = top_k_vector     or settings.top_k_vector
        top_k_graph_hops = top_k_graph_hops or settings.top_k_graph_hops

        logger.info(
            f"Running hybrid retrieval — "
            f"query: '{query[:60]}...', "
            f"entities: {entities}"
        )

        # Run both retrievers in parallel using a thread pool
        # ThreadPoolExecutor runs both functions concurrently in separate threads
        # This halves wall-clock retrieval time vs running them sequentially
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            # Submit both retrieval tasks to the thread pool simultaneously
            vector_future = executor.submit(
                self._retrieve_from_vector_store,
                query,
                top_k_vector,
            )
            graph_future = executor.submit(
                self._retrieve_from_graph,
                entities,
                top_k_graph_hops,
            )

            # Wait for both to complete and get their results
            # .result() blocks until the thread finishes and returns its value
            vector_results = vector_future.result()
            graph_results  = graph_future.result()

        logger.info(
            f"Retrieval complete — "
            f"{len(vector_results)} vector chunks, "
            f"{len(graph_results)} graph triples"
        )

        # Merge results from both systems using Reciprocal Rank Fusion
        merged_results = self._reciprocal_rank_fusion(
            vector_results,
            graph_results,
        )

        return {
            "vector_results": vector_results,
            "graph_results":  graph_results,
            "merged_results": merged_results,
            "query":          query,
            "entities_used":  entities,
        }

    def _retrieve_from_vector_store(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for the most semantically similar chunks.

        This runs in a background thread during hybrid retrieval.
        Returns the raw list from VectorStore.query().

        Args:
            query: The user's question.
            top_k: Number of chunks to return.

        Returns:
            List of chunk result dicts from VectorStore.query().
        """
        try:
            results = self.vector_store.query(query, top_k=top_k)
            logger.debug(f"Vector store returned {len(results)} chunks")
            return results
        except Exception as e:
            logger.error(f"Vector store retrieval failed: {e}")
            return []

    def _retrieve_from_graph(
        self,
        entities: List[str],
        max_hops: int,
    ) -> List[Dict[str, Any]]:
        """
        Traverses the Neo4j knowledge graph starting from the given entities.

        For each entity, runs a variable-depth Cypher traversal up to max_hops
        deep and returns all paths found as structured result dicts.

        This runs in a background thread during hybrid retrieval.

        Args:
            entities: Entity names to start traversal from.
                      Example: ["Apple", "TSMC"]
            max_hops: Maximum relationship hops to traverse.
                      3 hops covers ASML -> TSMC -> Apple -> revenue.

        Returns:
            List of graph result dicts, each representing one path:
            {
                "type":         "graph_triple",
                "source":       "TSMC",
                "relationship": "SUPPLIES_CHIPS_TO",
                "target":       "Apple",
                "properties":   {"component": "M3 chip"},
                "hop_distance": 1,
                "text":         "TSMC -[SUPPLIES_CHIPS_TO]-> Apple",
                "score":        1.0    <- normalised: 1 hop = score 1.0
            }
        """
        if not entities:
            logger.debug("No entities provided for graph traversal")
            return []

        graph_results = []

        try:
            with Neo4jClient() as client:
                for entity_name in entities:

                    # Cypher query for variable-depth traversal
                    # [*1..max_hops] means "follow between 1 and max_hops edges"
                    # DISTINCT prevents duplicate paths being returned
                    # We return the start node, the end node, the relationship
                    # type, and the hop length for each path found
                    cypher = """
                        MATCH path = (start {name: $entity_name})-[*1..$max_hops]-(end)
                        WITH start, end,
                             relationships(path)[0] AS first_rel,
                             length(path) AS hop_distance
                        RETURN DISTINCT
                            start.name       AS source,
                            type(first_rel)  AS relationship,
                            end.name         AS target,
                            hop_distance,
                            properties(first_rel) AS rel_properties
                        ORDER BY hop_distance ASC
                        LIMIT 20
                    """

                    results = client.run_query(
                        cypher,
                        parameters={
                            "entity_name": entity_name,
                            "max_hops":    max_hops,
                        },
                    )

                    for row in results:
                        source       = row.get("source", "")
                        relationship = row.get("relationship", "")
                        target       = row.get("target", "")
                        hop_distance = row.get("hop_distance", 1)
                        properties   = row.get("rel_properties", {})

                        # Build a human-readable text representation
                        # This text is what the synthesis agent reads
                        text = f"{source} -[{relationship}]-> {target}"
                        if properties:
                            props_str = ", ".join(
                                f"{k}: {v}" for k, v in properties.items()
                            )
                            text += f" ({props_str})"

                        # Score: closer hops are more relevant
                        # 1 hop = score 1.0, 2 hops = 0.5, 3 hops = 0.33
                        # This normalises hop distance to a 0-1 range
                        normalised_score = 1.0 / hop_distance

                        graph_results.append({
                            "type":         "graph_triple",
                            "source":       source,
                            "relationship": relationship,
                            "target":       target,
                            "properties":   properties,
                            "hop_distance": hop_distance,
                            "text":         text,
                            "score":        normalised_score,
                        })

            logger.debug(
                f"Graph traversal returned {len(graph_results)} triples "
                f"for entities: {entities}"
            )

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []

        return graph_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results:  List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merges vector and graph results using Reciprocal Rank Fusion.

        RRF formula:  score = 1 / (k + rank)
        where k = 60 (standard constant) and rank starts at 1.

        Results that appear highly ranked in both systems get their RRF scores
        summed, giving them a higher final score than results that only appear
        in one system. This is the core insight of RRF — agreement between
        retrieval systems is a strong signal of relevance.

        Args:
            vector_results: List of chunk dicts from ChromaDB.
            graph_results:  List of triple dicts from Neo4j.

        Returns:
            A single merged list sorted by final RRF score descending.
            Each item has a "rrf_score" field and a "retrieval_source" field
            indicating which system it came from ("vector", "graph", or "both").
        """

        # Dictionary to accumulate RRF scores for each result
        # Key: a unique identifier for the result
        # Value: dict with the result data and accumulated RRF score
        rrf_scores: Dict[str, Dict[str, Any]] = {}

        # Process vector results — rank starts at 1 for the best match
        for rank, result in enumerate(vector_results, start=1):

            # Use chunk_id as the unique key for vector results
            result_key = f"vector_{result.get('chunk_id', rank)}"

            # RRF score for this rank position
            rrf_score = 1.0 / (RRF_K + rank)

            rrf_scores[result_key] = {
                **result,                          # all fields from the original result
                "rrf_score":        rrf_score,
                "retrieval_source": "vector",
                "result_type":      "chunk",
            }

        # Process graph results — rank starts at 1 for the closest hop
        for rank, result in enumerate(graph_results, start=1):

            # Use a composite key for graph triples
            result_key = (
                f"graph_{result.get('source', '')}__"
                f"{result.get('relationship', '')}__"
                f"{result.get('target', '')}"
            )

            rrf_score = 1.0 / (RRF_K + rank)

            if result_key in rrf_scores:
                # This result appeared in both systems — sum the RRF scores
                # This is the RRF boosting effect for agreement between systems
                rrf_scores[result_key]["rrf_score"] += rrf_score
                rrf_scores[result_key]["retrieval_source"] = "both"
            else:
                rrf_scores[result_key] = {
                    **result,
                    "rrf_score":        rrf_score,
                    "retrieval_source": "graph",
                    "result_type":      "graph_triple",
                }

        # Convert the dictionary to a list and sort by RRF score descending
        # Higher RRF score = more relevant = appears first
        merged = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        logger.info(
            f"RRF fusion complete — "
            f"{len(merged)} total results merged. "
            f"Sources: "
            f"{sum(1 for r in merged if r['retrieval_source'] == 'vector')} vector, "
            f"{sum(1 for r in merged if r['retrieval_source'] == 'graph')} graph, "
            f"{sum(1 for r in merged if r['retrieval_source'] == 'both')} both"
        )

        return merged