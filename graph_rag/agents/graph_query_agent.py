# graph_rag/agents/graph_query_agent.py
#
# PURPOSE:
#   This is the second agent in the LangGraph pipeline.
#   It receives the entities extracted by entity_agent.py and traverses
#   the Neo4j knowledge graph starting from those entities.
#
#   Its job is to find all the relationship chains (paths) connected to
#   the entities in the user's question, up to N hops deep.
#
# HOW IT FITS IN THE PIPELINE:
#   entity_agent.py  -->  extracted_entities: ["ASML", "TSMC", "Apple"]
#                                    |
#                                    v
#                         graph_query_agent.py
#                                    |
#                              Neo4j AuraDB
#                          (graph traversal)
#                                    |
#                                    v
#                    graph_results: [
#                      {source: "ASML",  rel: "SUPPLIES_TO",    target: "TSMC"},
#                      {source: "TSMC",  rel: "SUPPLIES_TO",    target: "Apple"},
#                      {source: "TSMC",  rel: "MANUFACTURES_FOR", target: "Nvidia"},
#                      {source: "TSMC",  rel: "OPERATES_IN",    target: "Taiwan"},
#                      ...
#                    ]
#
# THIS RUNS IN PARALLEL WITH vector_agent.py:
#   LangGraph runs graph_query_agent and vector_agent simultaneously
#   (defined in graph_pipeline.py). Both write to different state fields
#   so there is no conflict. This is why we have separate agents for
#   graph retrieval and vector retrieval rather than combining them —
#   parallel execution halves the retrieval latency.
#
# CYPHER STRATEGY:
#   We run two types of Cypher queries per entity:
#
#   Query 1 — DIRECT RELATIONSHIPS (1 hop):
#     Finds all nodes directly connected to the entity.
#     Fast, always runs regardless of recommended_hops.
#     Gives us the immediate neighbourhood of each entity.
#
#   Query 2 — MULTI-HOP PATHS (up to recommended_hops):
#     Finds all paths up to N hops from the entity.
#     Reveals the cascade chains our demo questions depend on.
#     Only runs if recommended_hops > 1.
#
# DO YOU RUN THIS FILE?
#   No. Registered as a LangGraph node in graph_pipeline.py.


from typing import Dict, Any, List         # type hints
from loguru import logger                  # structured logging

from graph_rag.agents.state import GraphRAGState  # shared state schema
from graph_rag.graph.neo4j_client import Neo4jClient  # Neo4j connection
from graph_rag.utils.config import settings           # validated config


def graph_query_agent(state: GraphRAGState) -> Dict[str, Any]:
    """
    LangGraph node — traverses Neo4j to find relationship chains.

    Reads extracted_entities and recommended_hops from state, runs
    Cypher traversal queries against Neo4j AuraDB, and returns
    a partial state update with all discovered graph triples.

    Args:
        state: Current pipeline state. Must have:
               - extracted_entities : list of entity name strings
               - recommended_hops   : int traversal depth from entity_agent

    Returns:
        Partial state dict with:
          - graph_results      : list of relationship triple dicts
          - graph_paths_summary: human-readable summary of paths found
        Or on error:
          - error, error_source
    """

    entities         = state.get("extracted_entities", [])
    recommended_hops = state.get("recommended_hops", 2)

    logger.info(
        f"Graph query agent running — "
        f"entities: {entities}, hops: {recommended_hops}"
    )

    # If entity_agent found no entities, we cannot traverse the graph
    # Return empty results — the vector agent will still run in parallel
    # so the pipeline can continue with vector-only retrieval
    if not entities:
        logger.warning(
            "Graph query agent received no entities. "
            "Returning empty graph results. "
            "Pipeline will continue with vector retrieval only."
        )
        return {
            "graph_results":       [],
            "graph_paths_summary": "No entities found — graph traversal skipped.",
        }

    # Collect all graph triples across all entities
    all_graph_results: List[Dict[str, Any]] = []

    try:
        with Neo4jClient() as client:

            for entity_name in entities:

                logger.info(f"Traversing graph from entity: '{entity_name}'")

                # ----------------------------------------------------------
                # QUERY 1: Direct relationships (1 hop)
                # ----------------------------------------------------------
                # Find everything directly connected to this entity.
                # Returns both outgoing (entity -> other) and incoming
                # (other -> entity) relationships so we see both directions.
                # We use -[r]-() instead of -[r]->() to catch both directions.
                direct_results = _query_direct_relationships(
                    client, entity_name
                )
                all_graph_results.extend(direct_results)

                logger.debug(
                    f"  Direct relationships for '{entity_name}': "
                    f"{len(direct_results)} found"
                )

                # ----------------------------------------------------------
                # QUERY 2: Multi-hop paths (2+ hops)
                # ----------------------------------------------------------
                # Only run deeper traversal if entity_agent recommended it.
                # Deeper traversal is more expensive — we only do it when
                # the query intent suggests it is needed (risk_cascade, etc.)
                if recommended_hops > 1:
                    multihop_results = _query_multihop_paths(
                        client, entity_name, recommended_hops
                    )
                    all_graph_results.extend(multihop_results)

                    logger.debug(
                        f"  Multi-hop paths for '{entity_name}' "
                        f"(depth {recommended_hops}): "
                        f"{len(multihop_results)} found"
                    )

    except Exception as e:
        logger.error(f"Graph query agent failed: {e}")
        return {
            "error":        f"Graph query agent failed: {str(e)}",
            "error_source": "graph_query_agent",
            "graph_results": [],
            "graph_paths_summary": f"Graph traversal failed: {str(e)}",
        }

    # Deduplicate results — the same triple can appear from multiple
    # entity starting points (e.g. TSMC->Apple found starting from both
    # "TSMC" and "Apple")
    deduplicated_results = _deduplicate_triples(all_graph_results)

    # Build a human-readable summary of what was found
    # This goes into the final response so users can see the reasoning chain
    paths_summary = _build_paths_summary(deduplicated_results, entities)

    logger.success(
        f"Graph query agent complete — "
        f"{len(deduplicated_results)} unique triples found "
        f"across {len(entities)} entities"
    )

    return {
        "graph_results":       deduplicated_results,
        "graph_paths_summary": paths_summary,
    }


# -----------------------------------------------------------------------------
# PRIVATE HELPER FUNCTIONS
# These are only used inside this module (indicated by leading underscore)
# -----------------------------------------------------------------------------

def _query_direct_relationships(
    client: Neo4jClient,
    entity_name: str,
) -> List[Dict[str, Any]]:
    """
    Finds all nodes directly connected to the given entity (1 hop).

    Uses an undirected relationship match (-[r]-) so we find both
    outgoing and incoming relationships. Returns both directions
    because in our graph "Apple depends on TSMC" is stored as
    (Apple)-[DEPENDS_ON]->(TSMC), so querying from TSMC in the
    outgoing direction would miss it.

    Args:
        client:      An open Neo4jClient instance.
        entity_name: The entity name to start traversal from.

    Returns:
        List of triple dicts, one per relationship found.
    """

    # MATCH any node with this name
    # -[r]- matches relationships in both directions (undirected)
    # RETURN the start node name, relationship type, end node name
    cypher = """
        MATCH (start {name: $entity_name})-[r]-(end)
        RETURN
            start.name           AS source,
            type(r)              AS relationship,
            end.name             AS target,
            properties(r)        AS rel_properties,
            labels(start)[0]     AS source_type,
            labels(end)[0]       AS target_type,
            1                    AS hop_distance
        ORDER BY type(r)
        LIMIT 25
    """

    rows = client.run_query(
        cypher,
        parameters={"entity_name": entity_name}
    )

    return [_row_to_triple(row) for row in rows]


def _query_multihop_paths(
    client: Neo4jClient,
    entity_name: str,
    max_hops: int,
) -> List[Dict[str, Any]]:
    """
    Finds all relationship paths up to max_hops deep from the entity.

    Uses variable-length path matching [*2..max_hops] to find chains
    of relationships. Starts at hop 2 because hop 1 is already covered
    by _query_direct_relationships().

    Args:
        client:      An open Neo4jClient instance.
        entity_name: Starting entity for the traversal.
        max_hops:    Maximum depth to traverse.

    Returns:
        List of triple dicts representing paths found.
    """

    # Neo4j does NOT allow a $parameter inside a variable-length path range
    # like [*2..$max_hops] — it only accepts a literal number there. So we
    # inject max_hops directly into the query string with an f-string.
    # We cast to int first as a safety guard, since the value is now placed
    # into the query text rather than passed as a safe bound parameter.
    # (entity_name stays a proper $parameter — that one is fine and safe.)
    #
    # NOTE ON f-STRING BRACES: because this is an f-string, the literal
    # Cypher braces around the node pattern must be DOUBLED ({{ }}) so Python
    # does not treat them as an interpolation slot.
    max_hops = int(max_hops)

    # [*2..{max_hops}] means "follow between 2 and max_hops relationships"
    # We extract the first relationship in the path with relationships(path)[0]
    # and the last node with nodes(path)[-1] to build the triple
    # DISTINCT prevents duplicate paths
    cypher = f"""
        MATCH path = (start {{name: $entity_name}})-[*2..{max_hops}]-(end)
        WITH
            start,
            end,
            relationships(path)[0]  AS first_rel,
            length(path)            AS hop_distance
        RETURN DISTINCT
            start.name              AS source,
            type(first_rel)         AS relationship,
            end.name                AS target,
            properties(first_rel)   AS rel_properties,
            labels(start)[0]        AS source_type,
            labels(end)[0]          AS target_type,
            hop_distance
        ORDER BY hop_distance ASC
        LIMIT 30
    """

    rows = client.run_query(
        cypher,
        parameters={"entity_name": entity_name}
    )

    return [_row_to_triple(row) for row in rows]


def _row_to_triple(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a raw Neo4j query result row into a standardised triple dict.

    Every triple dict in graph_results has the same shape so downstream
    agents (rerank_agent, synthesis_agent) can process them uniformly
    without knowing which Cypher query produced them.

    Args:
        row: Raw dict from Neo4jClient.run_query() result.

    Returns:
        A standardised triple dict with these keys:
          type, source, source_type, relationship, target,
          target_type, properties, hop_distance, text, score
    """

    source       = row.get("source", "")
    relationship = row.get("relationship", "")
    target       = row.get("target", "")
    properties   = row.get("rel_properties", {}) or {}
    hop_distance = row.get("hop_distance", 1)
    source_type  = row.get("source_type", "Unknown")
    target_type  = row.get("target_type", "Unknown")

    # Build human-readable text for this triple
    # This is what the synthesis agent reads in the formatted context
    text = f"({source_type}){source} -[{relationship}]-> ({target_type}){target}"

    # Add properties inline if they exist
    # e.g. SUPPLIES_CHIPS_TO {component: M3 chip, process_node: N3}
    if properties:
        props_str = ", ".join(
            f"{k}: {v}" for k, v in properties.items() if v
        )
        if props_str:
            text += f" [{props_str}]"

    # Score based on hop distance — closer hops are more directly relevant
    # 1 hop = 1.0, 2 hops = 0.5, 3 hops = 0.33
    score = round(1.0 / max(hop_distance, 1), 4)

    return {
        "type":         "graph_triple",    # marks this as a graph result
        "source":       source,
        "source_type":  source_type,
        "relationship": relationship,
        "target":       target,
        "target_type":  target_type,
        "properties":   properties,
        "hop_distance": hop_distance,
        "text":         text,
        "score":        score,
    }


def _deduplicate_triples(
    triples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Removes duplicate triples from the results list.

    A duplicate is defined as two triples with the same
    (source, relationship, target) combination, regardless of which
    starting entity they were found from. We keep the one with the
    lower hop_distance (more direct connection).

    Args:
        triples: Raw list of triple dicts, possibly with duplicates.

    Returns:
        Deduplicated list sorted by hop_distance ascending.
    """

    # Use a dict keyed by (source, relationship, target) tuple
    # If we see the same triple twice, keep the one with lower hop_distance
    seen: Dict[tuple, Dict[str, Any]] = {}

    for triple in triples:
        key = (
            triple.get("source", "").lower(),
            triple.get("relationship", ""),
            triple.get("target", "").lower(),
        )

        if key not in seen:
            # First time seeing this triple — store it
            seen[key] = triple
        else:
            # Already seen — keep whichever has the lower hop_distance
            existing_hops = seen[key].get("hop_distance", 99)
            new_hops      = triple.get("hop_distance", 99)
            if new_hops < existing_hops:
                seen[key] = triple

    # Sort by hop_distance so closest relationships appear first
    deduplicated = sorted(
        seen.values(),
        key=lambda t: t.get("hop_distance", 99)
    )

    return deduplicated


def _build_paths_summary(
    triples: List[Dict[str, Any]],
    entities: List[str],
) -> str:
    """
    Builds a human-readable summary string of the graph paths found.

    This summary is stored in state as graph_paths_summary and returned
    to the API caller alongside the final answer so users can see the
    reasoning chain that led to the answer.

    Args:
        triples:  Deduplicated list of triple dicts.
        entities: The entity names used as traversal starting points.

    Returns:
        A formatted multi-line string summarising the paths.

    Example output:
        "Graph traversal from entities: [ASML, TSMC, Apple]
         Found 8 relationship paths:
           Hop 1: ASML -[SUPPLIES_EQUIPMENT_TO]-> TSMC
           Hop 1: TSMC -[SUPPLIES_CHIPS_TO]-> Apple
           Hop 1: TSMC -[MANUFACTURES_FOR]-> Nvidia
           Hop 2: ASML -[SUPPLIES_EQUIPMENT_TO]-> Samsung
           ..."
    """

    if not triples:
        return f"No graph paths found for entities: {entities}"

    lines = [
        f"Graph traversal from entities: {entities}",
        f"Found {len(triples)} relationship paths:",
    ]

    for triple in triples[:15]:   # cap at 15 for readability
        hop  = triple.get("hop_distance", "?")
        text = triple.get("text", "")
        lines.append(f"  Hop {hop}: {text}")

    if len(triples) > 15:
        lines.append(f"  ... and {len(triples) - 15} more paths")

    return "\n".join(lines)