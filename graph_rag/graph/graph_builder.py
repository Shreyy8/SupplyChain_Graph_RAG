# graph_rag/graph/graph_builder.py
#
# PURPOSE:
#   Takes the entities and relationships produced by entity_extractor.py
#   and writes them into Neo4j AuraDB using the Neo4jClient helper methods.
#
#   This file is the bridge between extraction and storage:
#
#   entity_extractor.py --> [entities + relationships dicts]
#                               |
#                               v
#                        graph_builder.py
#                               |
#                               v
#                        Neo4j AuraDB (nodes + edges)
#
#   After this file runs successfully, you can open the Neo4j Aura console,
#   click "Query", and see the full knowledge graph visualised as nodes
#   connected by labelled edges. That is the graph our agents will traverse.
#
# WHAT THIS FILE DOES STEP BY STEP:
#   1. Receives the full extraction result from entity_extractor.py
#   2. Writes all unique entities as nodes (one node per entity)
#   3. Writes all unique relationships as edges (one edge per relationship)
#   4. Creates indexes on the name property for fast lookups
#   5. Logs progress and a final summary
#
# DO YOU RUN THIS FILE?
#   No. Imported and called by scripts/run_ingestion.py.

'''
What This File Does 

Four steps happen inside build_graph():

Step 1 — Write all nodes. Loops through every entity from entity_extractor.py 
and calls client.write_node() for each one. 
Progress is logged every 10 nodes so during the demo students can watch the 
graph being built in real time.

Step 2 — Create indexes. After nodes are written, creates a Neo4j index on the name property 
for each entity type. This makes lookups fast.
Without an index, every MATCH (n {name: "Apple"}) query does a full table scan.

Step 3 — Write all relationships. Loops through every relationship and 
calls client.write_relationship(). This is done after nodes because the 
relationships use MATCH to find source and target nodes — if nodes do not exist yet, 
the relationship write silently fails.

Step 4 — Verify. After writing, calls verify_key_relationships() which runs five targeted 
Cypher queries to confirm the most important supply chain relationships are in the graph. 
This is your sanity check before running demo queries.

What You Will See in Neo4j Console After This Runs
Open your AuraDB console, click Query, and run:
cypher : ->
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
You will see a visual graph with nodes like Apple, TSMC, Nvidia, ASML connected by 
labelled arrows like SUPPLIES_CHIPS_TO, MANUFACTURES_FOR, OPERATES_IN. 
That is the knowledge graph our agents will traverse.

'''


from typing import Dict, Any, List         # type hints
from loguru import logger                  # structured logging

from graph_rag.graph.neo4j_client import Neo4jClient
# Neo4jClient provides write_node(), write_relationship(), run_query(), etc.


class GraphBuilder:
    """
    Writes extracted entities and relationships into Neo4j AuraDB.

    Receives the output of EntityExtractor.extract_from_chunks() and
    uses Neo4jClient to persist everything to the graph database.

    Usage:
        builder = GraphBuilder()
        builder.build_graph(extraction_results)
    """

    def __init__(self):
        """
        Initialises the GraphBuilder.
        Does not open a Neo4j connection yet — that happens in build_graph().
        """
        logger.info("GraphBuilder initialised")

    def build_graph(
        self,
        extraction_results: Dict[str, Any],
        clear_existing: bool = False,
    ) -> Dict[str, int]:
        """
        Main method — writes all entities and relationships into Neo4j.

        Opens a Neo4j connection, optionally clears the existing graph,
        writes all nodes, creates indexes, then writes all relationships.
        Logs progress throughout so you can watch it work during the demo.

        Args:
            extraction_results: The dict returned by
                                EntityExtractor.extract_from_chunks().
                                Must have "entities" and "relationships" keys.
            clear_existing:     If True, deletes all existing nodes and
                                relationships before writing new ones.
                                Set to True when re-running ingestion from scratch.
                                Default False to avoid accidental data loss.

        Returns:
            A summary dictionary:
            {
                "nodes_written":         42,
                "relationships_written": 35,
                "nodes_failed":          0,
                "relationships_failed":  2,
            }
        """

        entities      = extraction_results.get("entities", [])
        relationships = extraction_results.get("relationships", [])

        logger.info(
            f"Building graph with {len(entities)} entities "
            f"and {len(relationships)} relationships"
        )

        # Counters to track how many writes succeeded and failed
        nodes_written        = 0
        nodes_failed         = 0
        relationships_written = 0
        relationships_failed  = 0

        # Use the context manager so the connection is always properly closed
        with Neo4jClient() as client:

            # Optionally clear the graph before writing
            # Useful during development when re-running ingestion from scratch
            if clear_existing:
                logger.warning(
                    "clear_existing=True — deleting all existing graph data"
                )
                client.clear_graph()

            # ------------------------------------------------------------------
            # STEP 1: Write all entity nodes
            # ------------------------------------------------------------------
            logger.info(f"Writing {len(entities)} entity nodes to Neo4j...")

            for i, entity in enumerate(entities):

                try:
                    # write_node() uses MERGE so duplicates are handled safely
                    client.write_node(entity)
                    nodes_written += 1

                    # Log progress every 10 nodes so the demo shows activity
                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"  Nodes written: {i + 1}/{len(entities)}"
                        )

                except Exception as e:
                    # Log the failure but continue — one bad node should not
                    # stop the entire graph build
                    logger.error(
                        f"Failed to write node {entity.get('name', 'unknown')}: {e}"
                    )
                    nodes_failed += 1

            logger.success(
                f"Node writing complete — "
                f"{nodes_written} written, {nodes_failed} failed"
            )

            # ------------------------------------------------------------------
            # STEP 2: Create indexes for fast lookups
            # ------------------------------------------------------------------
            # An index on the name property speeds up MATCH queries that
            # look up nodes by name — which is what our graph query agent does
            # for every traversal. Without indexes, Neo4j does a full scan
            # which is slow on large graphs.
            #
            # We create one index per entity type.
            # IF NOT EXISTS prevents errors if the index already exists
            # (safe to run multiple times).

            logger.info("Creating Neo4j indexes on name property...")
            self._create_indexes(client)

            # ------------------------------------------------------------------
            # STEP 3: Write all relationships
            # ------------------------------------------------------------------
            # Relationships are written AFTER all nodes because write_relationship()
            # uses MATCH to find the source and target nodes. If we write
            # relationships before nodes, the MATCH finds nothing and the
            # relationship is silently skipped.

            logger.info(
                f"Writing {len(relationships)} relationships to Neo4j..."
            )

            for i, relationship in enumerate(relationships):

                try:
                    # write_relationship() uses MERGE so duplicates are safe
                    client.write_relationship(relationship)
                    relationships_written += 1

                    # Log progress every 10 relationships
                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"  Relationships written: {i + 1}/{len(relationships)}"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to write relationship "
                        f"{relationship.get('source')} -> "
                        f"{relationship.get('target')}: {e}"
                    )
                    relationships_failed += 1

            logger.success(
                f"Relationship writing complete — "
                f"{relationships_written} written, {relationships_failed} failed"
            )

            # ------------------------------------------------------------------
            # STEP 4: Final verification
            # ------------------------------------------------------------------
            # Query Neo4j to confirm the counts match what we tried to write
            node_count = client.get_node_count()
            rel_count  = client.get_relationship_count()

            logger.success(
                f"Graph build complete. "
                f"Neo4j now contains {node_count} nodes "
                f"and {rel_count} relationships."
            )

        # Return a summary for run_ingestion.py to log and display
        return {
            "nodes_written":          nodes_written,
            "relationships_written":  relationships_written,
            "nodes_failed":           nodes_failed,
            "relationships_failed":   relationships_failed,
            "total_nodes_in_graph":   node_count,
            "total_rels_in_graph":    rel_count,
        }

    def _create_indexes(self, client: Neo4jClient) -> None:
        """
        Creates Neo4j indexes on the name property for each entity type.

        Indexes make node lookups by name O(log n) instead of O(n).
        For our graph query agent which does MATCH (n {name: $name}) queries,
        this is a significant speedup even on a small graph.

        This is a private method — only called internally by build_graph().

        Args:
            client: An open Neo4jClient instance.
        """

        # List of entity types we create indexes for
        # These match the ENTITY_TYPES list in entity_extractor.py
        entity_types = [
            "Company",
            "Country",
            "Product",
            "Technology",
            "FinancialMetric",
            "RiskFactor",
        ]

        for entity_type in entity_types:
            try:
                # CREATE INDEX IF NOT EXISTS creates the index only if it does
                # not already exist — safe to call multiple times
                cypher = (
                    f"CREATE INDEX {entity_type.lower()}_name_idx "
                    f"IF NOT EXISTS "
                    f"FOR (n:{entity_type}) ON (n.name)"
                )
                client.run_query(cypher)
                logger.debug(f"Index created/verified for {entity_type}.name")

            except Exception as e:
                # Index creation failure is not fatal — queries still work,
                # just potentially slower
                logger.warning(
                    f"Could not create index for {entity_type}: {e}"
                )

    def verify_key_relationships(self) -> List[Dict[str, Any]]:
        """
        Runs a set of verification queries to confirm the most important
        relationships from our five PDFs were correctly written to Neo4j.

        This is useful after ingestion to quickly confirm the graph contains
        the supply chain relationships we care about before running demo queries.

        Returns:
            A list of verification result dicts, one per check.
            Each dict has "check", "found", and "result" keys.

        Example output:
            [
                {"check": "TSMC supplies Apple",  "found": True,  "result": "PASS"},
                {"check": "ASML supplies TSMC",   "found": True,  "result": "PASS"},
                {"check": "TSMC supplies Nvidia", "found": True,  "result": "PASS"},
            ]
        """

        # These are the key relationships we expect from our five PDFs
        # If any of these are missing, the extraction or graph build had issues
        key_checks = [
            {
                "check": "TSMC supplies chips to Apple",
                "cypher": """
                    MATCH (a {name: 'TSMC'})-[:SUPPLIES_CHIPS_TO]->(b {name: 'Apple'})
                    RETURN count(*) AS found
                """,
            },
            {
                "check": "ASML supplies equipment to TSMC",
                "cypher": """
                    MATCH (a {name: 'ASML'})-[:SUPPLIES_EQUIPMENT_TO]->(b {name: 'TSMC'})
                    RETURN count(*) AS found
                """,
            },
            {
                "check": "TSMC manufactures for Nvidia",
                "cypher": """
                    MATCH (a {name: 'TSMC'})-[:MANUFACTURES_FOR]->(b {name: 'Nvidia'})
                    RETURN count(*) AS found
                """,
            },
            {
                "check": "TSMC operates in Taiwan",
                "cypher": """
                    MATCH (a {name: 'TSMC'})-[:OPERATES_IN]->(b {name: 'Taiwan'})
                    RETURN count(*) AS found
                """,
            },
            {
                "check": "Apple depends on TSMC",
                "cypher": """
                    MATCH (a {name: 'Apple'})-[:DEPENDS_ON]->(b {name: 'TSMC'})
                    RETURN count(*) AS found
                """,
            },
        ]

        results = []

        with Neo4jClient() as client:
            for check in key_checks:
                try:
                    query_result = client.run_query(check["cypher"])

                    # query_result is a list with one dict: [{"found": 1}]
                    found_count = query_result[0]["found"] if query_result else 0
                    found       = found_count > 0

                    results.append({
                        "check":  check["check"],
                        "found":  found,
                        "result": "PASS" if found else "FAIL",
                    })

                    status = "PASS" if found else "FAIL"
                    logger.info(f"  [{status}] {check['check']}")

                except Exception as e:
                    results.append({
                        "check":  check["check"],
                        "found":  False,
                        "result": f"ERROR: {e}",
                    })
                    logger.error(f"  [ERROR] {check['check']}: {e}")

        return results