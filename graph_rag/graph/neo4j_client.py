# graph_rag/graph/neo4j_client.py
#
# PURPOSE:
#   Manages the connection to Neo4j AuraDB and provides a small set of
#   low-level helper methods for writing and reading data.
#
#   Think of this file as the "database driver layer" — it knows nothing
#   about our specific pipeline logic. It only knows how to:
#     - Open and close a connection to Neo4j
#     - Run a Cypher query and return results
#     - Write a node (entity) to the graph
#     - Write a relationship (edge) between two nodes
#     - Clear the entire graph (useful for re-running ingestion cleanly)
#     - Check if the connection is alive
#
#   graph_builder.py sits on top of this file and uses these helpers to
#   insert the entities and relationships extracted by entity_extractor.py.
#
# WHAT IS CYPHER?
#   Cypher is Neo4j's query language. It looks like drawing the graph
#   with ASCII art. For example:
#     MERGE (a:Company {name: "Apple"})
#   means "find or create a node labelled Company with name Apple".
#
#     MATCH (a:Company {name: "TSMC"})-[:SUPPLIES_CHIPS_TO]->(b:Company {name: "Apple"})
#   means "find TSMC, follow the SUPPLIES_CHIPS_TO edge, arrive at Apple".
#
#   We use MERGE instead of CREATE throughout this file.
#   MERGE = "create if it does not exist, otherwise find the existing one"
#   CREATE = "always create a new one (creates duplicates if run twice)"
#   MERGE makes ingestion idempotent — safe to run multiple times.
#
# DO YOU RUN THIS FILE?
#   No. Imported by graph_builder.py and graph_query_agent.py.

import neo4j
from neo4j import GraphDatabase            # official Neo4j Python driver
from neo4j.exceptions import ServiceUnavailable, AuthError
# ServiceUnavailable: raised when AuraDB cannot be reached (network issue)
# AuthError: raised when URI/username/password combination is wrong

from typing import List, Dict, Any, Optional  # type hints
from loguru import logger                  # structured logging

from graph_rag.utils.config import settings  # validated config values


class Neo4jClient:
    """
    Manages a connection to Neo4j AuraDB and provides helper methods
    for writing nodes, writing relationships, and running Cypher queries.

    This class uses a context manager pattern so the connection is always
    properly closed even if an error occurs:

        with Neo4jClient() as client:
            client.write_node(entity)

    Or it can be used directly:

        client = Neo4jClient()
        client.connect()
        client.write_node(entity)
        client.close()
    """

    def __init__(self):
        """
        Stores connection parameters from config.
        Does NOT open the connection yet — call connect() for that.
        """
        # Connection URI from .env — e.g. neo4j+s://xxxxxxxx.databases.neo4j.io
        self.uri = settings.neo4j_uri

        # Username from .env — always "neo4j" for AuraDB
        self.username = settings.neo4j_username

        # Password from .env — the generated password from AuraDB setup
        self.password = settings.neo4j_password

        # The driver object — None until connect() is called
        # GraphDatabase.driver() is the Neo4j connection object
        self.driver = None

    def connect(self) -> None:
        """
        Opens the connection to Neo4j AuraDB.

        Creates a GraphDatabase driver using the URI and credentials from
        config. The driver manages a connection pool internally — we do not
        need to manage individual connections ourselves.

        Raises:
            AuthError: if the username or password is wrong.
            ServiceUnavailable: if AuraDB cannot be reached.
        """
        try:
            logger.info(f"Connecting to Neo4j at {self.uri}")

            # GraphDatabase.driver() opens the connection pool
            # auth=(username, password) is passed as a tuple
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )

            # verify_connectivity() sends a test ping to AuraDB
            # This confirms the credentials are correct and the instance is running
            # It raises an exception immediately if something is wrong
            # rather than failing silently on the first real query
            self.driver.verify_connectivity()

            logger.success("Connected to Neo4j AuraDB successfully")

        except AuthError as e:
            logger.error(
                f"Neo4j authentication failed. "
                f"Check NEO4J_USERNAME and NEO4J_PASSWORD in your .env file.\n"
                f"Error: {e}"
            )
            raise

        except ServiceUnavailable as e:
            logger.error(
                f"Neo4j AuraDB is not reachable. "
                f"Check NEO4J_URI in your .env file and confirm the instance "
                f"is Running in the Neo4j Aura console.\n"
                f"Error: {e}"
            )
            raise

    def close(self) -> None:
        """
        Closes the connection to Neo4j and releases all resources.
        Always call this when finished — or use the context manager pattern.
        """
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        """
        Opens the connection when used as a context manager.
        Allows:  with Neo4jClient() as client:
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the connection when the with block exits.
        Called automatically even if an exception occurred inside the block.
        """
        self.close()
        # Return False so exceptions are not suppressed
        return False

    def run_query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs a Cypher query against Neo4j and returns the results.

        This is the core method that all other methods in this class
        ultimately use. It opens a session, runs the query, and returns
        the results as a list of dictionaries.

        Args:
            cypher:     The Cypher query string to execute.
                        Use $param_name placeholders for values — never
                        format values directly into the string (SQL injection risk).
            parameters: Dictionary of parameter values for the query.
                        Example: {"name": "Apple", "sector": "Technology"}
                        These map to $name and $sector in the Cypher string.

        Returns:
            A list of dictionaries, one per result row.
            Each dictionary maps field names to values.
            Returns an empty list if the query returns no results.

        Example:
            results = client.run_query(
                "MATCH (c:Company {name: $name}) RETURN c.name, c.sector",
                parameters={"name": "Apple"}
            )
            # results = [{"c.name": "Apple", "c.sector": "Technology"}]
        """
        # Default parameters to empty dict if none provided
        parameters = parameters or {}

        # Open a session — a session is a logical unit of work with Neo4j
        # The session handles transactions automatically for read/write queries
        with self.driver.session() as session:
            # session.run() executes the Cypher and returns a Result object
            result = session.run(cypher, parameters)

            # Convert the Result to a plain list of dicts so callers do not
            # need to know about Neo4j's internal result types
            # result.data() does this conversion for us
            return result.data()

    def write_node(self, entity: Dict[str, Any]) -> None:
        """
        Writes a single entity as a node in Neo4j.

        Uses MERGE so the node is created only if it does not already exist.
        If a node with the same name and label already exists, MERGE finds
        it and the ON CREATE SET block is skipped. This makes ingestion safe
        to run multiple times without creating duplicate nodes.

        Node structure in Neo4j after this call:
            (:Company {name: "Apple", entity_type: "Company"})
            (:Country {name: "Taiwan", entity_type: "Country"})
            (:Product  {name: "H100",  entity_type: "Product"})

        Args:
            entity: A dictionary with "name" and "type" keys.
                    Example: {"name": "TSMC", "type": "Company"}
        """
        entity_name = entity.get("name", "").strip()
        entity_type = entity.get("type", "Unknown").strip()

        # Skip entities with empty names — these are extraction artifacts
        if not entity_name:
            logger.warning(f"Skipping entity with empty name: {entity}")
            return

        # We cannot use a parameter for the node label in Cypher —
        # Cypher labels must be literals in the query string, not parameters.
        # So we format the label directly into the query string.
        # This is safe here because entity_type comes from our fixed
        # ENTITY_TYPES list, not from user input.
        cypher = f"""
            MERGE (n:{entity_type} {{name: $name}})
            ON CREATE SET
                n.entity_type = $entity_type,
                n.created_at  = timestamp()
            ON MATCH SET
                n.entity_type = $entity_type
        """

        self.run_query(cypher, parameters={
            "name":        entity_name,
            "entity_type": entity_type,
        })

        logger.debug(f"Wrote node: ({entity_type}) {entity_name}")

    def write_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Writes a relationship (edge) between two existing nodes in Neo4j.

        Uses MERGE on both the source and target nodes first, then MERGE
        on the relationship. This is safe even if the nodes were already
        written by write_node() — MERGE finds the existing nodes.

        Relationship structure in Neo4j after this call:
            (TSMC:Company)-[:SUPPLIES_CHIPS_TO {component: "M3 chip"}]->(Apple:Company)

        Args:
            relationship: A dictionary with keys:
                "source"     : name of the source entity (e.g. "TSMC")
                "target"     : name of the target entity (e.g. "Apple")
                "type"       : relationship type (e.g. "SUPPLIES_CHIPS_TO")
                "properties" : dict of extra properties (e.g. {"component": "M3 chip"})
        """
        source_name  = relationship.get("source", "").strip()
        target_name  = relationship.get("target", "").strip()
        rel_type     = relationship.get("type", "RELATED_TO").strip()
        properties   = relationship.get("properties", {})

        # Skip relationships where source or target name is missing
        if not source_name or not target_name:
            logger.warning(
                f"Skipping relationship with missing source or target: {relationship}"
            )
            return

        # Replace spaces with underscores in relationship type for valid Cypher
        # e.g. "SUPPLIES CHIPS TO" -> "SUPPLIES_CHIPS_TO"
        rel_type = rel_type.replace(" ", "_").upper()

        # Build a Cypher query that:
        # 1. Finds the source node by name (any label)
        # 2. Finds the target node by name (any label)
        # 3. MERGEs the relationship between them
        # 4. Sets any extra properties on the relationship
        #
        # We use MATCH with {name: $source} because the nodes were already
        # written by write_node() with their specific labels.
        # Using a generic MATCH without label is fine here because name is unique.
        cypher = f"""
            MATCH (source {{name: $source_name}})
            MATCH (target {{name: $target_name}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r += $properties
        """

        self.run_query(cypher, parameters={
            "source_name": source_name,
            "target_name": target_name,
            "properties":  properties,
        })

        logger.debug(
            f"Wrote relationship: ({source_name})-[{rel_type}]->({target_name})"
        )

    def clear_graph(self) -> None:
        """
        Deletes ALL nodes and relationships from the Neo4j database.

        Use this to start fresh before re-running ingestion.
        DETACH DELETE removes all relationships connected to each node
        before deleting the node itself (required by Neo4j — you cannot
        delete a node that still has relationships).

        WARNING: This is irreversible. All graph data will be lost.
        Only call this during development when you want a clean slate.
        """
        logger.warning(
            "Clearing all nodes and relationships from Neo4j. "
            "This cannot be undone."
        )

        # MATCH (n) matches every node in the database
        # DETACH DELETE n deletes the node and all its relationships
        self.run_query("MATCH (n) DETACH DELETE n")

        logger.success("Neo4j graph cleared successfully")

    def get_node_count(self) -> int:
        """
        Returns the total number of nodes currently in the graph.
        Useful for verifying that ingestion wrote the expected number of nodes.
        """
        results = self.run_query("MATCH (n) RETURN count(n) AS count")
        # results is a list with one dict: [{"count": 42}]
        return results[0]["count"] if results else 0

    def get_relationship_count(self) -> int:
        """
        Returns the total number of relationships currently in the graph.
        Useful for verifying that ingestion wrote the expected number of edges.
        """
        results = self.run_query("MATCH ()-[r]->() RETURN count(r) AS count")
        return results[0]["count"] if results else 0

    def health_check(self) -> Dict[str, Any]:
        """
        Checks the health of the Neo4j connection and returns basic stats.

        Called by the FastAPI /health endpoint to confirm the database
        is reachable and contains data.

        Returns:
            A dictionary with:
              "status"        : "healthy" or "unhealthy"
              "node_count"    : total nodes in the graph
              "relationship_count" : total relationships in the graph
              "message"       : human-readable status message
        """
        try:
            node_count = self.get_node_count()
            rel_count  = self.get_relationship_count()

            return {
                "status":             "healthy",
                "node_count":         node_count,
                "relationship_count": rel_count,
                "message":            (
                    f"Neo4j is running. "
                    f"{node_count} nodes, {rel_count} relationships."
                ),
            }

        except Exception as e:
            return {
                "status":             "unhealthy",
                "node_count":         0,
                "relationship_count": 0,
                "message":            f"Neo4j health check failed: {e}",
            }