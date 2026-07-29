# graph_rag/agents/entity_agent.py
#
# PURPOSE:
#   This is the first agent in the LangGraph pipeline.
#   It receives the user's raw question and extracts:
#     - Entity names  (Apple, TSMC, Taiwan)
#     - Entity types  (Company, Country, Company)
#     - Query intent  (risk_cascade, supply_chain, financial, competitive, general)
#     - Recommended graph traversal depth (1, 2, or 3 hops)
#
#   Its output populates the state fields that the next two agents depend on:
#     - graph_query_agent uses extracted_entities as Neo4j traversal start points
#     - vector_agent uses the original query directly (no change needed)
#     - synthesis_agent uses intent to frame the final answer correctly
#
# WHY DO WE NEED THIS AGENT?
#   Neo4j graph traversal requires concrete entity names to start from.
#   You cannot start a graph traversal with a natural language sentence.
#   This agent bridges the gap — it converts "How does Taiwan risk affect
#   Apple revenue?" into ["Apple", "Taiwan", "TSMC"] so the graph query
#   agent knows exactly which nodes to start traversing from.
#
#   The intent detection is also important. "What is Apple's revenue?"
#   needs a direct financial answer. "What happens if TSMC shuts down?"
#   needs a cascade reasoning answer. Different intents need different
#   framing in the synthesis prompt.
#
# HOW IT WORKS:
#   Sends the user's question to GPT-4o with a structured prompt that
#   asks for JSON output containing entities, types, intent, and
#   recommended hop count. Parses the JSON and writes it to state.
#
# DO YOU RUN THIS FILE?
#   No. graph_pipeline.py registers this as a node in the LangGraph
#   StateGraph. LangGraph calls it automatically when a query arrives.


import json                                   # parses GPT-4o JSON response
from typing import Dict, Any                  # type hints
from langchain_openai import ChatOpenAI       # GPT-4o wrapper
from langchain_core.prompts import PromptTemplate  # structures the prompt
from loguru import logger                     # structured logging

from graph_rag.agents.state import GraphRAGState  # shared state schema
from graph_rag.utils.config import settings        # validated config values


# -----------------------------------------------------------------------------
# INTENT DEFINITIONS
# These are the five query intents the agent can detect.
# Passed to the synthesis agent so it knows how to frame the answer.
# -----------------------------------------------------------------------------
INTENT_DEFINITIONS = """
- risk_cascade    : question about how a disruption to one entity affects others
                    Example: "What happens if TSMC shuts down?"
- supply_chain    : question about supplier/customer relationships
                    Example: "Who supplies chips to Apple?"
- financial       : question about revenue, margins, costs, or financial metrics
                    Example: "What was Apple's revenue in FY2024?"
- competitive     : question about competition between companies
                    Example: "How does Samsung compete with TSMC?"
- general         : any question that does not fit the above categories
                    Example: "Tell me about ASML"
"""

# -----------------------------------------------------------------------------
# ENTITY EXTRACTION PROMPT
# We ask GPT-4o to return strict JSON so we can parse it reliably.
# The prompt includes:
#   1. Allowed entity types (matches our Neo4j node labels)
#   2. Intent definitions (matches our synthesis agent framing logic)
#   3. Hop count reasoning (guides traversal depth)
#   4. A concrete example of the expected output format
# -----------------------------------------------------------------------------
ENTITY_AGENT_PROMPT = """
You are the first step in a financial intelligence Graph RAG pipeline.
Your job is to analyse the user's question and extract structured information
that will be used to query a Neo4j knowledge graph about semiconductor supply chains.

The knowledge graph contains these companies: Apple, TSMC, Nvidia, ASML, Samsung,
AMD, Qualcomm, Intel. And these countries: Taiwan, USA, Netherlands, Japan, China.

ALLOWED ENTITY TYPES:
- Company
- Country
- Product
- Technology
- FinancialMetric
- RiskFactor

INTENT TYPES:
{intent_definitions}

HOP COUNT GUIDELINES:
- Use 1 hop for simple direct relationship questions ("Who supplies Apple?")
- Use 2 hops for two-step relationship questions ("What countries is Apple exposed to?")
- Use 3 hops for complex cascade questions ("If ASML stops shipping, what happens to Apple?")

USER QUESTION:
{query}

Return ONLY a valid JSON object in this exact format — no explanation, no markdown:
{{
    "entities": [
        {{"name": "Apple",  "type": "Company"}},
        {{"name": "Taiwan", "type": "Country"}}
    ],
    "intent": "risk_cascade",
    "recommended_hops": 3,
    "reasoning": "The question asks about cascade risk from ASML through TSMC to Apple"
}}
"""


# -----------------------------------------------------------------------------
# ENTITY AGENT FUNCTION
# LangGraph node functions follow a strict signature:
#   - Input:  the current GraphRAGState dictionary
#   - Output: a partial state dictionary with only the fields this agent updates
# LangGraph merges this partial update into the full state automatically.
# -----------------------------------------------------------------------------

def entity_agent(state: GraphRAGState) -> Dict[str, Any]:
    """
    LangGraph node — extracts entities and intent from the user's question.

    Reads the 'query' field from state, calls GPT-4o to extract entities
    and detect intent, and returns a partial state update with the results.

    Args:
        state: The current pipeline state. Must have 'query' populated.
               All other fields may be None at this point.

    Returns:
        A partial state dictionary with these fields populated:
          - extracted_entities : list of entity name strings
          - entity_types       : list of entity type strings (parallel to entities)
          - intent             : detected query intent string
          - recommended_hops   : int, suggested Neo4j traversal depth
        Or, if an error occurs:
          - error              : error message string
          - error_source       : "entity_agent"
    """

    # Read the user's question from state
    query = state.get("query", "")

    logger.info(f"Entity agent running for query: '{query[:80]}...'")

    if not query:
        # No query in state — something went wrong upstream
        logger.error("Entity agent received empty query")
        return {
            "error":        "Empty query received by entity agent",
            "error_source": "entity_agent",
        }

    # Initialise the GPT-4o client
    # temperature=0 for deterministic entity extraction
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    # Build the extraction prompt by filling in the template variables
    prompt = PromptTemplate.from_template(ENTITY_AGENT_PROMPT)
    filled_prompt = prompt.format(
        intent_definitions=INTENT_DEFINITIONS,
        query=query,
    )

    try:
        # Call GPT-4o with the filled prompt
        response     = llm.invoke(filled_prompt)
        raw_response = response.content.strip()

        # Strip markdown code fences if GPT-4o wrapped the JSON
        if raw_response.startswith("```"):
            raw_response = raw_response.split("\n", 1)[1]
        if raw_response.endswith("```"):
            raw_response = raw_response.rsplit("```", 1)[0]

        # Parse the JSON response
        extracted = json.loads(raw_response.strip())

        # Pull out the entities list and split into parallel lists
        # extracted["entities"] is a list of {"name": X, "type": Y} dicts
        entities_raw = extracted.get("entities", [])

        # Build two parallel lists: names and types
        # These are kept separate for easier downstream use
        entity_names = [e.get("name", "").strip() for e in entities_raw]
        entity_types = [e.get("type", "Company").strip() for e in entities_raw]

        # Filter out any empty names that GPT-4o may have produced
        valid_pairs  = [
            (name, etype)
            for name, etype in zip(entity_names, entity_types)
            if name
        ]
        entity_names = [p[0] for p in valid_pairs]
        entity_types = [p[1] for p in valid_pairs]

        # Extract intent and hop count with safe defaults
        intent           = extracted.get("intent", "general")
        recommended_hops = int(extracted.get("recommended_hops", 2))
        reasoning        = extracted.get("reasoning", "")

        logger.success(
            f"Entity agent complete — "
            f"entities: {entity_names}, "
            f"intent: {intent}, "
            f"hops: {recommended_hops}"
        )
        logger.debug(f"Entity agent reasoning: {reasoning}")

        # Return only the fields this agent is responsible for
        # LangGraph merges this partial dict into the full state
        return {
            "extracted_entities": entity_names,
            "entity_types":       entity_types,
            "intent":             intent,
            "recommended_hops":   recommended_hops,
        }

    except json.JSONDecodeError as e:
        # GPT-4o returned something that is not valid JSON
        logger.error(
            f"Entity agent JSON parse error: {e}\n"
            f"Raw response: {raw_response[:300]}"
        )
        # Return a safe fallback — empty entities, general intent, 2 hops
        # The pipeline can continue with degraded quality rather than crashing
        return {
            "extracted_entities": [],
            "entity_types":       [],
            "intent":             "general",
            "recommended_hops":   2,
        }

    except Exception as e:
        logger.error(f"Entity agent failed: {e}")
        return {
            "error":        f"Entity agent failed: {str(e)}",
            "error_source": "entity_agent",
        }