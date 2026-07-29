# graph_rag/agents/synthesis_agent.py
#
# PURPOSE:
#   This is the fifth and final agent in the LangGraph pipeline.
#   It takes everything accumulated in state — the formatted context
#   built by rerank_agent, the graph paths, the user's question, and
#   the detected intent — and generates the final answer using GPT-4o.
#
#   This is the only agent the user ever sees the output of.
#   All the complexity of graph traversal, vector search, and reranking
#   is invisible. This agent converts it into a clean, cited,
#   analyst-quality response.
#
# WHAT THIS AGENT RECEIVES FROM STATE:
#   - query            : the user's original question
#   - formatted_context: structured evidence string from rerank_agent
#                        (graph triples + text chunks, ranked by RRF)
#   - intent           : query intent from entity_agent
#                        (risk_cascade, supply_chain, financial, etc.)
#   - extracted_entities: entity names from entity_agent
#   - graph_paths_summary: human-readable path summary from graph_query_agent
#
# WHAT THIS AGENT PRODUCES:
#   - final_answer     : the complete answer to the user's question
#   - sources          : list of source PDF filenames cited
#   - graph_paths_used : list of graph relationship paths used in reasoning
#
# INTENT-AWARE PROMPTING:
#   We use different prompt instructions depending on the detected intent.
#   A risk_cascade question needs the answer to trace the cascade chain
#   step by step. A financial question needs specific numbers cited.
#   A supply_chain question needs the relationship chain explained clearly.
#   Using the same generic prompt for all intents produces mediocre answers.
#   Intent-specific instructions produce analyst-quality answers.
#
# DO YOU RUN THIS FILE?
#   No. Registered as a LangGraph node in graph_pipeline.py.
#   Called automatically after rerank_agent completes.


import re                                      # extracts source filenames from context
from typing import Dict, Any, List             # type hints
from langchain_openai import ChatOpenAI        # GPT-4o wrapper
from langchain_core.prompts import PromptTemplate  # structures the prompt
from loguru import logger                      # structured logging

from graph_rag.agents.state import GraphRAGState  # shared state schema
from graph_rag.utils.config import settings        # validated config


# -----------------------------------------------------------------------------
# INTENT-SPECIFIC INSTRUCTIONS
# These are injected into the synthesis prompt based on the detected intent.
# Each gives GPT-4o specific guidance on how to structure its answer.
# -----------------------------------------------------------------------------
INTENT_INSTRUCTIONS = {
    "risk_cascade": """
You are answering a RISK CASCADE question.
Structure your answer as follows:
1. Identify the triggering event (what disruption or risk is being asked about)
2. Trace the cascade chain step by step — which entity is affected first,
   which is affected second, and so on, using the graph evidence
3. Quantify the financial impact at each step using the text evidence
4. State which companies face the highest total revenue risk and why
Use the graph relationship chains as the backbone of your reasoning.
""",

    "supply_chain": """
You are answering a SUPPLY CHAIN question.
Structure your answer as follows:
1. Identify the key entities in the supply chain
2. Explain the direction of each relationship (who supplies to whom)
3. Highlight any concentration risks (sole-source suppliers, geographic concentration)
4. Use specific relationship types from the graph evidence (SUPPLIES_TO, MANUFACTURES_FOR, etc.)
""",

    "financial": """
You are answering a FINANCIAL question.
Structure your answer as follows:
1. State the specific financial figures requested, with fiscal year context
2. Compare figures across companies or time periods if relevant
3. Explain what drives the financial performance using the text evidence
4. Always cite which document and page the figures come from
""",

    "competitive": """
You are answering a COMPETITIVE LANDSCAPE question.
Structure your answer as follows:
1. Identify which companies compete with each other and in which market
2. Explain the nature of the competition using graph relationship types
3. Note any companies that have dual roles (both competitor and supplier)
4. Quantify competitive positions using financial figures from the text evidence
""",

    "general": """
You are answering a GENERAL question about the semiconductor supply chain.
Provide a clear, well-structured answer using both the graph evidence
(relationship chains) and the text evidence (source document passages).
Cite specific sources for all factual claims.
""",
}

# -----------------------------------------------------------------------------
# SYNTHESIS PROMPT TEMPLATE
# -----------------------------------------------------------------------------
SYNTHESIS_PROMPT = """
You are a senior financial analyst specialising in semiconductor supply chains.
You have been provided with two types of evidence to answer the user's question:

1. GRAPH EVIDENCE: Structured relationship chains extracted from a knowledge graph
   These show how companies, countries, and products connect to each other.

2. TEXT EVIDENCE: Relevant passages from financial documents
   These provide narrative detail, financial figures, and risk context.

{intent_instructions}

EVIDENCE:
{formatted_context}

GRAPH PATHS FOUND:
{graph_paths_summary}

USER QUESTION:
{query}

INSTRUCTIONS:
- Base your answer ONLY on the evidence provided above
- Do not use knowledge from outside the provided evidence
- When citing financial figures, always mention the source document and fiscal year
- When describing relationships, use the exact relationship types from the graph evidence
- If the evidence is insufficient to fully answer the question, say so explicitly
- End your answer with a SOURCES section listing the documents you referenced

Now answer the question:
"""


def synthesis_agent(state: GraphRAGState) -> Dict[str, Any]:
    """
    LangGraph node — generates the final answer using GPT-4o.

    Reads all accumulated evidence from state, builds an intent-aware
    prompt, calls GPT-4o, and returns the final answer with citations
    and graph paths used.

    Args:
        state: Current pipeline state. Must have:
               - query             : user's original question
               - formatted_context : structured evidence from rerank_agent
               - intent            : query intent from entity_agent
               - graph_paths_summary: path summary from graph_query_agent

    Returns:
        Partial state dict with:
          - final_answer     : the complete answer string
          - sources          : list of cited source filenames
          - graph_paths_used : list of graph paths used in reasoning
        Or on error:
          - error, error_source
    """

    query             = state.get("query", "")
    formatted_context = state.get("formatted_context", "")
    intent            = state.get("intent", "general")
    graph_paths_summary = state.get("graph_paths_summary", "No graph paths available.")
    merged_results    = state.get("merged_results", []) or []

    logger.info(
        f"Synthesis agent running — "
        f"intent: {intent}, "
        f"context length: {len(formatted_context)} chars"
    )

    # Guard against empty context — means both retrievers failed
    if not formatted_context:
        logger.warning("Synthesis agent received empty context")
        return {
            "final_answer":     (
                "I was unable to find relevant evidence to answer your question. "
                "Please ensure the ingestion pipeline has been run successfully."
            ),
            "sources":          [],
            "graph_paths_used": [],
        }

    try:
        # Initialise GPT-4o
        # temperature=0.2 gives slightly more natural language than 0
        # while remaining factually grounded
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )

        # Get the intent-specific instructions
        # Default to "general" if intent is not in our map
        intent_instructions = INTENT_INSTRUCTIONS.get(
            intent,
            INTENT_INSTRUCTIONS["general"]
        )

        # Build the full synthesis prompt
        prompt = PromptTemplate.from_template(SYNTHESIS_PROMPT)
        filled_prompt = prompt.format(
            intent_instructions=intent_instructions,
            formatted_context=formatted_context,
            graph_paths_summary=graph_paths_summary,
            query=query,
        )

        logger.debug(
            f"Synthesis prompt length: {len(filled_prompt)} chars. "
            f"Calling GPT-4o..."
        )

        # Call GPT-4o with the complete prompt
        response     = llm.invoke(filled_prompt)
        final_answer = response.content.strip()

        # Extract source filenames cited in the answer
        # We scan both the formatted_context and the final_answer for
        # PDF filenames that appear in our document set
        sources = _extract_sources(formatted_context, final_answer)

        # Extract the graph relationship paths that contributed to the answer
        # These come from the graph results in merged_results
        graph_paths_used = _extract_graph_paths(merged_results)

        logger.success(
            f"Synthesis agent complete — "
            f"answer length: {len(final_answer)} chars, "
            f"sources: {len(sources)}, "
            f"graph paths used: {len(graph_paths_used)}"
        )

        return {
            "final_answer":     final_answer,
            "sources":          sources,
            "graph_paths_used": graph_paths_used,
        }

    except Exception as e:
        logger.error(f"Synthesis agent failed: {e}")
        return {
            "error":        f"Synthesis agent failed: {str(e)}",
            "error_source": "synthesis_agent",
            "final_answer": f"An error occurred while generating the answer: {str(e)}",
            "sources":      [],
            "graph_paths_used": [],
        }


# -----------------------------------------------------------------------------
# PRIVATE HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def _extract_sources(
    formatted_context: str,
    final_answer: str,
) -> List[str]:
    """
    Extracts unique source PDF filenames from the context and answer.

    Scans for filenames matching the pattern of our five synthetic PDFs
    (strings ending in .pdf) in both the formatted context and the
    generated answer. Returns a deduplicated list.

    Args:
        formatted_context: The structured evidence string from rerank_agent.
        final_answer:      The answer generated by GPT-4o.

    Returns:
        Deduplicated list of PDF filenames referenced.
        Example: ["apple_annual_overview_FY2024.pdf",
                  "tsmc_manufacturing_report_FY2023.pdf"]
    """

    # Regex pattern to match PDF filenames
    # Matches any string of word characters, underscores, and hyphens
    # followed by .pdf
    pdf_pattern = re.compile(r'[\w\-]+\.pdf', re.IGNORECASE)

    # Search both strings for PDF filename matches
    found_in_context = pdf_pattern.findall(formatted_context)
    found_in_answer  = pdf_pattern.findall(final_answer)

    # Combine and deduplicate while preserving order of first appearance
    seen    = set()
    sources = []
    for filename in found_in_context + found_in_answer:
        if filename.lower() not in seen:
            seen.add(filename.lower())
            sources.append(filename)

    return sources


def _extract_graph_paths(
    merged_results: List[Dict[str, Any]],
) -> List[str]:
    """
    Extracts the graph relationship paths from the merged results list.

    Filters merged_results for items of result_type "graph_triple" and
    returns their text representations. These are the relationship chains
    that contributed to the answer — shown to the user for explainability.

    Args:
        merged_results: The RRF-ranked list from rerank_agent.

    Returns:
        List of human-readable graph path strings.
        Example: [
            "(Company)TSMC -[SUPPLIES_CHIPS_TO]-> (Company)Apple [component: M3 chip]",
            "(Company)ASML -[SUPPLIES_EQUIPMENT_TO]-> (Company)TSMC",
        ]
    """

    graph_paths = []

    for result in merged_results:
        # Only include graph triple results, not text chunks
        if result.get("result_type") == "graph_triple":
            text = result.get("text", "")
            if text and text not in graph_paths:
                graph_paths.append(text)

    return graph_paths