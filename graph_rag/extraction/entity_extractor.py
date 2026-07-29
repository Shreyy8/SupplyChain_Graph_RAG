# graph_rag/extraction/entity_extractor.py
#
# PURPOSE:
#   This is the most important file in the ingestion pipeline.
#   It reads each text chunk produced by text_chunker.py and uses GPT-4o
#   to extract two things:
#     1. ENTITIES  -- the named things mentioned in the text
#                     (companies, countries, products, financial metrics)
#     2. RELATIONSHIPS -- the connections between those entities
#                     (SUPPLIES_TO, MANUFACTURES_FOR, OPERATES_IN, etc.)
#
#   The entities and relationships extracted here become the nodes and edges
#   of the Neo4j knowledge graph. The quality of this extraction directly
#   determines the quality of the entire Graph RAG pipeline.
#   Good extraction = good graph = good answers.
#   Bad extraction = corrupted graph = wrong answers.
#
# WHY GPT-4o FOR THIS STEP?
#   Named entity recognition (NER) from financial text is hard. The entities
#   we care about — companies, chip names, process nodes, revenue figures —
#   are domain-specific and not well handled by lightweight NLP models.
#   GPT-4o understands that "N3" means TSMC's 3-nanometer process node,
#   that "H100" is an Nvidia GPU, and that "EUV" is ASML's lithography tech.
#   A smaller model would miss or misclassify these constantly.
#
# OUTPUT FORMAT:
#   For each chunk, GPT-4o returns a JSON object with two lists:
#   {
#     "entities": [
#       {"name": "TSMC", "type": "Company"},
#       {"name": "Taiwan", "type": "Country"},
#       {"name": "H100", "type": "Product"}
#     ],
#     "relationships": [
#       {
#         "source": "TSMC",
#         "target": "Apple",
#         "type": "SUPPLIES_CHIPS_TO",
#         "properties": {"component": "M3 chip", "process_node": "N3"}
#       }
#     ]
#   }
#
# DO YOU RUN THIS FILE?
#   No. Imported and called by scripts/run_ingestion.py.


import json                                # parses GPT-4o's JSON response
import time                                # used for rate limit delays between API calls
from typing import List, Dict, Any         # type hints
from langchain_core.documents import Document     # LangChain text container (our chunks)
from langchain_openai import ChatOpenAI    # GPT-4o wrapper
from langchain_core.prompts import PromptTemplate  # structures the prompt sent to GPT-4o
from loguru import logger                  # structured logging

from graph_rag.utils.config import settings  # our validated config values


# -----------------------------------------------------------------------------
# ENTITY AND RELATIONSHIP TYPES
# -----------------------------------------------------------------------------
# We define the allowed entity types and relationship types explicitly.
# Giving GPT-4o a fixed vocabulary prevents it from inventing inconsistent
# labels like "semiconductor_company" vs "chip_maker" vs "fab" for the same
# concept. Consistency in labels is critical for graph traversal — Neo4j
# queries filter by exact type strings.

ENTITY_TYPES = [
    "Company",       # Apple, TSMC, Nvidia, ASML, Samsung, AMD, Qualcomm, Intel
    "Country",       # Taiwan, USA, Netherlands, Japan, South Korea, China
    "Product",       # A18 Pro, M4, H100, B200, EUV machine, TWINSCAN EXE:5000
    "Technology",    # EUV lithography, 3nm process node, VECTOR datatype
    "FinancialMetric",  # Revenue figures, margins, capex tied to a company and year
    "RiskFactor",    # Geopolitical risk, export controls, supply concentration
]

RELATIONSHIP_TYPES = [
    "SUPPLIES_CHIPS_TO",       # TSMC supplies chips to Apple
    "SUPPLIES_EQUIPMENT_TO",   # ASML supplies EUV machines to TSMC
    "SUPPLIES_COMPONENTS_TO",  # Samsung supplies OLED panels to Apple
    "MANUFACTURES_FOR",        # TSMC manufactures H100 for Nvidia
    "COMPETES_WITH",           # Samsung competes with TSMC in foundry market
    "OPERATES_IN",             # TSMC operates in Taiwan
    "DEPENDS_ON",              # Apple depends on TSMC for all chip supply
    "HAS_REVENUE",             # Apple has revenue of USD 391B in FY2024
    "HAS_RISK",                # TSMC has geopolitical risk from Taiwan tensions
    "USES_TECHNOLOGY",         # TSMC uses EUV lithography
    "ALTERNATIVE_FOR",         # Samsung foundry is a potential alternative for Nvidia
    "EXPORTS_TO",              # ASML exports EUV machines to TSMC
    "RESTRICTED_FROM",         # ASML is restricted from exporting to China
]


# -----------------------------------------------------------------------------
# EXTRACTION PROMPT
# -----------------------------------------------------------------------------
# This is the prompt template we send to GPT-4o for each chunk.
# The prompt is carefully designed to:
#   1. Give GPT-4o a clear task description
#   2. Constrain the output to our fixed vocabulary
#   3. Require strict JSON output so we can parse it reliably
#   4. Provide an example so GPT-4o knows the exact format expected

EXTRACTION_PROMPT_TEMPLATE = """
You are an expert financial analyst and knowledge graph builder.
Your task is to extract entities and relationships from the financial text below.

ENTITY TYPES you may use (use ONLY these exact strings):
{entity_types}

RELATIONSHIP TYPES you may use (use ONLY these exact strings):
{relationship_types}

RULES:
1. Extract only entities and relationships that are explicitly stated in the text.
   Do not infer or hallucinate relationships that are not directly mentioned.
2. Entity names must be clean and consistent:
   - Use full company names: "Apple" not "Apple Inc." or "AAPL"
   - Use "TSMC" not "Taiwan Semiconductor Manufacturing Company"
   - Use "Nvidia" not "NVIDIA Corporation"
   - Use "ASML" not "ASML Holding N.V."
3. For FinancialMetric entities, include the value and year in the name:
   Example: "Apple Revenue FY2024 USD 391B"
4. Only extract relationships where BOTH the source and target entities
   appear in your entities list.
5. Return ONLY valid JSON. No explanation, no markdown, no code fences.

EXAMPLE OUTPUT FORMAT:
{{
  "entities": [
    {{"name": "TSMC", "type": "Company"}},
    {{"name": "Apple", "type": "Company"}},
    {{"name": "Taiwan", "type": "Country"}},
    {{"name": "M3 chip", "type": "Product"}}
  ],
  "relationships": [
    {{
      "source": "TSMC",
      "target": "Apple",
      "type": "SUPPLIES_CHIPS_TO",
      "properties": {{
        "component": "M3 chip",
        "process_node": "N3"
      }}
    }},
    {{
      "source": "TSMC",
      "target": "Taiwan",
      "type": "OPERATES_IN",
      "properties": {{}}
    }}
  ]
}}

TEXT TO ANALYSE:
{text}

Return ONLY the JSON object. No other text.
"""


class EntityExtractor:
    """
    Extracts entities and relationships from text chunks using GPT-4o.

    This class wraps the GPT-4o API call and handles:
      - Building the extraction prompt for each chunk
      - Calling GPT-4o and parsing the JSON response
      - Handling API errors and malformed responses gracefully
      - Rate limiting to avoid hitting OpenAI's API limits
      - Deduplicating entities and relationships across all chunks

    Usage:
        extractor = EntityExtractor()
        results = extractor.extract_from_chunks(chunks)
        # results is a dict with "entities" and "relationships" lists
    """

    def __init__(self):
        """
        Initialises the GPT-4o client and the extraction prompt template.
        """

        # Initialise the GPT-4o chat model
        # temperature=0 means deterministic output — we want consistent
        # entity extraction, not creative variation between runs
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

        # Build the prompt template
        # PromptTemplate.from_template() identifies {variable} placeholders
        # and fills them in when we call prompt.format(...)
        self.prompt = PromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)

        # Format the entity and relationship type lists as newline-separated
        # strings so they are readable in the prompt
        self.entity_types_str = "\n".join(f"  - {t}" for t in ENTITY_TYPES)
        self.relationship_types_str = "\n".join(f"  - {t}" for t in RELATIONSHIP_TYPES)

        logger.info(
            f"EntityExtractor initialised with model: {settings.llm_model}"
        )

    def extract_from_chunk(self, chunk: Document) -> Dict[str, Any]:
        """
        Extracts entities and relationships from a single text chunk.

        Sends the chunk text to GPT-4o with our extraction prompt and
        parses the JSON response. Returns a dict with "entities" and
        "relationships" lists, plus metadata about which chunk this
        extraction came from.

        Args:
            chunk: A LangChain Document object (one text chunk from text_chunker.py)

        Returns:
            A dictionary with this structure:
            {
                "chunk_id": "apple...pdf__page_1__chunk_2",
                "source": "apple_annual_overview_FY2024.pdf",
                "entities": [{"name": "TSMC", "type": "Company"}, ...],
                "relationships": [{"source": "TSMC", "target": "Apple",
                                   "type": "SUPPLIES_CHIPS_TO",
                                   "properties": {...}}, ...]
            }
            Returns empty lists for entities and relationships if extraction fails.
        """

        # Build the full prompt by filling in the template variables
        filled_prompt = self.prompt.format(
            entity_types=self.entity_types_str,
            relationship_types=self.relationship_types_str,
            text=chunk.page_content,
        )

        try:
            # Call GPT-4o with the filled prompt
            # invoke() sends the message and returns a response object
            # .content gives us the string text of GPT-4o's reply
            response = self.llm.invoke(filled_prompt)
            raw_response = response.content.strip()

            # Parse the JSON response from GPT-4o
            # GPT-4o sometimes wraps JSON in markdown code fences like ```json
            # We strip those out before parsing to be safe
            clean_response = raw_response
            if clean_response.startswith("```"):
                # Remove opening fence (```json or ```)
                clean_response = clean_response.split("\n", 1)[1]
            if clean_response.endswith("```"):
                # Remove closing fence
                clean_response = clean_response.rsplit("```", 1)[0]

            # Parse the cleaned string as JSON
            extracted = json.loads(clean_response.strip())

            # Validate that the response has the expected structure
            # If GPT-4o returned something unexpected, default to empty lists
            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])

            # Log what was found in this chunk
            logger.debug(
                f"Chunk {chunk.metadata.get('chunk_id', 'unknown')}: "
                f"extracted {len(entities)} entities, "
                f"{len(relationships)} relationships"
            )

            return {
                "chunk_id":      chunk.metadata.get("chunk_id", "unknown"),
                "source":        chunk.metadata.get("source", "unknown"),
                "entities":      entities,
                "relationships": relationships,
            }

        except json.JSONDecodeError as e:
            # GPT-4o returned something that is not valid JSON
            # Log the error and return empty results rather than crashing
            logger.warning(
                f"JSON parse error for chunk "
                f"{chunk.metadata.get('chunk_id', 'unknown')}: {e}\n"
                f"Raw response was: {raw_response[:200]}..."
            )
            return {
                "chunk_id":      chunk.metadata.get("chunk_id", "unknown"),
                "source":        chunk.metadata.get("source", "unknown"),
                "entities":      [],
                "relationships": [],
            }

        except Exception as e:
            # Any other error (network issue, API timeout, etc.)
            logger.error(
                f"Extraction failed for chunk "
                f"{chunk.metadata.get('chunk_id', 'unknown')}: {e}"
            )
            return {
                "chunk_id":      chunk.metadata.get("chunk_id", "unknown"),
                "source":        chunk.metadata.get("source", "unknown"),
                "entities":      [],
                "relationships": [],
            }

    def extract_from_chunks(
        self,
        chunks: List[Document],
        delay_between_calls: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Extracts entities and relationships from all chunks and deduplicates them.

        Processes each chunk one at a time, waits delay_between_calls seconds
        between API calls to avoid hitting OpenAI's rate limits, then merges
        all results and removes duplicate entities and relationships.

        Args:
            chunks:                 List of chunk Documents from text_chunker.py.
            delay_between_calls:    Seconds to wait between GPT-4o API calls.
                                    Default 0.5s keeps us well under rate limits.

        Returns:
            A dictionary with:
              "entities"      : deduplicated list of all unique entities
              "relationships" : deduplicated list of all unique relationships
              "chunk_results" : raw per-chunk extraction results (for debugging)

        Example:
            {
                "entities": [
                    {"name": "TSMC", "type": "Company"},
                    {"name": "Apple", "type": "Company"},
                    ...
                ],
                "relationships": [
                    {"source": "TSMC", "target": "Apple",
                     "type": "SUPPLIES_CHIPS_TO", "properties": {...}},
                    ...
                ],
                "chunk_results": [...]
            }
        """

        logger.info(f"Starting entity extraction from {len(chunks)} chunks")

        # Store per-chunk results for debugging and transparency
        chunk_results = []

        # Process each chunk with a delay between calls
        for i, chunk in enumerate(chunks):

            logger.info(
                f"Extracting from chunk {i + 1}/{len(chunks)}: "
                f"{chunk.metadata.get('chunk_id', 'unknown')}"
            )

            # Extract entities and relationships from this chunk
            result = self.extract_from_chunk(chunk)
            chunk_results.append(result)

            # Wait between API calls to respect OpenAI rate limits
            # We skip the delay after the last chunk
            if i < len(chunks) - 1:
                time.sleep(delay_between_calls)

        # Deduplicate entities across all chunks
        # The same entity (e.g. "TSMC") will appear in dozens of chunks
        # We use a set of (name, type) tuples to track what we have seen
        unique_entities = []
        seen_entities = set()

        for result in chunk_results:
            for entity in result["entities"]:
                # Create a hashable key from name and type
                entity_key = (
                    entity.get("name", "").strip().lower(),
                    entity.get("type", "").strip(),
                )
                # Only add if we have not seen this entity before
                if entity_key not in seen_entities and entity_key[0]:
                    seen_entities.add(entity_key)
                    unique_entities.append(entity)

        # Deduplicate relationships across all chunks
        # A relationship is a triple: (source, target, type)
        # The same relationship may be mentioned in multiple chunks
        unique_relationships = []
        seen_relationships = set()

        for result in chunk_results:
            for rel in result["relationships"]:
                # Create a hashable key from source, target, and type
                rel_key = (
                    rel.get("source", "").strip().lower(),
                    rel.get("target", "").strip().lower(),
                    rel.get("type", "").strip(),
                )
                # Only add if we have not seen this relationship before
                if rel_key not in seen_relationships and all(rel_key):
                    seen_relationships.add(rel_key)
                    unique_relationships.append(rel)

        logger.success(
            f"Extraction complete — "
            f"{len(unique_entities)} unique entities, "
            f"{len(unique_relationships)} unique relationships "
            f"extracted from {len(chunks)} chunks"
        )

        return {
            "entities":      unique_entities,
            "relationships": unique_relationships,
            "chunk_results": chunk_results,
        }