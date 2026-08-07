# scripts/test_queries.py
#
# PURPOSE:
#   Runs the four demo multi-hop questions through the full pipeline
#   and prints the answers with sources and graph paths used.
#
#   These four questions are specifically chosen to demonstrate Graph RAG's
#   advantage over standard RAG. Each one requires multi-hop reasoning
#   across multiple documents that standard RAG cannot do reliably.
#
# DO YOU RUN THIS FILE?
#   YES. Run it after ingestion and with the API server NOT required:
#
#     python scripts/test_queries.py
#
#   This runs the pipeline directly (no HTTP) so you do not need the
#   API server running. Good for quick testing during development.
#
#   To test a single specific question:
#
#     python scripts/test_queries.py --question "Your question here"
#
# PREREQUISITES:
#   1. run_ingestion.py must have been run successfully
#   2. Neo4j AuraDB must be running
#   3. .env must be filled in correctly


import sys
import argparse
import time
from pathlib import Path
from loguru import logger

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_rag.pipeline.graph_pipeline import run_query


# -----------------------------------------------------------------------------
# THE FOUR DEMO QUESTIONS
# These questions are designed to demonstrate multi-hop reasoning.
# None of them can be answered reliably by standard RAG because
# each requires connecting facts across multiple documents.
# -----------------------------------------------------------------------------
DEMO_QUESTIONS = [
    {
        "id":       1,
        "question": (
            "If ASML stops shipping EUV lithography machines, "
            "which companies face the highest revenue risk and why? "
            "Trace the complete impact chain."
        ),
        # why_graph_rag — a note to a human explaining what makes this question hard for standard RAG
        "why_graph_rag": (
            "Requires 4-hop traversal: ASML -> TSMC -> Apple/Nvidia -> Revenue. "
            "No single document contains this complete chain."
        ),
    },
    {
        "id":       2,
        "question": (
            "Trace the complete supply chain path from a semiconductor "
            "equipment manufacturer in the Netherlands to an iPhone "
            "on a retail shelf. Name every company in the chain."
        ),
        "why_graph_rag": (
            "Requires following: ASML (Netherlands) -> TSMC (Taiwan) -> "
            "Apple (USA) -> iPhone. Spans three documents and three countries."
        ),
    },
    {
        "id":       3,
        "question": (
            "Samsung appears in multiple roles in this supply chain. "
            "Explain every role Samsung plays — as a supplier, "
            "as a competitor, and as a customer — and to which companies."
        ),
        "why_graph_rag": (
            "Samsung has circular relationships — supplier to Apple, "
            "competitor to TSMC, customer of ASML. Standard RAG returns "
            "fragments. Graph RAG returns all relationship types at once."
        ),
    },
    {
        "id":       4,
        "question": (
            "If Taiwan faced a military conflict for 90 days, "
            "quantify the total revenue at risk across all companies "
            "in this supply chain and explain the cascade order."
        ),
        "why_graph_rag": (
            "Requires: Taiwan risk -> TSMC (operates in Taiwan) -> "
            "Apple ($391B) + Nvidia ($60.9B) revenue figures from "
            "separate documents. Classic risk cascade multi-hop."
        ),
    },
]


def run_demo_query(question_data: dict, question_number: int, total: int):
    """
    Runs one demo question and prints the result in a formatted way.

    Args:
        question_data:   Dict with question, id, and why_graph_rag fields.
        question_number: Which question this is (for display).
        total:           Total number of questions being run.
    """

    question = question_data["question"]

    print()
    print("=" * 70)
    print(f"QUESTION {question_number}/{total}")
    print("=" * 70)
    print(f"Q: {question}")
    print()
    print(f"Why Graph RAG needed: {question_data['why_graph_rag']}")
    print("-" * 70)
    print("Running pipeline...")

    start_time = time.time()

    try:
        # Run the full pipeline for this question
        result = run_query(question)

        elapsed = time.time() - start_time

        # Print the answer
        print()
        print("ANSWER:")
        print("-" * 70)
        print(result.get("final_answer", "No answer generated."))

        # Print the graph paths used — this shows the multi-hop reasoning
        graph_paths = result.get("graph_paths_used", [])
        if graph_paths:
            print()
            print(f"GRAPH PATHS USED ({len(graph_paths)} relationship chains):")
            for path in graph_paths[:8]:   # show first 8 paths
                print(f"  {path}")
            if len(graph_paths) > 8:
                print(f"  ... and {len(graph_paths) - 8} more")

        # Print sources cited
        sources = result.get("sources", [])
        if sources:
            print()
            print(f"SOURCES CITED ({len(sources)}):")
            for source in sources:
                print(f"  {source}")

        # Print pipeline metadata
        print()
        print(f"PIPELINE METADATA:")
        print(f"  Intent detected:   {result.get('intent', 'unknown')}")
        print(f"  Entities found:    {result.get('extracted_entities', [])}")
        print(f"  Time elapsed:      {elapsed:.1f} seconds")

        if result.get("error"):
            print(f"  Warning:           {result.get('error')}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"PIPELINE ERROR after {elapsed:.1f}s: {e}")
        logger.error(f"Test query {question_number} failed: {e}")


def run_all_demo_questions():
    """
    Runs all four demo questions sequentially and prints results.
    Adds a 2-second pause between questions to avoid rate limiting.
    """

    print()
    print("=" * 70)
    print("FINSIGHT_GraphRAG — Demo Query Runner")
    print("=" * 70)
    print(f"Running {len(DEMO_QUESTIONS)} demo questions.")
    print("These questions demonstrate Graph RAG multi-hop reasoning.")
    print("=" * 70)

    total = len(DEMO_QUESTIONS)

    for i, question_data in enumerate(DEMO_QUESTIONS, start=1):
        run_demo_query(question_data, i, total)

        # Pause between questions to avoid hitting OpenAI rate limits
        if i < total:
            print()
            print(f"Pausing 2 seconds before next question...")
            time.sleep(2)

    print()
    print("=" * 70)
    print(f"All {total} demo questions complete.")
    print("=" * 70)


def run_single_question(question: str):
    """
    Runs a single custom question through the pipeline.

    Args:
        question: The question string to run.
    """

    print()
    print("=" * 70)
    print("FINSIGHT_GraphRAG — Single Query")
    print("=" * 70)
    print(f"Q: {question}")
    print("-" * 70)

    start_time = time.time()

    result  = run_query(question)
    elapsed = time.time() - start_time

    print()
    print("ANSWER:")
    print("-" * 70)
    print(result.get("final_answer", "No answer generated."))

    graph_paths = result.get("graph_paths_used", [])
    if graph_paths:
        print()
        print(f"GRAPH PATHS USED ({len(graph_paths)}):")
        for path in graph_paths[:8]:
            print(f"  {path}")

    sources = result.get("sources", [])
    if sources:
        print()
        print(f"SOURCES: {sources}")

    print()
    print(f"Intent: {result.get('intent')} | "
          f"Entities: {result.get('extracted_entities')} | "
          f"Time: {elapsed:.1f}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run demo queries through the FINSIGHT_GraphRAG pipeline"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Run a single custom question instead of the four demo questions",
    )
    args = parser.parse_args()

    if args.question:
        run_single_question(args.question)
    else:
        run_all_demo_questions()