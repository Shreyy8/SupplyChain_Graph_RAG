"""
test_questions.py - Evaluation Runner for Assignment 2 Section 7 Questions

Runs all 10 mandatory test questions from Assignment 2 and prints formatted outputs.
"""

import sys
import time
from dotenv import load_dotenv

load_dotenv()

from rag import query_rag

ASSIGNMENT_QUESTIONS = [
    {
        "id": 1,
        "type": "Single-document",
        "question": "Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?"
    },
    {
        "id": 2,
        "type": "Single-document",
        "question": "How many line stoppages happened in Q1, what was the total downtime, and what caused them?"
    },
    {
        "id": 3,
        "type": "Single-document",
        "question": "What is the approval authority for a purchase order worth ₹1.4 crore?"
    },
    {
        "id": 4,
        "type": "Single-document",
        "question": "What are the four supplier classification categories, and what qualifies a supplier as Critical?"
    },
    {
        "id": 5,
        "type": "Cross-document",
        "question": "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?"
    },
    {
        "id": 6,
        "type": "Cross-document",
        "question": "The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?"
    },
    {
        "id": 7,
        "type": "Cross-document",
        "question": "Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?"
    },
    {
        "id": 8,
        "type": "Cross-document",
        "question": "Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?"
    },
    {
        "id": 9,
        "type": "Cross-document",
        "question": "Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?"
    },
    {
        "id": 10,
        "type": "Deliberate trap question",
        "question": "What is the annual salary of the Head of Procurement?"
    }
]


def run_all_tests():
    print("=" * 80)
    print("ASSIGNMENT 2 — SUPPLY CHAIN RAG EVALUATION SUITE")
    print("=" * 80)

    for item in ASSIGNMENT_QUESTIONS:
        q_id = item["id"]
        q_type = item["type"]
        question = item["question"]

        print(f"\n[Question {q_id}] ({q_type})")
        print(f"Q: {question}")
        print("-" * 80)

        start_t = time.time()
        try:
            res = query_rag(question)
            elapsed = time.time() - start_t
            print("ANSWER:")
            print(res["answer"])
            print("\nSOURCES CITED:")
            for s in res["sources"]:
                print(f"  - {s['file']} (Page {s['page']})")
            print(f"(Time: {elapsed:.2f}s, Chunks Retrieved: {res.get('total_chunks_retrieved', 0)})")
        except Exception as e:
            print(f"FAILED: {e}")

        print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
