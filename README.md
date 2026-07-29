
# FINSIGHT_GraphRAG



GRAPH_RAG_PIPELINE/
├── api/           (main.py, routes/, schemas/)
├── data/          (pdfs/, chroma_db/)
├── graph_rag/
│   ├── agents/        (the 5 agents + state.py)
│   ├── extraction/    (entity_extractor.py)
│   ├── graph/         (graph_builder.py, neo4j_client.py)
│   ├── ingestion/     (pdf_loader.py, text_chunker.py)
│   ├── pipeline/      (graph_pipeline.py)
│   ├── retrieval/     (hybrid_retriever.py, vector_store.py)
│   └── utils/
├── scripts/       (run_ingestion.py, test_queries.py)
├── .env
└── README.md

A hybrid **Graph RAG** pipeline for semiconductor supply-chain intelligence.
It combines a **knowledge graph** (relationships) with **vector search**
(semantic similarity), fuses the two with reranking, and orchestrates the
whole retrieval as a **multi-agent workflow** — so it can answer questions
that require *tracing dependencies across multiple documents*, which
standard RAG cannot.

Built as a teaching demo: the domain is the real-world chip supply chain —
**Apple, TSMC, Nvidia, ASML, and Samsung**.

## Why hybrid (Graph + Vector)?

- **Vector RAG** finds passages that *look like* your question. Great for
  "what did TSMC report about capacity?" — a single-chunk lookup.
- **Graph RAG** follows the *relationships between facts*. Great for "if
  ASML stops shipping EUV machines, whose revenue is at risk?" — a
  multi-hop chain no single passage contains.

FINSIGHT uses **both**: the graph traverses the dependency chain, the
vectors pull supporting detail, and a reranker fuses the results before the
LLM writes the final answer.

## The questions it can answer (that break standard RAG)

- "If ASML stops shipping EUV machines, which companies face the highest
  revenue risk and why?"
- "Trace the complete supply chain path from a Dutch equipment factory to an
  iPhone on a retail shelf."
- "Which companies are simultaneously customers of TSMC and competitors to
  Apple?"
- "If Taiwan faces a 90-day blockade, quantify the revenue impact on each
  company in the graph."

Each requires multi-hop traversal across two or more documents.

## Architecture

```
        Question
           │
           ▼
   ┌───────────────┐
   │ entity_agent   │  extract the entities in the question
   └───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────┐
│graph_query│ │  vector  │   graph traversal (Neo4j)  +
│  _agent   │ │  _agent  │   semantic search (ChromaDB)
└──────────┘ └──────────┘
     └─────┬─────┘
           ▼
   ┌───────────────┐
   │ rerank_agent   │  fuse both result sets (Reciprocal Rank Fusion, k=60)
   └───────────────┘
           │
           ▼
   ┌───────────────┐
   │synthesis_agent │  write the final grounded answer
   └───────────────┘
           │
           ▼
        Answer
```

The five agents are orchestrated with **LangGraph**.

## Stack

| Layer           | Technology                    |
| --------------- | ----------------------------- |
| LLM             | OpenAI GPT-4o                 |
| Embeddings      | OpenAI text-embedding-3-small |
| Knowledge graph | Neo4j AuraDB                  |
| Vector store    | ChromaDB                      |
| Orchestration   | LangGraph (5 agents)          |
| Reranking       | Reciprocal Rank Fusion (k=60) |
| API             | FastAPI                       |

## The data

Five synthetic PDF documents (generated for the demo), each seeding a part
of the supply-chain graph:

1. **Apple FY2024** — Apple as a central node (→ TSMC, Samsung, ASML)
2. **TSMC FY2023** — TSMC as the hub (ASML upstream; Apple, Nvidia downstream)
3. **Nvidia FY2024** — Nvidia downstream of TSMC; the four-layer chain
4. **ASML FY2023** — ASML as the root upstream node
5. **Global Supply-Chain Risk Assessment** — ties it all together; the
   richest source of multi-hop paths

After ingestion the graph holds roughly **286 nodes and 253 relationships**
in Neo4j, and about **136 chunks** in ChromaDB.

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

```
# OpenAI
OPENAI_API_KEY=sk-your-key

# Neo4j Aura
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-generated-password
```

> Never commit `.env`. On Mac, if you hit an SSL certificate error when
> connecting, run the Python certificate installer:
> `/Applications/Python\ 3.x/Install\ Certificates.command`

## Run it (in order)

**1. Ingest — build the graph and vector index (run once):**

```bash
python3 scripts/run_ingestion.py
```

Expect roughly 286 nodes / 253 relationships in Neo4j and ~136 chunks in
ChromaDB.

**2. Test — run the demo multi-hop questions:**

```bash
python3 scripts/test_queries.py
```

**3. Serve — start the API:**

```bash
uvicorn app.main:app --reload
```

Then query the API with your own questions.

## Project structure

```
FINSIGHT_GraphRAG/
├── graph_rag/
│   ├── agents/          # entity, graph_query, vector, rerank, synthesis
│   ├── ingestion/       # PDF parsing, entity/relationship extraction
│   └── utils/
│       └── config.py    # settings (models, DB connections)
├── scripts/
│   ├── run_ingestion.py # build the graph + vector index (run once)
│   └── test_queries.py  # the demo multi-hop questions
├── data/                # the 5 synthetic PDFs
├── requirements.txt
└── .env                 # credentials (never commit)
```

## Notes / gotchas

- **Cypher params in variable-length paths:** Neo4j does not allow a
  parameter (`$max_hops`) inside a variable-length path pattern. Use an
  f-string literal instead (e.g. `{max_hops}` built into the query string).
- **LangChain v0.2+ imports:** imports from `langchain.schema`,
  `langchain.prompts`, and `langchain.text_splitter` moved to their
  `langchain_core` equivalents.
- **Run ingestion once.** Re-running rebuilds the graph; clear old data
  first if you don't want duplicates.

## What makes this a good teaching demo

The whole point is a question like *"if ASML shuts down, what happens to
Apple's revenue?"* — a **three-hop chain** (ASML → TSMC → Apple → revenue)
where no single document has the answer. Standard RAG retrieves documents;
Graph RAG follows the chain. FINSIGHT shows both working together.
