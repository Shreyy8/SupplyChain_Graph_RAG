
# Supply Chain Graph RAG with Neo4j Aura DB instance

**A hybrid Graph RAG pipeline for semiconductor supply-chain intelligence.**

---

## Table of contents

1. [The case study — what problem are we solving?](#1-the-case-study)
2. [Why Graph RAG (and not plain vector RAG)?](#2-why-graph-rag)
3. [System architecture](#3-system-architecture)
4. [Project structure — every folder and file](#4-project-structure)
5. [The five agents, explained](#5-the-five-agents)
6. [Setup and installation](#6-setup-and-installation)
7. [How to run it (in order)](#7-how-to-run-it)
8. [`run_ingestion.py` — what it does](#8-run_ingestion)
9. [`test_queries.py` — what it does](#9-test_queries)
10. [The demo questions and expected output](#10-demo-questions)
11. [Neo4j Aura query cookbook (copy-paste Cypher)](#11-cypher-cookbook)
12. [Known gotchas and fixes](#12-gotchas)

---

<a name="1-the-case-study"></a>

## 1. The case study — what problem are we solving?

### The company

Imagine **Meridian Semiconductor Advisory**, a research desk that advises
investors and manufacturers on **supply-chain risk** in the global chip
industry. Their analysts answer questions like: *"If one link in the chain
breaks, who gets hurt, how badly, and through what path?"*

Their world is a chain of dependencies:

```
ASML  ──(EUV machines)──▶  TSMC  ──(chips)──▶  Apple / Nvidia  ──▶  Revenue
```

- **ASML** (Netherlands) is the *only* company that makes the EUV
  lithography machines needed for advanced chips.
- **TSMC** (Taiwan) uses those machines to manufacture chips.
- **Apple** and **Nvidia** depend on TSMC to build their products.
- **Samsung** sits awkwardly across the chain — both a *supplier* to Apple
  and a *competitor*, and both a rival to TSMC and a customer of ASML.

### The problem with their current tooling

Meridian's analysts have a pile of documents — annual reports, technology
reports, a supply-chain risk assessment. They tried a normal RAG chatbot
over these PDFs. It works for simple lookups ("what was TSMC's revenue?")
but **falls apart on the questions that actually matter**:

> *"If ASML stops shipping EUV machines, which companies face the highest
> revenue risk, and why?"*

A vector RAG system fails here because **the answer does not exist in any
single document**. The ASML report talks about EUV exports. The TSMC report
says TSMC buys from ASML and sells to Apple. The Apple report says Apple
depends on TSMC. To answer the question you must **connect facts across four
documents** — a chain the chatbot cannot follow, because it only retrieves
passages that individually *look like* the question.

### What SUPPLY-SIGHT does about it

SUPPLY-SIGHT stores the analysts' knowledge as a **graph of relationships** and
answers questions by **walking the chain** — while *also* pulling supporting
text from the documents. The result is a system that can trace a disruption
from ASML all the way to Apple's revenue, cite the documents that support
each hop, and explain *why*.

---

<a name="2-why-graph-rag"></a>

## 2. Why Graph RAG (and not plain vector RAG)?

There are two ways to "retrieve" for RAG:

|                | **Vector RAG** (vanilla)                         | **Graph RAG**                                    |
| -------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| Retrieves by   | *Similarity* — passages that look like the question | *Relationships* — walking connections between facts |
| Good at        | "Find the fact about X"                                | "Connect facts: X → Y → Z"                           |
| Struggles with | Multi-hop / cross-document reasoning                   | Simple single-passage lookups                          |
| Data shape     | Chunks + embeddings                                    | Nodes + relationships                                  |

Both are "RAG" — they differ only in *how* they retrieve. FINSIGHT is
**hybrid**: it uses *both*, because real questions need both the relationship
chain (graph) and the supporting detail (vectors).

### The multi-hop test

These questions all require **traversal across two or more documents** —
they break vector RAG and showcase Graph RAG:

- "If ASML stops shipping EUV machines, which companies face the highest
  revenue risk?" — a 4-hop chain: ASML → TSMC → Apple/Nvidia → Revenue.
- "Which companies are simultaneously customers of TSMC and competitors of
  Apple?" — requires seeing two *different* relationships on the same node.
- "Trace the path from a Dutch equipment factory to an iPhone on a shelf."

No single chunk answers these. Following the graph does.

---

<a name="3-system-architecture"></a>

## 3. System architecture

```
                         User question
                              │
                              ▼
                   ┌────────────────────┐
                   │   entity_agent      │  extract entities + intent + hop depth
                   └────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
      ┌────────────────────┐   ┌────────────────────┐
      │ graph_query_agent   │   │   vector_agent      │
      │  traverse Neo4j     │   │  search ChromaDB    │   ← run IN PARALLEL
      └────────────────────┘   └────────────────────┘
                 └────────────┬────────────┘
                              ▼
                   ┌────────────────────┐
                   │   rerank_agent      │  fuse both lists with RRF (k=60),
                   │                     │  format the top evidence
                   └────────────────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │  synthesis_agent    │  GPT-4o writes the final cited answer
                   └────────────────────┘
                              │
                              ▼
                           Answer
```

### The stack

| Layer           | Technology                               |
| --------------- | ---------------------------------------- |
| LLM             | OpenAI**GPT-4o**                   |
| Embeddings      | OpenAI**text-embedding-3-small**   |
| Knowledge graph | **Neo4j AuraDB**                   |
| Vector store    | **ChromaDB** (persisted locally)   |
| Orchestration   | **LangGraph** (5 agents)           |
| Fusion          | **Reciprocal Rank Fusion**, k = 60 |
| API             | **FastAPI**                        |

---

<a name="4-project-structure"></a>

## 4. Project structure — every folder and file

```
GRAPH_RAG_PIPELINE/
│
├── api/                          FastAPI web layer
│   ├── main.py                   app entry point (uvicorn target: api.main:app)
│   ├── routes/
│   │   ├── health.py             health-check endpoint
│   │   ├── ingest.py             trigger ingestion via API
│   │   └── query.py              the /query endpoint (ask a question)
│   └── schemas/
│       └── models.py             Pydantic request/response models
│
├── data/
│   ├── pdfs/                     the 5 source PDFs (see below)
│   └── chroma_db/                persisted ChromaDB vector index (auto-created)
│
├── graph_rag/                    the core pipeline package
│   ├── agents/                   the 5 LangGraph agents + shared state
│   │   ├── state.py              GraphRAGState — the shared "whiteboard"
│   │   ├── entity_agent.py       (1) extract entities + intent
│   │   ├── graph_query_agent.py  (2) traverse Neo4j
│   │   ├── vector_agent.py       (3) semantic search over ChromaDB
│   │   ├── rerank_agent.py       (4) fuse graph + vector with RRF
│   │   └── synthesis_agent.py    (5) GPT-4o writes the final answer
│   │
│   ├── extraction/
│   │   └── entity_extractor.py   LLM-powered entity/relationship extraction from text
│   ├── graph/
│   │   ├── neo4j_client.py       manages the Neo4j Aura connection
│   │   └── graph_builder.py      writes nodes/relationships into Neo4j
│   ├── ingestion/
│   │   ├── pdf_loader.py         reads the PDFs
│   │   └── text_chunker.py       splits text into overlapping chunks
│   ├── pipeline/
│   │   └── graph_pipeline.py     wires the 5 agents into a LangGraph graph
│   ├── retrieval/
│   │   ├── vector_store.py       ChromaDB wrapper (embed + search)
│   │   └── hybrid_retriever.py   graph + vector retrieval helper
│   └── utils/
│       └── config.py             loads + validates settings from .env
│
├── scripts/
│   ├── run_ingestion.py          BUILD everything (run once)
│   └── test_queries.py           run the demo questions end to end
│
├── .env                          credentials + config (NEVER commit)
├── .env.example                  template for .env
└── README.md
```

### The five source documents (`data/pdfs/`)

| Document                                       | Seeds into the graph                                          |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `apple_annual_overview_FY2024.pdf`           | Apple as a central node (→ TSMC, Samsung)                    |
| `tsmc_manufacturing_report_FY2023.pdf`       | TSMC as the hub (ASML upstream; Apple, Nvidia downstream)     |
| `nvidia_annual_review_FY2024.pdf`            | Nvidia downstream of TSMC; the 4-layer chain                  |
| `asml_technology_report_FY2023.pdf`          | ASML as the root upstream node                                |
| `global_semiconductor_supply_chain_risk.pdf` | Ties it all together — the richest source of multi-hop paths |

After ingestion the graph holds roughly **286 nodes and 253 relationships**
in Neo4j and about **136 chunks** in ChromaDB.

---

<a name="5-the-five-agents"></a>

## 5. The five agents, explained

Each agent is a node in the LangGraph pipeline. They share one state object
(`GraphRAGState` in `state.py`) — each reads what it needs and writes its
own fields.

**1. `entity_agent.py` — understand the question.**
Uses GPT-4o to extract the entities in the question (e.g. `["ASML", "TSMC", "Apple"]`), classify the *intent* (simple lookup vs. risk-cascade), and
recommend how many hops of graph traversal are needed. Writes
`extracted_entities`, `intent`, and `recommended_hops`.

**2. `graph_query_agent.py` — walk the graph.**
Takes those entities and runs Cypher against Neo4j. Two queries per entity:
a **1-hop** query for the immediate neighbourhood, and a **multi-hop** query
(`[*2..max_hops]`) for the cascade chains. Deduplicates and writes
`graph_results`.

**3. `vector_agent.py` — search the documents.**
Embeds the question and does a semantic similarity search over ChromaDB,
returning the most relevant text chunks with source/page citations. Writes
`vector_results`. **Runs in parallel with the graph agent** — they touch
different state fields, so LangGraph executes them simultaneously to halve
retrieval latency.

**4. `rerank_agent.py` — fuse the two result sets.**
This is where graph and vector evidence get merged with **Reciprocal Rank
Fusion (RRF)**. RRF scores each result by `1 / (k + rank)` where `k = 60`,
summing contributions across both lists — so a fact that ranks well in
*both* the graph and vector lists is boosted. It then formats the top
results into a clean context string (a GRAPH EVIDENCE section and a TEXT
EVIDENCE section). Writes `merged_results` and `formatted_context`.

> **Why RRF?** The two retrievers score on incompatible scales (graph =
> hop-distance, vector = cosine similarity). RRF ignores raw scores and uses
> only *rank/position*, which *is* comparable — so it fuses them without any
> fragile score-normalisation. `k=60` dampens the gap between top ranks so
> no single list can dominate.

**5. `synthesis_agent.py` — write the answer.**
Takes the formatted context and the original question, and prompts GPT-4o to
write the final answer, grounded in the retrieved evidence and citing the
graph chain and documents. Writes `final_answer`.

---

<a name="6-setup-and-installation"></a>

## 6. Setup and installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in:

```
# OpenAI
OPENAI_API_KEY=sk-your-key

# Neo4j Aura (from your instance's Connect dropdown)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-generated-password

# Paths and retrieval settings
PDF_DIR=./data/pdfs
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=finsight_chunks
CHUNK_SIZE=300
CHUNK_OVERLAP=100
TOP_K_VECTOR=5
TOP_K_GRAPH_HOPS=3
```

> **Never commit `.env`.** It holds two secrets (OpenAI + Neo4j). Add it to
> `.gitignore` before your first `git add`.

### 3. Network note (important on corporate machines)

Neo4j connects over **port 7687**. Many corporate firewalls block it. If you
get `ServiceUnavailable: Unable to retrieve routing information` or a
timeout, port 7687 is likely blocked — confirm on a phone hotspot. Fixes:
have IT whitelist outbound 7687 to `*.databases.neo4j.io`, or use the Aura
browser Query console (HTTPS/443) for the graph parts.

### 4. Mac SSL note

If you hit an SSL certificate error connecting to Aura on macOS, run the
Python certificate installer:
`/Applications/Python\ 3.x/Install\ Certificates.command`.

---

<a name="7-how-to-run-it"></a>

## 7. How to run it (in order)

Run all commands from the **project root**.

```bash
# 1. Build the graph + vector index (run ONCE)
python3 scripts/run_ingestion.py

# 2. Verify the pipeline with the demo questions
python3 scripts/test_queries.py

# 3. Serve it as a live API
uvicorn api.main:app --reload
#    then open http://localhost:8000/docs
```

The order matters: ingest builds the data, test verifies the pipeline in
isolation, the API exposes it. If step 2 fails, you know it's the pipeline,
not the API.

---

<a name="8-run_ingestion"></a>

## 8. `run_ingestion.py` — what it does

This is the **build** step. It runs once and populates both data stores.
Its pipeline:

1. **Load PDFs** (`pdf_loader.py`) — reads all 5 documents from `PDF_DIR`.
2. **Chunk text** (`text_chunker.py`) — splits each document into
   overlapping chunks (`CHUNK_SIZE=300`, `CHUNK_OVERLAP=100`).
3. **Embed + index** (`vector_store.py`) — embeds each chunk with
   text-embedding-3-small and stores it in ChromaDB (`~136 chunks`).
4. **Extract entities/relationships** (`entity_extractor.py`) — uses GPT-4o
   to pull entities (ASML, TSMC…) and relationships (SUPPLIES_TO,
   DEPENDS_ON…) from the text.
5. **Build the graph** (`graph_builder.py`, `neo4j_client.py`) — writes
   those nodes and relationships into Neo4j (`~286 nodes, ~253 rels`).
6. **Self-verify** — reads back key relationships to confirm the graph
   built correctly. You will see lines like:

```
Connected to Neo4j AuraDB successfully
[PASS] TSMC supplies chips to Apple
[PASS] ASML supplies equipment to TSMC
[PASS] TSMC manufactures for Nvidia
[PASS] TSMC operates in Taiwan
[PASS] Apple depends on TSMC
Neo4j connection closed
```

Each `[PASS]` means that relationship exists in the graph — these are the
exact hops the demo questions traverse, so passing them confirms the graph
can answer them.

**Expected end state:** `data/chroma_db/` contains `chroma.sqlite3` plus a
UUID folder, and Neo4j holds ~286 nodes / ~253 relationships.

---

<a name="9-test_queries"></a>

## 9. `test_queries.py` — what it does

This is the **verify** step. It holds the demo questions (as a
`DEMO_QUESTIONS` list) and runs each one through the full five-agent
pipeline, printing the answer.

Each question in the list is a dict with three keys:

- `id` — a number.
- `question` — the actual question sent to the pipeline.
- `why_graph_rag` — a **human-readable note** explaining why this question
  needs graph traversal (e.g. *"Requires 4-hop traversal: ASML → TSMC →
  Apple/Nvidia → Revenue. No single document contains this complete
  chain."*). This is documentation for *you* / the audience, not something
  the pipeline consumes — it makes the test set self-explaining for a demo.

Running it exercises entity extraction → parallel graph+vector retrieval →
RRF rerank → synthesis, so a clean run proves the whole system works.

---

<a name="10-demo-questions"></a>

## 10. The demo questions and expected output

These are the showcase questions. Each requires multi-hop reasoning that
vector RAG alone cannot do.

**Q1 — Cascade risk:**

> "If ASML stops shipping EUV lithography machines, which companies face the
> highest revenue risk and why? Trace the complete impact chain."

*Expected:* an answer naming TSMC (directly, as ASML's customer), then Apple
and Nvidia (downstream of TSMC), tracing ASML → TSMC → Apple/Nvidia →
revenue, with the graph chain and supporting document citations.

**Q2 — Path tracing:**

> "Trace the complete supply chain path from a Dutch equipment factory to an
> iPhone on a retail shelf."

*Expected:* ASML (Netherlands) → TSMC (chips) → Apple (iPhone) → retail,
drawn from the graph.

**Q3 — Dual-role reasoning:**

> "Which companies are simultaneously customers of TSMC and competitors of
> Apple?"

*Expected:* Samsung surfaces — it buys from / competes across the chain.
This needs the graph to hold multiple relationships on one node.

**Q4 — Quantified impact:**

> "If Taiwan faces a 90-day blockade, quantify the revenue impact on each
> company in the graph."

*Expected:* an impact summary per company, drawing on the risk-assessment
document's figures, following TSMC's Taiwan concentration through the chain.

> **What "good output" looks like:** the answer should reference a
> *relationship chain* (not just isolated facts), and cite both graph
> evidence and document passages. If an answer is vague or only cites one
> document, the graph half may not be contributing — check the graph agent.

---

<a name="11-cypher-cookbook"></a>

## 11. Neo4j Aura query cookbook (copy-paste Cypher)

Run these directly in the **Aura Query tab** (browser) to inspect and
demonstrate the graph. Each is copy-paste ready, with the expected result.

### See the whole graph (with relationships)

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

*Returns:* the full graph drawn visually — all nodes and the arrows between
them. (Use this, not `MATCH (n) RETURN n`, which hides the arrows.)

### Count nodes and relationships

```cypher
MATCH (n) RETURN count(n) AS nodes
```

*Returns:* ~286.

```cypher
MATCH ()-[r]->() RETURN count(r) AS relationships
```

*Returns:* ~253.

### What node types (labels) exist?

```cypher
MATCH (n) RETURN DISTINCT labels(n) AS node_type, count(*) AS count ORDER BY count DESC
```

*Returns:* a table of labels (Company, Country, Product, …) and how many of
each.

### What relationship types exist?

```cypher
MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS count ORDER BY count DESC
```

*Returns:* SUPPLIES_CHIPS_TO, SUPPLIES_EQUIPMENT_TO, DEPENDS_ON,
MANUFACTURES_FOR, OPERATES_IN, … with counts.

### Everything directly connected to TSMC (1 hop)

```cypher
MATCH (t {name: "TSMC"})-[r]-(other)
RETURN t.name AS from, type(r) AS relationship, other.name AS to
```

*Returns:* TSMC's immediate neighbourhood — its suppliers (ASML), customers
(Apple, Nvidia), and location (Taiwan).

### The ASML → TSMC → customer cascade (the multi-hop showcase)

```cypher
MATCH path = (asml {name: "ASML"})-[*1..3]-(affected)
RETURN path
```

*Returns:* every entity within 3 hops of ASML, drawn as paths — the visual
version of "who is exposed if ASML fails."

### Which companies depend (directly or indirectly) on TSMC?

```cypher
MATCH (c:Company)-[:DEPENDS_ON|SUPPLIES_CHIPS_TO|MANUFACTURES_FOR*1..3]-(t {name: "TSMC"})
RETURN DISTINCT c.name AS exposed_company
```

*Returns:* Apple, Nvidia (and any others linked through the chain).

### Find the dual-role node (Samsung)

```cypher
MATCH (s {name: "Samsung"})-[r]-(other)
RETURN s.name AS samsung, type(r) AS relationship, other.name AS other
```

*Returns:* all of Samsung's edges at once — supplier, competitor, customer —
showing why it's the interesting "friend and threat" node.

### Trace a specific path: ASML to Apple

```cypher
MATCH path = shortestPath((a {name: "ASML"})-[*]-(b {name: "Apple"}))
RETURN path
```

*Returns:* the shortest chain of relationships linking ASML to Apple (e.g.
ASML → TSMC → Apple).

### Reset the graph (start clean before re-ingesting)

```cypher
MATCH (n) DETACH DELETE n
```

*Returns:* nothing — deletes all nodes and relationships. Run this if you
want to re-ingest from scratch without duplicates. **Destructive — use with
care.**

> **Teaching tip:** run a demo question in `test_queries.py`, then run the
> matching Cypher above in the Aura browser to *show the path the answer
> walked*. Seeing the graph light up next to the natural-language answer is
> the moment Graph RAG clicks for a room.

---

<a name="12-gotchas"></a>

## 12. Known gotchas and fixes

**Cypher parameters in variable-length paths.**
Neo4j does **not** allow a `$parameter` inside a variable-length path range
like `[*2..$max_hops]` — it only accepts a literal number. Fix: build that
value into the query with an f-string (`[*2..{max_hops}]`), keeping other
values as proper `$parameters`. (This bit us in `graph_query_agent.py`.)

**LangChain v0.2+ import moves.**
Imports from `langchain.schema`, `langchain.prompts`, and
`langchain.text_splitter` moved to their `langchain_core` equivalents.

**Run ingestion once.**
Re-running `run_ingestion.py` rebuilds the graph; clear it first (the reset
Cypher above) if you don't want duplicates.

**Always run from the project root.**
The `.env` paths (`./data/pdfs`, `./data/chroma_db`) are relative to the
root. Launching a script from inside `scripts/` breaks them.

**Port 7687 / firewall.**
Covered in setup — the most common connection failure on corporate networks.

---

The whole point is a question like *"if ASML shuts down, what happens to
Apple's revenue?"* — a chain no single document contains. Vector RAG
retrieves documents; Graph RAG follows the chain. FINSIGHT shows **both
working together**, and lets you *prove* the reasoning by drawing the exact
path in the Neo4j browser next to the answer.
