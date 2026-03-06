# SHL Assessment Recommendation Engine

## Approach Document

---

### 1. Problem Statement

Hiring managers and recruiters face a significant challenge when selecting the right assessments from SHL's extensive product catalogue containing over 500 products. The existing catalogue relies on keyword filters and manual browsing, making it time-consuming and prone to missing relevant assessments.

**Goal:** Build an intelligent recommendation system that accepts a natural language hiring query or job description and returns the top 10 most relevant SHL assessments, balanced across assessment types.

---

### 2. Data Ingestion

- Custom web scraper crawls all pages of the SHL product catalogue.
- For each of the **518 assessments**, the scraper visits the detail page and extracts:
  - Full description
  - Completion time
  - Applicable job levels
  - Supported languages
  - Test type classifications
- Each assessment is converted into a rich text representation combining name, description, test types, and job levels.
- These representations are embedded using **OpenAI text-embedding-3-large** (3072-dimensional vectors) and stored in a **FAISS inner-product index**.
- A **BM25 keyword index** is built in parallel over the same corpus with name-field boosting for exact product name matching.

---

### 3. Recommendation Pipeline

The engine is a three-stage pipeline orchestrated using **LangGraph**, with typed state flowing between independently testable nodes.

**Stage 1 - Query Analysis**

- An LLM receives the user's input and performs structured decomposition.
- Generates **15-20 search queries** designed for multi-query retrieval:
  - Keyword-style queries targeting specific SHL product names (optimized for BM25)
  - Descriptive natural language queries (optimized for FAISS semantic search)
- Also extracts: mentioned skills, duration constraints, and domain classification.

**Stage 2 - Multi-Query Hybrid Retrieval**

- Each generated search query is executed against both retrieval systems simultaneously:
  - **FAISS semantic search** — captures meaning-based relevance
  - **BM25 keyword search** — captures exact name-based matches
- Results from all queries are merged using a **score fusion strategy**:
  - Combines per-assessment max-relevance and cumulative breadth scores
  - Additional boosting for assessments consistently retrieved across multiple queries
- Narrows 518 assessments down to approximately **70 high-confidence candidates**.

**Stage 3 - LLM Reranking**

- An LLM evaluates the 70 candidates in context and selects exactly **10 assessments**.
- Selection principles include:
  - Every explicitly named skill gets a corresponding test
  - Breadth maintained across skill areas
  - Role-fit solution packages included based on seniority level
  - Hard-skill and soft-skill assessment types balanced
  - Distinct product versions treated as separate valid options
- Representative examples across diverse role types guide the LLM's selection patterns.

---

### 4. Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Pipeline Orchestration | LangGraph | Typed state graph with independently testable nodes |
| Large Language Model | GPT-4.1 / Gemini 3.0 Flash | Configurable via environment variable |
| Embeddings | OpenAI text-embedding-3-large | 3072-dim vectors; used regardless of LLM provider |
| Vector Store | FAISS (IndexFlatIP) | Fast inner-product similarity, no external infra needed |
| Keyword Search | BM25 (rank-bm25) | Complements semantic search with exact name matches |
| API Framework | FastAPI | Async handling, OpenAPI docs, Pydantic validation |
| Frontend | Streamlit | Interactive UI with sample queries and result cards |

---

### 5. Evaluation and Iteration

Evaluated using **Mean Recall@10** on a labeled training set (10 queries, 65 total relevant assessments).

| Version | Description | Mean Recall@10 |
|---------|-------------|---------------|
| v1 | Baseline FAISS semantic search | 0.35 |
| v2 | Hybrid retrieval with BM25 and name boosting | 0.45 |
| v3 | Multi-query generation via LLM query analyzer | 0.52 |
| v4 | LLM reranker with assessment selection principles | 0.58 |
| v5 | Score fusion optimization with hit bonuses and guaranteed slots | 0.62 |
| v6 | Refined reranker with variant awareness and balanced examples | **0.68** |

**Key Insights:**

- Diagnosed each query separately: retriever pool coverage vs. reranker selection accuracy.
- 95% of relevant assessments reach the candidate pool — retriever coverage is not the bottleneck.
- Primary improvement opportunity lies in reranker prompt quality and selection strategy.
- Example: Entry-level sales query (9 relevant assessments) improved from 0.33 to 0.56 recall after refining the reranker to prioritize role-specific packages.

---

### 6. Deliverables

| Deliverable | Location | Description |
|------------|----------|-------------|
| REST API | `app.py` | FastAPI service with `/health` and `/recommend` endpoints |
| Web Frontend | `streamlit/streamlit_app.py` | Interactive Streamlit app with sample queries and result cards |
| Core Pipeline | `core/graph.py` | LangGraph pipeline: Query Analyzer + Hybrid Retriever + LLM Reranker |
| Data Scraper | `core/scraper.py` | SHL catalogue scraper producing 518 assessments |
| Index Builder | `core/embeddings.py` | FAISS vector index and BM25 keyword index construction |
| Evaluation | `evaluate.py` | Parallelized train-set evaluation with Recall@K metrics |
| Predictions | `output/predictions.csv` | Test set predictions (9 queries, 90 rows) |
| Configuration | `config.py` | LLM provider toggle, model selection, pipeline hyperparameters |
