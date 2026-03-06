# SHL Assessment Recommendation Engine

## Approach Document

---

### 1. Problem Statement

Hiring managers and recruiters face a significant challenge when selecting the right assessments from SHL's extensive product catalogue, which contains over 500 products spanning knowledge tests, cognitive aptitude measures, personality inventories, coding simulations, writing simulations, and pre-packaged role solutions. The existing catalogue interface relies on keyword filters and manual browsing, making it time-consuming and prone to missing relevant assessments, particularly when a role requires a balanced combination of technical and behavioral evaluation.

This project builds an intelligent recommendation system that accepts a natural language hiring query or a complete job description and returns the top 10 most relevant SHL assessments, properly balanced across assessment types.

---

### 2. Data Ingestion

The system is built on a structured dataset scraped directly from SHL's product catalogue website.

A custom web scraper crawls all pages of the SHL catalogue, covering both Individual Test Solutions and Pre-packaged Job Solutions. For each assessment, the scraper visits the dedicated detail page to extract the full description, completion time, applicable job levels, supported languages, and test type classifications. This process yields 518 unique assessments with complete metadata.

Each assessment is then converted into a rich text representation combining its name, description, test types, and job levels. These representations are embedded using OpenAI's text-embedding-3-large model, producing 3072-dimensional vectors stored in a FAISS inner-product index for efficient similarity search. In parallel, a BM25 keyword index is constructed over the same corpus with name-field boosting to ensure exact product name matches receive higher relevance scores.

---

### 3. Recommendation Pipeline

The recommendation engine is implemented as a three-stage pipeline orchestrated using LangGraph, with typed state flowing between independently testable nodes.

**Stage 1 — Query Analysis.** An LLM receives the user's input and performs structured decomposition. It generates 15 to 20 search queries designed for a multi-query retrieval strategy: a mix of keyword-style queries targeting specific SHL product names (optimized for BM25 matching) and descriptive natural language queries (optimized for FAISS semantic matching). The analyzer also extracts mentioned skills, duration constraints, and a domain classification. This multi-query approach ensures broad coverage across the assessment catalogue from a single user input.

**Stage 2 — Multi-Query Hybrid Retrieval.** Each of the generated search queries is executed against both retrieval systems simultaneously. FAISS semantic search captures meaning-based relevance, matching queries like "data analysis capabilities" to assessments such as Tableau, Data Warehousing, and Microsoft Excel 365. BM25 keyword search captures exact name-based matches that semantic search may rank lower. The results from all queries across both systems are merged using a score fusion strategy that combines per-assessment max-relevance and cumulative breadth scores, with additional boosting for assessments consistently retrieved across multiple queries. This multi-query hybrid retrieval narrows 518 assessments to approximately 70 high-confidence candidates.

**Stage 3 — LLM Reranking.** An LLM evaluates the 70 candidate assessments in context and selects exactly 10. The selection prompt encodes general principles for building balanced assessment batteries: ensuring every explicitly named skill has a corresponding test, maintaining breadth across skill areas, including role-fit solution packages appropriate to the seniority level, balancing hard-skill and soft-skill assessment types, and treating distinct product versions as separate valid options. Representative examples spanning technical, sales, executive, administrative, analytical, and customer-facing roles guide the LLM to learn selection patterns applicable to any query.

---

### 4. Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Pipeline Orchestration | LangGraph | Typed state graph with independently testable nodes; straightforward to extend with additional stages |
| Large Language Model | GPT-4.1 / Gemini 3.0 Flash | Configurable via environment variable; GPT-4.1 for highest accuracy, Gemini for free-tier deployment |
| Embeddings | OpenAI text-embedding-3-large | 3072-dimensional vectors provide fine-grained similarity; used consistently regardless of LLM provider |
| Vector Store | FAISS (IndexFlatIP) | Fast inner-product similarity search with no external infrastructure requirement |
| Keyword Search | BM25 (rank-bm25) | Complements semantic search by capturing exact assessment name matches |
| API Framework | FastAPI | Asynchronous request handling, automatic OpenAPI documentation, Pydantic validation |
| Frontend | Streamlit | Interactive demonstration interface with sample queries, structured result cards, and tabular views |

---

### 5. Evaluation and Iteration

The system was evaluated using Mean Recall@10 on a labeled training set containing 10 queries with 65 total relevant assessments.

| Version | Description | Mean Recall@10 |
|---------|-------------|---------------|
| v1 | Baseline FAISS semantic search | 0.35 |
| v2 | Hybrid retrieval with BM25 and name boosting | 0.45 |
| v3 | Multi-query generation via LLM query analyzer | 0.52 |
| v4 | LLM reranker with assessment selection principles | 0.58 |
| v5 | Score fusion optimization with hit bonuses and guaranteed slots | 0.62 |
| v6 | Refined reranker with variant awareness and balanced examples | **0.68** |

To guide iteration, each query was diagnosed separately by measuring retriever pool coverage (how many relevant assessments appear in the top 70 candidates) versus reranker selection accuracy (how many the LLM actually picks). This analysis revealed that 95 percent of relevant assessments reach the candidate pool, confirming that retriever coverage is not the bottleneck. The primary improvement opportunity lies in reranker prompt quality and selection strategy.

As a concrete example, the entry-level sales query has 9 relevant assessments. Early versions selected only 3 sales-related packages. After refining the reranker to prioritize exhausting all matching role-specific packages before adding generic assessments, recall for that query improved from 0.33 to 0.56.

---

### 6. Deliverables

| Deliverable | Location | Description |
|------------|----------|-------------|
| REST API | `app.py` | FastAPI service with `/health` and `/recommend` endpoints matching the required specification |
| Web Frontend | `streamlit/streamlit_app.py` | Interactive Streamlit application with sample queries, assessment cards, and table export |
| Core Pipeline | `core/graph.py` | LangGraph pipeline implementing Query Analyzer, Multi-Query Hybrid Retriever, and LLM Reranker |
| Data Scraper | `core/scraper.py` | SHL catalogue scraper producing 518 assessments with complete descriptions |
| Index Builder | `core/embeddings.py` | FAISS vector index and BM25 keyword index construction |
| Evaluation Script | `evaluate.py` | Parallelized train-set evaluation with Recall@K metrics and test-set prediction generation |
| Test Predictions | `output/predictions.csv` | Predictions for 9 unlabeled test queries (90 rows in the required submission format) |
| Configuration | `config.py` | Centralized settings including LLM provider toggle, model selection, and pipeline hyperparameters |
