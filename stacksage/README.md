# StackSage ⚡ Stack Overflow Intelligence Engine

> Production-grade RAG over Stack Exchange Q&A. No LangChain. No LlamaIndex.

## Problem
Developers waste hours digging through Stack Overflow threads that are
outdated, buried, or irrelevant.  Existing semantic search returns individual
posts without synthesising across the best answers.

## Solution
StackSage runs a 7-stage RAG pipeline over curated Stack Overflow data,
returning a synthesised, source-cited answer with judge-scored quality
and full Grafana observability.

## Architecture
```
User Query
    │
    ▼
┌─────────────────┐
│  Query Rewriter │  LLM → 2-3 sub-queries
└────────┬────────┘
    ┌────┴────┐
    ▼          ▼
┌──────┐  ┌─────────┐
│ BM25 │  │  Dense  │  Elasticsearch + Qdrant
└──────┘  └─────────┘
    └────┬─────┘
         ▼
┌─────────────────┐
│   RRF Fusion    │  Reciprocal Rank Fusion
└────────┬────────┘
         ▼
┌─────────────────┐
│  Metadata Filter│  tags · score · date
└────────┬────────┘
         ▼
┌─────────────────┐
│ Cross-Encoder   │  ms-marco-MiniLM-L-6-v2
│   Reranker      │
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM Generator  │  any OpenAI-compatible API / Ollama
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM-as-Judge   │  relevance · accuracy · completeness
└─────────────────┘
```

## Tech Stack

| Layer       | Tool                           | Why                          |
|-------------|--------------------------------|------------------------------|
| Dense index | Qdrant                         | Fast ANN, payload filters    |
| BM25 index  | Elasticsearch                  | Keyword precision            |
| Embeddings  | all-MiniLM-L6-v2               | Fast, 384-dim, free          |
| Reranker    | ms-marco-MiniLM-L-6-v2         | Cross-encoder accuracy       |
| LLM         | Any OpenAI-compatible / Ollama | Provider-agnostic            |
| API         | FastAPI                        | Async, typed, auto-docs      |
| UI          | Streamlit                      | Rapid, clean interface       |
| Logging     | PostgreSQL                     | Structured query history     |
| Monitoring  | Grafana                        | Live latency + quality dash  |

## Setup

```bash
git clone <repo> && cd stacksage
cp .env.example .env          # set LLM_API_KEY (or leave blank for Ollama)
make setup                    # start services + install deps
# place sede_export.csv in data/raw/  (see SEDE SQL in .env.example)
make ingest-sede
make run
```

- UI:       http://localhost:8501
- API docs: http://localhost:8000/docs
- Grafana:  http://localhost:3000  (admin / stacksage)

## Evaluation Results

> Populate after `make eval`

| Method              | Hit@5 | Hit@10 | MRR |
|---------------------|-------|--------|-----|
| BM25 only           |  —    |  —     |  —  |
| Dense only          |  —    |  —     |  —  |
| Hybrid + Reranking  |  —    |  —     |  —  |

## Limitations
- Corpus scoped to Python tags; extend via TARGET_TAGS
- Cross-encoder adds ~500 ms on CPU — GPU or caching recommended at scale
- LLM may hallucinate on edge-case questions outside corpus
