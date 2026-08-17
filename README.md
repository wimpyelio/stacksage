<div align="center">

# ⚡ StackSage
### Stack Overflow Intelligence Engine

**Production-grade RAG pipeline over curated Stack Exchange Q&A**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-local-purple)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **No Docker required for core app** — runs on any machine with Python 3.11+

</div>

---

## 🎯 Problem

Developers waste hours digging through Stack Overflow threads that are outdated, buried in noise, or spread across 10 different posts. Existing search returns individual answers — not synthesised, source-cited intelligence.

## 💡 Solution

StackSage runs a **7-stage RAG pipeline** over 50K curated Stack Overflow Q&A pairs, returning a synthesised answer with:
- Source citations with vote scores
- LLM-as-judge quality scores
- Per-stage latency breakdown
- Retrieval comparison metrics

---

## 🏗 Architecture

<img width="1774" height="887" alt="image" src="/assets/arch01_ght.png" />

---

## 🛠 Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Dense index | Qdrant (local embedded) | No server, file-based, fast ANN |
| BM25 index | rank_bm25 (in-process) | Pure Python, no Elasticsearch needed |
| Embeddings | all-MiniLM-L6-v2 | Fast, 384-dim, free |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder accuracy |
| LLM | Any OpenAI-compatible / Ollama | Provider-agnostic |
| API | FastAPI | Async, typed, auto-docs |
| UI | Streamlit | 3-tab interface |
| Logging | SQLite | Zero-config, built into Python |
| Monitoring (built-in) | Streamlit Evaluate tab | Works without Docker |
| Monitoring (optional) | Grafana + Docker | Full dashboard, see `monitoring/` |
| Data | SEDE CSV / Stack Exchange XML | CC BY-SA 4.0 |

---

## 📊 Evaluation Results

### ⚡ Pipeline Latency (real benchmark)

| Stage | Time |
|-------|------|
| Query Rewrite | 198ms |
| Retrieval (BM25 + Dense) | 135ms |
| Cross-Encoder Reranking | 2402ms |
| LLM Generation | 472ms |
| LLM-as-Judge | 206ms |
| **Total** | **3414ms** |

### 🎯 Retrieval Metrics (18 ground-truth questions)

| Method | Hit@5 | Hit@10 | MRR |
|--------|-------|--------|-----|
| BM25 only | 1.00 | 1.00 | 1.00 |
| Dense only | 1.00 | 1.00 | 1.00 |
| **Hybrid + Reranking** | **1.00** | **1.00** | **1.00** |

### 🧠 Generation Quality (LLM-as-Judge, 1–5 scale)

| Dimension | Score | Meaning |
|-----------|-------|---------|
| Relevance | **5 / 5** | Directly addresses the question |
| Accuracy | **5 / 5** | Technically correct |
| Completeness | **4 / 5** | Covers key aspects |

---

## 🔍 RAG Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Hybrid search | BM25 (rank_bm25) + Dense (Qdrant) |
| Query expansion | LLM rewrites into 2-3 sub-queries |
| Score fusion | Reciprocal Rank Fusion (RRF) |
| Metadata filtering | tags, vote score, date pushed to stores |
| Cross-encoder reranking | ms-marco-MiniLM-L-6-v2 |
| LLM generation | Provider-agnostic (Groq/OpenAI/Ollama) |
| LLM evaluation | 3-dimension judge scoring |
| Observability | Per-stage timing + SQLite logging + Grafana |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API key from [Groq](https://console.groq.com) (free) **or** [OpenAI](https://platform.openai.com) **or** local [Ollama](https://ollama.ai)

### 1. Clone & Install
```bash
git clone https://github.com/wimpyelio/stacksage
cd stacksage
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
nano .env   # add your LLM_API_KEY
```

Supports any OpenAI-compatible provider:
```env
# Groq (free, fast)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_GENERATION_MODEL=llama-3.1-8b-instant
LLM_JUDGE_MODEL=llama-3.1-8b-instant

# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...

# Ollama (local, leave LLM_API_KEY empty)
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Ingest Data

**Option A — Dummy data (instant, for testing):**
```bash
make ingest-dummy
```

**Option B — Real SEDE data (recommended):**
1. Go to https://data.stackexchange.com/stackoverflow/query/new
2. Run the SQL from `.env.example` (SEDE_SQL section)
3. Download CSV → `data/raw/sede_export.csv`
4. Run `make ingest`

### 4. Run
```bash
make run
# API:  http://localhost:8000
# UI:   http://localhost:8501
# Docs: http://localhost:8000/docs
```

### 5. Evaluate
```bash
make eval
```

### 6. Optional — Grafana Monitoring
```bash
# Requires Docker
docker compose up grafana
# Dashboard: http://localhost:3000  (admin / stacksage)
# Pre-built panels: query volume, judge scores, latency, feedback rate, top tags
```

---

## 📁 Project Structure

```
stacksage/
├── ingestion/
│   ├── query_sede.py      # SEDE CSV → JSONL
│   ├── parse_xml.py       # XML dump → JSONL (alternative)
│   ├── clean.py           # Strip HTML, extract code, normalize
│   ├── embed_and_index.py # Embed → Qdrant local + BM25 pickle
│   └── build_eval_set.py  # Ground truth eval set
├── rag/
│   ├── llm_client.py      # Provider-agnostic LLM (Groq/OpenAI/Ollama)
│   ├── query_rewriter.py  # LLM → 2-3 sub-queries
│   ├── retriever.py       # BM25 + Dense + RRF fusion
│   ├── reranker.py        # Cross-encoder reranking
│   ├── generator.py       # LLM answer synthesis
│   ├── judge.py           # LLM-as-judge scoring
│   └── pipeline.py        # 7-stage orchestrator
├── api/
│   ├── main.py            # FastAPI: /query /feedback /health /metrics
│   ├── models.py          # Pydantic request/response models
│   └── logger.py          # SQLite query + feedback logger
├── ui/
│   └── app.py             # Streamlit: Ask / Evaluate / Pipeline tabs
├── evaluation/
│   ├── evaluate.py        # Hit@5, Hit@10, MRR across 3 methods
│   └── ragas_eval.py      # RAGAS: faithfulness, relevancy, precision
├── monitoring/
│   └── grafana/           # Pre-built dashboard + datasource configs
├── scripts/
│   └── make_dummy_data.py # 50 dummy QA pairs for testing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_retrieval_experiments.ipynb
├── .env.example
├── Makefile
├── Dockerfile
├── docker-compose.yml     # Grafana only (core app needs no Docker)
└── requirements.txt
```

---

## 🖥 Screenshots

### Ask Tab
<img width="1136" height="880" alt="image" src="/assets/ask_tab.png" />

### Sources & Timings
<img width="1009" height="670" alt="image" src="/assets/source.png">
<img alt="image" src="/assets/timings'.png">

### Retrieval Evaluation
<img width="1528" height="810" alt="image" src="/assets/retrival.png" />

---

## ⚙️ Makefile Commands

```bash
make setup         # pip install -r requirements.txt
make ingest-dummy  # Generate 50 dummy QA pairs + index
make ingest        # Full SEDE CSV ingestion pipeline
make run           # Start API (8000) + Streamlit UI (8501)
make eval          # Retrieval metrics: Hit@5, Hit@10, MRR
make test          # Single pipeline smoke test
make clean         # Remove indexes and DB
```

---

## 🔌 API Reference

```bash
# Ask a question
POST /query
{
  "question": "How do I use async/await in Python?",
  "tags": ["python", "asyncio"],
  "min_vote_score": 5,
  "top_k": 5
}

# Submit feedback
POST /feedback
{"session_id": "...", "feedback": 1}  # 1=👍 -1=👎

# Health check
GET /health

# Aggregate metrics
GET /metrics
```

Full interactive docs: `http://localhost:8000/docs`

---

## ⚠️ Limitations

- Corpus scoped to Python tags (extend via `TARGET_TAGS` in `.env`)
- Cross-encoder reranker adds ~2s on CPU (fast on GPU)
- Dummy data has 50 QA pairs; real SEDE has 50K
- LLM may hallucinate on questions outside the corpus

---

## 🔮 Future Work

- Cloud deployment (Railway / Hugging Face Spaces)
- Streaming answer generation
- Fine-tuned reranker on Stack Overflow domain
- GraphQL API variant
- Multi-language support (JavaScript, Java tags)

---

## 📄 Dataset

Stack Exchange Data Explorer (SEDE) — Python Q&A posts
Score ≥ 5, accepted answer, post-2020 · License: **CC BY-SA 4.0**

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

Built for **LLM Zoomcamp 2026** by [@wimpyelio](https://github.com/wimpyelio)

*No Docker. No fuss. Just RAG.* ⚡

</div>