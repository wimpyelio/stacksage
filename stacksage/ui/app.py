"""StackSage Streamlit UI — full implementation."""
import os, time
import httpx
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")
GRAFANA = "http://localhost:3000"

st.set_page_config(page_title="StackSage ⚡", page_icon="⚡", layout="wide")

PIPELINE_DIAGRAM = """
Query → [Rewriter] → 2-3 sub-queries
                          │
              ┌───────────┴───────────┐
         [BM25/ES]              [Dense/Qdrant]
              └───────────┬───────────┘
                     [RRF Fusion]
                          │
                  [Metadata Filter]
                  tags · score · date
                          │
                [Cross-Encoder Rerank]
                          │
                   [LLM Generator]
                          │
                   [LLM-as-Judge]
                   rel · acc · comp
"""

TAG_OPTIONS = ["python","pandas","fastapi","sqlalchemy","pydantic","numpy","matplotlib","django","flask","pytest"]
DATE_MAP    = {"Any": None, "2020+": "2020-01-01", "2022+": "2022-01-01",
               "2023+": "2023-01-01", "2024+": "2024-01-01"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ StackSage")
    st.caption("Stack Overflow Intelligence Engine")

    # Active model info
    try:
        h = httpx.get(f"{API}/health", timeout=3)
        st.success("API online") if h.status_code == 200 else st.warning("API degraded")
    except:
        st.error("API offline")

    llm_provider = os.getenv("LLM_API_KEY","")
    provider_label = "Ollama (local)" if not llm_provider or llm_provider=="your_api_key_here"                      else f"OpenAI-compat ({os.getenv('LLM_BASE_URL','').split('/')[2]})"
    st.caption(f"🤖 LLM: `{provider_label}`")
    st.caption(f"📐 Generation: `{os.getenv('LLM_GENERATION_MODEL','llama-3.1-8b-instant')}`")
    st.caption(f"⚖️  Judge: `{os.getenv('LLM_JUDGE_MODEL','llama-3.1-70b-versatile')}`")
    st.divider()

    selected_tags = st.multiselect("Filter by tags", TAG_OPTIONS, default=[])
    min_score     = st.slider("Min question score", 0, 100, 5)
    date_label    = st.selectbox("Posts after", list(DATE_MAP.keys()))
    after_date    = DATE_MAP[date_label]
    top_k         = st.slider("Sources to retrieve", 1, 10, 5)
    st.divider()

    st.markdown(f"[📊 Open Grafana Dashboard]({GRAFANA})", unsafe_allow_html=True)
    st.divider()

    if st.button("Refresh metrics"):
        try:
            m = httpx.get(f"{API}/metrics", timeout=5).json()
            st.metric("Total queries",   m.get("total_queries", 0))
            st.metric("Avg relevance",   m.get("avg_judge_relevance", "—"))
            st.metric("Avg latency ms",  m.get("avg_total_latency_ms", "—"))
            st.metric("👍 rate %",       m.get("feedback_positive_rate", "—"))
        except Exception as e:
            st.warning(f"Metrics unavailable: {e}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_query, tab_eval, tab_pipeline = st.tabs(["🔍 Ask", "📊 Evaluate", "🗺 Pipeline"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Query
# ═════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.header("Ask a technical question")
    question = st.text_area("Question", height=80,
                            placeholder="How do I merge two dicts in Python 3.9+?")

    if st.button("Ask StackSage ⚡", type="primary", disabled=not question.strip()):
        payload = {
            "question":       question.strip(),
            "tags":           selected_tags,
            "min_vote_score": min_score,
            "after_date":     after_date,
            "top_k":          top_k,
        }
        with st.spinner("Running 7-stage RAG pipeline…"):
            try:
                resp = httpx.post(f"{API}/query", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API error: {e}"); st.stop()

        # Answer
        st.subheader("Answer")
        st.markdown(data["answer"])

        # Judge scores
        js = data.get("judge_scores") or {}
        if any(js.get(k) for k in ("relevance","accuracy","completeness")):
            c1,c2,c3 = st.columns(3)
            c1.metric("Relevance",    f"{js.get('relevance','—')}/5")
            c2.metric("Accuracy",     f"{js.get('accuracy','—')}/5")
            c3.metric("Completeness", f"{js.get('completeness','—')}/5")
            if js.get("reasoning"): st.caption(f"Judge: {js['reasoning']}")

        # Sources
        with st.expander(f"📚 Sources ({len(data.get('sources',[]))})", expanded=True):
            for i,s in enumerate(data.get("sources",[]),1):
                st.markdown(
                    f"**{i}. [{s['question_title']}]({s['question_url']})** "
                    f"— score {s['question_score']} | reranker {s.get('reranker_score',0):.3f}"
                )
                if s.get("tags"): st.caption("Tags: " + ", ".join(s["tags"]))

        # Sub-queries
        rq = data.get("rewritten_queries",[])
        if rq:
            with st.expander("🔄 Rewritten sub-queries"):
                for q in rq: st.markdown(f"- {q}")

        # Timings
        tim = data.get("timings",{})
        with st.expander("⏱ Pipeline timings"):
            cols = st.columns(6)
            for col,(k,lab) in zip(cols,[
                ("rewrite_ms","Rewrite"),("retrieval_ms","Retrieval"),
                ("reranker_ms","Reranker"),("generation_ms","Generation"),
                ("judge_ms","Judge"),("total_ms","Total"),
            ]):
                col.metric(lab, f"{tim.get(k,0)} ms")

        # Feedback
        st.divider()
        st.caption("Was this helpful?")
        sid = data.get("session_id","")
        fc1,fc2,_ = st.columns([1,1,8])
        if fc1.button("👍"):
            httpx.post(f"{API}/feedback", json={"session_id":sid,"feedback":1}, timeout=5)
            st.toast("Thanks! 👍")
        if fc2.button("👎"):
            httpx.post(f"{API}/feedback", json={"session_id":sid,"feedback":-1}, timeout=5)
            st.toast("Thanks for the feedback.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation
# ═════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("Retrieval Evaluation")
    st.caption("Runs BM25-only, Dense-only, and Hybrid+Reranking against ground truth.")

    n_q = st.slider("Number of questions", 10, 100, 50)

    if st.button("▶ Run Retrieval Evaluation", type="primary"):
        with st.spinner("Evaluating… this takes a few minutes"):
            try:
                from dotenv import load_dotenv; load_dotenv()
                from ingestion.embed_and_index import get_qdrant, get_es, get_embedder
                from rag.retriever import HybridRetriever
                from evaluation.evaluate import evaluate_retrieval

                retriever = HybridRetriever(get_qdrant(), get_es(), get_embedder())
                results   = evaluate_retrieval(
                    "data/eval/ground_truth.jsonl", retriever, n=n_q
                )
                st.success("Evaluation complete!")

                # Table
                rows = []
                for method, v in results.items():
                    rows.append({"Method": method,
                                 "Hit@5":  v["hit_rate_at_5"],
                                 "Hit@10": v["hit_rate_at_10"],
                                 "MRR":    v["mrr"]})
                st.dataframe(rows, use_container_width=True)

                # Bar chart
                import pandas as pd
                df = pd.DataFrame(rows).set_index("Method")
                st.bar_chart(df[["Hit@5","Hit@10","MRR"]])

            except FileNotFoundError:
                st.error("data/eval/ground_truth.jsonl not found. Run `make ingest-sede` first.")
            except Exception as e:
                st.error(f"Evaluation error: {e}")

    # Show cached results if present
    cached = Path("evaluation/results/retrieval_metrics.json")
    if cached.exists():
        with st.expander("Last saved results"):
            import json as _json
            st.json(_json.loads(cached.read_text()))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Pipeline diagram
# ═════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.header("7-Stage RAG Pipeline")
    st.code(PIPELINE_DIAGRAM, language=None)
    st.markdown("""
| Stage | Component | Detail |
|-------|-----------|--------|
| 1 | **Query Rewriter** | LLM → 2-3 sub-queries |
| 2 | **BM25 Search** | Elasticsearch multi-match |
| 3 | **Dense Search** | Qdrant ANN (all-MiniLM-L6-v2) |
| 4 | **RRF Fusion** | score = Σ 1/(60+rank) |
| 5 | **Metadata Filter** | tags · score · date (pushed to stores) |
| 6 | **Cross-Encoder Rerank** | ms-marco-MiniLM-L-6-v2 |
| 7 | **LLM Generation** | Any OpenAI-compatible / Ollama |
| 8 | **LLM-as-Judge** | relevance · accuracy · completeness |
""")
