"""Retrieval + generation evaluation for StackSage.

Usage:
    python evaluation/evaluate.py \
        --ground-truth data/eval/ground_truth.jsonl \
        --output evaluation/results/ \
        --n-questions 100
"""
import argparse, json, logging, time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
OUT    = Path("evaluation/results")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def _hit(retrieved_ids: list[str], gold_id: str, k: int) -> int:
    return int(gold_id in retrieved_ids[:k])

def _rr(retrieved_ids: list[str], gold_id: str) -> float:
    try:    return 1.0 / (retrieved_ids.index(gold_id) + 1)
    except: return 0.0

def evaluate_retrieval(
    ground_truth_path: str,
    retriever,           # HybridRetriever instance
    n: int = 100,
    min_vote_score: int = 5,
) -> dict:
    """Compute Hit@5, Hit@10, MRR for BM25-only, Dense-only, and Hybrid+Rerank.

    For each ground-truth question we treat the gold question_id as the
    relevant document and measure whether retrieval surfaces it.
    """
    from rag.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()

    gt = _load_gt(ground_truth_path, n)
    results = {m: {"hit5":[],"hit10":[],"rr":[]}
               for m in ("bm25","dense","hybrid")}

    for i, item in enumerate(gt):
        q   = item["question_title"]
        gid = item["question_id"]

        bm25  = retriever.search_bm25 (q, [], min_vote_score, None, 10)
        dense = retriever.search_dense(q, [], min_vote_score, None, 10)
        fused = retriever.reciprocal_rank_fusion(bm25, dense)
        reranked = reranker.rerank(q, fused[:20], top_k=10)

        for method, docs in [("bm25",bm25),("dense",dense),("hybrid",reranked)]:
            ids = [d.get("question_id","") for d in docs]
            results[method]["hit5"].append(_hit(ids, gid, 5))
            results[method]["hit10"].append(_hit(ids, gid, 10))
            results[method]["rr"].append(_rr(ids, gid))

        if (i+1) % 10 == 0:
            logger.info("Retrieval eval: %d/%d", i+1, len(gt))

    summary = {}
    for method, vals in results.items():
        n_q = len(vals["hit5"]) or 1
        summary[method] = {
            "hit_rate_at_5":  round(sum(vals["hit5"])  / n_q, 4),
            "hit_rate_at_10": round(sum(vals["hit10"]) / n_q, 4),
            "mrr":            round(sum(vals["rr"])     / n_q, 4),
            "n_questions":    n_q,
        }
    _save(summary, OUT/"retrieval_metrics.json")
    _print_table(summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Generation metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_generation(
    ground_truth_path: str,
    pipeline,    # StackSagePipeline instance
    n: int = 100,
) -> dict:
    """Run full pipeline on n questions; score each with LLM-as-judge.
    Returns avg relevance, accuracy, completeness + latency stats."""
    gt = _load_gt(ground_truth_path, n)
    scores = {"relevance":[],"accuracy":[],"completeness":[],"total_ms":[]}

    for i, item in enumerate(gt):
        try:
            result = pipeline.run(item["question_title"])
            js  = result.get("judge_scores") or {}
            tim = result.get("timings",{})
            for k in ("relevance","accuracy","completeness"):
                v = js.get(k)
                if v is not None: scores[k].append(float(v))
            if tim.get("total_ms"):
                scores["total_ms"].append(tim["total_ms"])
        except Exception as e:
            logger.error("Gen eval error on item %d: %s", i, e)
        if (i+1) % 10 == 0:
            logger.info("Generation eval: %d/%d", i+1, len(gt))

    def _avg(lst): return round(sum(lst)/len(lst), 4) if lst else None

    summary = {
        "avg_relevance":    _avg(scores["relevance"]),
        "avg_accuracy":     _avg(scores["accuracy"]),
        "avg_completeness": _avg(scores["completeness"]),
        "avg_latency_ms":   _avg(scores["total_ms"]),
        "n_questions":      len(gt),
    }
    _save(summary, OUT/"generation_metrics.json")
    logger.info("Generation metrics: %s", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt(path: str, n: int) -> list[dict]:
    items = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    logger.info("Loaded %d ground-truth items (using %d)", len(items), min(n,len(items)))
    return items[:n]

def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved → %s", path)

def _print_table(summary: dict) -> None:
    print(f"\n{'Method':<10} {'Hit@5':>7} {'Hit@10':>8} {'MRR':>7}")
    print("-" * 35)
    for m, v in summary.items():
        print(f"{m:<10} {v['hit_rate_at_5']:>7.4f} {v['hit_rate_at_10']:>8.4f} {v['mrr']:>7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from dotenv import load_dotenv; load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output",       default=str(OUT))
    ap.add_argument("--n-questions",  type=int, default=100)
    ap.add_argument("--mode", choices=["retrieval","generation","both"], default="both")
    args = ap.parse_args()

    OUT = Path(args.output)

    if args.mode in ("retrieval","both"):
        from ingestion.embed_and_index import get_qdrant, get_es, get_embedder
        from rag.retriever import HybridRetriever
        retriever = HybridRetriever(get_qdrant(), get_es(), get_embedder())
        evaluate_retrieval(args.ground_truth, retriever, args.n_questions)

    if args.mode in ("generation","both"):
        from rag.pipeline import StackSagePipeline
        pipeline = StackSagePipeline()
        evaluate_generation(args.ground_truth, pipeline, args.n_questions)
