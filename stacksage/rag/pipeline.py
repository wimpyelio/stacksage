"""Top-level StackSage pipeline — wires all 7 RAG stages."""
import logging, os, time, uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from rag.llm_client    import LLMClient
from rag.query_rewriter import QueryRewriter
from rag.retriever      import HybridRetriever
from rag.reranker       import CrossEncoderReranker
from rag.generator      import AnswerGenerator
from rag.judge          import LLMJudge
from ingestion.embed_and_index import get_embedder, get_qdrant, get_es

logger = logging.getLogger(__name__)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class StackSagePipeline:
    """
    7-stage RAG pipeline:
    Query Rewriting → Hybrid Retrieval → RRF → Metadata Filter
    → Cross-Encoder Reranking → LLM Generation → LLM-as-Judge

    All stage failures are caught — the pipeline never raises.
    """

    def __init__(self) -> None:
        gen_llm   = LLMClient(role="generation")
        judge_llm = LLMClient(role="judge")

        self.rewriter  = QueryRewriter(gen_llm)
        self.retriever = HybridRetriever(get_qdrant(), get_es(), get_embedder())
        self.reranker  = CrossEncoderReranker()
        self.generator = AnswerGenerator(gen_llm)
        self.judge     = LLMJudge(judge_llm)
        logger.info("StackSagePipeline ready  (gen=%s  judge=%s)",
                    gen_llm.active_model, judge_llm.active_model)

    def run(self, question: str, tags: Optional[list[str]] = None,
            min_vote_score: int = 5, after_date: Optional[str] = None,
            top_k: int = 5) -> dict:
        """Execute full pipeline.

        Returns:
            answer            : str
            sources           : list[dict]
            rewritten_queries : list[str]
            judge_scores      : dict  {relevance, accuracy, completeness, reasoning}
            timings           : dict  per-stage ms + total_ms
            session_id        : str
        """
        session_id  = str(uuid.uuid4())
        t_total     = time.perf_counter()
        timings: dict[str, int] = {}

        # ── Stage 1: Query Rewriting ──────────────────────────────────────────
        t = time.perf_counter()
        try:    queries = self.rewriter.rewrite(question)
        except Exception as e:
            logger.error("Rewrite error: %s", e); queries = [question]
        timings["rewrite_ms"] = _ms(t)

        # ── Stage 2-3: Hybrid Retrieval + RRF ────────────────────────────────
        t = time.perf_counter()
        try:
            candidates = self.retriever.retrieve(
                queries, tags=tags, min_vote_score=min_vote_score,
                after_date=after_date, top_k=40,
            )
        except Exception as e:
            logger.error("Retrieval error: %s", e); candidates = []
        timings["retrieval_ms"] = _ms(t)

        # ── Stage 4: Metadata filter already applied inside retriever ─────────
        # (question_score, tags, after_date filters pushed to ES+Qdrant)

        # ── Stage 5: Cross-Encoder Reranking ─────────────────────────────────
        t = time.perf_counter()
        try:    sources = self.reranker.rerank(question, candidates, top_k=top_k)
        except Exception as e:
            logger.error("Reranker error: %s", e); sources = candidates[:top_k]
        timings["reranker_ms"] = _ms(t)

        # ── Stage 6: LLM Generation ───────────────────────────────────────────
        t = time.perf_counter()
        answer = self.generator.generate(question, sources)
        timings["generation_ms"] = _ms(t)

        # ── Stage 7: LLM-as-Judge ─────────────────────────────────────────────
        t = time.perf_counter()
        judge_scores = self.judge.score(question, answer)
        timings["judge_ms"] = _ms(t)

        timings["total_ms"] = _ms(t_total)
        logger.info(
            "[%s] done — %dms | judge=%s/%s/%s | docs=%d",
            session_id[:8], timings["total_ms"],
            judge_scores.get("relevance"), judge_scores.get("accuracy"),
            judge_scores.get("completeness"), len(sources),
        )
        return {
            "answer":             answer,
            "sources":            sources,
            "rewritten_queries":  queries,
            "judge_scores":       judge_scores,
            "timings":            timings,
            "session_id":         session_id,
        }


# Quick smoke-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    p = StackSagePipeline()
    r = p.run("How do I use async context managers in Python?")
    print("\n=== ANSWER ===")
    print(r["answer"][:800])
    print("\n=== JUDGE ===", r["judge_scores"])
    print("=== TIMINGS ===", r["timings"])
