"""StackSage pipeline — no Docker needed."""
import logging, os, time, uuid
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from rag.llm_client      import LLMClient
from rag.query_rewriter  import QueryRewriter
from rag.retriever       import HybridRetriever
from rag.reranker        import CrossEncoderReranker
from rag.generator       import AnswerGenerator
from rag.judge           import LLMJudge
from ingestion.embed_and_index import get_embedder, get_qdrant

logger = logging.getLogger(__name__)

def _ms(t): return int((time.perf_counter()-t)*1000)


class StackSagePipeline:
    def __init__(self) -> None:
        gen_llm   = LLMClient(role="generation")
        judge_llm = LLMClient(role="judge")
        self.rewriter  = QueryRewriter(gen_llm)
        self.retriever = HybridRetriever(get_qdrant(), get_embedder())
        self.reranker  = CrossEncoderReranker()
        self.generator = AnswerGenerator(gen_llm)
        self.judge     = LLMJudge(judge_llm)
        logger.info("StackSagePipeline ready (no Docker needed)")

    def run(self, question: str, tags: Optional[list[str]]=None,
            min_vote_score: int=5, after_date: Optional[str]=None, top_k: int=5) -> dict:
        sid = str(uuid.uuid4()); t0 = time.perf_counter(); tim = {}
        t = time.perf_counter()
        try:    queries = self.rewriter.rewrite(question)
        except: queries = [question]
        tim["rewrite_ms"] = _ms(t)
        t = time.perf_counter()
        candidates = self.retriever.retrieve(queries,tags,min_vote_score,after_date,40)
        tim["retrieval_ms"] = _ms(t)
        t = time.perf_counter()
        sources = self.reranker.rerank(question, candidates, top_k=top_k)
        tim["reranker_ms"] = _ms(t)
        t = time.perf_counter()
        answer = self.generator.generate(question, sources)
        tim["generation_ms"] = _ms(t)
        t = time.perf_counter()
        judge_scores = self.judge.score(question, answer)
        tim["judge_ms"] = _ms(t)
        tim["total_ms"] = _ms(t0)
        logger.info("[%s] %dms | docs=%d", sid[:8], tim["total_ms"], len(sources))
        return {"answer":answer,"sources":sources,"rewritten_queries":queries,
                "judge_scores":judge_scores,"timings":tim,"session_id":sid}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    p = StackSagePipeline()
    r = p.run("How do I use async context managers in Python?")
    print("\n=== ANSWER ===\n", r["answer"][:500])
    print("=== JUDGE ===", r["judge_scores"])
    print("=== TIMING ===", r["timings"])
