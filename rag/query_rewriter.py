"""Expand a user question into sub-queries via LLM."""
import json, logging
from rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

_PROMPT = """You are a Stack Overflow search expert.
Rewrite the developer question into 2-3 focused sub-queries for retrieval.

Rules:
- Each sub-query targets a distinct aspect of the question
- Use technical terms found on Stack Overflow
- Max 20 words per sub-query
- Return ONLY a JSON array of strings — nothing else

Question: {question}

JSON array:"""


class QueryRewriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, question: str) -> list[str]:
        """Return 2-3 retrieval-optimised sub-queries.
        Falls back to [question] on any error."""
        try:
            raw = self.llm.complete(_PROMPT.format(question=question),
                                    temperature=0.3, max_tokens=256)
            # Strip markdown fences if model added them
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            queries = json.loads(raw)
            if isinstance(queries, list) and queries:
                logger.debug("Rewrote to %d sub-queries", len(queries))
                return [str(q).strip() for q in queries if str(q).strip()]
        except Exception as e:
            logger.warning("Query rewrite failed (%s) — using original", e)
        return [question]
